#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>

using namespace std;

// for first layer
#define IMAGE_SIZE 32
#define FILTER_1_SIZE 5
#define FILTER_1_CHANNELS 6

// for second layer
#define POOL_1_SIZE 2
#define POOL_1_STRIDE 2

// for third layer
#define FILTER_2_SIZE 5
#define FILTER_2_CHANNELS 16

// for fourth layer
#define POOL_2_SIZE 2
#define POOL_2_STRIDE 2

// for fifth layer
#define FILTER_3_SIZE 5
#define FILTER_3_CHANNELS 120

// for sixth layer
#define FC_1_SIZE 84

// for seventh layer
#define FC_2_SIZE 10

// error hadler
void handleError(cudaError_t err, string errMsg)
{
  if (err == cudaSuccess)
    return;
  cout << errMsg << ": " << cudaGetErrorString(err) << '\n';
  exit(EXIT_FAILURE);
}

// read image data from file
bool read_img_data(float *arr)
{

  ifstream infile("image_data.txt");
  if (!infile.good())
  {
    return false;
  }

  for (int i = 0; i < IMAGE_SIZE; i++)
  {
    for (int j = 0; j < IMAGE_SIZE; j++)
    {
      infile >> arr[i * IMAGE_SIZE + j];
    }
  }

  infile.close();
  return true;
}

// image
struct Image
{
  int dim;
  int channels;
  size_t size;
  float *img;

  Image() {}
  Image(int _dim, int _channels)
  {
    dim = _dim;
    channels = _channels;
    size = dim * dim * channels * sizeof(float);
    img = (float *)malloc(size);
    init_image();
  }
  Image(int _dim, int _channels, float *_img)
  {
    dim = _dim;
    channels = _channels;
    size = dim * dim * channels * sizeof(float);
    img = _img;
  }

  void init_image()
  {
    for (int k = 0; k < channels; k++)
      for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
          img[(k * dim * dim) + (i * dim) + j] = rand() % 256;
  }

  void print_image()
  {
    for (int k = 0; k < channels; k++)
    {
      for (int i = 0; i < dim; i++)
      {
        for (int j = 0; j < dim; j++)
          cout << img[(k * dim * dim) + (i * dim) + j] << " ";
      }
      cout << "\n";
    }
    cout << "\n";
  }
};

// filter
struct Filter
{
  int dimxy;
  int dimz;
  int channels;
  int size;
  int *filter;

  Filter() {}
  Filter(int _dimxy, int _dimz, int _channels)
  {
    dimxy = _dimxy;
    dimz = _dimz;
    channels = _channels;
    size = channels * dimz * dimxy * dimxy * sizeof(int);
    filter = (int *)malloc(size);
    init_filter();
  }
  Filter(int _dimxy, int _dimz, int _channels, int *_filter)
  {
    dimxy = _dimxy;
    dimz = _dimz;
    channels = _channels;
    size = channels * dimz * dimxy * dimxy * sizeof(int);
    filter = _filter;
  }

  // initialize filter
  void init_filter()
  {
    for (int z = 0; z < channels; z++)
      for (int k = 0; k < dimz; k++)
        for (int i = 0; i < dimxy; i++)
          for (int j = 0; j < dimxy; j++)
            filter[(z * dimz * dimxy * dimxy) + (k * dimxy * dimxy) + (i * dimxy) + j] = rand() % 3 - 1;
  }
};

// convolutional kernel
__global__ void convolutional_kernel(float *input, int *filter, float *output, int img_size, int img_channels, int filter_size, int filter_channels)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ch_num = blockIdx.z;

  int output_size = img_size - filter_size + 1;

  if ((row >= output_size) || (col >= output_size) || (ch_num >= filter_channels))
  {
    return; // return if thread lie outside the feature map
  }

  float sum = 0.0;
  for (int k = 0; k < img_channels; k++)
  {
    for (int i = 0; i < filter_size; i++)
    {
      for (int j = 0; j < filter_size; j++)
      {
        sum += input[(k * img_size * img_size) + ((row + i) * img_size) + (col + j)] * filter[(ch_num * img_channels * filter_size * filter_size) + (k * filter_size * filter_size) + (i * filter_size) + j];
      }
    }
  }

  // tanh activation
  sum = tanhf(sum);
  output[(ch_num * output_size * output_size) + (row * output_size) + col] = sum;
}

// average pool kernel
__global__ void avg_pool_kernel(float *input, float *output, int img_size, int img_channels, int pool_size, int pool_stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ch_num = blockIdx.z;

  int output_size = img_size / pool_stride;

  if ((row >= output_size) || (col >= output_size) || (ch_num >= img_channels))
    return; // return if thread lie outside feature map

  float sum = 0.0;
  for (int i = 0; i < pool_size; i++)
  {
    for (int j = 0; j < pool_size; j++)
    {
      sum += input[(ch_num * img_size * img_size) + ((row * pool_stride + i) * img_size) + (col * pool_stride + j)];
    }
  }
  sum /= pool_size * pool_size;
  output[(ch_num * output_size * output_size) + (row * output_size) + col] = sum;
}

// hosts to verify the results

// convolutional host
void convo_host(Image *image, Filter *filter, Image *feature_map)
{
  for (int ch_num = 0; ch_num < feature_map->channels; ch_num++)
  {
    for (int row = 0; row < feature_map->dim; row++)
    {
      for (int col = 0; col < feature_map->dim; col++)
      {
        float sum = 0.0;
        for (int k = 0; k < filter->dimz; k++)
        {
          for (int i = 0; i < filter->dimxy; i++)
          {
            for (int j = 0; j < filter->dimxy; j++)
            {
              sum += image->img[(k * image->dim * image->dim) + ((row + i) * image->dim) + (col + j)] * filter->filter[(ch_num * filter->dimz * filter->dimxy * filter->dimxy) + (k * filter->dimxy * filter->dimxy) + (i * filter->dimxy) + j];
            }
          }
        }
        // tanh activation
        sum = tanhf(sum);

        float kernel_value = feature_map->img[(ch_num * feature_map->dim * feature_map->dim) + (row * feature_map->dim) + col];

        if (kernel_value - sum > 0.0001)
        {
          cout << "error at convo2 " << ch_num << " " << row << " " << col << '\n';
          cout << "output: " << kernel_value << " sum: " << sum << '\n';
          return;
        }
      }
    }
  }
}

// avg pool host
void avg_pool_host(Image *image, Image *feature_map, int pool_size, int pool_stride)
{
  for (int ch_num = 0; ch_num < feature_map->channels; ch_num++)
  {
    for (int row = 0; row < feature_map->dim; row++)
    {
      for (int col = 0; col < feature_map->dim; col++)
      {
        float sum = 0;
        for (int i = 0; i < pool_size; i++)
        {
          for (int j = 0; j < pool_size; j++)
          {
            sum += image->img[(ch_num * image->dim * image->dim) + ((row * pool_stride + i) * image->dim) + (col * pool_stride + j)];
          }
        }
        sum /= pool_size * pool_size;
        float kernel_output = feature_map->img[(ch_num * feature_map->dim * feature_map->dim) + (row * feature_map->dim) + col];
        if (kernel_output != sum)
        {
          cout << "error at avg pool2" << ch_num << " " << row << " " << col << '\n';
          cout << "output: " << kernel_output << " sum: " << sum << '\n';
          return;
        }
      }
    }
  }
}

// layers
Image *convolutional_layer(Image *image, int filter_dim, int filter_channels)
{
  Filter *filter = new Filter(filter_dim, image->channels, filter_channels);
  Image *feature_map = new Image(image->dim - filter->dimxy + 1, filter->channels);

  // writing for device
  float *d_img, *d_feature_map;
  int *d_filter;

  // allocating device memory
  handleError(cudaMalloc((void **)&d_img, image->size), "Failed to allocate device memory for input image in convo-layer");

  handleError(cudaMalloc((void **)&d_filter, filter->size), "Failed to allocate device memory for filter in convo-layer");

  handleError(cudaMalloc((void **)&d_feature_map, feature_map->size), "Failed to allocate device memory for convolutional layer feature map");

  // copying data to device
  handleError(cudaMemcpy(d_img, image->img, image->size, cudaMemcpyHostToDevice), "Failed to copy input image to device in convo-layer");
  handleError(cudaMemcpy(d_filter, filter->filter, filter->size, cudaMemcpyHostToDevice), "Failed to copy filter to device in convo-layer");

  // creating events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // launching kernel

  dim3 blockDim(32, 32);
  dim3 gridDim((feature_map->dim + blockDim.x - 1) / blockDim.x, (feature_map->dim + blockDim.y - 1) / blockDim.y, feature_map->channels);

  cudaEventRecord(start);
  // convo1 first layer

  convolutional_kernel<<<gridDim, blockDim>>>(d_img, d_filter, d_feature_map, image->dim, image->channels, filter->dimxy, filter->channels);

  cudaEventRecord(stop);
  handleError(cudaGetLastError(), "Failed to launch convolutional kernel");

  // copying data back to host
  handleError(cudaMemcpy(feature_map->img, d_feature_map, feature_map->size, cudaMemcpyDeviceToHost), "Failed to copy feature map to host in convo-layer");

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("convolutional kernel execution time: %.4f µs\n", milliseconds * 1000);

  // checking results
  convo_host(image, filter, feature_map);

  return feature_map;
}

Image *avg_pooling_layer(Image *image, int stride, int pooling_dim)
{
  Image *feature_map = new Image(image->dim / stride, image->channels);

  float *d_img, *d_feature_map;

  // allocating device memory
  handleError(cudaMalloc((void **)&d_img, image->size), "Failed to allocate device memory for image in pooling-layer");

  handleError(cudaMalloc((void **)&d_feature_map, feature_map->size), "Failed to allocate device memory for feature map in pooling-layer");

  handleError(cudaMemcpy(d_img, image->img, image->size, cudaMemcpyHostToDevice), "Failed to copy image to device in pooling-layer");

  // creating events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // launching kernel
  dim3 blockDim2(32, 32);
  dim3 gridDim2((feature_map->dim + blockDim2.x - 1) / blockDim2.x, (feature_map->dim + blockDim2.y - 1) / blockDim2.y, feature_map->channels);

  cudaEventRecord(start);

  avg_pool_kernel<<<gridDim2, blockDim2>>>(d_img, d_feature_map, image->dim, image->channels, pooling_dim, stride);

  cudaEventRecord(stop);
  handleError(cudaGetLastError(), "Failed to launch avg-pooling kernel");

  // copying data back to host
  handleError(cudaMemcpy(feature_map->img, d_feature_map, feature_map->size, cudaMemcpyDeviceToHost), "Failed to copy feature map to host in avg-pooling layer");

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("avg-pooling kernel execution time: %.4f µs\n", milliseconds * 1000);

  // checking results
  avg_pool_host(image, feature_map, pooling_dim, stride);

  return feature_map;
}

// main function
int main(int argc, char const *argv[])
{
  srand(time(0));

  // image data
  float *img = (float *)malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
  bool success = read_img_data(img);
  if (!success)
  {
    cout << "Failed to read image data\n";
    return 0;
  }

  // takes this image as input
  Image *image = new Image(IMAGE_SIZE, 1, img);
  cout << "layer1: ";
  Image *layer1_feature_map = convolutional_layer(image, FILTER_1_SIZE, FILTER_1_CHANNELS);
  cout << "layer2: ";
  Image *layer2_feature_map = avg_pooling_layer(layer1_feature_map, POOL_1_STRIDE, POOL_1_SIZE);
  cout << "layer3: ";
  Image *layer3_feature_map = convolutional_layer(layer2_feature_map, FILTER_2_SIZE, FILTER_2_CHANNELS);
  cout << "layer4: ";
  Image *layer4_feature_map = avg_pooling_layer(layer3_feature_map, POOL_2_STRIDE, POOL_2_SIZE);
  cout << "layer5: ";
  Image *layer5_feature_map = convolutional_layer(layer4_feature_map, FILTER_3_SIZE, FILTER_3_CHANNELS);
  cout << "layer6: ";
  Image *layer6_feature_map = convolutional_layer(layer5_feature_map, 1, FC_1_SIZE);
  cout << "layer7: ";
  Image *layer7_feature_map = convolutional_layer(layer6_feature_map, 1, FC_2_SIZE);

  cout << "printing layer 7 feature map" << endl;
  cout << layer7_feature_map->size / sizeof(int) << endl;

  layer7_feature_map->print_image();

  // free the host memory
  delete image;
  delete layer1_feature_map;
  delete layer2_feature_map;
  delete layer3_feature_map;
  delete layer4_feature_map;
  delete layer5_feature_map;
  delete layer6_feature_map;
  delete layer7_feature_map;

  handleError(cudaDeviceReset(), "Failed to deinitialize the device!");

  cout << "\n";
  return 0;
}