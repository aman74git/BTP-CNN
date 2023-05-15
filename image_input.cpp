#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

#define IMAGE_SIZE 32

void write_array_to_file(Mat img)
{
  std::ofstream outfile("image_data.txt");

  int rows = img.rows;
  int cols = img.cols;

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      outfile << img.at<float>(i, j) << " ";
    }
    outfile << std::endl;
  }

  outfile.close();
}

int main(int argc, char const *argv[])
{
  string path;
  if (argc < 2)
  {
    cout << "please provide image path" << endl;
    return -1;
  }
  else
    path = argv[1];

  Mat image = imread(path, IMREAD_COLOR);

  // check if image doesnot exists
  if (image.empty())
  {
    cout << "Image not found" << endl;
    return -1;
  }

  // resize image to 16*16
  resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));

  // Convert image to grayscale
  Mat grayscale;
  cvtColor(image, grayscale, COLOR_RGB2GRAY);

  // Normalize pixel values to range [0, 1]
  grayscale.convertTo(grayscale, CV_32FC1, 1.0 / 255.0);

  write_array_to_file(grayscale);
  return 0;
}
