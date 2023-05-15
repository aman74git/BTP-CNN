CC = g++
CUDACC = nvcc
SRC=../image_input.cpp
CPP_EXECUTABLE=ImageInput
SRC2=../lenet-5-architecture.cu
CUDA_EXECUTABLE=LeNet-5

LIBS=`pkg-config opencv4 --cflags --libs`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

all: ${CPP_EXECUTABLE} ${CUDA_EXECUTABLE} lenet-5.sh

${CPP_EXECUTABLE}: ${SRC}
	${CC} ${SRC} -o ${CPP_EXECUTABLE} ${LIBS}

${CUDA_EXECUTABLE}: ${SRC2}
	${CUDACC} ${SRC2} -o ${CUDA_EXECUTABLE}

lenet-5.sh:
	echo "#!/bin/bash" > $@
	echo "./${CPP_EXECUTABLE} \$$1" >> $@
	echo "./${CUDA_EXECUTABLE}" >> $@
	chmod +x $@
	echo "if [ -f image_data.txt ]; then rm image_data.txt; fi" >> $@