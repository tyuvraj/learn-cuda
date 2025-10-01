Practising CUDA with the help of PPMP lectures by [GPU Computing, Spring 2021, Izzat El Hajj](http://youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)


| Filename| Kernel Description | Learning |
|---------|--------------------|----------|
| [vecadd.cu](vecadd.cu) | Vector addition kernel that performs element-wise addition of two input vectors | How to run a simple 1D grid cuda kernel, how to load data from host memory to device memory |
| [rgb2gray.cu](rgb2gray.cu) | Color conversion kernel that converts RGB images to grayscale using standard luminance formula | How to run a cuda kernel with 2D grid |
| [simple_blur.cu](simple_blur.cu)    | Image blur kernel that applies a simple averaging filter to blur an image | How to read data from different indexes in tthe same thread, how to place proper if-else guards to avoid out of bound data reads |
| [simple_matmul.cu](simple_matmul.cu)  | Basic matrix multiplication kernel with naive implementation using global memory access (supports rectangular matrices) | How for large matrices running mat mul on GPU can be 1000x faster than CPU |
| [tiled_matmul.cu](tiled_matmul.cu)   | Tiled matrix multiplication kernel using shared memory optimization with 32x32 tiles for better performance (supports rectangular matrices) | Tiling based matmul to reduce number of data fetch from the global memory |

