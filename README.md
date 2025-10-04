Practising CUDA with the help of PPMP lectures by [GPU Computing, Spring 2021, Izzat El Hajj](http://youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)

| Filename                             | Kernel Description                                                                                                                            | Learning                                                                                                                        |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [vecadd.cu](vecadd.cu)               | Vector addition kernel that performs element-wise addition of two input vectors                                                               | How to run a simple 1D grid cuda kernel, how to load data from host memory to device memory                                     |
| [rgb2gray.cu](rgb2gray.cu)           | Color conversion kernel that converts RGB images to grayscale using standard luminance formula                                                | How to run a cuda kernel with 2D grid                                                                                           |
| [simple_blur.cu](simple_blur.cu)     | Image blur kernel that applies a simple averaging filter to blur an image                                                                     | How to read data from different indexes in the same thread, how to place proper if-else guards to avoid out of bound data reads |
| [simple_matmul.cu](simple_matmul.cu) | Basic matrix multiplication kernel with naive implementation using global memory access (supports rectangular matrices)                       | How for large matrices running mat mul on GPU can be 1000x faster than CPU                                                      |
| [tiled_matmul.cu](tiled_matmul.cu)   | Tiled matrix multiplication kernel using shared memory optimization with 32x32 tiles for better performance (supports rectangular matrices)   | Tiling based matmul to reduce number of data fetch from the global memory                                                       |
| [coarse_matmul.cu](coarse_matmul.cu) | Tiled matrix multiplication kernel using shared memory optimization with 32x32 tiles and work per thread of 4 (supports rectangular matrices) | Memory Coarsing                                                                                                                 |

## Mat mul different runtimes on Nvidia GPU 4060 TI

### Square Mat Mul

```
M = 256, N = 256, K = 256;
matA_size = M x K; // input matrix A
matB_size = K x N; // input matrix B
matC_size = M x N; // output matrix C
TILE_SIZE 32
COARSE_FACTOR 4
```

- Simple matmul (without tiling) runtime: **0.03 ms**

- Tiled matmul runtime (without data access guards): **0.023 ms**

- Tiled matmul with work per thread of 2 (with data access guards): **0.021 ms**
- Tiled matmul with work per thread of 4 (without data access guards): **0.040 ms**

### Rectangular Mat Mul

```
u_int32_t M = 255, N = 257, K = 256;
matA_size = M x K; // input matrix A
matB_size = K x N; // input matrix B
matC_size = M x N; // output matrix C
TILE_SIZE 32
COARSE_FACTOR 4
```

- Simple matmul (without tiling) runtime: **0.045 ms**

- Tiled matmul runtime (with data access guards): **0.034 ms**

- Tiled matmul with work per thread of 4 (with data access guards): **0.041 ms**

> Note: When the ratio of block size to max matrix length is smaller W.P.T based approach is not recommended as it leads to lower thread occupancy.

---

### Square Mat Mul with different BLOCK_SIZE and WPT

```
M = 1024, N = 1024, K = 1024;
matA_size = M x K; // input matrix A
matB_size = K x N; // input matrix B
matC_size = M x N; // output matrix C
```

#### BLOCK_SIZE 16

- Simple matmul (without tiling) & BLOCK_SIZE 16 runtime: **1.675 ms**

- Tiled matmul runtime (without data access guards): **1.221 ms**

- Tiled matmul with work per thread of 2 (with data access guards): **1.175 ms** (best!)
- Tiled matmul with work per thread of 4 (with data access guards): **1.170 ms** (best!)
- Tiled matmul with work per thread of 8 (with data access guards): **1.208 ms**

#### BLOCK_SIZE 32

- Simple matmul (without tiling) & BLOCK_SIZE 32 runtime: **1.717 ms**

- Tiled matmul runtime (with data access guards): **1.274 ms**

- Tiled matmul with work per thread of 2 (with data access guards): **1.262 ms**
- Tiled matmul with work per thread of 4 (with data access guards): **1.246 ms**
- Tiled matmul with work per thread of 8 (with data access guards): **1.234 ms**

> I tried other sizes of 2048 and 4096 on all of them I saw BLOCK_SIZE 16 with WPT of 4 take the least amount of time.
