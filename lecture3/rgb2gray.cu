#include <iostream>

__host__ __device__ u_char rgb2gray(
    u_char r, u_char g, u_char b
) {
    return (r*3/10) + (g*6/10) + (b*1/10);
}

__global__ void rgb2gray_kernel(
    u_char* red, u_char* green, u_char* blue, u_char* gray, u_int32_t width, u_int32_t height
){

    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y_index = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int pixel_index = width * x_index + y_index;

    if(x_index < width && y_index < height) {
        gray[pixel_index] = rgb2gray(red[pixel_index], green[pixel_index], blue[pixel_index]); 
    }
}


int main() {
    const u_int32_t width = 256;
    const u_int32_t height = 256;
    auto N = width * height;
    const auto N_bytes = N*sizeof(u_char);
    u_char *r = new u_char[N];
    u_char *g = new u_char[N];
    u_char *b = new u_char[N];
    u_char *gray = new u_char[N];

    for (int i=0; i< N; i++){
        r[i] = u_char(255*(float(rand())/RAND_MAX));
        g[i] = u_char(255*(float(rand())/RAND_MAX));
        b[i] = u_char(255*(float(rand())/RAND_MAX));
    }

    u_char *r_d, *g_d, *b_d, *gray_d;
    cudaMalloc((void**)&r_d, N_bytes);
    cudaMalloc((void**)&g_d, N_bytes);
    cudaMalloc((void**)&b_d, N_bytes);
    cudaMalloc((void**)&gray_d, N_bytes);


    cudaMemcpy(r_d, r, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N_bytes, cudaMemcpyHostToDevice);
    

    const dim3 blockSize = dim3(32, 32);
    const dim3 gridSize = dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        1
    );

    rgb2gray_kernel<<<gridSize, blockSize>>>(
        r_d,
        g_d,
        b_d,
        gray_d,
        width,
        height
    );

    cudaMemcpy(gray, gray_d, N_bytes, cudaMemcpyDeviceToHost);

    std::cout << "gpu output at index 0: " <<float(gray[0]) 
    << " cpu output at index 0: " << float(rgb2gray(r[0], g[0], b[0])) 
    << std::endl;
    
    cudaFree(r_d);
    cudaFree(g_d);
    cudaFree(b_d);
    cudaFree(gray_d);

    
    delete[] r;
    delete[] g;
    delete[] b;
    delete[] gray;

}