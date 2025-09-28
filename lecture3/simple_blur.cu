#include <iostream>
#define KERNEL_SIZE 3

__host__ __device__ u_char get_pixel_value(
    u_char* gray, 
    int32_t row, 
    int32_t col, 
    u_int32_t width, 
    u_int32_t height) {
    if (row >= 0 && row < height && col >= 0 && col < width) {
        const u_int32_t input_index = width * row + col;
        return u_char(gray[input_index]);
    }
    return 0.0;
}

__host__ __device__ u_char calc_blur_value(
    u_char* gray, 
    int32_t row, 
    int32_t col, 
    u_int32_t width, 
    u_int32_t height) {
    int32_t average = 0;
    int kernel_offset = (KERNEL_SIZE-1)/2; // considering kernel size to be always odd
    for (int i = -kernel_offset; i <= kernel_offset; i++) {
        for (int j = -kernel_offset; j <= kernel_offset; j++){
            const int32_t rowIn = row + i;
            const int32_t colIn = col + j;
            average += get_pixel_value(gray, rowIn, colIn, width, height);
        }
    }
    return average/(KERNEL_SIZE*KERNEL_SIZE);
}

__global__ void blur_kernel(u_char* gray, u_char* blur, u_int32_t width, u_int32_t height) {
    int32_t row = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t col = blockDim.y * blockIdx.y + threadIdx.y;
    u_int32_t output_index = width * row + col;

    if (row < height && col < width){
        blur[output_index] = calc_blur_value(gray, row, col, width, height);
    }
}

int main() {
    const u_int32_t width = 256;
    const u_int32_t height = 256;
    const u_int32_t N = width * height;
    const u_int32_t N_bytes = N*sizeof(u_char);
    u_char *gray = new u_char[N];
    u_char *blur = new u_char[N];
    
    for (int i=0; i< N; i++){
        gray[i] =  u_char(255*(float(rand())/RAND_MAX));
    }

    u_char *gray_d, *blur_d;

    cudaMalloc((void **)&gray_d, N_bytes);
    cudaMalloc((void **)&blur_d, N_bytes);

    cudaMemcpy(gray_d, gray, N_bytes, cudaMemcpyHostToDevice);

    const dim3 blockSize(32, 32);
    const dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        1
    );

    // call blur kernel
    blur_kernel<<<gridSize, blockSize>>>(
        gray_d,
        blur_d,
        width,
        height
    );

    cudaMemcpy(blur, blur_d, N_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(gray_d);
    cudaFree(blur_d);


    for (int i = 0; i < 256; i++){
        for (int j = 0; j < 256; j++) {
            auto cpu_val = float(get_pixel_value(blur, i, j, width, height));
            auto gpu_val = float(calc_blur_value(gray, i, j, width, height));
            auto diff = abs(cpu_val-gpu_val);
            if (diff != 0){
                std::cout << "Miss match at " << i << "," << j << " "
                << " CPU: " << cpu_val 
                << " GPU: " << gpu_val << std::endl; 
            }
        }
    }
    delete[] gray;
    delete[] blur;
}