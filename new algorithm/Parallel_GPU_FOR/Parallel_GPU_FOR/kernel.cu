#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// 读取小端格式的四字节无符号整数
uint32_t read_uint32_le(std::ifstream& stream) {
    uint32_t value;
    char bytes[4];
    stream.read(bytes, 4);
    value = (static_cast<uint32_t>(static_cast<unsigned char>(bytes[3])) << 24) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[2])) << 16) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[1])) << 8) |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[0]));
    return value;
}

// 读取一个整数数组
std::vector<uint32_t> read_array(std::ifstream& stream) {
    uint32_t length = read_uint32_le(stream);
    std::vector<uint32_t> array(length);
    for (uint32_t i = 0; i < length; ++i) {
        array[i] = read_uint32_le(stream);
    }
    return array;
}

// CUDA内核函数
__global__ void compress_kernel(float* d_compress, const float* d_floatArray, uint32_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        d_compress[0] = d_floatArray[0];  // 明确设置compress[0]
    }
    if (i > 0 && i < length) {
        d_compress[i] = d_floatArray[i] - d_floatArray[i - 1];
    }
}

int main() {
    for (int j = 0; j < 10; j++) {
        std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array = read_array(file);
        std::vector<float> floatArray;
        // 转换整数数组到浮点数组
        for (uint32_t value : array) {
            floatArray.push_back(static_cast<float>(value));
        }
        uint32_t length = floatArray.size();
        std::vector<float> compress(length);
        compress[0] = floatArray[0];

        
        float* d_floatArray;
        float* d_compress;

        // 分配设备内存
        cudaMalloc(&d_floatArray, length * sizeof(float));
        cudaMalloc(&d_compress, length * sizeof(float));

        // 传输数据到设备
        cudaMemcpy(d_floatArray, floatArray.data(), length * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (length + blockSize - 1) / blockSize;

        auto beforeTime = std::chrono::steady_clock::now();

        // 调用CUDA内核函数
        compress_kernel << <numBlocks, blockSize >> > (d_compress, d_floatArray, length);

        // 同步设备
        cudaDeviceSynchronize();

        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << " seconds" << std::endl;

        // 传输结果回主机
        cudaMemcpy(compress.data(), d_compress, length * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 释放设备内存
        cudaFree(d_floatArray);
        cudaFree(d_compress);
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/compress_cuda_for.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (float value : compress) {
            f << value << ' ';
        }
        f.close();*/
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/array_cuda.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (uint32_t value : array) {
            f << value << ' ';
        }
        f.close();
        file.close();*/
    }
    return 0;
}
