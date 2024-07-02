#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <map>
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
__global__ void process_data_kernel(const uint32_t* d_array, uint32_t* d_highParts, uint32_t* d_lowParts, uint32_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        d_highParts[i] = d_array[i] >> 16;
        d_lowParts[i] = d_array[i] & 65535;
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
        uint32_t length = array.size();

        // 分配设备内存
        uint32_t* d_array;
        uint32_t* d_highParts;
        uint32_t* d_lowParts;

        cudaMalloc(&d_array, length * sizeof(uint32_t));
        cudaMalloc(&d_highParts, length * sizeof(uint32_t));
        cudaMalloc(&d_lowParts, length * sizeof(uint32_t));

        // 传输数据到设备
        cudaMemcpy(d_array, array.data(), length * sizeof(uint32_t), cudaMemcpyHostToDevice);

        int blockSize = 512;
        int numBlocks = (length + blockSize - 1) / blockSize;

        auto beforeTime = std::chrono::steady_clock::now();

        // 调用CUDA内核函数
        process_data_kernel << <numBlocks, blockSize >> > (d_array, d_highParts, d_lowParts, length);

        // 同步设备
        cudaDeviceSynchronize();

        // 分配主机内存用于接收结果
        std::vector<uint32_t> highParts(length);
        std::vector<uint32_t> lowParts(length);

        // 传输结果回主机
        cudaMemcpy(highParts.data(), d_highParts, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(lowParts.data(), d_lowParts, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // 合并结果
        std::map<uint32_t, std::vector<uint32_t>> final_compress;
        for (uint32_t i = 0; i < length; ++i) {
            final_compress[highParts[i]].push_back(lowParts[i]);
        }

        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << " seconds" << std::endl;

        // 释放设备内存
        cudaFree(d_array);
        cudaFree(d_highParts);
        cudaFree(d_lowParts);

        std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/compress_cuda_rbm.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        /*for (const auto& pair : final_compress) {
            f << "High Part: " << pair.first << " -> Low Parts: ";
            for (size_t i = 0; i < pair.second.size(); ++i) {
                f << pair.second[i];
                if (i != pair.second.size() - 1) f << ", ";
            }
            f << std::endl;
        }
        f.close();*/
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/array.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (uint32_t value : array) {
            f << value << ' ';
        }
        f.close();*/

        file.close();
    }
    return 0;
}
