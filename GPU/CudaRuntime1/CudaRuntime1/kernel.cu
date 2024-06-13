#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <cuda_runtime.h>

__device__ bool binary_search_device(const uint32_t* list, uint32_t length, uint32_t element) {
    uint32_t left = 0;
    uint32_t right = length - 1;
    while (left <= right) {
        uint32_t mid = left + (right - left) / 2;
        if (list[mid] == element) {
            return true;
        }
        else if (list[mid] < element) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }
    return false;
}

__global__ void find_intersections(const uint32_t* list1, uint32_t len1, const uint32_t* list2, uint32_t len2, uint32_t* result, uint32_t* result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len1) {
        uint32_t element = list1[idx];
        if (binary_search_device(list2, len2, element)) {
            uint32_t pos = atomicAdd(result_count, 1);
            result[pos] = element;
        }
    }
}

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

std::vector<uint32_t> read_array(std::ifstream& stream) {
    uint32_t length = read_uint32_le(stream);
    std::vector<uint32_t> array(length);
    for (uint32_t i = 0; i < length; ++i) {
        array[i] = read_uint32_le(stream);
    }
    return array;
}

int main() {
    for (int j = 0; j < 10; j++) {
        std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array1 = read_array(file);
        file.seekg(1733008, std::ios::beg);
        std::vector<uint32_t> array2 = read_array(file);
        file.close();

        uint32_t* d_array1;
        uint32_t* d_array2;
        uint32_t* d_result;
        uint32_t* d_result_count;

        size_t size1 = array1.size() * sizeof(uint32_t);
        size_t size2 = array2.size() * sizeof(uint32_t);
        size_t result_size = std::min(array1.size(), array2.size()) * sizeof(uint32_t);

        cudaMalloc(&d_array1, size1);
        cudaMalloc(&d_array2, size2);
        cudaMalloc(&d_result, result_size);
        cudaMalloc(&d_result_count, sizeof(uint32_t));

        cudaMemcpy(d_array1, array1.data(), size1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_array2, array2.data(), size2, cudaMemcpyHostToDevice);
        cudaMemset(d_result_count, 0, sizeof(uint32_t));

        int blockSize = 256;//每个block的线程数量
        int numBlocks = (array1.size() + blockSize - 1) / blockSize;//block数量
        auto beforeTime = std::chrono::steady_clock::now();
        find_intersections << <numBlocks, blockSize >> > (d_array1, array1.size(), d_array2, array2.size(), d_result, d_result_count);
        cudaDeviceSynchronize();
        auto afterTime = std::chrono::steady_clock::now();

        uint32_t result_count;
        cudaMemcpy(&result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::vector<uint32_t> result(result_count);
        cudaMemcpy(result.data(), d_result, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << " seconds" << std::endl;

        cudaFree(d_array1);
        cudaFree(d_array2);
        cudaFree(d_result);
        cudaFree(d_result_count);
    }
    return 0;
}
