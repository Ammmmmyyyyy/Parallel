#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>

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

// 并行处理每个数据块以找到交集
void process_data(const std::vector<uint32_t>& array1, const std::vector<uint32_t>& array2, std::vector<uint32_t>& result) {
#pragma omp parallel
    {

        std::vector<uint32_t> local_result;


#pragma omp for nowait
        for (int i = 0; i < array1.size(); ++i) {
            if (std::binary_search(array2.begin(), array2.end(), array1[i])) {
                local_result.push_back(array1[i]);
            }
        }

#pragma omp critical
        result.insert(result.end(), local_result.begin(), local_result.end());
    }
}

int main() {
    int desired_threads = 4;  // 设定线程数
    omp_set_num_threads(desired_threads);

    for (int i = 0; i < 10; i++) {
        std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "Unable to open file" << std::endl;
            return 1;
        }

        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array1 = read_array(file);
        file.seekg(1733008, std::ios::beg);
        std::vector<uint32_t> array2 = read_array(file);
        file.close();

        // 预处理：对数组进行排序
        std::sort(array1.begin(), array1.end());
        std::sort(array2.begin(), array2.end());

        std::vector<uint32_t> final_result;
        auto beforeTime = std::chrono::steady_clock::now();

        // 使用 OpenMP 并行处理数据
        process_data(array1, array2, final_result);

        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        int actual_threads = omp_get_max_threads();  // 获取实际使用的线程数

        std::cout << "Intersection size: " << final_result.size() << " , number of thread:" << actual_threads << ", Time: " << time << " seconds\n";
    }
    return 0;
}