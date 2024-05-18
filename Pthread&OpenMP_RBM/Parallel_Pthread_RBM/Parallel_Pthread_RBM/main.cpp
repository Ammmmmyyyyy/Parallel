#include <iostream>
#include <fstream>
#include <vector>
#include<chrono>
#include <thread>
#include <map>
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

void process_data(const std::vector<uint32_t>& array, std::map<uint32_t, std::vector<uint32_t>>& compress) {
    uint32_t length = array.size();
    for (uint32_t i = 0; i < length; i++) {
        uint32_t HighPart = array[i] >> 16;
        uint32_t LowPart = array[i] & 65535;

        if (compress.find(HighPart) == compress.end()) {
            compress[HighPart] = std::vector<uint32_t>();
        }

        compress[HighPart].push_back(LowPart);
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
        auto beforeTime = std::chrono::steady_clock::now();
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<std::map<uint32_t, std::vector<uint32_t>>> compress_parts(num_threads);

        size_t part_length = array.size() / num_threads;
        std::vector<std::vector<uint32_t>> subarrays(num_threads);

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * part_length;
            size_t end = (i == num_threads - 1) ? array.size() : (start + part_length);
            subarrays[i] = std::vector<uint32_t>(array.begin() + start, array.begin() + end);
            threads.push_back(std::thread(process_data, std::ref(subarrays[i]), std::ref(compress_parts[i])));
        }

        for (auto& thread : threads) {
            thread.join();
        }
        // 合并结果
        std::map<uint32_t, std::vector<uint32_t>> final_compress;
        for (auto& part : compress_parts) {
            for (auto& entry : part) {
                final_compress[entry.first].insert(final_compress[entry.first].end(), entry.second.begin(), entry.second.end());
            }
        }
        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << "seconds" << std::endl;
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/thread_rbm.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (const auto& pair : final_compress) {
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