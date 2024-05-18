#include <iostream>
#include <fstream>
#include <vector>
#include<chrono>
#include <omp.h>
#include <map>
#include <smmintrin.h>
using namespace std;
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
    size_t thelength = length - (length % 4);
    __m128i* thearray = (__m128i*) array.data();

    for (size_t i = 0; i < thelength / 4; ++i) {
        __m128i chunk = _mm_load_si128(&thearray[i]);

        __m128i high_parts = _mm_srli_epi32(chunk, 16);

        __m128i low_mask = _mm_set1_epi32(0xFFFF);
        __m128i low_parts = _mm_and_si128(chunk, low_mask);



        for (int j = 0; j < 4; ++j) {
            uint32_t high_part;
            uint32_t low_part;
            switch (j) {
            case 0:
                high_part = _mm_extract_epi32(high_parts, 0);
                low_part = _mm_extract_epi32(low_parts, 0);
                break;
            case 1:
                high_part = _mm_extract_epi32(high_parts, 1);
                low_part = _mm_extract_epi32(low_parts, 1);
                break;
            case 2:
                high_part = _mm_extract_epi32(high_parts, 2);
                low_part = _mm_extract_epi32(low_parts, 2);
                break;
            case 3:
                high_part = _mm_extract_epi32(high_parts, 3);
                low_part = _mm_extract_epi32(low_parts, 3);
                break;
            }

            // 更新 map
            if (!compress.count(high_part)) {
                compress[high_part] = vector<uint32_t>();
            }
            compress[high_part].push_back(low_part);
        }
    }

    for (uint32_t i = thelength; i < length; i++) {
        uint32_t HighPart = array[i] >> 16;
        uint32_t LowPart = array[i] & (65535);

        if (compress.find(HighPart) == compress.end()) {
            compress[HighPart] = vector<uint32_t>();
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
        size_t num_threads = omp_get_max_threads();
        std::vector<std::map<uint32_t, std::vector<uint32_t>>> compress_parts(num_threads);
        size_t part_length = array.size() / num_threads;
        std::vector<std::vector<uint32_t>> subarrays(num_threads);
        #pragma omp parallel for 
        for (int i = 0; i < num_threads; ++i) {
            size_t start = i * part_length;
            size_t end = (i == num_threads - 1) ? array.size() : (start + part_length);
            subarrays[i] = std::vector<uint32_t>(array.begin() + start, array.begin() + end);
            process_data(subarrays[i], compress_parts[i]);
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
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/openmp_SIMD_rbm.txt", std::ios::app);
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