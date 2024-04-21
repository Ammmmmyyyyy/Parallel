#include <iostream>
#include <fstream>
#include <vector>
#include <immintrin.h>
#include <chrono>

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
        for (int value : array) {
            floatArray.push_back(static_cast<float>(value));
        }
        uint32_t length = floatArray.size();
        //std::vector<float> test = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        __m512 compress, arr1, arr2;
        std::vector<float> comp(length);
        comp[0] = floatArray[0];
        uint32_t i;
        auto beforeTime = std::chrono::steady_clock::now();
        for (i = 0; i < length ; i += 16) {
            arr1 = _mm512_loadu_ps(&floatArray[i]);
            arr2 = _mm512_loadu_ps(&floatArray[i + 1]);
            compress = _mm512_sub_ps(arr2, arr1);
            _mm512_storeu_ps(&comp[i + 1], compress);
        }
        for (; i < length - 1; i++)
        {
            comp[i + 1] = floatArray[i + 1] - floatArray[i];
        }
        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << "seconds" << std::endl;

        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/compress_SIMD.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (float value : comp) {
            f << value << ' ';
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
