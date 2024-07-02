#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>

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

void perform_intersection(const std::vector<uint32_t>& small_array, const std::vector<uint32_t>& large_array, std::vector<uint32_t>& result, int start, int end) {
    std::vector<uint32_t> local_result;
    std::set_intersection(small_array.begin(), small_array.end(),
        large_array.begin() + start, large_array.begin() + end,
        std::back_inserter(local_result));
    std::lock_guard<std::mutex> lock(std::mutex);
    result.insert(result.end(), local_result.begin(), local_result.end());
}

int main() {
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

        std::vector<uint32_t> small_array;
        std::vector<uint32_t> large_array;
        if (array1.size() > array2.size()) {
            small_array = array2;
            large_array = array1;
        }
        else {
            small_array = array1;
            large_array = array2;
        }

        std::vector<uint32_t> final_result;
        std::mutex mutex;
        int num_threads = 4;//std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        int total_elements = large_array.size();
        int chunk_size = total_elements / num_threads;

        auto beforeTime = std::chrono::steady_clock::now();

        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? total_elements : start + chunk_size;
            threads[i] = std::thread(perform_intersection, std::ref(small_array), std::ref(large_array), std::ref(final_result), start, end);
        }

        for (auto& th : threads) {
            th.join();
        }

        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();

        std::cout << "Intersection size: " << final_result.size() << ", time: " << time << " seconds,  Number of Threads: " << num_threads << std::endl;
    }
    return 0;
}




