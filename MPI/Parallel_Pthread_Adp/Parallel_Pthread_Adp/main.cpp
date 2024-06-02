#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <thread>
#include <mutex>

// 全局变量和互斥锁
std::vector<uint32_t> final_result;
std::mutex result_mutex;

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

// 查找元素是否存在于列表中
bool find(const std::vector<uint32_t>& list, uint32_t element) {
    return binary_search(list.begin(), list.end(), element);
}

// 获取第一个未找到的元素
uint32_t first_unfound_element(const std::vector<uint32_t>& list, const std::unordered_set<uint32_t>& found_elements) {
    for (uint32_t element : list) {
        if (found_elements.find(element) == found_elements.end()) {
            return element;
        }
    }
    return -1; // assuming -1 is not a valid element
}

void process_data(std::vector<std::vector<uint32_t>> &lists)
{
    std::unordered_set<uint32_t> S;  // 用于存储交集结果
    std::vector<uint32_t> local_result;
    while (true) {
        // 如果任何一个列表为空，退出循环
        bool any_empty = false;
        for (const auto& list : lists) {
            if (list.empty()) {
                any_empty = true;
                break;
            }
        }
        if (any_empty) break;

        // 动态调整列表顺序
        sort(lists.begin(), lists.end(), [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
            return a.size() < b.size();
            });

        // 查找第一个未找到的元素
        uint32_t e = first_unfound_element(lists[0], S);
        if (e == -1) break;  // 如果没有未找到的元素，退出循环

        // 查找其他列表是否包含该元素
        bool found = true;
        for (size_t i = 1; i < lists.size(); i++) {
            if (!find(lists[i], e)) {
                found = false;
                break;
            }
        }

        // 如果所有列表都包含该元素，将其添加到结果集中
        if (found) {
            S.insert(e);
            local_result.push_back(e);
        }

        // 从所有列表中移除e
        for (auto& list : lists) {
            list.erase(remove(list.begin(), list.end(), e), list.end());
        }
    }

    // 使用互斥锁保护对全局变量的访问
    std::lock_guard<std::mutex> lock(result_mutex);
    final_result.insert(final_result.end(), local_result.begin(), local_result.end());
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
        auto beforeTime = std::chrono::steady_clock::now();
        size_t num_threads = 28;
        std::vector<std::thread> threads;
        std::vector<std::vector<uint32_t>> results_parts(num_threads);//存储结果的部分

        std::vector<std::vector<uint32_t>> bucket(num_threads * 2);
        uint32_t bucket_range = std::max(array1.back(), array2.back()) / num_threads;
        for (uint32_t elem : array1) {
            int bucket_index = elem / bucket_range;
            if (bucket_index >= num_threads) bucket_index = num_threads - 1;
            bucket[bucket_index].push_back(elem);
        }
        for (uint32_t elem : array2) {
            int bucket_index = num_threads + elem / bucket_range;
            if (bucket_index >= 2 * num_threads) bucket_index = 2 * num_threads - 1;
            bucket[bucket_index].push_back(elem);
        }

        std::vector<std::vector<std::vector<uint32_t>>> lists(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            std::vector<uint32_t> local_chunk1 = bucket[i];
            std::vector<uint32_t> local_chunk2 = bucket[i + num_threads];
            lists[i].push_back(local_chunk1);
            lists[i].push_back(local_chunk2);
            threads.push_back(std::thread(process_data, std::ref(lists[i])));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        
        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << "seconds" << std::endl;
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/thread_Adp.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (const auto& s : final_result) {
            f << s << " ";
        }
        f.close();*/
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/array_30000_3.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (uint32_t value : array3) {
            f << value << ' ';
        }
        f.close();*/

        file.close();
    }
    return 0;
}