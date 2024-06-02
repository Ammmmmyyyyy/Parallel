#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <thread>
#include <mutex>

// ȫ�ֱ����ͻ�����
std::vector<uint32_t> final_result;
std::mutex result_mutex;

// ��ȡС�˸�ʽ�����ֽ��޷�������
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

// ��ȡһ����������
std::vector<uint32_t> read_array(std::ifstream& stream) {
    uint32_t length = read_uint32_le(stream);
    std::vector<uint32_t> array(length);
    for (uint32_t i = 0; i < length; ++i) {
        array[i] = read_uint32_le(stream);
    }
    return array;
}

// ����Ԫ���Ƿ�������б���
bool find(const std::vector<uint32_t>& list, uint32_t element) {
    return binary_search(list.begin(), list.end(), element);
}

// ��ȡ��һ��δ�ҵ���Ԫ��
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
    std::unordered_set<uint32_t> S;  // ���ڴ洢�������
    std::vector<uint32_t> local_result;
    while (true) {
        // ����κ�һ���б�Ϊ�գ��˳�ѭ��
        bool any_empty = false;
        for (const auto& list : lists) {
            if (list.empty()) {
                any_empty = true;
                break;
            }
        }
        if (any_empty) break;

        // ��̬�����б�˳��
        sort(lists.begin(), lists.end(), [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
            return a.size() < b.size();
            });

        // ���ҵ�һ��δ�ҵ���Ԫ��
        uint32_t e = first_unfound_element(lists[0], S);
        if (e == -1) break;  // ���û��δ�ҵ���Ԫ�أ��˳�ѭ��

        // ���������б��Ƿ������Ԫ��
        bool found = true;
        for (size_t i = 1; i < lists.size(); i++) {
            if (!find(lists[i], e)) {
                found = false;
                break;
            }
        }

        // ��������б�������Ԫ�أ�������ӵ��������
        if (found) {
            S.insert(e);
            local_result.push_back(e);
        }

        // �������б����Ƴ�e
        for (auto& list : lists) {
            list.erase(remove(list.begin(), list.end(), e), list.end());
        }
    }

    // ʹ�û�����������ȫ�ֱ����ķ���
    std::lock_guard<std::mutex> lock(result_mutex);
    final_result.insert(final_result.end(), local_result.begin(), local_result.end());
}

int main() {
    for (int j = 0; j < 10; j++) {
        std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "�޷����ļ�" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array1 = read_array(file);
        file.seekg(1733008, std::ios::beg);
        std::vector<uint32_t> array2 = read_array(file);
        auto beforeTime = std::chrono::steady_clock::now();
        size_t num_threads = 28;
        std::vector<std::thread> threads;
        std::vector<std::vector<uint32_t>> results_parts(num_threads);//�洢����Ĳ���

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
            std::cerr << "�޷����ļ�" << std::endl;
            return 0;
        }
        for (const auto& s : final_result) {
            f << s << " ";
        }
        f.close();*/
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/array_30000_3.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "�޷����ļ�" << std::endl;
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