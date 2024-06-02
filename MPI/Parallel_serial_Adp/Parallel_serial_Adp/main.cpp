#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <algorithm>
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

int main() {
    for (int j = 0; j < 10; j++) {
        std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "�޷����ļ�" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array1 = read_array(file);
        /*file.seekg(272840, std::ios::beg);
        std::vector<uint32_t> array2 = read_array(file);*/
        file.seekg(1733008, std::ios::beg);
        std::vector<uint32_t> array3 = read_array(file);
        auto beforeTime = std::chrono::steady_clock::now();
        // ����ȡ�����������һ���б���
        std::vector<std::vector<uint32_t>> lists = { array1, /*array2 ,*/  array3};
        std::unordered_set<uint32_t> S;  // ���ڴ洢�������
        std::vector<uint32_t> result;    // ���ڴ洢����������

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
                result.push_back(e);
            }

            // �������б����Ƴ�e
            for (auto& list : lists) {
                list.erase(remove(list.begin(), list.end(), e), list.end());
            }
        }
        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << "seconds" << std::endl;
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/serial_Adp.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "�޷����ļ�" << std::endl;
            return 0;
        }
        for (const auto& s : result) {
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