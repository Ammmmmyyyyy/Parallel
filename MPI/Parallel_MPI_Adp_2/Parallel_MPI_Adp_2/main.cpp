#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <mpi.h>
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


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);// ��ȡ����������
    //std::cout << world_size << std::endl;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);// ��ȡ��ǰ�������� ID
    //std::cout << world_rank << std::endl;
    std::vector<uint32_t> array1, array2;
    for (int j = 0; j < 10; j++) {
        if (world_rank == 0) {
            std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
            if (!file) {
                std::cerr << "�޷����ļ�" << std::endl;
                MPI_Finalize();
                return 1;
            }
            file.seekg(32832, std::ios::beg);
            array1 = read_array(file);
            file.seekg(1733008, std::ios::beg);
            array2 = read_array(file);
            file.close();
        }
        // �㲥�����С
        uint32_t size1 = array1.size(), size2 = array2.size();
        MPI_Bcast(&size1, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&size2, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (world_rank != 0) {
            array1.resize(size1);
            array2.resize(size2);
        }

        // �㲥��������
        MPI_Bcast(array1.data(), size1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(array2.data(), size2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        std::vector<std::vector<uint32_t>> bucket(world_size * 2);
        if (world_rank == 0) {
            uint32_t bucket_range = std::max(array1.back(),array2.back()) / world_size;
            for (uint32_t elem : array1) {
                int bucket_index = elem / bucket_range;
                if (bucket_index >= world_size) bucket_index = world_size - 1;
                bucket[bucket_index].push_back(elem);
            }
            for (uint32_t elem : array2) {
                int bucket_index = world_size + elem / bucket_range;
                if (bucket_index >= 2 * world_size) bucket_index = 2 * world_size - 1;
                bucket[bucket_index].push_back(elem);
            }
        }
        for (int i = 0; i < world_size * 2; i++) {
            uint32_t bucket_size = bucket[i].size();
            MPI_Bcast(&bucket_size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
            if (world_rank != 0) {
                bucket[i].resize(bucket_size);
            }
            MPI_Bcast(bucket[i].data(), bucket_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        }

        // ÿ�����̴����Լ��Ĳ���
        std::vector<uint32_t> local_chunk1 = bucket[world_rank];
        std::vector<uint32_t> local_chunk2 = bucket[world_rank + world_size];

        auto beforeTime = std::chrono::steady_clock::now();
        // ����ȡ�����������һ���б���
        std::vector<std::vector<uint32_t>> lists = { local_chunk1, local_chunk2 };
        std::unordered_set<uint32_t> S;  // ���ڴ洢�������
        std::vector<uint32_t> result;    // ���ڴ洢�������
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
        // �������ռ����
        std::vector<uint32_t> global_result ;
        if (world_rank == 0) {
            global_result.insert(global_result.end(), result.begin(), result.end());
            for (int i = 1; i < world_size; ++i) {
                uint32_t size;
                MPI_Recv(&size, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// ���ս����С
                std::vector<uint32_t> local_result(size);
                MPI_Recv(local_result.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// ���ս������
                global_result.insert(global_result.end(), local_result.begin(), local_result.end());
            }
            auto afterTime = std::chrono::steady_clock::now();
            double time = std::chrono::duration<double>(afterTime - beforeTime).count();
            std::cout << "time = " << time << " seconds" << std::endl;
        }
        else {
            // �������̷��ͽ����������
            uint32_t size = result.size();
            MPI_Send(&size, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);// ���ͽ����С
            MPI_Send(result.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);// ���ͽ������
        }


        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/MPI_Adp.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "�޷����ļ�" << std::endl;
            return 0;
        }
        for (const auto& s : global_result) {
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


    }
    MPI_Finalize();
    return 0;
}