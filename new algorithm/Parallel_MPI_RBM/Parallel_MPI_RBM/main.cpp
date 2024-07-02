#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <mpi.h>
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

void pack_map(const std::map<uint32_t, std::vector<uint32_t>>& compress, std::vector<uint32_t>& packed) {
    packed.push_back(compress.size());
    for (const auto& entry : compress) {
        packed.push_back(entry.first);
        packed.push_back(entry.second.size());
        for (const auto& value : entry.second) {
            packed.push_back(value);
        }
    }
}

void unpack_map(const std::vector<uint32_t>& packed, std::map<uint32_t, std::vector<uint32_t>>& compress) {
    auto it = packed.begin();
    uint32_t num_keys = *it++;
    for (uint32_t i = 0; i < num_keys; ++i) {
        uint32_t key = *it++;
        uint32_t num_values = *it++;
        std::vector<uint32_t> values(it, it + num_values);
        it += num_values;
        compress[key] = values;
    }
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);// 获取处理器数量
    //std::cout << world_size << std::endl;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);// 获取当前处理器的 ID
    //std::cout << world_rank << std::endl;
    std::vector<uint32_t> array;
    for (int j = 0; j < 10; j++) {
        if (world_rank == 0) {
            std::ifstream file("D:/BaiduNetdiskDownload/data/ExpIndex_Query/ExpIndex", std::ios::binary);
            if (!file) {
                std::cerr << "无法打开文件" << std::endl;
                return 1;
            }
            file.seekg(32832, std::ios::beg);
            array = read_array(file);
            file.close();
        }

        // 广播数组大小
        uint32_t size = array.size();
        MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (world_rank != 0) {
            array.resize(size);
        }

        // 广播数组内容
        MPI_Bcast(array.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        auto beforeTime = std::chrono::steady_clock::now();

        std::map<uint32_t, std::vector<uint32_t>> compress;

        uint32_t chunk_size = size / world_size;
        uint32_t start = world_rank * chunk_size;
        uint32_t end = (world_rank == world_size - 1) ? size : start + chunk_size;

        std::vector<uint32_t> subarray = std::vector<uint32_t>(array.begin() + start, array.begin() + end);
        for (uint32_t i = 0; i < subarray.size(); i++) {
            uint32_t HighPart = subarray[i] >> 16;
            uint32_t LowPart = subarray[i] & 65535;

            if (compress.find(HighPart) == compress.end()) {
                compress[HighPart] = std::vector<uint32_t>();
            }

            compress[HighPart].push_back(LowPart);
        }

        std::vector<uint32_t> packed_local_compress;
        pack_map(compress, packed_local_compress);
        // 合并结果
        std::map<uint32_t, std::vector<uint32_t>> final_compress;

        if (world_rank == 0) {
            for (auto& entry : compress) {
                final_compress[entry.first].insert(final_compress[entry.first].end(), entry.second.begin(), entry.second.end());
            }
            for (int i = 1; i < world_size; i++) {
                uint32_t size1;
                MPI_Recv(&size1, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// 接收结果大小
                std::vector<uint32_t> packed(size1);
                MPI_Recv(packed.data(), size1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::map<uint32_t, std::vector<uint32_t>> local_result;
                unpack_map(packed, local_result);
                for (auto& entry : local_result) {
                    final_compress[entry.first].insert(final_compress[entry.first].end(), entry.second.begin(), entry.second.end());
                }
            }
            auto afterTime = std::chrono::steady_clock::now();
            double time = std::chrono::duration<double>(afterTime - beforeTime).count();
            std::cout << "time = " << time << " seconds" << std::endl;
        }
        else {
            // 其他进程发送结果给主进程
            uint32_t packed_size = packed_local_compress.size();
            MPI_Send(&packed_size, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(packed_local_compress.data(), packed_size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
        }
        std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/compress_rbm.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        /*for (const auto& pair : final_compress) {
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

       
    }
    MPI_Finalize();
    return 0;
}