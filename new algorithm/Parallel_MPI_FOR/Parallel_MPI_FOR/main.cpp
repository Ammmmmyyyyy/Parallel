#include <iostream>
#include <fstream>
#include <vector>
#include<chrono>
#include <mpi.h>
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);// 获取处理器数量
    //std::cout << world_size << std::endl;

    int world_rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);// 获取当前处理器的 ID
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
        // 划分数据到不同进程
        uint32_t length = array.size();
        MPI_Bcast(&length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        if (world_rank != 0) {
            array.resize(length);
        }

        // 广播数组内容
        MPI_Bcast(array.data(), length, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        uint32_t chunk_size = length / world_size;
        uint32_t start = world_rank * chunk_size;
        uint32_t end = (world_rank == world_size - 1) ? length : start + chunk_size;

        // 确保每个进程都有正确的边界值
        uint32_t previous_value = 0;
        if (world_rank != 0) {
            MPI_Recv(&previous_value, 1, MPI_UINT32_T, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::vector<float> floatArray(array.begin() + start, array.begin() + end);
        std::vector<float> compress(end - start);
        compress[0] = floatArray[0] - previous_value;

        auto beforeTime = std::chrono::steady_clock::now();
        for (uint32_t i = 1; i < compress.size(); i++) {
            compress[i] = floatArray[i] - floatArray[i - 1];
        }
        

        // 发送当前进程的最后一个值给下一个进程
        if (world_rank != world_size - 1) {
            MPI_Send(&array[end - 1], 1, MPI_UINT32_T, world_rank + 1, 0, MPI_COMM_WORLD);
        }

        // 合并压缩数据到主进程
        std::vector<float> final_compress;
        if (world_rank == 0) {
            final_compress.insert(final_compress.end(), compress.begin(), compress.end());
            for (int i = 1; i < world_size; i++) {
                uint32_t size;
                MPI_Recv(&size, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// 接收结果大小
                std::vector<float> local_result(size);
                MPI_Recv(local_result.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// 接收结果数据
                final_compress.insert(final_compress.end(), local_result.begin(), local_result.end());
            }
            auto afterTime = std::chrono::steady_clock::now();
            double time = std::chrono::duration<double>(afterTime - beforeTime).count();
            std::cout << "time = " << time << " seconds" << std::endl;
        }
        else {
            // 其他进程发送结果给主进程
            uint32_t size = compress.size();
            MPI_Send(&size, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);// 发送结果大小
            MPI_Send(compress.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);// 发送结果数据
        }
        
        /*std::ofstream f("D:/BaiduNetdiskDownload/data/ExpIndex_Query/compress_mpi.txt", std::ios::app);
        if (!f.is_open()) {
            std::cerr << "无法打开文件" << std::endl;
            return 0;
        }
        for (float value : final_compress) {
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

    }
    MPI_Finalize();
    return 0;
}
