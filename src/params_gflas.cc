#include <iostream>
#include <map>

#include "params_gflags.h"
DEFINE_string(input_file, "../data/back", "Input file path");
DEFINE_string(output_file, "intrinsic", "Output file path");
DEFINE_string(input_file_extension, "jpg", "Input file extension");
DEFINE_string(output_file_format, "txt", "Output file format");
DEFINE_string(output_file_prefix, "back", "Output file prefix");
DEFINE_int32(width, 11, "Number of inner corners in the chessboard width");
DEFINE_int32(height, 8, "Number of inner corners in the chessboard height");
DEFINE_int32(taylor_order, 4, "Degree of polynomial expansion");
DEFINE_double(square_size, 100, "Chessboard square size in mm");
void ParseShortFlags(int* argc, char*** argv) {
  // 定义简写标志与全名标志的映射关系
  std::map<std::string, std::string> short_to_long = {
      {"-w", "width"},
      {"-h", "height"},
      {"-i", "input_file"},
      {"-o", "output_file"},
      {"-e", "input_file_extension"},
      {"-f", "output_file_format"},
      {"-p", "output_file_prefix"},
      {"-s", "square_size"}};

  for (int i = 1; i < *argc; ++i) {
    std::string arg = (*argv)[i];
    if (short_to_long.count(arg)) {
      // 获取对应的全名标志
      std::string long_flag = short_to_long[arg];
      if (i + 1 < *argc) {
        // 设置标志值
        gflags::SetCommandLineOption(long_flag.c_str(), (*argv)[i + 1]);
        // 删除简写标志及其值
        for (int j = i; j < *argc - 2; ++j) {
          (*argv)[j] = (*argv)[j + 2];
        }
        *argc -= 2;
        --i;
      } else {
        std::cerr << "Error: Missing value for " << arg << std::endl;
        exit(1);
      }
    }
  }
}