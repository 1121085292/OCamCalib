#pragma once

#include "gflags/gflags.h"

DECLARE_string(input_file);
DECLARE_string(output_file);
DECLARE_string(input_file_extension);
DECLARE_string(output_file_format);
DECLARE_string(output_file_prefix);
DECLARE_int32(width);
DECLARE_int32(height);
DECLARE_int32(taylor_order);
DECLARE_double(square_size);

void ParseShortFlags(int* argc, char*** argv);