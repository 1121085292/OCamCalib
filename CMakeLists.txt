cmake_minimum_required(VERSION 3.0)
project(calibration)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fPIC -Wall")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories("include")

add_executable(calibration 
        src/calibration.cc
        src/chessboard.cc
        src/params_gflas.cc
        src/prova_cost.cc
        src/utils.cc
        src/main.cc
)

target_link_libraries(calibration
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
)

add_executable(undistort src/undistort.cc src/utils.cc)
target_link_libraries(undistort ${OpenCV_LIBS})