cmake_minimum_required(VERSION 3.10)
project(customop)
add_definitions(-std=c++17)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS}
    /home/ll/Documents/app/onnxruntime-linux-x64-1.22.0/include)
link_directories(/home/ll/Documents/app/onnxruntime-linux-x64-1.22.0/lib)

set(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/custom_op.cc)

# add_library(${PROJECT_NAME} ${SRCS})
add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME}
onnxruntime
)
