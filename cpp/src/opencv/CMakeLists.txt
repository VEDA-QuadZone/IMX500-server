cmake_minimum_required(VERSION 3.14)
project(brighten)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(brighten pybinding.cpp filter.cpp)

target_include_directories(brighten PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(brighten PRIVATE ${OpenCV_LIBS})
