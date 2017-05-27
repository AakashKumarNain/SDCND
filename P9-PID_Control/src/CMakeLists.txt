cmake_minimum_required(VERSION 3.7)
project(src)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        json.hpp
        main.cpp
        PID.cpp
        PID.h)

add_executable(src ${SOURCE_FILES})