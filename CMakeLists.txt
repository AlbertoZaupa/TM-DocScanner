cmake_minimum_required(VERSION 3.22)
project(server_app)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(server_app main.cpp lib/paper_algorithm/binarization.h lib/utility.cpp lib/utility.h lib/processing.cpp lib/processing.h)
target_link_libraries( server_app  ${OpenCV_LIBS})
