cmake_minimum_required(VERSION 2.8)
project(resize_demo)
#SET(CMAKE_VERBOSE_MAKEFILE on)
SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_C_COMPILER "gcc")

#set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_BUILD_TYPE "Debug" )
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC" )
set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
#set( CMAKE_CXX_FLAGS "-std=c++11 -funwind-tables -pthread -ggdb -g -march=native" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -fsanitize=address -fno-omit-frame-pointer -fuse-ld=gold -funwind-tables -pthread -ggdb -g -march=native" )

#set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(OpenCV_DIR "/usr/local/include/opencv3.2.0/share/OpenCV")
find_package( OpenCV REQUIRED )

set(LOCAL_SOURCE_PATH ${PROJECT_SOURCE_DIR}/)

include_directories(
${LOCAL_SOURCE_PATH}
${LOCAL_SOURCE_PATH}/include
)

add_executable(run_main ${LOCAL_SOURCE_PATH}/main.cpp )

target_link_libraries(run_main ${OpenCV_LIBS})
