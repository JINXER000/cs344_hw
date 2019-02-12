############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# minimum required cmake version
cmake_minimum_required(VERSION 2.8)
project(cs344_hw)
SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)
if(CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-std=c++11)
message(STATUS "optional:-std=c++11")
endif(CMAKE_COMPILER_IS_GNUCXX)

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS "-arch=sm_30;-g;-G" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

find_package(OpenCV REQUIRED)
if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()
include_directories(
        ${PROJECT_SOURCE_DIR}/include

)

SET(CMAKE_BUILD_TYPE Debug)

file( GLOB  hdr ./include/*.h )
file( GLOB  cu  ./src/*.cu)
file( GLOB  srcpp  ./src/*.cpp)

    
CUDA_ADD_EXECUTABLE(HW3 ${srcpp} ${hdr}
    ${cu} )
    #./src/HW3.cu ./src/student_func.cu)
target_link_libraries(
    HW3
    ${OpenCV_LIBS}
    -lpthread
    )
