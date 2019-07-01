cmake_minimum_required (VERSION 3.6)

project (retinaface_ncnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
find_package( OpenCV REQUIRED )
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
link_directories("${CMAKE_CURRENT_LIST_DIR}/lib")

add_executable(retinaface_ncnn src/anchor_generator.cpp  src/config.cpp  src/detect.cpp  src/tools.cpp 
src/anchor_generator.h src/config.h src/tools.h)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
target_link_libraries(retinaface_ncnn ${OpenCV_LIBS} ncnn)

#find_package(Boost REQUIRED COMPONENTS filesystem)
#if(NOT Boost_FOUND)
#    message("Not found Boost")
#endif()
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(retinaface_ncnn ${Boost_LIBRARIES})
