cmake_minimum_required(VERSION 2.8)

project(SignObjectsDetection)

include_directories(headers)
include_directories(alglib)

include_directories(SYSTEM ${OPENCV_DIR})
include_directories(OpenCV_DIR)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(SRC_LIST SignObjectsDetection.cpp
             sources/auto_correct.cpp 
             sources/detection_object.cpp 
             sources/mouse_click.cpp 
             sources/thread_pool.cpp)

set(HEADER headers/auto_correct.h
           headers/detection_objects.h 
           headers/mouse_click.h 
           headers/thread_pool.h 
           headers/targetver.h)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${HEADER})

add_subdirectory(alglib)
target_link_libraries(${PROJECT_NAME} alglib ${OpenCV_LIBS} ${OpenCV_LIBRARIES})