cmake_minimum_required(VERSION 2.8)

project(alglib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(SOURCE_LIB alglibinternal.cpp
               alglibmisc.cpp
               ap.cpp 
               dataanalysis.cpp 
               diffequations.cpp 
               fasttransforms.cpp 
               integration.cpp 
               interpolation.cpp 
               linalg.cpp 
               optimization.cpp 
               solvers.cpp 
               specialfunctions.cpp 
               statistics.cpp
               stdafx.cpp)

add_library(alglib STATIC ${SOURCE_LIB})