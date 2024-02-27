cmake_minimum_required(VERSION 3.25)
project(DAT)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
# Visual Studio compiler with static runtime libraries
if (MSVC AND MT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif ()

include_directories(include)

find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(DAT main.cpp)
target_link_libraries(DAT ${Boost_LIBRARIES})
target_link_libraries(DAT optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(DAT ${GUROBI_LIBRARY})

enable_testing()

add_subdirectory(test)
