cmake_minimum_required(VERSION 3.24)

get_filename_component(proj_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

project(${proj_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O1 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mlp STATIC
  src/mlp.cpp
)

target_include_directories(mlp
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/ann-lib"
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include/"
)

add_executable(${CMAKE_PROJECT_NAME}
  src/main.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME}
  PUBLIC mlp
)

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(third_party/Unity)
  add_subdirectory(test)
endif()
