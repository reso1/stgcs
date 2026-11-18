cmake_minimum_required(VERSION 3.20)
project(stgcs VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

find_package(drake CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CDD REQUIRED IMPORTED_TARGET cddlib)
add_library(stgcs
  src/geometry.cc src/trajectory.cc src/global_trajopt.cc src/graph.cc src/optimization.cc
  src/search.cc src/heuristic.cc src/multi_robot.cc)
add_library(stgcs::stgcs ALIAS stgcs)
target_compile_features(stgcs PUBLIC cxx_std_20)
target_include_directories(stgcs PUBLIC include PRIVATE src)
target_link_libraries(stgcs PUBLIC drake::drake PRIVATE PkgConfig::CDD)
target_compile_options(stgcs PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stgcs_planner src/planner_main.cc)
target_link_libraries(stgcs_planner PRIVATE stgcs::stgcs)
target_compile_options(stgcs_planner PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stgcs_trajopt src/trajopt_main.cc)
target_link_libraries(stgcs_trajopt PRIVATE stgcs::stgcs)
set_target_properties(stgcs_planner stgcs_trajopt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
