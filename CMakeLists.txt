cmake_minimum_required(VERSION 3.22.1)
project(voxel_mapping VERSION 0.1.0 LANGUAGES CXX CUDA)

option(USE_NVTX "Enable NVTX profiling support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75;86)

set(CCCL_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/thrust
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/cub
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/libcudacxx/include
)

set(CUCO_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cuco/include
)

find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

add_library(voxel_mapping SHARED
src/voxel_mapping.cu
src/voxel_mapping_impl.cu
src/update_generator.cu
src/gpu_hash_map.cu
src/grid_processor.cu
src/extraction_result.cpp
)

if(USE_NVTX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") # Jetson ARM 64-bit
    set(NVTX_LIBRARY_PATH /usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvToolsExt.so)
    set(NVTX_INCLUDE_DIR /usr/local/cuda-12.6/include)
  else() # Desktop x86_64 
    set(NVTX_LIBRARY_PATH /usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvToolsExt.so)
    set(NVTX_INCLUDE_DIR /usr/local/cuda-12.6/include)
  endif()

  if(EXISTS ${NVTX_LIBRARY_PATH})
    message(STATUS "NVTX enabled. Found at ${NVTX_LIBRARY_PATH}")
    set(NVTX_LIBRARY ${NVTX_LIBRARY_PATH})

    target_link_libraries(voxel_mapping PRIVATE ${NVTX_LIBRARY})
    target_compile_definitions(voxel_mapping PRIVATE USE_NVTX)
    target_include_directories(voxel_mapping PRIVATE ${NVTX_INCLUDE_DIR})

  else()
    message(FATAL_ERROR "NVTX requested but not found at ${NVTX_LIBRARY_PATH}")
  endif()
endif()

target_link_libraries(voxel_mapping
  PRIVATE
    CUDA::cudart
    spdlog::spdlog
    fmt::fmt
)

target_include_directories(voxel_mapping
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CUCO_INCLUDE_DIR}
    ${CCCL_INCLUDE_DIRS}
)

target_compile_options(voxel_mapping PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS voxel_mapping
  EXPORT export_voxel_mapping
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
    EXPORT export_voxel_mapping
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE voxel_mapping::
    DESTINATION share/${PROJECT_NAME}
)

configure_package_config_file(
  "cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "share/${PROJECT_NAME}"
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "share/${PROJECT_NAME}"
)