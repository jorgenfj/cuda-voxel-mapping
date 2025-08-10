#ifndef VOXEL_MAPPING_IMPL_HPP
#define VOXEL_MAPPING_IMPL_HPP

#include "voxel-mapping/voxel_mapping.hpp"
#include <cuda_runtime.h>
#include <shared_mutex>
#include <memory>

#include "voxel-mapping/gpu_hash_map.cuh"
#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/grid_processor.cuh"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/internal_types.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/extraction_result_impl.hpp"

namespace voxel_mapping {

class VoxelMappingImpl {
public:
    VoxelMappingImpl(const VoxelMappingParams& params);
    ~VoxelMappingImpl();

    void integrate_depth(const float* depth_image, const float* transform);

    void query_free_chunk_capacity();

    void set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height);
    AABB get_current_aabb() const;
    Frustum get_frustum() const;

    template <ExtractionType Type>
    ExtractionResult extract_grid_data(const AABB& aabb, const SliceZIndices& slice_indices) {
    ExtractionResult result;

    const int aabb_size_x = aabb.size.x;
    const int aabb_size_y = aabb.size.y;
    
    int num_z_layers;
    if constexpr (Type == ExtractionType::Block) {
        num_z_layers = aabb.size.z;
    } else {
        num_z_layers = slice_indices.count;
    }

    const size_t total_elements = static_cast<size_t>(aabb_size_x) * aabb_size_y * num_z_layers;

    if (total_elements == 0) {
        return result;
    }

    auto impl = std::make_unique<ExtractionResultTyped<VoxelType>>();
    impl->size_bytes_ = total_elements * sizeof(VoxelType);

    CHECK_CUDA_ERROR(cudaMalloc(&impl->d_data_, impl->size_bytes_));
    CHECK_CUDA_ERROR(cudaHostAlloc(&impl->h_pinned_data_, impl->size_bytes_, cudaHostAllocDefault));
    
    ExtractOp extract_op = {static_cast<VoxelType*>(impl->d_data_), aabb.size.x, aabb.size.y};

    {
        std::shared_lock lock(map_mutex_);
        voxel_map_->launch_map_extraction_kernel<Type>(
            aabb, 
            slice_indices, 
            extract_op,
            extract_stream_
        );
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        impl->h_pinned_data_,
        impl->d_data_,
        impl->size_bytes_,
        cudaMemcpyDeviceToHost,
        extract_stream_
    ));

    CHECK_CUDA_ERROR(cudaEventCreate(&impl->event_));
    CHECK_CUDA_ERROR(cudaEventRecord(impl->event_, extract_stream_));

    result.pimpl_ = std::move(impl);
    return result;
}

template <ExtractionType Type>
ExtractionResult extract_edt_data(const AABB& aabb, const SliceZIndices& slice_indices) {
    ExtractionResult result;

    const int aabb_size_x = aabb.size.x;
    const int aabb_size_y = aabb.size.y;
    
    int num_z_layers;
    int max_dim_sq = 0;
    if constexpr (Type == ExtractionType::Block) {
        num_z_layers = aabb.size.z;
        max_dim_sq = std::max({aabb_size_x, aabb_size_y, aabb.size.z});
        max_dim_sq *= max_dim_sq;
    } else {
        num_z_layers = slice_indices.count;
        max_dim_sq = std::max(aabb_size_x, aabb_size_y);
        max_dim_sq *= max_dim_sq;
    }

    const size_t total_elements = static_cast<size_t>(aabb_size_x) * aabb_size_y * num_z_layers;

    if (total_elements == 0) {
        return result;
    }
    auto impl = std::make_unique<ExtractionResultTyped<int>>();
    impl->size_bytes_ = total_elements * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&impl->d_data_, impl->size_bytes_));
    CHECK_CUDA_ERROR(cudaHostAlloc(&impl->h_pinned_data_, impl->size_bytes_, cudaHostAllocDefault));

    ExtractBinaryOp extract_op = {static_cast<int*>(impl->d_data_), aabb_size_x, aabb_size_y, free_threshold_, max_dim_sq};

    {
        std::shared_lock lock(map_mutex_);
        voxel_map_->launch_map_extraction_kernel<Type>(
            aabb, 
            slice_indices, 
            extract_op,
            extract_stream_
        );
    }

    if constexpr (Type == ExtractionType::Block) {
        grid_processor_->launch_3d_edt_kernels(
            static_cast<int*>(impl->d_data_), aabb_size_x, aabb_size_y, num_z_layers, extract_stream_
        );
    } else {
        grid_processor_->launch_edt_slice_kernels(
            static_cast<int*>(impl->d_data_), aabb_size_x, aabb_size_y, num_z_layers, extract_stream_
        );
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        impl->h_pinned_data_,
        impl->d_data_,
        impl->size_bytes_,
        cudaMemcpyDeviceToHost,
        extract_stream_
    ));

    CHECK_CUDA_ERROR(cudaEventCreate(&impl->event_));
    CHECK_CUDA_ERROR(cudaEventRecord(impl->event_, extract_stream_));

    result.pimpl_ = std::move(impl);
    return result;
}

private:
    float resolution_;
    int occupancy_threshold_;
    int free_threshold_;
    mutable std::shared_mutex map_mutex_;
    cudaStream_t insert_stream_ = nullptr;
    cudaStream_t extract_stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;
    std::unique_ptr<GridProcessor> grid_processor_;
};

}
#endif