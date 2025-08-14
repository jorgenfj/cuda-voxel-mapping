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

struct ExtractionBuffer {
    void* d_data = nullptr;
    void* h_pinned_data = nullptr;
    cudaEvent_t event = nullptr;
    size_t current_size_bytes = 0;

    ~ExtractionBuffer() {
        if (d_data) cudaFree(d_data);
        if (h_pinned_data) cudaFreeHost(h_pinned_data);
        if (event) cudaEventDestroy(event);
    }
};

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
        set_extraction_params<Type>(aabb, slice_indices);
        if(extraction_buffer_.current_size_bytes == max_extraction_buffer_size_bytes_) {
            spdlog::warn("Extraction buffer size ({}) is at maximum capacity. Consider increasing max_extraction_buffer_size_bytes in the config.", extraction_buffer_.current_size_bytes);
        }
        extract_op_ = {static_cast<VoxelType*>(extraction_buffer_.d_data), extraction_aabb_cuda_.aabb_current_size.x, extraction_aabb_cuda_.aabb_current_size.y};
        if (!extract_grid_graph_is_initialized_) {
            setup_extract_grid_graph<Type, ExtractOp>();
        }

        update_grid_graph_nodes<Type, ExtractOp>();

        CHECK_CUDA_ERROR(cudaGraphLaunch(extract_grid_exec_graph_, extract_stream_));

        auto impl = std::make_unique<ExtractionResultTyped<VoxelType>>(
            extraction_buffer_.h_pinned_data,
            extraction_buffer_.current_size_bytes,
            extraction_buffer_.event
        );
        ExtractionResult result;
        result.pimpl_ = std::move(impl);
        return result;   
    }
    
private:

    template <ExtractionType Type>
    void set_extraction_params(const AABB& aabb, const SliceZIndices& slice_indices) {
        int size_z = (Type == ExtractionType::Block) ? aabb.size.z : slice_indices.count;
        size_t requested_size_bytes = aabb.size.x * aabb.size.y * size_z * sizeof(VoxelType);
        
        if (requested_size_bytes > max_extraction_buffer_size_bytes_) {
            size_t max_z = max_extraction_buffer_size_bytes_ / (aabb.size.x * aabb.size.y * sizeof(VoxelType));

            if (static_cast<int>(max_z) < aabb.size.z) {
                size_z = static_cast<int>(max_z);
                spdlog::warn(
                    "Requested AABB volume ({}) is too large for the buffer ({}). "
                    "The Z dimension has been clipped to {}.",
                    requested_size_bytes, max_extraction_buffer_size_bytes_, max_z
                );
            }
        }
        extraction_buffer_.current_size_bytes = min(max_extraction_buffer_size_bytes_, aabb.size.x * aabb.size.y * size_z * sizeof(VoxelType));

        extraction_aabb_cuda_.aabb_min_index = {aabb.min_corner_index.x, aabb.min_corner_index.y, aabb.min_corner_index.z};
        extraction_aabb_cuda_.aabb_current_size = {aabb.size.x, aabb.size.y, size_z};
        extraction_slice_indices_ = slice_indices;
        extraction_slice_indices_.count = size_z;
    }

    template <ExtractionType Type, typename ExtractOp>
    void setup_extract_grid_graph() {
        CHECK_CUDA_ERROR(cudaGraphCreate(&extract_grid_graph_, 0));

        extract_grid_kernel_node_ = voxel_map_->add_extraction_kernel_node<Type, ExtractOp>(
            extract_grid_graph_,
            {},
            &extraction_aabb_cuda_,
            &extraction_slice_indices_,
            &extract_op_,
            extract_grid_kernel_node_params_
        );

        cudaMemcpy3DParms memcpy3d_params = {};

        size_t total_copy_bytes = extraction_buffer_.current_size_bytes;

        memcpy3d_params.srcPtr = make_cudaPitchedPtr(extraction_buffer_.d_data, total_copy_bytes, total_copy_bytes, 1);
        memcpy3d_params.dstPtr = make_cudaPitchedPtr(extraction_buffer_.h_pinned_data, total_copy_bytes, total_copy_bytes, 1);
        memcpy3d_params.extent = make_cudaExtent(total_copy_bytes, 1, 1);
        memcpy3d_params.kind = cudaMemcpyDeviceToHost;
            
        CHECK_CUDA_ERROR(cudaGraphAddMemcpyNode(
            &extract_grid_memcpy_node_,
            extract_grid_graph_,
            &extract_grid_kernel_node_,
            1,
            &memcpy3d_params));

        cudaGraphNode_t event_node;
        CHECK_CUDA_ERROR(cudaGraphAddEventRecordNode(
            &event_node,
            extract_grid_graph_,
            &extract_grid_memcpy_node_,
            1,
            extraction_buffer_.event
        ));
        
        CHECK_CUDA_ERROR(cudaGraphInstantiate(&extract_grid_exec_graph_, extract_grid_graph_, nullptr, nullptr, 0));
        extract_grid_graph_is_initialized_ = true;
    }

    template <ExtractionType Type, typename ExtractOp>
    void update_grid_graph_nodes() {
        voxel_map_->update_extraction_kernel_node<Type, ExtractOp>(
            extract_grid_exec_graph_,
            extract_grid_kernel_node_,
            &extraction_aabb_cuda_,
            &extraction_slice_indices_,
            &extract_op_,
            extract_grid_kernel_node_params_
        );

        cudaMemcpy3DParms memcpy3d_params = {};
        
        size_t total_copy_bytes = extraction_buffer_.current_size_bytes;

        memcpy3d_params.srcPtr = make_cudaPitchedPtr(extraction_buffer_.d_data, total_copy_bytes, total_copy_bytes, 1);
        memcpy3d_params.dstPtr = make_cudaPitchedPtr(extraction_buffer_.h_pinned_data, total_copy_bytes, total_copy_bytes, 1);
        memcpy3d_params.extent = make_cudaExtent(total_copy_bytes, 1, 1);
        memcpy3d_params.kind = cudaMemcpyDeviceToHost;

        CHECK_CUDA_ERROR(cudaGraphExecMemcpyNodeSetParams(extract_grid_exec_graph_, extract_grid_memcpy_node_, &memcpy3d_params));
    }

    void setup_insert_graph(const float* depth_image, const float* transform);
    float resolution_;
    int occupancy_threshold_;
    int free_threshold_;
    ExtractionBuffer extraction_buffer_;
    size_t max_extraction_buffer_size_bytes_;
    mutable std::shared_mutex map_mutex_;
    cudaStream_t insert_stream_ = nullptr;
    cudaStream_t extract_stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;
    std::unique_ptr<GridProcessor> grid_processor_;

    cudaGraph_t insert_graph_ = nullptr;
    cudaGraphExec_t insert_exec_graph_ = nullptr;
    bool insert_graph_is_initialized_ = false;

    cudaGraph_t extract_grid_graph_ = nullptr;
    cudaGraphExec_t extract_grid_exec_graph_ = nullptr;
    bool extract_grid_graph_is_initialized_ = false;
    cudaGraphNode_t extract_grid_memcpy_node_;
    void* extract_grid_kernel_node_params_[5];
    cudaGraphNode_t extract_grid_kernel_node_;

    cudaGraph_t extract_grid_slice_graph_ = nullptr;
    cudaGraphExec_t extract_grid_slice_exec_graph_ = nullptr;
    bool extract_grid_slice_graph_is_initialized_ = false;

    cudaGraph_t extract_edt_slice_graph_ = nullptr;
    cudaGraphExec_t extract_edt_slice_exec_graph_ = nullptr;
    bool extract_edt_slice_graph_is_initialized_ = false;

    cudaGraph_t extract_edt_block_graph_ = nullptr;
    cudaGraphExec_t extract_edt_block_exec_graph_ = nullptr;
    bool extract_edt_block_graph_is_initialized_ = false;

    AABB_CUDA extraction_aabb_cuda_;
    SliceZIndices extraction_slice_indices_;
    ExtractOp extract_op_;
};

}
#endif