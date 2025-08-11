#include "voxel-mapping/voxel_mapping_impl.cuh"
#ifdef USE_NVTX
    #include <nvtx3/nvToolsExt.h>
#endif

namespace voxel_mapping {

VoxelMappingImpl::VoxelMappingImpl(const VoxelMappingParams& params)
    : resolution_(params.resolution), occupancy_threshold_(params.occupancy_threshold),
        free_threshold_(params.free_threshold)
{
    if (resolution_ <= 0) {
        throw std::invalid_argument("Resolution must be positive");
    }
    CHECK_CUDA_ERROR(cudaStreamCreate(&insert_stream_));
    CHECK_CUDA_ERROR(cudaStreamCreate(&extract_stream_));
    voxel_map_ = std::make_unique<GpuHashMap>(params.chunk_capacity, params.log_odds_occupied, params.log_odds_free, params.log_odds_min, params.log_odds_max, params.occupancy_threshold, params.free_threshold);
    spdlog::info("Voxel map initialized on GPU with initial capacity {}", params.chunk_capacity);
    update_generator_ = std::make_unique<UpdateGenerator>(resolution_, params.min_depth, params.max_depth);
    spdlog::info("Update generator initialized with resolution: {}, min_depth: {}, max_depth: {}", resolution_, params.min_depth, params.max_depth);
    grid_processor_ = std::make_unique<GridProcessor>(params.occupancy_threshold, params.free_threshold, params.edt_max_distance);
    spdlog::info("Grid processor initialized with occupancy threshold: {}, free threshold: {}, EDT max distance: {}", params.occupancy_threshold, params.free_threshold, params.edt_max_distance);
}

VoxelMappingImpl::~VoxelMappingImpl() {
    if (insert_stream_) cudaStreamDestroy(insert_stream_);
    if (extract_stream_) cudaStreamDestroy(extract_stream_);
    if (insert_graph_is_initialized_) {
        cudaGraphExecDestroy(insert_exec_graph_);
        cudaGraphDestroy(insert_graph_);
    }
}

void VoxelMappingImpl::setup_insert_graph(const float* depth_image, const float* transform) {
    CHECK_CUDA_ERROR(cudaGraphCreate(&insert_graph_, 0));

    std::vector<cudaGraphNode_t> dependencies;

    update_generator_->compute_active_aabb(transform);
    const AABB_CUDA& aabb_update = update_generator_->get_aabb_update_struct();
    UpdateType* d_aabb_ptr = update_generator_->get_device_aabb_ptr();
    dependencies = update_generator_->add_nodes_to_insertion_graph(insert_graph_, dependencies, aabb_update);

    dependencies = voxel_map_->add_nodes_to_insertion_graph(
        insert_graph_,
        dependencies,
        aabb_update,
        d_aabb_ptr
    );

    CHECK_CUDA_ERROR(cudaGraphInstantiate(&insert_exec_graph_, insert_graph_, nullptr, nullptr, 0));
    insert_graph_is_initialized_ = true;
}

void VoxelMappingImpl::integrate_depth(const float* depth_image, const float* transform) {
    #ifdef USE_NVTX
        nvtx3::scoped_range r{"Integrate Depth"};
    #endif
    if (!insert_graph_is_initialized_) {
        setup_insert_graph(depth_image, transform);
    }
    
    update_generator_->compute_active_aabb(transform);
    const AABB_CUDA& aabb_update = update_generator_->get_aabb_update_struct();
    UpdateType* d_aabb_ptr = update_generator_->get_device_aabb_ptr();
    
    float* h_pinned_depth = update_generator_->get_host_pinned_depth_ptr();
    float* h_pinned_transform = update_generator_->get_host_pinned_transform_ptr();
    memcpy(h_pinned_depth, depth_image, update_generator_->get_depth_buffer_size());
    memcpy(h_pinned_transform, transform, 16 * sizeof(float));

    update_generator_->update_insertion_graph_nodes(
        insert_exec_graph_, aabb_update);

    voxel_map_->update_insertion_graph_nodes(
        insert_exec_graph_,
        aabb_update, d_aabb_ptr);

    CHECK_CUDA_ERROR(cudaGraphLaunch(insert_exec_graph_, insert_stream_));
}

void VoxelMappingImpl::set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height) {
    update_generator_->set_camera_properties(fx, fy, cx, cy, width, height);
}

AABB VoxelMappingImpl::get_current_aabb() const {
    return update_generator_->get_aabb();
}

Frustum VoxelMappingImpl::get_frustum() const {
    return update_generator_->get_frustum();
}

void VoxelMappingImpl::query_free_chunk_capacity() {
    #ifdef USE_NVTX
        nvtx3::scoped_range r{"query_free_chunk_capacity"};
    #endif
    uint32_t current_freelist_count;
    voxel_map_->get_freelist_counter(&current_freelist_count);
    size_t freelist_capacity = voxel_map_->get_freelist_capacity();
    uint32_t threshold = static_cast<uint32_t>(freelist_capacity * 0.95);

    if (current_freelist_count >= threshold) {
        int3 current_chunk_pos = update_generator_->get_current_chunk_position();
        
        spdlog::info("Freelist usage ({}) is above 95% threshold ({}). Clearing distant chunks.", 
                        current_freelist_count, threshold);

        {
            std::unique_lock lock(map_mutex_);
            voxel_map_->clear_chunks(current_chunk_pos);
        }
    }
}

} // namespace voxel_mapping