#include "voxel-mapping/voxel_mapping_impl.cuh"
#ifdef USE_NVTX
    #include <nvtx3/nvToolsExt.h>
#endif

namespace voxel_mapping {

VoxelMappingImpl::VoxelMappingImpl(const VoxelMappingParams& params)
    : resolution_(params.resolution), occupancy_threshold_(params.occupancy_threshold),
        free_threshold_(params.free_threshold), max_extraction_buffer_size_bytes_(params.max_extraction_buffer_size_bytes)
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

    size_t buffer_size = params.max_extraction_buffer_size_bytes;
    CHECK_CUDA_ERROR(cudaMalloc(&extraction_buffer_.d_data, buffer_size));
    CHECK_CUDA_ERROR(cudaHostAlloc(&extraction_buffer_.h_pinned_data, buffer_size, cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaEventCreate(&extraction_buffer_.event));
    extraction_buffer_.capacity_bytes = buffer_size;
    spdlog::info("Extraction buffer preallocated with a maximum size of {} bytes", buffer_size);
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

void VoxelMappingImpl::setup_extract_grid_block_graph() {
    // 1. Create a placeholder AABB_CUDA and functor.
    extraction_aabb_cuda_ = { {0, 0, 0}, {0, 0, 0} };
    extract_op_ = { static_cast<VoxelType*>(extraction_buffer_.d_data), 0, 0 };

    // 2. Create the graph
    CHECK_CUDA_ERROR(cudaGraphCreate(&extract_grid_block_graph_, 0));

    // 3. Add the kernel node.
    // The GpuHashMap::add_extraction_kernel_node function must be modified to use AABB_CUDA.
    grid_block_kernel_node_ = voxel_map_->add_extraction_kernel_node<ExtractionType::Block, ExtractOp>(
        extract_grid_block_graph_,
        {}, // No dependencies for the first node
        &extraction_aabb_cuda_,
        &extraction_slice_indices_,
        &extract_op_,
        extract_stream_
    );

    // 4. Add the memcpy node
    cudaMemcpyNodeParams memcpy_params = {};
    memcpy_params.dst = extraction_buffer_.h_pinned_data;
    memcpy_params.src = extraction_buffer_.d_data;
    memcpy_params.count = 0; // Placeholder size
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    
    CHECK_CUDA_ERROR(cudaGraphAddMemcpyNode(
        &grid_block_memcpy_node_,
        extract_grid_block_graph_,
        &grid_block_kernel_node_, // Depends on the kernel node
        1,
        &memcpy_params
    ));

    // 5. Add the event record node
    cudaEventRecordNodeParams event_params = {};
    event_params.event = extraction_buffer_.event;
    event_params.stream = extract_stream_;
    
    cudaGraphNode_t event_node;
    CHECK_CUDA_ERROR(cudaGraphAddEventRecordNode(
        &event_node,
        extract_grid_block_graph_,
        &grid_block_memcpy_node_, // Depends on the memcpy node
        1,
        &event_params
    ));

    // 6. Instantiate the graph
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&extract_grid_block_exec_graph_, extract_grid_block_graph_, nullptr, nullptr, 0));
    extract_grid_block_graph_is_initialized_ = true;
}

void VoxelMappingImpl::update_grid_block_graph_nodes() {
    // 1. Convert user-provided AABB to the device-side struct
    extraction_aabb_cuda_arg_.aabb_min_index = {
        aabb.min_corner_index.x,
        aabb.min_corner_index.y,
        aabb.min_corner_index.z
    };
    extraction_aabb_cuda_arg_.aabb_current_size = {
        aabb.size.x,
        aabb.size.y,
        aabb.size.z
    };

    // 2. Update kernel parameters.
    dim3 block_dim(32, 8, 1);
    dim3 grid_dim(
        (aabb.size.x + block_dim.x - 1) / block_dim.x,
        (aabb.size.y + block_dim.y - 1) / block_dim.y,
        (aabb.size.z + block_dim.z - 1) / block_dim.z
    );

    cudaKernelNodeParams kernel_params = {};
    kernel_params.func = (void*)query_map_and_process_kernel<ExtractionType::Block, ExtractOp>;
    kernel_params.gridDim = grid_dim;
    kernel_params.blockDim = block_dim;
    kernel_params.sharedMemBytes = 0;

    void* kernel_args[] = {
        &voxel_map_->get_map_ref(), // You'll need a getter for the map reference
        &extraction_aabb_cuda_arg_.aabb_min_index,
        &extraction_aabb_cuda_arg_.aabb_current_size,
        &extraction_slice_indices_arg_,
        &grid_block_extract_op_arg_
    };
    kernel_params.kernelParams = kernel_args;

    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(extract_grid_block_exec_graph_, grid_block_kernel_node_, &kernel_params));

    // 3. Update memcpy parameters.
    const size_t size_bytes = static_cast<size_t>(aabb.size.x) * aabb.size.y * aabb.size.z * sizeof(VoxelType);
    cudaMemcpyNodeParams memcpy_params = {};
    memcpy_params.dst = extraction_buffer_.h_pinned_data;
    memcpy_params.src = extraction_buffer_.d_data;
    memcpy_params.count = size_bytes;
    memcpy_params.kind = cudaMemcpyDeviceToHost;

    CHECK_CUDA_ERROR(cudaGraphExecMemcpyNodeSetParams(extract_grid_block_exec_graph_, grid_block_memcpy_node_, &memcpy_params));
}

ExtractionResult VoxelMappingImpl::extract_grid_block_data(const AABB& aabb) {
    set_extraction_params(aabb, SliceZIndices{0, 0, 0}); // No slices for block extraction
    if (!extract_grid_block_graph_is_initialized_) {
        setup_extract_grid_block_graph();
    }
    
    update_grid_block_graph_nodes();
    
    CHECK_CUDA_ERROR(cudaGraphLaunch(extract_grid_block_exec_graph_, extract_stream_));

    ExtractionResult result;
    auto impl = std::make_unique<ExtractionResultTyped<VoxelType>>();
    impl->h_pinned_data_ = extraction_buffer_.h_pinned_data;
    impl->event_ = extraction_buffer_.event;
    impl->size_bytes_ = static_cast<size_t>(aabb.size.x) * aabb.size.y * aabb.size.z * sizeof(VoxelType);
    result.pimpl_ = std::move(impl);
    
    return result;
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