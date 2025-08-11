#include "voxel-mapping/gpu_hash_map.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/internal_types.cuh"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

namespace voxel_mapping {

static __constant__ uint32_t d_freelist_capacity;
static __constant__ VoxelType d_log_odds_occupied;
static __constant__ VoxelType d_log_odds_free;
static __constant__ VoxelType d_log_odds_min;
static __constant__ VoxelType d_log_odds_max;
static __constant__ VoxelType d_occupancy_threshold;
static __constant__ VoxelType d_free_threshold;

GpuHashMap::GpuHashMap(size_t capacity, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold) {
    if (capacity == 0) {
        spdlog::error("Capacity must be greater than zero.");
        throw std::invalid_argument("Capacity must be greater than zero.");
    }
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        spdlog::error("Capacity exceeds the limit of the internal counter.");
        throw std::invalid_argument("Capacity exceeds the limit of the internal counter.");
    }

    freelist_capacity_ = capacity;

    map_capacity_ = static_cast<size_t>(static_cast<double>(freelist_capacity_) / 0.9);

    ChunkKey empty_key_sentinel = std::numeric_limits<ChunkKey>::max();
    ChunkKey erased_key_sentinel = std::numeric_limits<ChunkKey>::max() - 1;
    ChunkPtr empty_value_sentinel = nullptr;

    d_voxel_map_ = std::make_unique<ChunkMap>(
        map_capacity_,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel},
        cuco::erased_key{erased_key_sentinel}
    );

    size_t chunk_size_elements = chunk_dim() * chunk_dim() * chunk_dim();
    size_t chunk_size_bytes = sizeof(VoxelType) * chunk_size_elements;
    size_t pool_size_bytes = chunk_size_bytes * freelist_capacity_;

    CHECK_CUDA_ERROR(cudaMalloc(&global_memory_pool_, pool_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_, sizeof(ChunkPtr) * freelist_capacity_));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_counter_, sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemset(global_memory_pool_, default_voxel_value(), pool_size_bytes));

    std::vector<ChunkPtr> h_chunk_ptrs(freelist_capacity_);
    for (size_t i = 0; i < freelist_capacity_; ++i) {
        h_chunk_ptrs[i] = global_memory_pool_ + (i * chunk_size_elements);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_,
        h_chunk_ptrs.data(),
        sizeof(ChunkPtr) * freelist_capacity_,
        cudaMemcpyHostToDevice
    ));

    uint32_t initial_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_counter_,
        &initial_count,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_freelist_capacity, &freelist_capacity_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_occupied, &log_odds_occupied, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_free, &log_odds_free, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_min, &log_odds_min, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_max, &log_odds_max, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
}

GpuHashMap::~GpuHashMap() = default;

void GpuHashMap::get_freelist_counter(uint32_t* freelist_counter) {
    CHECK_CUDA_ERROR(cudaMemcpy(freelist_counter, freelist_counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__device__ inline ChunkPtr allocate_chunk_from_pool(
    ChunkPtr* freelist, uint32_t* counter, uint32_t freelist_capacity) {
    uint32_t index = *((volatile uint32_t*)counter);

    while (index < freelist_capacity) {
        uint32_t old_value = atomicCAS(counter, index, index + 1);
        if (old_value == index) {
            return freelist[index];
        }
        index = old_value;
    }
    return nullptr;
}

__device__ inline void atomic_add_and_clamp(VoxelType* address, VoxelType val, VoxelType min_val, VoxelType max_val) {
    VoxelType old_val = *address;
    VoxelType assumed_val;
    
    do {
        assumed_val = old_val;
        VoxelType new_val = min(max(assumed_val + val, min_val), max_val);
        old_val = atomicCAS(address, assumed_val, new_val);
    } while (assumed_val != old_val);
}

__device__ inline void update_voxels(ChunkPtr chunk_ptr, uint32_t intra_chunk_idx, UpdateType update_type) {
    if (update_type == UpdateType::Unknown) return;
    VoxelType update_value = (update_type == UpdateType::Free) ? d_log_odds_free : d_log_odds_occupied;
    atomic_add_and_clamp(
        &chunk_ptr[intra_chunk_idx],
        update_value,
        d_log_odds_min,
        d_log_odds_max
    );
}

__global__ void update_map_from_aabb(
    ChunkMapRef map_ref,
    ChunkPtr* freelist,
    uint32_t* freelist_counter,
    UpdateType* d_aabb,
    int3 aabb_min_index,
    int3 aabb_size)
    {
        auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        
        int aabb_x = blockIdx.x * blockDim.x + threadIdx.x;
        int aabb_y = blockIdx.y * blockDim.y + threadIdx.y;
        int aabb_z = blockIdx.z;
        bool is_in_bounds = (aabb_x < aabb_size.x && aabb_y < aabb_size.y && aabb_z < aabb_size.z);
        ChunkKey my_key = 0;
        uint32_t intra_chunk_idx = 0;
        UpdateType update_type = UpdateType::Unknown;
        if (is_in_bounds) {
            int global_x = aabb_min_index.x + aabb_x;
            int global_y = aabb_min_index.y + aabb_y;
            int global_z = aabb_min_index.z + aabb_z;
            my_key = get_chunk_key(global_x, global_y, global_z);
            intra_chunk_idx = get_intra_chunk_index(global_x, global_y, global_z);
            update_type = d_aabb[block_1d_index(aabb_x, aabb_y, aabb_z, aabb_size.x, aabb_size.y)];
        }
        
        unsigned int active_mask = group.ballot(is_in_bounds && update_type != UpdateType::Unknown);
        while (active_mask != 0) {
            int leader_lane = __ffs(active_mask) - 1;
            ChunkKey current_key = group.shfl(my_key, leader_lane);
            ChunkPtr final_chunk_ptr = invalid_chunk_ptr();
            
            auto it = map_ref.find(group, current_key);
            group.sync();
            
            if (it != map_ref.end()) {
                final_chunk_ptr = it->second;
            } else {
                ChunkMap::value_type pair_to_insert{current_key, invalid_chunk_ptr()};
                auto result = map_ref.insert_and_find(group, pair_to_insert);
                
                bool was_inserted = result.second;
                auto insert_it = result.first;
                
                if (was_inserted) {
                    if (group.thread_rank() == leader_lane) {
                        ChunkPtr new_chunk_ptr = allocate_chunk_from_pool(freelist, freelist_counter, d_freelist_capacity);
                        if (new_chunk_ptr != nullptr) {
                            auto ref = cuda::atomic_ref<ChunkPtr, cuda::thread_scope_device>{insert_it->second};
                            ref.exchange(new_chunk_ptr, cuda::memory_order_release);
                            final_chunk_ptr = new_chunk_ptr;
                        }
                    }
                    final_chunk_ptr = group.shfl(final_chunk_ptr, leader_lane);
                } else {
                    if (insert_it != map_ref.end()) {
                        final_chunk_ptr = insert_it->second;
                    }
                }
            }
            
            if (my_key == current_key && final_chunk_ptr != invalid_chunk_ptr() && final_chunk_ptr != nullptr) {
                update_voxels(final_chunk_ptr, intra_chunk_idx, update_type);
            }
            
            unsigned int processed_mask = group.ballot(my_key == current_key);
            active_mask &= ~processed_mask;
        }
    }

std::vector<cudaGraphNode_t> GpuHashMap::add_nodes_to_insertion_graph(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& preceding_dependencies,
    const AABB_CUDA& aabb_update,
    UpdateType* d_aabb_ptr)
{
    dim3 threads(32, 8, 1);
    dim3 blocks(
        (aabb_update.aabb_current_size.x + threads.x - 1) / threads.x,
        (aabb_update.aabb_current_size.y + threads.y - 1) / threads.y,
        aabb_update.aabb_current_size.z
    );

    ChunkMapRef map_ref = d_voxel_map_->ref(cuco::op::find, cuco::op::insert_and_find, cuco::op::erase);
    insertion_kernel_args_[0] = &map_ref;
    insertion_kernel_args_[1] = (void*)&freelist_;
    insertion_kernel_args_[2] = (void*)&freelist_counter_;
    insertion_kernel_args_[3] = (void*)&d_aabb_ptr;
    insertion_kernel_args_[4] = (void*)&aabb_update.aabb_min_index;
    insertion_kernel_args_[5] = (void*)&aabb_update.aabb_current_size;

    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)update_map_from_aabb;
    kernel_params.gridDim = blocks;
    kernel_params.blockDim = threads;
    kernel_params.kernelParams = insertion_kernel_args_;

    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &update_map_node_,
        graph,
        preceding_dependencies.data(),
        preceding_dependencies.size(),
        &kernel_params));

    return {update_map_node_};
}


void GpuHashMap::update_insertion_graph_nodes(
    cudaGraphExec_t executable_graph,
    const AABB_CUDA& aabb_update,
    UpdateType* d_aabb_ptr)
{
    ChunkMapRef map_ref = d_voxel_map_->ref(cuco::op::find, cuco::op::insert_and_find, cuco::op::erase);
    insertion_kernel_args_[0] = &map_ref;

    insertion_kernel_args_[3] = (void*)&d_aabb_ptr;
    insertion_kernel_args_[4] = (void*)&aabb_update.aabb_min_index;
    insertion_kernel_args_[5] = (void*)&aabb_update.aabb_current_size;

    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)update_map_from_aabb;

    dim3 threads(32, 8, 1);
    dim3 blocks(
        (aabb_update.aabb_current_size.x + threads.x - 1) / threads.x,
        (aabb_update.aabb_current_size.y + threads.y - 1) / threads.y,
        aabb_update.aabb_current_size.z
    );

    kernel_params.gridDim = blocks;
    kernel_params.blockDim = threads;
    kernel_params.kernelParams = insertion_kernel_args_;

    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(
        executable_graph,
        update_map_node_,
        &kernel_params
    ));
}

void GpuHashMap::clear_chunks(const int3& current_chunk_pos) {
    std::vector<ChunkKey> h_keys;
    std::vector<ChunkPtr> h_values;
    retrieve_all_chunks(h_keys, h_values);
    if (h_keys.empty()) return;
    
    std::vector<ChunkInfo> sorted_chunks = prioritize_chunks_for_clearing(h_keys, h_values, current_chunk_pos);
    
    std::vector<ChunkKey> keys_to_erase;
    std::vector<ChunkPtr> ptrs_to_deallocate;
    size_t num_to_clear_goal = freelist_capacity_ * 0.1; 
    size_t cleared_count = 0;
    
    for (const auto& chunk : sorted_chunks) {
        if (chunk.is_invalid || (cleared_count < num_to_clear_goal)) {
            keys_to_erase.push_back(chunk.key);
            if (!chunk.is_invalid) {
                ptrs_to_deallocate.push_back(chunk.ptr);
            }
            cleared_count++;
        }
    }
    execute_chunk_removal(keys_to_erase, ptrs_to_deallocate);
}

void GpuHashMap::retrieve_all_chunks(std::vector<ChunkKey>& h_keys, std::vector<ChunkPtr>& h_values) {
    size_t num_pairs = d_voxel_map_->size();
    if (num_pairs == 0) return;
    
    thrust::device_vector<ChunkKey> d_keys(num_pairs);
    thrust::device_vector<ChunkPtr> d_values(num_pairs);
    d_voxel_map_->retrieve_all(d_keys.begin(), d_values.begin());
    
    h_keys.resize(num_pairs);
    h_values.resize(num_pairs);
    thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());
}

std::vector<ChunkInfo> GpuHashMap::prioritize_chunks_for_clearing(
    const std::vector<ChunkKey>& h_keys,
    const std::vector<ChunkPtr>& h_values,
    const int3& current_chunk_pos)
    {
        std::vector<ChunkInfo> all_chunks;
        all_chunks.reserve(h_keys.size());
        
        for (size_t i = 0; i < h_keys.size(); ++i) {
            int3 chunk_indices = unpack_key_to_indices(h_keys[i]);
            float dx = static_cast<float>(chunk_indices.x - current_chunk_pos.x);
            float dy = static_cast<float>(chunk_indices.y - current_chunk_pos.y);
            float dz = static_cast<float>(chunk_indices.z - current_chunk_pos.z);
            
            all_chunks.push_back({
                .key = h_keys[i],
                .ptr = h_values[i],
                .distance_sq = dx * dx + dy * dy + dz * dz,
                .is_invalid = (h_values[i] == invalid_chunk_ptr())
            });
        }
        
        std::sort(all_chunks.begin(), all_chunks.end(), 
        [](const ChunkInfo& a, const ChunkInfo& b) {
            if (a.is_invalid != b.is_invalid) {
                return a.is_invalid;
            }
            return a.distance_sq > b.distance_sq;
        }
    );
    return all_chunks;
}

__global__ void deallocate_chunks_to_pool(
    ChunkPtr* ptrs_to_deallocate,
    uint32_t num_ptrs_to_deallocate,
    ChunkPtr* freelist,
    uint32_t* freelist_counter) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_ptrs_to_deallocate) {
        uint32_t freelist_idx = atomicSub(freelist_counter, 1);
        
        freelist[freelist_idx - 1] = ptrs_to_deallocate[idx];
    }
}

void GpuHashMap::execute_chunk_removal(
    const std::vector<ChunkKey>& keys_to_erase,
    const std::vector<ChunkPtr>& ptrs_to_deallocate)
{
    if (keys_to_erase.empty()) {
        return;
    }

    thrust::device_vector<ChunkKey> d_keys_to_erase = keys_to_erase;
    d_voxel_map_->erase(d_keys_to_erase.begin(), d_keys_to_erase.end());

    if (!ptrs_to_deallocate.empty()) {
        size_t chunk_size_bytes = sizeof(VoxelType) * chunk_dim() * chunk_dim() * chunk_dim();
        uint8_t reset_value = default_voxel_value();
        
        for (ChunkPtr ptr : ptrs_to_deallocate) {
            CHECK_CUDA_ERROR(cudaMemsetAsync(ptr, reset_value, chunk_size_bytes));
        }

        uint32_t num_to_deallocate = ptrs_to_deallocate.size();
        thrust::device_vector<ChunkPtr> d_ptrs_to_deallocate = ptrs_to_deallocate;

        dim3 block_dim(256);
        dim3 grid_dim((num_to_deallocate + block_dim.x - 1) / block_dim.x);
        
        deallocate_chunks_to_pool<<<grid_dim, block_dim>>>(
            thrust::raw_pointer_cast(d_ptrs_to_deallocate.data()),
            num_to_deallocate,
            freelist_,
            freelist_counter_
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

} // namespace voxel_mapping