#ifndef GPU_HASH_MAP_CUH
#define GPU_HASH_MAP_CUH

#include <cuco/static_map.cuh>
#include <voxel-mapping/types.hpp>
#include <voxel-mapping/host_macros.hpp>
#include "voxel-mapping/internal_types.cuh"
#include "voxel-mapping/map_utils.cuh"
#include <memory>

namespace voxel_mapping {

/**
 * @brief A generic kernel that traverses a region, queries the voxel map, and applies a user-defined operation.
 * * @tparam ExtractionTag A tag type (e.g., Block, Slice) to
 * determine how global_z is calculated.
 * @tparam Functor A callable type (e.g., a lambda) that processes the retrieved voxel value.
 * @param map_ref A constant reference to the chunk map.
 * @param aabb_min_index The minimum index of the AABB in world coordinates.
 * @param aabb_size The size of the AABB in grid coordinates (size.z used only with BlockExtractionTag).
 * @param z_indices A struct containing an array of Z indices and their count (used only with SliceExtractionTag).
 * @param op The callable object to execute for each voxel. It is passed the VoxelType and the
 * local coordinates (aabb_x, aabb_y, aabb_z) to write the result to the output.
 */
template <ExtractionType Tag, typename Functor> __global__ void extract_from_map(
    ConstChunkMapRef map_ref,
    int3 aabb_min_index,
    int3 aabb_size,
    SliceZIndices z_indices,
    Functor op) {
    auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    int aabb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int aabb_y = blockIdx.y * blockDim.y + threadIdx.y;
    int aabb_z = blockIdx.z;

    bool is_in_bounds = (aabb_x < aabb_size.x && aabb_y < aabb_size.y);

    int global_z;

    if constexpr (Tag == ExtractionType::Slice) {
        if (aabb_z < z_indices.count) {
            global_z = z_indices.indices[aabb_z];
        } else {
            is_in_bounds = false;
        }
    } else {
        if (aabb_z < aabb_size.z) {
            global_z = aabb_min_index.z + aabb_z;
        } else {
            is_in_bounds = false;
        }
    }

    ChunkKey my_key = 0;
    uint32_t output_idx = 0;
    
    int global_x = aabb_min_index.x + aabb_x;
    int global_y = aabb_min_index.y + aabb_y;

    if (is_in_bounds) {
        output_idx = block_1d_index(aabb_x, aabb_y, aabb_z, aabb_size.x, aabb_size.y);
        my_key = get_chunk_key(global_x, global_y, global_z);
    }
    
    unsigned int active_mask = group.ballot(is_in_bounds);
    
    while(active_mask != 0) {
        int leader_lane = __ffs(active_mask) - 1;
        ChunkKey current_key = group.shfl(my_key, leader_lane);
        
        auto it = map_ref.find(group, current_key);
        
        if (my_key == current_key) {
            VoxelType log_odds = default_voxel_value();
            if (it != map_ref.end() && it->second != invalid_chunk_ptr() && it->second != nullptr) {
                uint32_t intra_chunk_idx = get_intra_chunk_index(global_x, global_y, global_z);
                log_odds = it->second[intra_chunk_idx];
            }
            op(log_odds, aabb_x, aabb_y, aabb_z);
        }

        unsigned int processed_mask = group.ballot(my_key == current_key);
        active_mask &= ~processed_mask;
    }
}

class GpuHashMap {
    public:
        /**
         * @brief Constructs a cuco static map for voxel mapping using chunkidx as the key and ChunkPtr as the value.
         * This map is used to manage the preallocated chunks in the global memory pool, by distributing ChunkPtrs to Chunks on demand.
         * @param capacity The number of chunks to preallocate in the global memory pool. Hash map size is set to capacity / 0.90 to ensure a max load factor of 0.90.
         * @param log_odds_occupied Log-odds update value for occupied voxels.
         * @param log_odds_free Log-odds update value for free voxels.
         * @param log_odds_min Clamped minimum log-odds value for voxels.
         * @param log_odds_max Clamped maximum log-odds value for voxels.
         * @param occupancy_threshold Threshold for occupancy to consider a voxel occupied.
         * @param free_threshold Threshold for occupancy to consider a voxel free.
         */
        GpuHashMap(size_t capacity, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold);
        ~GpuHashMap();

        GpuHashMap(const GpuHashMap&) = delete;
        GpuHashMap& operator=(const GpuHashMap&) = delete;

        GpuHashMap(GpuHashMap&&) = default;
        GpuHashMap& operator=(GpuHashMap&&) = default;

        /**
         * @brief Creates a kernel node for extraction and adds it to a CUDA graph.
         * @tparam Tag The extraction type (Block or Slice).
         * @tparam Functor The functor type used for processing the voxels.
         * @param graph The CUDA graph to add the node to.
         * @param dependencies An optional vector of nodes that this node depends on.
         * @param aabb_placeholder A placeholder AABB for graph creation.
         * @param slice_indices_placeholder A placeholder SliceZIndices for graph creation.
         * @param extract_op_placeholder A placeholder functor for graph creation.
         * @return The handle to the created kernel node.
         */
        template <ExtractionType Tag, typename Functor>
        cudaGraphNode_t add_extraction_kernel_node(
            cudaGraph_t graph,
            const std::vector<cudaGraphNode_t>& dependencies,
            const AABB_CUDA* aabb_cuda_ptr,
            const SliceZIndices* slice_indices_ptr,
            Functor* extract_op_ptr,
            void** kernel_args) {

            dim3 block_dim(32, 8, 1);

            dim3 grid_dim;
            if constexpr (Tag == ExtractionType::Block) {
                grid_dim = dim3(
                    (aabb_cuda_ptr->aabb_current_size.x + block_dim.x - 1) / block_dim.x,
                    (aabb_cuda_ptr->aabb_current_size.y + block_dim.y - 1) / block_dim.y,
                    aabb_cuda_ptr->aabb_current_size.z
                );
            } else { // ExtractionType::Slice
                grid_dim = dim3(
                    (aabb_cuda_ptr->aabb_current_size.x + block_dim.x - 1) / block_dim.x,
                    (aabb_cuda_ptr->aabb_current_size.y + block_dim.y - 1) / block_dim.y,
                    slice_indices_ptr->count
                );
            }

            cudaKernelNodeParams kernel_params = {};
            kernel_params.func = (void*)extract_from_map<Tag, Functor>;
            kernel_params.gridDim = grid_dim;
            kernel_params.blockDim = block_dim;
            kernel_params.sharedMemBytes = 0;

            kernel_args[0] = &map_ref_extraction_;
            kernel_args[1] = (void*)&aabb_cuda_ptr->aabb_min_index;
            kernel_args[2] = (void*)&aabb_cuda_ptr->aabb_current_size;
            kernel_args[3] = (void*)slice_indices_ptr;
            kernel_args[4] = (void*)extract_op_ptr;

            kernel_params.kernelParams = kernel_args;

            cudaGraphNode_t kernel_node;
            CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
                &kernel_node,
                graph,
                dependencies.data(),
                dependencies.size(),
                &kernel_params
            ));

            return kernel_node;
        }

        /**
         * @brief Updates the kernel parameters for an extraction kernel node.
         * @tparam Tag The extraction type (Block or Slice).
         * @tparam Functor The functor type used for processing the voxels.
         * @param exec_graph The executable graph containing the node.
         * @param kernel_node The handle to the kernel node to update.
         * @param aabb_cuda_ptr Pointer to the updated AABB_CUDA struct.
         * @param slice_indices_ptr Pointer to the updated SliceZIndices struct.
         * @param extract_op_ptr Pointer to the updated functor.
         */
        template <ExtractionType Tag, typename Functor>
        void update_extraction_kernel_node(
            cudaGraphExec_t exec_graph,
            cudaGraphNode_t kernel_node,
            const AABB_CUDA* aabb_cuda_ptr,
            const SliceZIndices* slice_indices_ptr,
            Functor* extract_op_ptr,
            void** kernel_args) {

            dim3 block_dim(32, 8, 1);
            dim3 grid_dim;
            if constexpr (Tag == ExtractionType::Block) {
                grid_dim = dim3(
                    (aabb_cuda_ptr->aabb_current_size.x + block_dim.x - 1) / block_dim.x,
                    (aabb_cuda_ptr->aabb_current_size.y + block_dim.y - 1) / block_dim.y,
                    aabb_cuda_ptr->aabb_current_size.z
                );
            } else { // ExtractionType::Slice
                grid_dim = dim3(
                    (aabb_cuda_ptr->aabb_current_size.x + block_dim.x - 1) / block_dim.x,
                    (aabb_cuda_ptr->aabb_current_size.y + block_dim.y - 1) / block_dim.y,
                    slice_indices_ptr->count
                );
            }

            cudaKernelNodeParams kernel_params = {};
            kernel_params.func = (void*)extract_from_map<Tag, Functor>;
            kernel_params.gridDim = grid_dim;
            kernel_params.blockDim = block_dim;
            kernel_params.sharedMemBytes = 0;

            kernel_args[0] = &map_ref_extraction_;
            kernel_args[1] = (void*)&aabb_cuda_ptr->aabb_min_index;
            kernel_args[2] = (void*)&aabb_cuda_ptr->aabb_current_size;
            kernel_args[3] = (void*)slice_indices_ptr;
            kernel_args[4] = (void*)extract_op_ptr;

            kernel_params.kernelParams = kernel_args;

            CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, kernel_node, &kernel_params));
        }

        /**
         * @brief Clears chunks from the hash map that either have an invalid chunk pointer or is far away from the current chunk position.
         * Clears chunks equal to 10% of the total chunk capacity.
         * @param current_chunk_pos The current chunk position in 3D grid coordinates.
         */
        void clear_chunks(const int3& current_chunk_pos);

        /**
         * @brief Retrieves the counter for freelist allocations.
         * This counter indicates how many chunks are currently allocated and used in the hash map.
         * @param freelist_counter Pointer to a uint32_t variable where the counter will be stored.
         */
        void get_freelist_counter(uint32_t* freelist_counter);

        /**
         * @brief Retrieves the capacity of the freelist (number of chunks).
         * The freelist is used to manage chunk allocations and deallocations.
         * @return The capacity of the freelist.
         */
        size_t get_freelist_capacity() const {
            return freelist_capacity_;
        }

        /**
         * @brief Adds the map update node to the CUDA graph.
         * This function creates a new CUDA graph node for the map update operation and adds it to the provided graph.
         * @param graph The CUDA graph to which the node will be added.
         * @param preceding_dependencies A vector of CUDA graph nodes that this node depends on.
         * @param aabb_update The per-frame AABB used for kernel execution.
         * @param d_aabb_ptr Pointer to the device memory where the AABB grid is stored
         * @return A vector of CUDA graph nodes that were added to the graph.
         * This vector includes the new map update node as the last dependency of the graph.
         */
        std::vector<cudaGraphNode_t> add_nodes_to_insertion_graph(
            cudaGraph_t graph,
            const std::vector<cudaGraphNode_t>& preceding_dependencies,
            const AABB_CUDA& aabb_update,
            UpdateType* d_aabb_ptr);

        /**
         * @brief Updates the parameters of the map update kernel node in the executable graph.
         * @param executable_graph The executable graph to update.
         * @param aabb_update The per-frame AABB used for kernel execution.
         * @param d_aabb_ptr Pointer to the device memory where the AABB grid is stored.
         */
        void update_insertion_graph_nodes(
            cudaGraphExec_t executable_graph,
            const AABB_CUDA& aabb_update,
            UpdateType* d_aabb_ptr);


    private:
        /**
         * @brief Creates a new chunk map with the specified capacity.
         * This function initializes a new chunk map with the given capacity and returns a unique pointer to it.
         * @param capacity The number of max entries in the map.
         * @return A unique pointer to the newly created ChunkMap.
         */
        std::unique_ptr<ChunkMap> create_chunk_map(size_t capacity);

        /**
         * @brief Helper function to retrieve all chunks from the hash map.
         * This function retrieves all chunk keys and their corresponding pointers from the hash map.
         * @param h_keys Vector to store the chunk keys.
         * @param h_values Vector to store the chunk pointers.
         */
        void retrieve_all_chunks(std::vector<ChunkKey>& h_keys, std::vector<ChunkPtr>& h_values);

        /**
         * @brief Helper function to prioritize chunks for clearing based on distance from the current chunk position and pointer validity.
         * This function sorts the chunks by their squared distance to the current chunk position and returns the top N chunks,
         * prioritizing chunks that have invalid pointers.
         * @param h_keys Vector of chunk keys.
         * @param h_values Vector of chunk pointers.
         * @param current_chunk_pos The current chunk position in 3D grid coordinates.
         * @return A vector of ChunkInfo containing the prioritized chunks for clearing.
         */
        std::vector<ChunkInfo> prioritize_chunks_for_clearing(
            const std::vector<ChunkKey>& h_keys,
            const std::vector<ChunkPtr>& h_values,
            const int3& current_chunk_pos);

        /**
         * @brief Executes the chunk removal operation by erasing chunks from the hash map,
         * setting the chunk memory to the default value, and deallocating the chunk pointers by returning them to the freelist.
         * @param keys_to_erase Vector of chunk keys to erase.
         * @param ptrs_to_deallocate Vector of chunk pointers to deallocate.
         */
        void execute_chunk_removal(
            const std::vector<ChunkKey>& keys_to_erase,
            const std::vector<ChunkPtr>& ptrs_to_deallocate);

        std::unique_ptr<ChunkMap> d_voxel_map_;
        ChunkMapRef map_ref_insertion_;
        ConstChunkMapRef map_ref_extraction_;

        VoxelType* global_memory_pool_ = nullptr;
        ChunkPtr* freelist_ = nullptr;
        uint32_t* freelist_counter_ = nullptr;
        uint32_t freelist_capacity_ = 0;
        size_t map_capacity_ = 0;

        cudaGraphNode_t update_map_node_;
        void* insertion_kernel_args_[6];

};

} // namespace voxel_mapping

#endif // GPU_HASH_MAP_CUH