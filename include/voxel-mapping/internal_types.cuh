#ifndef INTERNAL_TYPES_CUH
#define INTERNAL_TYPES_CUH

#include "voxel-mapping/types.hpp"
#include <cstdint>
#include <cuco/static_map.cuh>

namespace voxel_mapping {

/**
 * @brief Unique identifier for a voxel chunk, represented as a 64-bit unsigned integer.
 */
using ChunkKey = uint64_t;

/**
 * @brief Pointer to the first voxel in a chunk, represented as a pointer to VoxelType.
 */
using ChunkPtr = VoxelType*;

/**
 * @brief Defines the type of update to be applied to a voxel.
 */
enum class UpdateType : uint8_t {
    Unknown,  ///< Default value for initialized voxels. Represents no update.
    Free,     ///< The voxel is part of free space (ray passed through it).
    Occupied, ///< The voxel is occupied by a surface.
};

/**
 * @brief Represents the type of extraction operation to be performed on the voxel grid.
 */
enum class ExtractionType {
    Block, ///< Extracts a contiguous block of voxels within a specified axis-aligned bounding box (AABB).
    Slice  ///< Extracts non-contiguous slices of voxels along the Z-axis, defined by a set of indices.
};

/**
 * @brief Represents an update to an axis-aligned bounding box (AABB) region.
 */
struct AABB_CUDA {
    ///< Minimum index (corner) of the AABB in 3D grid coordinates.
    int3 aabb_min_index;
    ///< Current size of the AABB in grid units.
    int3 aabb_current_size;
};

/**
 * @brief Represents a chunk in the voxel mapping system for host-side processing.
 */
struct ChunkInfo {
    ///< Unique identifier for the chunk.
    ChunkKey key;
    ///< Pointer to the chunk's data on the GPU.
    ChunkPtr ptr;
    ///< Cached squared distance to a reference point.
    float distance_sq;
    ///< Flag indicating if the chunk's pointer is invalid (unallocated).
    bool is_invalid;
};

/**
 * @brief Type alias for a hashmap with {key, value} pairs are {ChunkKey, ChunkPtr}.
 * This map is used to manage voxel chunks allowing for on-demand allocation of chunks.
 */
using ChunkMap = cuco::static_map<
    ChunkKey,
    ChunkPtr,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<ChunkKey>,
    cuco::linear_probing<32, cuco::default_hash_function<ChunkKey>>
>;

/**
 * @brief Type alias for a reference to a ChunkMap.
 * This reference allows for read and write operations on the ChunkMap.
 */
using ChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find, cuco::op::insert_and_find));

/**
 * @brief Type alias for a constant reference to a ChunkMap.
 * This reference allows for read-only operations on the ChunkMap.
 */
using ConstChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find));

} // namespace voxel_mapping

#endif // INTERNAL_TYPES_CUH