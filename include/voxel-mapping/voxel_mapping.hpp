#ifndef VOXEL_MAPPING_HPP
#define VOXEL_MAPPING_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/extraction_result.hpp"

namespace voxel_mapping {

class VoxelMappingImpl;

/**
 * @brief Holds all configuration parameters for the VoxelMapping class.
 */
struct VoxelMappingParams {
    /// @brief The number of voxel chunks to pre-allocate memory for.
    size_t chunk_capacity;
    /// @brief The resolution of the voxel grid in meters per voxel.
    float resolution;
    /// @brief The minimum depth for processing insertions from a depth image.
    float min_depth;
    /// @brief The maximum depth for processing insertions from a depth image.
    float max_depth;
    /// @brief Log-odds update value for occupied voxels.
    VoxelType log_odds_occupied;
    /// @brief Log-odds update value for free voxels.
    VoxelType log_odds_free;
    /// @brief Clamped minimum log-odds value for any voxel.
    VoxelType log_odds_min;
    /// @brief Clamped maximum log-odds value for any voxel.
    VoxelType log_odds_max;
    /// @brief Log-odds value above which a voxel is considered occupied.
    VoxelType occupancy_threshold;
    /// @brief Log-odds value below which a voxel is considered free.
    VoxelType free_threshold;
    /// @brief Maximum distance for EDT computation.
    int edt_max_distance;
    /// @brief Maximum size of the extraction buffer in bytes.
    size_t max_extraction_buffer_size_bytes;
};

/**
 * @brief VoxelMapping is the main interface for voxel mapping operations.
 * It provides methods to set camera properties, integrate depth images, extract voxel blocks,
 * and manage the voxel map's state.
 */
class VoxelMapping {
public:
    /**
     * @brief Constructs a VoxelMapping object that acts as the interface to mapping operations.
     * @param params A struct containing all configuration parameters for the map.
     */
    VoxelMapping(const VoxelMappingParams& params);
    
    ~VoxelMapping();

    VoxelMapping(VoxelMapping&&);
    VoxelMapping& operator=(VoxelMapping&&);
    VoxelMapping(const VoxelMapping&) = delete;
    VoxelMapping& operator=(const VoxelMapping&) = delete;

    /**
     * @brief Sets the necessary camera properties for the voxel mapping to begin processing depth images.
     * @param fx Focal length in the x direction.
     * @param fy Focal length in the y direction.
     * @param cx Optical center x-coordinate.
     * @param cy Optical center y-coordinate.
     * @param width Width of the image.
     * @param height Height of the image.
     */
    void set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height);

    /**
     * @brief Integrates a depth image into the voxel map using the provided transformation.
     * The sensor frame required the z-direction to be defined as the view/forward direction aligning with camera conventions.
     * @param depth_image Pointer to the host memory containing the depth image.
     * @param transform Pointer to the host memory containing the world-to-camera transformation matrix.
     */
    void integrate_depth(const float* depth_image,  const float* transform);

    /**
     * @brief Extracts a 3D block of voxels from the voxel map based on the provided AABB.
     * @param aabb The AABB defining the region to extract defined by its minimum corner index and size.
     * @return An ExtractionResult object containing the data for the specified AABB.
     */
    ExtractionResult extract_grid_block(const AABB& aabb);

    /**
     * @brief Extracts slices of voxels from the voxel map based on the provided AABB and slice indices.
     * @param aabb The AABB defining the region to extract defined by its minimum corner index and size.
     * @param slice_indices The indices of the Z slices to extract.
     * @return An ExtractionResult object containing the data where slices are stacked in the order they are defined in slice_indices.indices.
     */
    ExtractionResult extract_grid_slices(const AABB& aabb, const SliceZIndices& slice_indices);

    /**
     * @brief Extracts a 3D block of the Euclidean Distance Transform (EDT) from the voxel map based on the provided AABB.
     * @param aabb The AABB defining the region to extract defined by its minimum corner index and size.
     * @return An ExtractionResult object containing the EDT data for the specified AABB.
     */
    ExtractionResult extract_edt_block(const AABB& aabb);

    /**
     * @brief Extracts slices of the Euclidean Distance Transform (EDT) from the voxel map based on the provided AABB and slice indices.
     *  This serves as a more efficient alternative to extracting the full EDT block, since this does not calculate distances along the z-dimension.
     * @param aabb The AABB defining the region to extract defined by its minimum corner index and size.
     * @param slice_indices The indices of the Z slices to extract.
     * @return An ExtractionResult object containing the EDT data for the specified slices with slices stacked in the order they are defined in slice_indices.indices.
     */
    ExtractionResult extract_edt_slices(const AABB& aabb, const SliceZIndices& slice_indices);
    /**
     * @brief Returns the current AABB's minimum index in world coordinates and its size in grid coordinates.
     * @return AABB struct containing:
     * - min_corner_index: A vector containing the minimum index {x, y, z} of the AABB in world coordinates.
     * - aabb_size: A vector containing the size {x, y, z} of the AABB in grid coordinates.
     */
    AABB get_current_aabb() const;

    /**
     * @brief Returns the current frustum of the camera.
     * @return A Frustum struct containing:
     * - near_plane: The near plane of the frustum, defined by four points in world coordinates.
     * - far_plane: The far plane of the frustum, defined by four points in world coordinates.
     */
    Frustum get_frustum() const;

    /**
     * @brief Checks if the memory pool is nearing capacity and, if so, clears distant and invalid chunks.
     */
    void query_free_chunk_capacity();
    
private:
    std::unique_ptr<VoxelMappingImpl> pimpl_;
};

} // namespace voxel_mapping

#endif // VOXEL_MAPPING_HPP