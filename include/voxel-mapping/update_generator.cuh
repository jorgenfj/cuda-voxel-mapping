#ifndef UPDATE_GENERATOR_CUH
#define UPDATE_GENERATOR_CUH

#include "voxel-mapping/internal_types.cuh"
#include <cstdint>

namespace voxel_mapping {

struct FrustumIntBounds {
    int3 frustum_min_i;
    int3 frustum_max_i;
    int3 near_plane_max_i;
};

class UpdateGenerator {
public:

    /**
     * @brief Constructs an UpdateGenerator responsible for performing raycasting and generating updates for the AABB.
     * @param voxel_resolution The resolution of the voxels in meters.
     * @param min_depth The minimum depth value to consider for updates.
     * @param max_depth The maximum depth value to consider for updates.
     */
    UpdateGenerator(float voxel_resolution, float min_depth, float max_depth);
    ~UpdateGenerator();

    UpdateGenerator(const UpdateGenerator&) = delete;
    UpdateGenerator& operator=(const UpdateGenerator&) = delete;

    UpdateGenerator(UpdateGenerator&&) = default;
    UpdateGenerator& operator=(UpdateGenerator&&) = default;

    /**
     * @brief Sets the camera properties for the update generator and copies them to their respective device constants.
     * @param fx The focal length in the x direction.
     * @param fy The focal length in the y direction.
     * @param cx The optical center x-coordinate.
     * @param cy The optical center y-coordinate.
     * @param width The width of the image.
     * @param height The height of the image.
     */
    void set_camera_properties(
        float fx, float fy, float cx, float cy,
        uint32_t width, uint32_t height);

    /**
     * @brief Get the defining properties of the AABB
     * @return A AABB object containing:
     * - min_corner_index: The minimum index {x, y, z} of the AABB in world coordinates.
     * - size: The size {x, y, z} of the AABB in grid coordinates.
     */
    AABB get_aabb() const {
    AABB aabb;
    aabb.min_corner_index = {aabb_cuda_.aabb_min_index.x, aabb_cuda_.aabb_min_index.y, aabb_cuda_.aabb_min_index.z};
    aabb.size = {aabb_cuda_.aabb_current_size.x, aabb_cuda_.aabb_current_size.y, aabb_cuda_.aabb_current_size.z};
    return aabb;
    }

    /**
     * @brief Get the frustum of the camera in world coordinates.
     * @return A Frustum object containing the near and far planes of the camera frustum.
     */
    Frustum get_frustum() const {
        return frustum_;
    }

    /**
     * @brief Get the current chunk position in world coordinates.
     * @return An int3 containing the current chunk position {x, y, z}.
     */
    int3 get_current_chunk_position() const {
        return current_chunk_pos_;
    }

    /**
     * @brief Get the device pointers for depth data.
     * @return Pointers to the device memory for depth data.
     */
    float* get_device_depth_ptr() const {
        return d_depth_;
    }

    /**
     * @brief Get the device pointer for the transformation matrix.
     * @return Pointer to the device memory for the transformation matrix.
     */
    float* get_device_transform_ptr() const {
        return d_transform_;
    }

    /**
     * @brief Get the host pinned pointers for depth data.
     * @return Pointers to the host pinned memory for depth data.
     */
    float* get_host_pinned_depth_ptr() { return h_pinned_depth_; }

    /**
     * @brief Get the host pinned pointer for the transformation matrix.
     * @return Pointer to the host pinned memory for the transformation matrix.
     */
    float* get_host_pinned_transform_ptr() { return h_pinned_transform_; }

    /**
     * @brief Get the size of the depth buffer in bytes.
     * @return The size of the depth buffer in bytes.
     */
    size_t get_depth_buffer_size() { return depth_buffer_size_bytes_; }

    /**
     * @brief Get the device pointer for the AABB update type.
     * @return Pointer to the device memory for the AABB update type.
     */
    UpdateType* get_device_aabb_ptr() const {
        return d_aabb_;
    }

    /**
     * @brief Get the AABB update structure used for CUDA operations.
     * @return A reference to the AABB_CUDA structure containing the AABB data.
     */
    const AABB_CUDA& get_aabb_update_struct() const {
        return aabb_cuda_;
    }

    /**
     * @brief Orchestrates the calculation of the current AABB min index and size
     *
     * This function calls the necessary helpers to compute the AABB's bounding
     * indices and stores the results in the class's member variables.
     * @param transform The current world-to-camera transformation matrix.
     */
    void compute_active_aabb(const float* transform);
    
    /**
     * @brief Adds nodes to the insertion graph for the current AABB update.
     *
     * This function creates CUDA graph nodes updating voxels in the map. It returns a vector of CUDA graph nodes that
     * can be used in subsequent operations.
     *
     * @param graph The CUDA graph to which the nodes will be added.
     * @param preceding_dependencies A vector of CUDA graph nodes that must be completed before these nodes can execute.
     * @param aabb_update The AABB_CUDA structure containing the AABB update data.
     * @return A vector of CUDA graph nodes representing the insertion operations.
     */
    std::vector<cudaGraphNode_t> add_nodes_to_insertion_graph(
        cudaGraph_t graph,
        const std::vector<cudaGraphNode_t>& preceding_dependencies,
        const AABB_CUDA& aabb_update);

    /**
     * @brief Updates the insertion graph nodes with the current AABB update.
     *
     * This function updates the CUDA graph nodes for insertion based on the
     * current AABB update. It is used to manage the insertion of new data into
     * the voxel map.
     *
     * @param executable_graph The CUDA graph execution object to update.
     * @param aabb_update The AABB_CUDA structure containing the AABB update data.
     */
    void update_insertion_graph_nodes(
        cudaGraphExec_t executable_graph, const AABB_CUDA& aabb_update);
    
    
        
        
private:
    
    /**
     * @brief Sets the frustum planes based on the provided transformation matrix.
     * The near and far planes are defined by four points in world coordinates,
     * with distances set by min_depth_ and max_depth_.
     * @param transform Pointer to the transformation matrix in host memory.
     */
    void set_frustum(const float* transform);
    
    /**
     * @brief Calculates the integer bounds of the frustum in grid coordinates.
     * This function computes the minimum and maximum indices of the frustum
     * in the voxel grid, as well as the maximum indices of the near plane.
     * @return A FrustumIntBounds object containing the calculated bounds.
     */
    FrustumIntBounds get_frustum_int_bounds() const;
    
    /**
     * @brief Sets the AABB CUDA structure based on the current frustum.
     * This function calculates the minimum index and current size of the AABB
     * in grid coordinates, and updates the aabb_cuda_ member variable.
     * This function ensures the near plane is always included in the AABB.
     */
    void set_aabb_cuda();

    /**
     * @brief Sets the current chunk position based on the provided transformation matrix.
     * The chunk position is calculated by flooring the translation components of the transformation
     * divided by the voxel resolution.
     * @param transform Pointer to the transformation matrix in host memory.
     */
    void set_current_chunk_position(const float* transform);
    
    float voxel_resolution_;
    float min_depth_;
    float max_depth_;
    
    UpdateType* d_aabb_ = nullptr;
    float* h_pinned_transform_ = nullptr;
    float* d_transform_ = nullptr;
    float* h_pinned_depth_ = nullptr;
    float* d_depth_ = nullptr;
    size_t depth_buffer_size_bytes_ = 0;
    uint32_t image_width_;
    uint32_t image_height_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    uint32_t aabb_max_total_size_ = 0;
    int3 aabb_max_size_ = {0, 0, 0};
    Frustum frustum_;
    int3 current_chunk_pos_ = {0, 0, 0};
    AABB_CUDA aabb_cuda_;
    
    void* graph_kernel_args_[5];
    cudaGraphNode_t mark_free_node_ = nullptr;
    cudaGraphNode_t mark_occupied_node_ = nullptr;
};
    
} // namespace voxel_mapping

#endif // UPDATE_GENERATOR_CUH