#ifndef RAYCASTING_UTILS_CUH
#define RAYCASTING_UTILS_CUH

#include "voxel-mapping/internal_types.cuh"
#include "voxel-mapping/map_utils.cuh"

namespace voxel_mapping {

/**
 * @brief Defines indices for a 4x4 float[16] column-major matrix
 *
 */
namespace tf_index {

    inline constexpr __device__ int basis_xx() { return 0; }
    inline constexpr __device__ int basis_xy() { return 1; }
    inline constexpr __device__ int basis_xz() { return 2; }
    inline constexpr __device__ int basis_xw() { return 3; }

    inline constexpr __device__ int basis_yx() { return 4; }
    inline constexpr __device__ int basis_yy() { return 5; }
    inline constexpr __device__ int basis_yz() { return 6; }
    inline constexpr __device__ int basis_yw() { return 7; }

    inline constexpr __device__ int basis_zx() { return 8; }
    inline constexpr __device__ int basis_zy() { return 9; }
    inline constexpr __device__ int basis_zz() { return 10; }
    inline constexpr __device__ int basis_zw() { return 11; }

    inline constexpr __device__ int t_x() { return 12; }
    inline constexpr __device__ int t_y() { return 13; }
    inline constexpr __device__ int t_z() { return 14; }
    inline constexpr __device__ int t_w() { return 15; }
}

/**
 * @brief Calculates the 1D index for a pixel in a row-major 2D image.
 * 
 */
__device__ inline uint32_t image_1d_index(int x, int y, int size_x) {
    return (y * size_x) + x;
}

/**
 * @brief Transforms a pixel coordinate into a 3D world-space point.
 */
__device__ inline float3 pixel_to_world_space(
    uint32_t x, uint32_t y, float depth, const float* transform,
    float fx, float fy, float cx, float cy)
{
    float cam_x = (x - cx) * depth / fx;
    float cam_y = (y - cy) * depth / fy;
    
    float wx = transform[tf_index::basis_xx()] * cam_x + transform[tf_index::basis_yx()] * cam_y + transform[tf_index::basis_zx()] * depth + transform[tf_index::t_x()];
    float wy = transform[tf_index::basis_xy()] * cam_x + transform[tf_index::basis_yy()] * cam_y + transform[tf_index::basis_zy()] * depth + transform[tf_index::t_y()];
    float wz = transform[tf_index::basis_xz()] * cam_x + transform[tf_index::basis_yz()] * cam_y + transform[tf_index::basis_zz()] * depth + transform[tf_index::t_z()];
    float w = transform[tf_index::basis_xw()] * cam_x + transform[tf_index::basis_yw()] * cam_y + transform[tf_index::basis_zw()] * depth + transform[tf_index::t_w()];
    if (w != 1.0f && w != 0.0f) {
        return {wx / w, wy / w, wz / w};
    }
    return {wx, wy, wz};
}

/**
 * @brief Traverses a ray from start to end, marking voxels in the AABB as Free.
 */
__device__ inline void mark_ray_as_free(
    const float3& start_point, const float3& end_point, UpdateType* d_aabb,
    const int3& min_aabb_index, const int3& aabb_current_size, float resolution)
{
    float step_size = resolution * 0.5f;
    float3 direction = {end_point.x - start_point.x, end_point.y - start_point.y, end_point.z - start_point.z};
    float distance = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    
    if (distance < 1e-6f) return;
    int steps = static_cast<int>(floorf(distance / step_size));
    if (steps == 0) return;

    for (int i = 0; i < steps; ++i) {
        float t = static_cast<float>(i) / steps;
        float3 p = {start_point.x + t * direction.x, start_point.y + t * direction.y, start_point.z + t * direction.z};
        
        int grid_x = static_cast<int>(floorf(p.x / resolution)) - min_aabb_index.x;
        int grid_y = static_cast<int>(floorf(p.y / resolution)) - min_aabb_index.y;
        int grid_z = static_cast<int>(floorf(p.z / resolution)) - min_aabb_index.z;


        if (grid_x >= 0 && grid_x < aabb_current_size.x && grid_y >= 0 && grid_y < aabb_current_size.y && grid_z >= 0 && grid_z < aabb_current_size.z) {
            uint32_t idx = block_1d_index(grid_x, grid_y, grid_z, aabb_current_size.x, aabb_current_size.y);
            d_aabb[idx] = UpdateType::Free;
        }
    }
}

/**
 * @brief Marks the voxel at the ray's endpoint as Occupied.
 */
__device__ inline void mark_endpoint_as_occupied(
    const float3& world_point, UpdateType* d_aabb, const int3& min_aabb_index,
    const int3& aabb_current_size, float resolution)
{
    int end_x = static_cast<int>(floorf(world_point.x / resolution)) - min_aabb_index.x;
    int end_y = static_cast<int>(floorf(world_point.y / resolution)) - min_aabb_index.y;
    int end_z = static_cast<int>(floorf(world_point.z / resolution)) - min_aabb_index.z;

    if (end_x >= 0 && end_x < aabb_current_size.x && end_y >= 0 && end_y < aabb_current_size.y && end_z >= 0 && end_z < aabb_current_size.z) {
        uint32_t idx = block_1d_index(end_x, end_y, end_z, aabb_current_size.x, aabb_current_size.y);
        d_aabb[idx] = UpdateType::Occupied;
    }
}

} // namespace voxel_mapping

#endif // RAYCASTING_UTILS_CUH