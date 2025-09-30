#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/raycasting_utils.cuh"
#include "voxel-mapping/internal_types.cuh"
#include <limits.h>
#include <cassert>

namespace voxel_mapping {

static __constant__ float d_fx;
static __constant__ float d_fy;
static __constant__ float d_cx;
static __constant__ float d_cy;
static __constant__ uint32_t d_image_width;
static __constant__ uint32_t d_image_height;
static __constant__ float d_resolution;
static __constant__ float d_min_depth;
static __constant__ float d_max_depth;

UpdateGenerator::UpdateGenerator(float voxel_resolution, float min_depth, float max_depth) :
    voxel_resolution_(voxel_resolution),
    min_depth_(min_depth),
    max_depth_(max_depth)
{
    CHECK_CUDA_ERROR(cudaMalloc(&d_transform_, 16 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_pinned_transform_, 16 * sizeof(float), cudaHostAllocWriteCombined));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_resolution, &voxel_resolution_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_min_depth, &min_depth_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_max_depth, &max_depth_, sizeof(float), 0, cudaMemcpyHostToDevice));
}

UpdateGenerator::~UpdateGenerator() {
    cudaFreeHost(h_pinned_depth_);
    cudaFreeHost(h_pinned_transform_);
    cudaFree(d_depth_);
    cudaFree(d_transform_);
    cudaFree(d_aabb_);
}

void UpdateGenerator::set_camera_properties(
    float fx, float fy, float cx, float cy,
    uint32_t width, uint32_t height) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    image_width_ = width;
    image_height_ = height;

    cudaFreeHost(h_pinned_depth_);
    cudaFree(d_depth_);
    cudaFree(d_aabb_);

    depth_buffer_size_bytes_ = static_cast<size_t>(image_width_) * image_height_ * sizeof(float);
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_pinned_depth_, depth_buffer_size_bytes_, cudaHostAllocWriteCombined));
    CHECK_CUDA_ERROR(cudaMalloc(&d_depth_, depth_buffer_size_bytes_));

    float frustum_width = (image_width_ / fx_) * max_depth_;
    float frustum_height = (image_height_ / fy_) * max_depth_;

    float space_diagonal = sqrtf(frustum_width*frustum_width + frustum_height*frustum_height + max_depth_*max_depth_);
    int aabb_max_dim_size = static_cast<int>(ceil(space_diagonal / voxel_resolution_)) + 1;
    
    aabb_max_size_ = {aabb_max_dim_size, aabb_max_dim_size, aabb_max_dim_size};
    uint32_t aabb_max_total_size = static_cast<uint32_t>(aabb_max_size_.x) * aabb_max_size_.y * aabb_max_size_.z;

    CHECK_CUDA_ERROR(cudaMalloc(&d_aabb_, aabb_max_total_size * sizeof(UpdateType)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fx, &fx_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fy, &fy_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cx, &cx_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cy, &cy_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &image_width_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &image_height_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

void UpdateGenerator::compute_active_aabb(const float* transform) {
    set_frustum(transform);
    set_aabb_cuda();
    set_current_chunk_position(transform);
}

void UpdateGenerator::set_frustum(const float* transform) {
    float3 corners_cam[8];
    
    float near_x_min = (0.0f - cx_) * min_depth_ / fx_;
    float near_x_max = (image_width_ - cx_) * min_depth_ / fx_;
    float near_y_min = (0.0f - cy_) * min_depth_ / fy_;
    float near_y_max = (image_height_ - cy_) * min_depth_ / fy_;
    corners_cam[0] = {near_x_min, near_y_min, min_depth_};
    corners_cam[1] = {near_x_max, near_y_min, min_depth_};
    corners_cam[2] = {near_x_max, near_y_max, min_depth_};
    corners_cam[3] = {near_x_min, near_y_max, min_depth_};
    
    float far_x_min = (0.0f - cx_) * max_depth_ / fx_;
    float far_x_max = (image_width_ - cx_) * max_depth_ / fx_;
    float far_y_min = (0.0f - cy_) * max_depth_ / fy_;
    float far_y_max = (image_height_ - cy_) * max_depth_ / fy_;
    corners_cam[4] = {far_x_min, far_y_min, max_depth_};
    corners_cam[5] = {far_x_max, far_y_min, max_depth_};
    corners_cam[6] = {far_x_max, far_y_max, max_depth_};
    corners_cam[7] = {far_x_min, far_y_max, max_depth_};
    
    for(int i = 0; i < 8; ++i) {
        float3 p_cam = corners_cam[i];
        
        float wx = transform[tf_index::basis_xx()] * p_cam.x + transform[tf_index::basis_yx()] * p_cam.y + transform[tf_index::basis_zx()] * p_cam.z + transform[tf_index::t_x()];
        float wy = transform[tf_index::basis_xy()] * p_cam.x + transform[tf_index::basis_yy()] * p_cam.y + transform[tf_index::basis_zy()] * p_cam.z + transform[tf_index::t_y()];
        float wz = transform[tf_index::basis_xz()] * p_cam.x + transform[tf_index::basis_yz()] * p_cam.y + transform[tf_index::basis_zz()] * p_cam.z + transform[tf_index::t_z()];
        
        voxel_mapping::Vec3f world_corner = {wx, wy, wz};
        
        switch (i) {
            case 0: frustum_.near_plane.bl = world_corner; break;
            case 1: frustum_.near_plane.br = world_corner; break;
            case 2: frustum_.near_plane.tr = world_corner; break;
            case 3: frustum_.near_plane.tl = world_corner; break;
            
            case 4: frustum_.far_plane.bl = world_corner; break;
            case 5: frustum_.far_plane.br = world_corner; break;
            case 6: frustum_.far_plane.tr = world_corner; break;
            case 7: frustum_.far_plane.tl = world_corner; break;
        }
    }
}

FrustumIntBounds UpdateGenerator::get_frustum_int_bounds() const {
    FrustumIntBounds bounds;
    bounds.frustum_min_i = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
    bounds.frustum_max_i = {std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest()};
    bounds.near_plane_max_i = {std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest()};

    auto update_bounds = [&](const voxel_mapping::Vec3f& corner, bool is_near_plane) {
        int ix = static_cast<int>(std::floor(corner.x / voxel_resolution_));
        int iy = static_cast<int>(std::floor(corner.y / voxel_resolution_));
        int iz = static_cast<int>(std::floor(corner.z / voxel_resolution_));
        
        bounds.frustum_min_i.x = std::min(bounds.frustum_min_i.x, ix);
        bounds.frustum_min_i.y = std::min(bounds.frustum_min_i.y, iy);
        bounds.frustum_min_i.z = std::min(bounds.frustum_min_i.z, iz);
        
        bounds.frustum_max_i.x = std::max(bounds.frustum_max_i.x, ix);
        bounds.frustum_max_i.y = std::max(bounds.frustum_max_i.y, iy);
        bounds.frustum_max_i.z = std::max(bounds.frustum_max_i.z, iz);
        
        if (is_near_plane) {
            bounds.near_plane_max_i.x = std::max(bounds.near_plane_max_i.x, ix);
            bounds.near_plane_max_i.y = std::max(bounds.near_plane_max_i.y, iy);
            bounds.near_plane_max_i.z = std::max(bounds.near_plane_max_i.z, iz);
        }
    };
    
    update_bounds(frustum_.near_plane.tl, true);
    update_bounds(frustum_.near_plane.tr, true);
    update_bounds(frustum_.near_plane.bl, true);
    update_bounds(frustum_.near_plane.br, true);
    
    update_bounds(frustum_.far_plane.tl, false);
    update_bounds(frustum_.far_plane.tr, false);
    update_bounds(frustum_.far_plane.bl, false);
    update_bounds(frustum_.far_plane.br, false);
    
    return bounds;
}

void UpdateGenerator::set_aabb_cuda() {
    FrustumIntBounds bounds = get_frustum_int_bounds();
    
    int3 final_origin_i = bounds.frustum_min_i;
    
    int required_near_size_x = bounds.near_plane_max_i.x - bounds.frustum_min_i.x + 1;
    if (required_near_size_x > aabb_max_size_.x) {
        final_origin_i.x = bounds.near_plane_max_i.x - aabb_max_size_.x + 1;
    }
    
    int required_near_size_y = bounds.near_plane_max_i.y - bounds.frustum_min_i.y + 1;
    if (required_near_size_y > aabb_max_size_.y) {
        final_origin_i.y = bounds.near_plane_max_i.y - aabb_max_size_.y + 1;
    }
    
    int required_near_size_z = bounds.near_plane_max_i.z - bounds.frustum_min_i.z + 1;
    if (required_near_size_z > aabb_max_size_.z) {
        final_origin_i.z = bounds.near_plane_max_i.z - aabb_max_size_.z + 1;
    }
    
    int3 aabb_current_size;
    
    int required_full_size_x = bounds.frustum_max_i.x - final_origin_i.x + 1;
    aabb_current_size.x = std::min(required_full_size_x, aabb_max_size_.x);
    
    int required_full_size_y = bounds.frustum_max_i.y - final_origin_i.y + 1;
    aabb_current_size.y = std::min(required_full_size_y, aabb_max_size_.y);
    
    int required_full_size_z = bounds.frustum_max_i.z - final_origin_i.z + 1;
    aabb_current_size.z = std::min(required_full_size_z, aabb_max_size_.z);
    
    aabb_cuda_.aabb_min_index = final_origin_i;
    aabb_cuda_.aabb_current_size = aabb_current_size;
}

void UpdateGenerator::set_current_chunk_position(const float* transform) {
    int global_vx = static_cast<int>(std::floor(transform[tf_index::t_x()] / voxel_resolution_));
    int global_vy = static_cast<int>(std::floor(transform[tf_index::t_y()] / voxel_resolution_));
    int global_vz = static_cast<int>(std::floor(transform[tf_index::t_z()] / voxel_resolution_));

    current_chunk_pos_.x = floor_div(global_vx, chunk_dim());
    current_chunk_pos_.y = floor_div(global_vy, chunk_dim());
    current_chunk_pos_.z = floor_div(global_vz, chunk_dim());
}

__global__ void mark_free_space_kernel(
    const float* d_depth, const float* d_transform, UpdateType* d_aabb,
    int3 min_aabb_index, int3 aabb_current_size)
    {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= d_image_width || y >= d_image_height) return;
    float depth = d_depth[image_1d_index(x, y, d_image_width)];
    if (depth < d_min_depth || depth <= 0.0f) return;
    if (depth > d_max_depth) depth = d_max_depth;

    float3 world_point = pixel_to_world_space(x, y, depth, d_transform, d_fx, d_fy, d_cx, d_cy);
    float3 start_point = {d_transform[tf_index::t_x()], d_transform[tf_index::t_y()], d_transform[tf_index::t_z()]};
    mark_ray_as_free(start_point, world_point, d_aabb, min_aabb_index, aabb_current_size, d_resolution);
}

__global__ void mark_occupied_space_kernel(
    const float* d_depth, const float* d_transform, UpdateType* d_aabb,
    int3 min_aabb_index, int3 aabb_current_size)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= d_image_width || y >= d_image_height) return;
    float depth = d_depth[image_1d_index(x, y, d_image_width)];
    if (depth < d_min_depth || depth <= 0.0f || depth > d_max_depth) return;

    float3 world_point = pixel_to_world_space(x, y, depth, d_transform, d_fx, d_fy, d_cx, d_cy);
    mark_endpoint_as_occupied(world_point, d_aabb, min_aabb_index, aabb_current_size, d_resolution);
}

std::vector<cudaGraphNode_t> UpdateGenerator::add_nodes_to_insertion_graph(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& preceding_dependencies,
    const AABB_CUDA& aabb_update) {

    cudaGraph_t captured_graph;
    cudaStream_t capture_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&capture_stream));

    CHECK_CUDA_ERROR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));

    size_t transform_size_bytes = 16 * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_transform_, h_pinned_transform_, transform_size_bytes, cudaMemcpyHostToDevice, capture_stream));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_depth_, h_pinned_depth_, depth_buffer_size_bytes_, cudaMemcpyHostToDevice, capture_stream));

    uint32_t aabb_total_size = static_cast<uint32_t>(aabb_max_size_.x) * aabb_max_size_.y * aabb_max_size_.z;
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_aabb_, static_cast<int>(UpdateType::Unknown), aabb_total_size * sizeof(UpdateType), capture_stream));

    CHECK_CUDA_ERROR(cudaStreamEndCapture(capture_stream, &captured_graph));
    CHECK_CUDA_ERROR(cudaStreamDestroy(capture_stream));

    cudaGraphNode_t memory_ops_node;
    CHECK_CUDA_ERROR(cudaGraphAddChildGraphNode(&memory_ops_node, graph, preceding_dependencies.data(), preceding_dependencies.size(), captured_graph));

    std::vector<cudaGraphNode_t> free_kernel_deps = {memory_ops_node};

    graph_kernel_args_[0] = (void *)&d_depth_;
    graph_kernel_args_[1] = (void *)&d_transform_;
    graph_kernel_args_[2] = (void *)&d_aabb_;
    graph_kernel_args_[3] = (void *)&aabb_update.aabb_min_index;
    graph_kernel_args_[4] = (void *)&aabb_update.aabb_current_size;

    cudaKernelNodeParams kernel_node_params;
    kernel_node_params.func = (void*)mark_free_space_kernel;
    kernel_node_params.gridDim = dim3((image_width_ + 15) / 16, (image_height_ + 15) / 16);
    kernel_node_params.blockDim = dim3(16, 16);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = (void **)graph_kernel_args_;
    kernel_node_params.extra = NULL;

    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &mark_free_node_,
        graph,
        free_kernel_deps.data(),
        free_kernel_deps.size(),
        &kernel_node_params));

    std::vector<cudaGraphNode_t> occupied_kernel_deps = {mark_free_node_};
 
    kernel_node_params.func = (void*)mark_occupied_space_kernel;
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &mark_occupied_node_,
        graph,
        occupied_kernel_deps.data(),
        occupied_kernel_deps.size(),
        &kernel_node_params));

    CHECK_CUDA_ERROR(cudaGraphDestroy(captured_graph));

    return {mark_occupied_node_};
}

void UpdateGenerator::update_insertion_graph_nodes(
    cudaGraphExec_t executable_graph, const AABB_CUDA& aabb_update) {
    graph_kernel_args_[3] = (void*)&aabb_update.aabb_min_index;
    graph_kernel_args_[4] = (void*)&aabb_update.aabb_current_size;

    cudaKernelNodeParams node_update_params = {0};
    node_update_params.gridDim = dim3((image_width_ + 15) / 16, (image_height_ + 15) / 16);
    node_update_params.blockDim = dim3(16, 16);
    node_update_params.sharedMemBytes = 0;
    node_update_params.kernelParams = (void**)graph_kernel_args_;
    node_update_params.extra = nullptr;

    node_update_params.func = (void*)mark_free_space_kernel;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(executable_graph, mark_free_node_, &node_update_params));

    node_update_params.func = (void*)mark_occupied_space_kernel;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(executable_graph, mark_occupied_node_, &node_update_params));

}

} // namespace voxel_mapping