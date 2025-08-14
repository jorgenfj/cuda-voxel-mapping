#include "voxel-mapping/grid_processor.cuh"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include <cuda_runtime.h>

namespace voxel_mapping {

static __constant__ VoxelType d_occupancy_threshold;
static __constant__ VoxelType d_free_threshold;
static __constant__ int d_esdf_max_distance;

GridProcessor::GridProcessor(int occupancy_threshold, int free_threshold, int esdf_max_distance)
    : occupancy_threshold_(occupancy_threshold), free_threshold_(free_threshold) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold_, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold_, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_esdf_max_distance, &esdf_max_distance, sizeof(int), 0, cudaMemcpyHostToDevice));
}

/**
 * @brief Performs a single 1D pass of a squared Euclidean Distance Transform with a maximum distance.
 * @tparam Dim The dimension (X, Y, or Z) along which to perform the pass.
 * @param input Pointer to the input 3D grid in device memory.
 * @param out Pointer to the output 3D grid in device memory.
 * @param size_x The size of the grid in the X dimension.
 * @param size_y The size of the grid in the Y dimension.
 * @param size_z The size of the grid in the Z dimension.
 * @param max_distance The maximum distance (in voxels) to search in each direction.
 */
template <Dimension Dim>
__global__ void edt_1d_pass_kernel(int* input, int* out, int size_x, int size_y, int size_z) {
    int p1, p2, line_length;

    if constexpr (Dim == Dimension::X) {
        p1 = blockIdx.x; p2 = blockIdx.y; line_length = size_x;
        if (p1 >= size_y || p2 >= size_z) return;
    } else if constexpr (Dim == Dimension::Y) {
        p1 = blockIdx.x; p2 = blockIdx.y; line_length = size_y;
        if (p1 >= size_x || p2 >= size_z) return;
    } else { // Dimension::Z
        p1 = blockIdx.x; p2 = blockIdx.y; line_length = size_z;
        if (p1 >= size_x || p2 >= size_y) return;
    }

    extern __shared__ int line_shmem[];
    for (int i = threadIdx.x; i < line_length; i += blockDim.x) {
        size_t idx;
        if constexpr (Dim == Dimension::X) idx = (size_t)p2 * size_y * size_x + (size_t)p1 * size_x + i;
        if constexpr (Dim == Dimension::Y) idx = (size_t)p2 * size_y * size_x + (size_t)i  * size_x + p1;
        if constexpr (Dim == Dimension::Z) idx = (size_t)i  * size_y * size_x + (size_t)p2 * size_x + p1;
        line_shmem[i] = input[idx];
    }
    __syncthreads();

    for (int current_idx = threadIdx.x; current_idx < line_length; current_idx += blockDim.x) {
        int value = line_shmem[current_idx];

        for (int i = 1, d = 1; i <= d_esdf_max_distance && current_idx + i < line_length; i++) {
            value = min(value, line_shmem[current_idx + i] + d);
            d += 1 + 2 * i;
        }
        for (int i = 1, d = 1; i <= d_esdf_max_distance && current_idx - i >= 0; i++) {
            value = min(value, line_shmem[current_idx - i] + d);
            d += 1 + 2 * i;
        }

        size_t out_idx;
        if constexpr (Dim == Dimension::X) out_idx = (size_t)p2 * size_y * size_x + (size_t)p1 * size_x + current_idx;
        if constexpr (Dim == Dimension::Y) out_idx = (size_t)p2 * size_y * size_x + (size_t)current_idx * size_x + p1;
        if constexpr (Dim == Dimension::Z) out_idx = (size_t)current_idx * size_y * size_x + (size_t)p2 * size_x + p1;
        out[out_idx] = value;
    }
}

template <ExtractionType Type>
void GridProcessor::launch_edt_kernels_internal(int* d_grid, int size_x, int size_y, int size_z, cudaStream_t stream) {
    dim3 grid_dim_x(size_y, size_z);
    dim3 block_dim_x(256);
    size_t shared_mem_x = size_x * sizeof(int);
    edt_1d_pass_kernel<Dimension::X><<<grid_dim_x, block_dim_x, shared_mem_x, stream>>>(
        d_grid, d_grid, size_x, size_y, size_z);

    dim3 grid_dim_y(size_x, size_z);
    dim3 block_dim_y(256);
    size_t shared_mem_y = size_y * sizeof(int);
    edt_1d_pass_kernel<Dimension::Y><<<grid_dim_y, block_dim_y, shared_mem_y, stream>>>(
        d_grid, d_grid, size_x, size_y, size_z);

    if constexpr (Type == ExtractionType::Block) {
        dim3 grid_dim_z(size_x, size_y);
        dim3 block_dim_z(256);
        size_t shared_mem_z = size_z * sizeof(int);
        edt_1d_pass_kernel<Dimension::Z><<<grid_dim_z, block_dim_z, shared_mem_z, stream>>>(
            d_grid, d_grid, size_x, size_y, size_z);
    }
}

void GridProcessor::launch_edt_slice_kernels(int* d_edt_slices, int size_x, int size_y, int num_slices, cudaStream_t stream) {
    launch_edt_kernels_internal<ExtractionType::Slice>(d_edt_slices, size_x, size_y, num_slices, stream);
}

void GridProcessor::launch_3d_edt_kernels(int* d_grid, int size_x, int size_y, int size_z, cudaStream_t stream) {
    launch_edt_kernels_internal<ExtractionType::Block>(d_grid, size_x, size_y, size_z, stream);
}

cudaGraphNode_t GridProcessor::add_edt_slice_nodes(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& dependencies,
    int* d_grid_ptr,
    int size_x,
    int size_y,
    int size_z) {
    
    int* d_grid_ptr_ = d_grid_ptr;
    int size_x_ = size_x;
    int size_y_ = size_y;
    int size_z_ = size_z;

    void* kernel_args_x_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_y_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};

    cudaKernelNodeParams x_params = {};
    x_params.func = (void*)edt_1d_pass_kernel<Dimension::X>;
    x_params.gridDim = dim3(size_y, size_z);
    x_params.blockDim = dim3(256);
    x_params.sharedMemBytes = size_x * sizeof(int);
    x_params.kernelParams = kernel_args_x_;
    
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &edt_slice_x_node_, graph, dependencies.data(), dependencies.size(), &x_params));

    cudaKernelNodeParams y_params = {};
    y_params.func = (void*)edt_1d_pass_kernel<Dimension::Y>;
    y_params.gridDim = dim3(size_x, size_z);
    y_params.blockDim = dim3(256);
    y_params.sharedMemBytes = size_y * sizeof(int);
    y_params.kernelParams = kernel_args_y_;
    
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &edt_slice_y_node_, graph, &edt_slice_x_node_, 1, &y_params));
    return edt_slice_y_node_;
}

void GridProcessor::update_edt_slice_nodes(
    cudaGraphExec_t exec_graph,
    int* d_grid_ptr,
    int size_x,
    int size_y,
    int size_z) {

    int* d_grid_ptr_ = d_grid_ptr;
    int size_x_ = size_x;
    int size_y_ = size_y;
    int size_z_ = size_z;

    void* kernel_args_x_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_y_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};

    cudaKernelNodeParams x_params = {};
    x_params.func = (void*)edt_1d_pass_kernel<Dimension::X>;
    x_params.gridDim = dim3(size_y, size_z);
    x_params.blockDim = dim3(256);
    x_params.sharedMemBytes = size_x * sizeof(int);
    x_params.kernelParams = kernel_args_x_;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, edt_slice_x_node_, &x_params));

    cudaKernelNodeParams y_params = {};
    y_params.func = (void*)edt_1d_pass_kernel<Dimension::Y>;
    y_params.gridDim = dim3(size_x, size_z);
    y_params.blockDim = dim3(256);
    y_params.sharedMemBytes = size_y * sizeof(int);
    y_params.kernelParams = kernel_args_y_;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, edt_slice_y_node_, &y_params));
}

cudaGraphNode_t GridProcessor::add_edt_block_nodes(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& dependencies,
    int* d_grid_ptr,
    int size_x,
    int size_y,
    int size_z) {

    int* d_grid_ptr_ = d_grid_ptr;
    int size_x_ = size_x;
    int size_y_ = size_y;
    int size_z_ = size_z;

    void* kernel_args_x_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_y_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_z_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};

    cudaKernelNodeParams x_params = {};
    x_params.func = (void*)edt_1d_pass_kernel<Dimension::X>;
    x_params.gridDim = dim3(size_y, size_z);
    x_params.blockDim = dim3(256);
    x_params.sharedMemBytes = size_x * sizeof(int);
    x_params.kernelParams = kernel_args_x_;
    
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &edt_block_x_node_, graph, dependencies.data(), dependencies.size(), &x_params));

    cudaKernelNodeParams y_params = {};
    y_params.func = (void*)edt_1d_pass_kernel<Dimension::Y>;
    y_params.gridDim = dim3(size_x, size_z);
    y_params.blockDim = dim3(256);
    y_params.sharedMemBytes = size_y * sizeof(int);
    y_params.kernelParams = kernel_args_y_;
    
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &edt_block_y_node_, graph, &edt_block_x_node_, 1, &y_params));

    cudaKernelNodeParams z_params = {};
    z_params.func = (void*)edt_1d_pass_kernel<Dimension::Z>;
    z_params.gridDim = dim3(size_x, size_y);
    z_params.blockDim = dim3(256);
    z_params.sharedMemBytes = size_z * sizeof(int);
    z_params.kernelParams = kernel_args_z_;

    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(
        &edt_block_z_node_, graph, &edt_block_y_node_, 1, &z_params));

    return edt_block_z_node_;
}

void GridProcessor::update_edt_block_nodes(
    cudaGraphExec_t exec_graph,
    int* d_grid_ptr,
    int size_x,
    int size_y,
    int size_z) {

    int* d_grid_ptr_ = d_grid_ptr;
    int size_x_ = size_x;
    int size_y_ = size_y;
    int size_z_ = size_z;

    void* kernel_args_x_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_y_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};
    void* kernel_args_z_[5] = {&d_grid_ptr_, &d_grid_ptr_, &size_x_, &size_y_, &size_z_};

    cudaKernelNodeParams x_params = {};
    x_params.func = (void*)edt_1d_pass_kernel<Dimension::X>;
    x_params.gridDim = dim3(size_y, size_z);
    x_params.blockDim = dim3(256);
    x_params.sharedMemBytes = size_x * sizeof(int);
    x_params.kernelParams = kernel_args_x_;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, edt_block_x_node_, &x_params));

    cudaKernelNodeParams y_params = {};
    y_params.func = (void*)edt_1d_pass_kernel<Dimension::Y>;
    y_params.gridDim = dim3(size_x, size_z);
    y_params.blockDim = dim3(256);
    y_params.sharedMemBytes = size_y * sizeof(int);
    y_params.kernelParams = kernel_args_y_;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, edt_block_y_node_, &y_params));

    cudaKernelNodeParams z_params = {};
    z_params.func = (void*)edt_1d_pass_kernel<Dimension::Z>;
    z_params.gridDim = dim3(size_x, size_y);
    z_params.blockDim = dim3(256);
    z_params.sharedMemBytes = size_z * sizeof(int);
    z_params.kernelParams = kernel_args_z_;
    
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(exec_graph, edt_block_z_node_, &z_params));
}

} // namespace voxel_mapping

