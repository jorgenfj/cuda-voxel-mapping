#ifndef EXTRACTOR_CUH
#define EXTRACTOR_CUH

#include <cuda_runtime.h>
#include "voxel-mapping/internal_types.cuh"

namespace voxel_mapping {

enum class Dimension { X, Y, Z };

class GridProcessor {
public:
    /**
     * @brief Constructor for the GridProcessor responsible for processing extracted voxel blocks before the results are copied to the host.
     * @param occupancy_threshold Threshold for occupancy.
     * @param free_threshold Threshold for free space.
     */
    GridProcessor(int occupancy_threshold, int free_threshold, int edt_max_distance);
    
    /**
     * @brief Launches the kernel for performing 2d Euclidean Distance Transform (EDT) on a set of slices.
     * @param d_edt_slices Pointer to the device memory where the EDT slices will be stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param num_slices Number of slices to extract.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_edt_slice_kernels(int* d_edt_slices, int size_x, int size_y, int num_slices, cudaStream_t stream);

    /**
     * @brief Launches the kernel for performing 3d Euclidean Distance Transform (EDT) on a block of voxels.
     * @param d_edt_block Pointer to the device memory where the EDT block will be stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param size_z Size of the grid in the Z dimension.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_3d_edt_kernels(int* d_edt_block, int size_x, int size_y, int size_z, cudaStream_t stream);

    /**
     * @brief Adds the 2D EDT (slice) kernel nodes to a pre-existing graph.
     * @param graph The CUDA graph to add the nodes to.
     * @param dependencies An optional vector of nodes that the first EDT node depends on.
     * @param d_grid_ptr A pointer to the device memory grid.
     * @param size_x The size of the grid in X.
     * @param size_y The size of the grid in Y.
     * @param size_z The size of the grid in Z.
     * @param out_nodes A vector to store the handles of the nodes that were added.
     */
    cudaGraphNode_t add_edt_slice_nodes(
        cudaGraph_t graph,
        const std::vector<cudaGraphNode_t>& dependencies,
        int* d_grid_ptr,
        int size_x,
        int size_y,
        int size_z);

    /**
     * @brief Updates the kernel parameters for the 2D EDT (slice) nodes in an executable graph.
     * @param exec_graph The executable CUDA graph.
     * @param d_grid_ptr The updated pointer to the device grid data.
     * @param size_x The updated size of the grid in X.
     * @param size_y The updated size of the grid in Y.
     * @param size_z The updated size of the grid in Z (number of slices).
     * @param edt_x_node The handle to the X-pass kernel node.
     * @param edt_y_node The handle to the Y-pass kernel node.
     */
    void update_edt_slice_nodes(
        cudaGraphExec_t exec_graph,
        int* d_grid_ptr,
        int size_x,
        int size_y,
        int size_z);

    cudaGraphNode_t add_edt_block_nodes(
        cudaGraph_t graph,
        const std::vector<cudaGraphNode_t>& dependencies,
        int* d_grid_ptr,
        int size_x,
        int size_y,
        int size_z);

    void update_edt_block_nodes(
        cudaGraphExec_t exec_graph,
        int* d_grid_ptr,
        int size_x,
        int size_y,
        int size_z);

private:
    /**
     * @brief Launches the appropriate kernel for extracting a block of voxels based on the specified extraction type.
     * @param d_grid Pointer to the device memory where the voxel grid is stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param size_z Size of the grid in the Z dimension.
     * @param stream CUDA stream for asynchronous execution.
     */
    template <ExtractionType Type>
    void launch_edt_kernels_internal(int* d_grid, int size_x, int size_y, int size_z, cudaStream_t stream);

    int occupancy_threshold_;
    int free_threshold_;

    cudaGraph_t edt_slice_graph_ = nullptr;
    cudaGraphExec_t edt_slice_exec_graph_ = nullptr;
    cudaGraphNode_t edt_slice_x_node_ = nullptr;
    cudaGraphNode_t edt_slice_y_node_ = nullptr;

    // Graph components for 3D (Block) EDT
    cudaGraph_t edt_block_graph_ = nullptr;
    cudaGraphExec_t edt_block_exec_graph_ = nullptr;
    cudaGraphNode_t edt_block_x_node_ = nullptr;
    cudaGraphNode_t edt_block_y_node_ = nullptr;
    cudaGraphNode_t edt_block_z_node_ = nullptr;

    // Kernel argument placeholders (owned by GridProcessor)
    int* d_grid_ptr_ = nullptr;
    int size_x_ = 0;
    int size_y_ = 0;
    int size_z_ = 0;
    void* kernel_args_x_[5];
    void* kernel_args_y_[5];
    void* kernel_args_z_[5];

};

} // namespace voxel_mapping

#endif // EXTRACTOR_CUH