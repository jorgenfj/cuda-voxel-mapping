#include "voxel-mapping/voxel_mapping.hpp"
#include "voxel-mapping/voxel_mapping_impl.cuh"

namespace voxel_mapping {

VoxelMapping::VoxelMapping(const VoxelMappingParams& params)
    : pimpl_(std::make_unique<VoxelMappingImpl>(params)) {
}

VoxelMapping::~VoxelMapping() = default;
VoxelMapping::VoxelMapping(VoxelMapping&&) = default;
VoxelMapping& VoxelMapping::operator=(VoxelMapping&&) = default;

void VoxelMapping::integrate_depth(const float* depth_image, const float* transform) {
    pimpl_->integrate_depth(depth_image, transform);
}

ExtractionResult VoxelMapping::extract_grid_block(const AABB& aabb) {
    return pimpl_->extract_grid_data<ExtractionType::Block>(aabb, SliceZIndices{});
}

ExtractionResult VoxelMapping::extract_grid_slices(const AABB& aabb, const SliceZIndices& slice_indices) {
    return pimpl_->extract_grid_data<ExtractionType::Slice>(aabb, slice_indices);
}

ExtractionResult VoxelMapping::extract_edt_block(const AABB& aabb) {
    // return pimpl_->extract_edt_data<ExtractionType::Block>(aabb, SliceZIndices{});
}

ExtractionResult VoxelMapping::extract_edt_slices(const AABB& aabb, const SliceZIndices& slice_indices) {
    // return pimpl_->extract_edt_data<ExtractionType::Slice>(aabb, slice_indices);
}

void VoxelMapping::query_free_chunk_capacity() {
    pimpl_->query_free_chunk_capacity();
}

void VoxelMapping::set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height) {
    pimpl_->set_camera_properties(fx, fy, cx, cy, width, height);
}

AABB VoxelMapping::get_current_aabb() const {
    return pimpl_->get_current_aabb();
}

Frustum VoxelMapping::get_frustum() const {
    return pimpl_->get_frustum();
}

} // namespace voxel_mapping