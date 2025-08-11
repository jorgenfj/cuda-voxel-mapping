#ifndef EXTRACTION_RESULT_IMPL_HPP
#define EXTRACTION_RESULT_IMPL_HPP

#include "voxel-mapping/extraction_result.hpp"
#include <cuda_runtime.h>
#include <typeinfo>
#include <memory>

namespace voxel_mapping {

class ExtractionResultBase {
public:
    virtual ~ExtractionResultBase() = default;
    virtual void wait() = 0;
    virtual const void* data() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual const std::type_info& type() const = 0;
};

template<typename T>
class ExtractionResultTyped : public ExtractionResultBase {
public:
    ExtractionResultTyped(void* h_pinned_data, size_t size_bytes, cudaEvent_t event)
        : h_pinned_data_(h_pinned_data), size_bytes_(size_bytes), event_(event) {}

    ~ExtractionResultTyped() = default;

    void wait() override {
        if (event_) cudaEventSynchronize(event_);
    }
    const void* data() const override { return h_pinned_data_; }
    size_t size_bytes() const override { return size_bytes_; }
    const std::type_info& type() const override { return typeid(T); }

private:
    void* h_pinned_data_ = nullptr;
    size_t size_bytes_ = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace voxel_mapping

#endif // EXTRACTION_RESULT_IMPL_HPP