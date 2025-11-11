#include <cmath>        // ceil
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>    // runtime_error
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "fma_opencl.hpp"

namespace {
constexpr auto TARGET_DEVICE_TYPE = CL_DEVICE_TYPE_ALL;
}

namespace MyCl
{
int get_platform_num(void)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return platforms.size();
}

int get_device_num(int platform_index)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms.at(platform_index).getDevices(TARGET_DEVICE_TYPE, &devices);

    return devices.size();
}

}  // namespace MyCl

namespace
{

void rethrow_exception(const cl::BuildError& build_error)
{
    auto [device, build_log] = build_error.getBuildLog().at(0);
    auto msg = std::string(build_error.what()) + "() returns " + std::to_string(build_error.err()) + ". build log: " + build_log;
    throw std::runtime_error(msg);
}
void rethrow_exception(const cl::Error& cl_error)
{
    auto msg = std::string(cl_error.what()) + " returns " + std::to_string(cl_error.err());
    throw std::runtime_error(msg);
}

using CoarseRead = cl::SVMTraitCoarse<cl::SVMTraitReadOnly<>>;
using CoarseWrite = cl::SVMTraitCoarse<cl::SVMTraitWriteOnly<>>;
using CoarseReadAllocator = cl::SVMAllocator<float, CoarseRead>;
using CoarseWriteAllocator = cl::SVMAllocator<float, CoarseWrite>;

template <class Alloc>
class SvmArea
{
    using T = Alloc::value_type;
    using pointer_type = T*;
    class UnmapTraits : public std::allocator_traits<Alloc>
    {
        public:
            static pointer_type allocate(Alloc allocator, std::size_t count) {
                std::size_t size = sizeof(T) * count;
                typename Alloc::const_pointer hint = 0;
                bool is_map = false;
                return allocator.allocate(size, hint, is_map);
            }
    };
    void alloc(Alloc allocator, std::size_t count) {
        allocator_ = allocator;
        count_ = count;
        p_ = UnmapTraits::allocate(allocator_, count_);
    }
    void dealloc(void) {
        if (p_) {
            UnmapTraits::deallocate(allocator_, p_, count_);
        }
    }
public:
    SvmArea(void): p_(nullptr), count_(0) {}
    SvmArea(Alloc allocator, std::size_t count): p_(nullptr) {
        alloc(allocator, count);
    }
    SvmArea(const SvmArea& other) {
        dealloc();
        alloc(other.allocator_, other.count_);
    }
    SvmArea& operator = (const SvmArea& other) {
        dealloc();
        alloc(other.allocator_, other.count_);
        return *this;
    }
    ~SvmArea() {
        dealloc();
    }
    pointer_type get(void) {
        return p_;
    }
    const pointer_type get(void) const {
        return p_;
    }
private:
    Alloc allocator_;
    pointer_type p_;
    std::size_t count_;
};

cl::CommandQueue get_queue(int platform_index, int device_index)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    auto err = cl::Platform::get(&platforms);

    err = platforms.at(platform_index).getDevices(TARGET_DEVICE_TYPE, &devices);
    auto device = devices.at(device_index);

    cl::Context context(device);

    return cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
}

cl::Context get_context(cl::CommandQueue& queue)
{
    cl::Context context;
    queue.getInfo(CL_QUEUE_CONTEXT, &context);
    return context;
}

constexpr const char* kernel_source = R"DCS(
kernel void fma(global const float* A, global const float* B, global const float* C, global float* result, int N)
{
    int index = get_global_id(0) * get_global_size(1) + get_global_id(1) * get_global_size(2) + get_global_id(2);
    if (index < N) {
        result[index] = fma(A[index], B[index], C[index]);
    }
}
)DCS";

constexpr const char* build_options = "-cl-std=CL2.0";

cl::Kernel get_fma_kernel(cl::CommandQueue& queue)
{
    cl::Context context;
    cl::Device device;

    queue.getInfo(CL_QUEUE_DEVICE, &device);
    context = get_context(queue);

    cl::Program program(context, kernel_source);
    program.build(device, build_options);

    return cl::Kernel(program, "fma");
}

}  // namespace


namespace MyCl
{

class Fma::Impl
{
    public:
        Impl(int platform_index, int device_index, std::size_t N = 0): data_count_(N), queue_(get_queue(platform_index, device_index)), kernel_(get_fma_kernel(queue_))
        {
            if (N > 0) {
                set_size(N);
            }
        }
        void set_size(std::size_t N)
        {
            assert(N > 0);
            assert(N <= 64*1024*1024);
            cl::Context context = get_context(queue_);
            device_a_ = SvmArea(CoarseReadAllocator(context), N);
            device_b_ = SvmArea(CoarseReadAllocator(context), N);
            device_c_ = SvmArea(CoarseReadAllocator(context), N);
            device_result_ = SvmArea(CoarseWriteAllocator(context), N);
            data_count_ = N;
            global_work_size_ = cl::NDRange(std::ceil((float)N / (64*1024)), std::ceil((float)N / 64), (N > 64)? 64: N);
        }
        void kick(const value_type* A, const value_type* B, const value_type* C, value_type* result)
        {
            queue_.enqueueMemcpySVM(device_a_.get(), A, CL_NON_BLOCKING, sizeof(value_type) * data_count_);
            queue_.enqueueMemcpySVM(device_b_.get(), B, CL_NON_BLOCKING, sizeof(value_type) * data_count_);
            queue_.enqueueMemcpySVM(device_c_.get(), C, CL_NON_BLOCKING, sizeof(value_type) * data_count_, nullptr, &copy_input_event_);
            kernel_.setArg(0, device_a_.get());
            kernel_.setArg(1, device_b_.get());
            kernel_.setArg(2, device_c_.get());
            kernel_.setArg(3, device_result_.get());
            kernel_.setArg(4, (int)data_count_);
            queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, global_work_size_, cl::NullRange);
            queue_.enqueueMemcpySVM(result, device_result_.get(), CL_NON_BLOCKING, sizeof(value_type) * data_count_, nullptr, &copy_output_event_);
            copy_input_event_.wait();
        }
        bool completed(void)
        {
            return copy_output_event_.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
        }
    private:
        std::size_t data_count_;
        cl::CommandQueue queue_;
        cl::Kernel kernel_;
        cl::NDRange global_work_size_;
        cl::Event copy_input_event_, copy_output_event_;
        SvmArea<CoarseReadAllocator> device_a_, device_b_, device_c_;
        SvmArea<CoarseWriteAllocator> device_result_;
};

Fma::Fma(int platform_index, int device_index, std::size_t N): pimpl_(nullptr)
{
    try {
        pimpl_ = std::unique_ptr<Impl>(new Impl(platform_index, device_index, N));
    } catch (const cl::BuildError& e) {
        rethrow_exception(e);
    } catch (const cl::Error& e) {
        rethrow_exception(e);
    }
}

Fma::~Fma() = default;

void Fma::set_size(std::size_t N)
{
    try {
        if (pimpl_) {
            pimpl_->set_size(N);
        }
    } catch (const cl::BuildError& e) {
        rethrow_exception(e);
    } catch (const cl::Error& e) {
        rethrow_exception(e);
    }
}

void Fma::kick(const value_type* A, const value_type* B, const value_type* C, value_type* result)
{
    try {
        if (pimpl_) {
            pimpl_->kick(A, B, C, result);
        }
    } catch (const cl::BuildError& e) {
        rethrow_exception(e);
    } catch (const cl::Error& e) {
        rethrow_exception(e);
    }
}

bool Fma::completed(void)
{
    try {
        if (pimpl_) {
            return pimpl_->completed();
        }
    } catch (const cl::BuildError& e) {
        rethrow_exception(e);
    } catch (const cl::Error& e) {
        rethrow_exception(e);
    }
    return false;
}

}  // namespace MyCl
