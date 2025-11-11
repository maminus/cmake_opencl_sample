#pragma once
#include <cstddef>    // size_t
#include <string>
#include <memory>     // unique_ptr
#include <fma_opencl_export.h>

namespace MyCl
{

int FMA_OPENCL_EXPORT get_platform_num(void);

int FMA_OPENCL_EXPORT get_device_num(int platform_index);

class FMA_OPENCL_EXPORT Fma
{
    public:
		using value_type = float;

        Fma(int platform_index, int device_index, std::size_t N = 0);
        ~Fma();

        void set_size(std::size_t N);
        void kick(const value_type* A, const value_type* B, const value_type* C, value_type* result);
        bool completed(void);

    private:
        Fma(const Fma& other) = delete;
        Fma& operator=(const Fma& other) = delete;

        class Impl;
        std::unique_ptr<Impl> pimpl_;
};

}  // namespace MyCl
