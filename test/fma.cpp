#include <vector>
#include <numeric>			// iota
#include <gtest/gtest.h>
#include "fma_opencl.hpp"

using namespace MyCl;
using value_type = Fma::value_type;

TEST(FunctionalTest, PlatformCount)
{
    ASSERT_TRUE(get_platform_num() > 0);
}

TEST(FunctionalTest, DeviceCount)
{
    ASSERT_TRUE(get_device_num(0) > 0);
}

TEST(FunctionalTest, SingleData)
{
    value_type A = 1.5f;
    value_type B = 2.0f;
    value_type C = 1.0f;
    value_type result;

    try {
        Fma fma(0, 0, 1);
        fma.kick(&A, &B, &C, &result);
        while(!fma.completed());
        EXPECT_EQ(4.0f, result);
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}

TEST(FunctionalTest, continuousCycle)
{
    value_type A = 1.5f;
    value_type B = 2.0f;
    value_type C = 1.0f;
    value_type result;

    try {
        Fma fma(0, 0, 1);
        fma.kick(&A, &B, &C, &result);
        while(!fma.completed());
        EXPECT_EQ(4.0f, result);

        A = 3.0f; B = 4.0f; C = 2.0f;
        fma.kick(&A, &B, &C, &result);
        while(!fma.completed());
        EXPECT_EQ(14.0f, result);
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}


class DataSizeTest : public ::testing::TestWithParam<std::size_t>{};

TEST_P(DataSizeTest, SingleCall)
{
    auto data_size = GetParam();

    std::vector<value_type> A(data_size), B(data_size), C(data_size), result(data_size);
	std::iota(std::begin(A), std::end(A),  1.0f);
	std::iota(std::begin(B), std::end(B),  2.0f);
	std::iota(std::begin(C), std::end(C), -1.0f);

    try {
        Fma fma(0, 0, data_size);

        fma.kick(A.data(), B.data(), C.data(), result.data());

        while(!fma.completed());

        // @C++23 --> std::views::zip_transform([](auto a, auto b, auto c, auto res){ return std::pair{a * b + c, res}; }, A, B, C, result)
        for (auto i = 0; i < data_size; i++) {
            ASSERT_FLOAT_EQ(A[i] * B[i] + C[i], result[i]) << "index:" << i;
        }
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}

//INSTANTIATE_TEST_CASE_P(BoundarySizeTest, DataSizeTest, testing::Values(2));
INSTANTIATE_TEST_CASE_P(BoundarySizeTest, DataSizeTest, testing::Values(63, 64+1, 64*1024+1));
