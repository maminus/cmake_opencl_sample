cmake_minimum_required(VERSION 3.26)  # copy -t(3.26), TARGET_RUNTIME_DLLS(3.21), FetchContent_MakeAvailable(3.14), FetchContent(3.11), gtest_discover_tests(3.10), GIT_SHALLOW(3.6)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(test_fma fma.cpp)
target_link_libraries(test_fma gtest_main fma_opencl)

# https://stackoverflow.com/questions/69978314/cmake-with-gtest-on-windows-build-starts-test-but-shared-libs-cannot-be-found
# https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:TARGET_RUNTIME_DLLS
if (WIN32)
add_custom_command(TARGET test_fma POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:test_fma> $<TARGET_RUNTIME_DLLS:test_fma>
  COMMAND_EXPAND_LISTS
)
endif ()

include(GoogleTest)
gtest_discover_tests(test_fma)
