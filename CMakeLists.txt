#[[
# ビルド・テスト手順
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release

# インストール手順
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DBUILD_TESTING=OFF
cmake --build build --config Release
cmake --install build

Windows: git, cmake, Visual Studio Communityが必要
Linux: git, cmake, g++, build-essentialが必要。実行にはさらにintel-opencl-icdなどの環境に合わせたopencl-icdパッケージが必要
]]

cmake_minimum_required(VERSION 3.24)  # FetchContent_MakeAvailable(3.14, 3.24), install中のFILE_SET(3.23), RUNTIME_DEPENDENCY_SET(3.21), GIT_SUBMODULES_RECURSE(3.17), FetchContent(3.11), GIT_SHALLOW(3.6)

project(FMA_OpenCL CXX)


include(FetchContent)

# ----------------------------------------------------
# OpenCL-SDKをFetchContentで取得
# ----------------------------------------------------
if (WIN32)
  FetchContent_Declare(OpenCL_SDK
    URL https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2025.07.23/OpenCL-SDK-v2025.07.23-Win-x64.zip
  )
else ()
  # Ubuntuならocl-icd-libopencl1,opencl-c-headers,opencl-clhpp-headersパッケージがあればFetchContentせずfind_package()のみでOK。今回はcmakeの練習のためFetchContentを使う

  # https://github.com/KhronosGroup/OpenCL-SDK/issues/47#issuecomment-1141073908
  FetchContent_Declare(OpenCL_SDK
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
    GIT_TAG v2025.07.23
    GIT_SUBMODULES_RECURSE TRUE   # 再帰的なクローンを有効にする（OpenCL-SDKはサブモジュールを使用）
    GIT_SHALLOW TRUE              # コミット履歴は不要なので--depth 1扱い
  )
  set(OPENCL_SDK_BUILD_SAMPLES OFF)  # FetchContent_DeclareでCMAKE_ARGSが効かないのでワークアラウンド https://discourse.cmake.org/t/fetchcontent-declare-with-cmake-args-does-not-work/11227/7
endif ()

# Windowsの場合:ダウンロード+unzip、Linuxの場合: git clone＋add_subdirectory
FetchContent_MakeAvailable(OpenCL_SDK)

if (WIN32)
  # Windowsはダウンロードしかしていないのでfind_packageする
  find_package(OpenCL REQUIRED
    HINTS ${opencl_sdk_SOURCE_DIR}  # build/_deps/opencl_sdk-src
  )
  # message(NOTICE "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})  # build
  if (OpenCL_FOUND)
    message(NOTICE "OpenCL FOUND")
  endif ()
endif ()


# ----------------------------------------------------
# 実行可能ターゲットの定義とリンク
# ----------------------------------------------------

add_library(fma_opencl SHARED)

include(GenerateExportHeader)
generate_export_header(fma_opencl)

# ソースコード
target_sources(fma_opencl
  PRIVATE
    src/main.cpp
  INTERFACE
    FILE_SET HEADERS
      TYPE HEADERS
      FILES
	    # ここでファイル名のみにしてBASE_DIRSを指定しても結局CMAKE_CURRENT_SOURCE_DIRからの相対パス扱いされて意味がない...(実質ファイル名のみの指定ができない)
        include/fma_opencl.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/fma_opencl_export.h
)
target_include_directories(fma_opencl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)  # ここで$<TARGET_PROPERTY:INTERFACE_INCLUDE_DIRECTORIES>を使おうとしても"Dependency loop found."とか言われてエラーになる
if (NOT WIN32)
  target_include_directories(fma_opencl PUBLIC
    ${opencl_sdk_SOURCE_DIR}/external/OpenCL-CLHPP/include
    ${opencl_sdk_SOURCE_DIR}/external/OpenCL-Headers
  )
endif ()

include(GNUInstallDirs)

# https://cmake.org/cmake/help/latest/command/install.html
install(TARGETS fma_opencl
  # EXPORT fma_ocl_exports    # install TARGETS target fma_opencl is exported but not all of its interface
  # RUNTIME_DEPENDENCY_SET fma_ocl_dlls
  RUNTIME                     # デフォルトのDESTINATIONはbin。WindowsのDLL
  LIBRARY                     # デフォルトのDESTINATIONはlib。LinuxのDLL
  ARCHIVE                     # デフォルトのDESTINATIONはlib。静的ライブラリもしくはWindows版DLLに対するlib
  # ここでFILE_SETを指定するとFILESで指定した親ディレクトリを含めたファイルがインストールされてしまう
)
install(FILES
  $<TARGET_PROPERTY:fma_opencl,HEADER_SET_HEADERS>
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/my_proj"
)
# install(EXPORT fma_ocl_exports DESTINATION cmake)  # EXPORTを定義できていないのでダメ
# install(RUNTIME_DEPENDENCY_SET fma_ocl_dlls)  # Visual Studio関連DLLまで探そうとしてエラーになる...
install(FILES
  $<TARGET_RUNTIME_DLLS:fma_opencl>
  DESTINATION $<$<PLATFORM_ID:Windows>:${CMAKE_INSTALL_BINDIR}>$<$<PLATFORM_ID:Linux>:${CMAKE_INSTALL_LIBDIR}>
)
# install(IMPORTED_RUNTIME_ARTIFACTS)だとfma_opencl.dllがインストールされてしまう

# OpenCL-SDKが提供するインポート済みターゲット OpenCL::OpenCL をリンクする
target_link_libraries(fma_opencl PRIVATE OpenCL::OpenCL)

set_target_properties(fma_opencl PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# ----------------------------------------------------
# テスト
# ----------------------------------------------------
option(BUILD_TESTING "Enable testing and build tests" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif ()
