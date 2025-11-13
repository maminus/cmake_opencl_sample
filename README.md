[![ci](https://github.com/maminus/cmake_opencl_sample/actions/workflows/ci.yml/badge.svg)](https://github.com/maminus/cmake_opencl_sample/actions/workflows/ci.yml)

# cmake_opencl_sample

CMakeを使ってOpenCLを呼び出すDLLを開発するプロジェクトの習作

## 実装している機能

* float配列同士の積和算

```cpp
#include <fma_opencl_export.h>

int main()
{
    int platform_index = 0;
    int device_index = 0;
    std::size_t data_size = 10;

    // 入出力データ置き場
    std::vector<value_type> A(data_size), B(data_size), C(data_size), result(data_size);

    std::iota(std::begin(A), std::end(A),  1.0f);
	std::iota(std::begin(B), std::end(B),  2.0f);
	std::iota(std::begin(C), std::end(C), -1.0f);

    try {
        // OpenCLの準備
        Fma fma(platform_index, device_index, data_size);

        // OpenCLで計算を実行
        fma.kick(A.data(), B.data(), C.data(), result.data());

        // 計算完了待ち
        while(!fma.completed());

    } catch (const std::exception& e) {
        // 失敗したら例外が飛んでくる
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

## 対応していないこと

* staticリンクライブラリとDLLを切り替える
* バージョン番号付与
* cmakeファイル生成

## 必要なソフト

* Windows
  - git
  - cmake
  - Visual Studio Community
  - OpenCLのランタイム
* Ubuntu
  - git
  - cmake
  - g++
  - build-essential
  - intel-opencl-icdなどの環境に合わせたopencl-icdパッケージ

## ビルド・テスト手順

```bash
git clone https://github.com/maminus/cmake_opencl_sample.git
cd cmake_opencl_sample
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

※configure時のCMAKE_BUILD_TYPEは不要かも

## ビルド・インストール手順

```bash
git clone https://github.com/maminus/cmake_opencl_sample.git
cd cmake_opencl_sample
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DBUILD_TESTING=OFF
cmake --build build --config Release
cmake --install build
```

※configure時のCMAKE_BUILD_TYPEは不要かも
