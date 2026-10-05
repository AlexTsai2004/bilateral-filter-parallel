# 雙邊濾波平行化（Bilateral Filter Parallelization）

對一張 4096×4096 灰階影像（`dog_gray.jpg`）執行雙邊濾波（Bilateral Filter），並比較循序版與四種平行化實作的效能與輸出品質。

| 參數 | 值 |
|---|---|
| 影像大小 `N` | 4096 × 4096 |
| 視窗半徑 `RADIUS` | 5（11×11 視窗） |
| 空間標準差 `SIGMA_D` | 3.0 |
| 灰階標準差 `SIGMA_R` | 20.0 |

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `bilateral_serial.c` | 循序版基準，輸出 `out_serial.jpg`（其他版本的品質比較都以它為準） |
| `bilateral_openmp.c` | OpenMP 版：外層 row 迴圈加上 `#pragma omp parallel for` |
| `bilateral_mpi.c` | MPI 靜態分割：1-D row decomposition，`MPI_Sendrecv` 交換 ghost rows，`Scatterv`/`Gatherv` 分配與收集 |
| `bilateral_mpi_dynamic.c` | MPI 動態 Master–Worker：rank 0 以 32 rows 為一個工作單位分派給閒置的 worker（至少需 2 個行程） |
| `bilateral_cuda.ipynb` | CUDA 版（Google Colab）：分別以 4×4、16×16、32×32 block size 測量 H2D / Kernel / D2H 時間 |
| `compare_quality.c` | 計算兩張 JPEG 之間的 MSE / PSNR / MAE |
| `雙邊濾波平行化報告.pdf` | 期末報告 |

## 編譯與執行

需要 libjpeg（`libjpeg-dev`）。以下是原始碼標頭中的 Intel 編譯器指令，換成 `gcc` / `mpicc`（OpenMP 用 `-fopenmp`）也可以。

```bash
# 循序版（請先執行，其他版本需要 out_serial.jpg 做品質比較）
icc -O2 -xHost -o bilateral_serial bilateral_serial.c -lm -ljpeg
./bilateral_serial

# OpenMP
icc -O2 -xHost -qopenmp -o bilateral_openmp bilateral_openmp.c -lm -ljpeg
OMP_NUM_THREADS=8 ./bilateral_openmp

# MPI 靜態分割
mpiicc -O2 -xHost -o bilateral_mpi_static bilateral_mpi.c -lm -ljpeg
mpirun -n 8 ./bilateral_mpi_static
mpirun -n 8 --hostfile hostfile ./bilateral_mpi_static   # 多節點

# MPI 動態 Master–Worker
mpiicc -O2 -xHost -o bilateral_mpi_dynamic bilateral_mpi_dynamic.c -lm -ljpeg
mpirun -n 8 ./bilateral_mpi_dynamic

# 品質比較
icc -O2 -o compare_quality compare_quality.c -lm -ljpeg
./compare_quality out_serial.jpg out_openmp.jpg
```

CUDA 版：在 Google Colab 開啟 `bilateral_cuda.ipynb`，執行階段選 GPU，把 `dog_gray.jpg`（以及做品質比較用的 `out_serial.jpg`）上傳到 `/content/` 後依序執行各 cell。

## 輸出

每個程式都會輸出一行 CSV 格式的計時結果，再加上計算、通訊與總時間：

```
<版本>,<執行緒或行程數>,<N>,<RADIUS>,<秒數>
CPU computation time: ...
Communication time: ...
Total wall time: ...
```

MPI 版會讀取 Slurm 的 `SLURM_NNODES`，自動標示 `single` 或 `multi` 節點。OpenMP 與 MPI 版另外會產生 `quality_<版本>_T<n>.txt` / `_P<n>.txt` 品質報告。

## 版本與修正

- `original` 分支：原始繳交版本。
- `main` 分支：修正以下問題（差異見 [original...main](https://github.com/AlexTsai2004/bilateral-filter-parallel/compare/original...main)）：
  - **品質報告比較基準錯誤**：原本拿未量化的 float 輸出去比 JPEG 解碼後的 serial 結果，量到的是 JPEG 壓縮誤差而不是平行化誤差。報告中 CPU 版本的 MSE 0.1368 / PSNR 56.77 dB 就是這個原因造成的。修正後改成 JPEG 對 JPEG 比較，演算法相同時 MSE 應為 0。
  - **MPI ghost row 交換被串成一條鏈**：改成兩階段 `MPI_Sendrecv` 搭配 `MPI_PROC_NULL`，通訊時間不再隨行程數線性增加。
  - **CUDA 冷啟動偏差**：加入不計時的 warm-up，避免第一個 block size（4×4）的時間被 context 初始化與 page fault 拉高；另外加上 `CUDA_CHECK` 錯誤檢查。

> 注意：`雙邊濾波平行化報告.pdf` 中的數據是修正前量測的，品質指標與 MPI / CUDA 計時需要重新執行後才會反映修正。
