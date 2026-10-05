/*
 * bilateral_openmp.c — OpenMP parallel version
 * Compile: icc -O2 -xHost -qopenmp -o bilateral_openmp bilateral_openmp.c -lm -ljpeg
 * Run:     OMP_NUM_THREADS=8 ./bilateral_openmp
 *
 * The only change from the serial version is the single #pragma on the outer loop.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <jpeglib.h>

#define N       4096
#define RADIUS  5
#define SIGMA_D 3.0f
#define SIGMA_R 20.0f

static void save_jpg(const char *path, const float *src, int n)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot create %s\n", path); return; }
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);
    cinfo.image_width      = (JDIMENSION)n;
    cinfo.image_height     = (JDIMENSION)n;
    cinfo.input_components = 1;
    cinfo.in_color_space   = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    unsigned char *row = (unsigned char *)malloc((size_t)n);
    while (cinfo.next_scanline < cinfo.image_height) {
        int y = (int)cinfo.next_scanline;
        for (int x = 0; x < n; x++) {
            float v = src[(size_t)y * n + x];
            row[x] = (v < 0.5f) ? 0 : (v > 254.5f) ? 255 : (unsigned char)(v + 0.5f);
        }
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    free(row);
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
}

static int load_jpg(const char *path, float *dst, int n)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&cinfo);
    if ((int)cinfo.output_width != n || (int)cinfo.output_height != n) {
        fprintf(stderr, "Expected %dx%d, got %dx%d\n",
                n, n, cinfo.output_width, cinfo.output_height);
        jpeg_destroy_decompress(&cinfo); fclose(fp); return -1;
    }
    unsigned char *row = malloc(n);
    for (int y = 0; cinfo.output_scanline < cinfo.output_height; y++) {
        jpeg_read_scanlines(&cinfo, &row, 1);
        for (int x = 0; x < n; x++) dst[(size_t)y * n + x] = (float)row[x];
    }
    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return 0;
}

/* Compare the saved test JPEG against the saved serial JPEG.
 * Both sides must go through the same 8-bit quantisation + JPEG codec;
 * comparing the raw float output against a decoded JPEG would measure
 * JPEG compression loss instead of parallel-vs-serial differences.     */
static void write_quality_report(const char *txt_path, const char *label,
                                  const char *tst_path, int n)
{
    float *ref = (float *)malloc((size_t)n * n * sizeof(float));
    float *tst = (float *)malloc((size_t)n * n * sizeof(float));
    if (!ref || !tst) {
        fprintf(stderr, "malloc failed in write_quality_report\n");
        free(ref); free(tst); return;
    }
    if (load_jpg("out_serial.jpg", ref, n) != 0) {
        fprintf(stderr, "Cannot load out_serial.jpg — run bilateral_serial first\n");
        free(ref); free(tst); return;
    }
    if (load_jpg(tst_path, tst, n) != 0) {
        free(ref); free(tst); return;
    }

    size_t npix = (size_t)n * n;
    double mse = 0.0, mae = 0.0;
    long   ndiff = 0;
    for (size_t i = 0; i < npix; i++) {
        double d = (double)ref[i] - (double)tst[i];
        mse += d * d;
        mae += (d < 0.0 ? -d : d);
        unsigned char r8 = (ref[i] < 0.5f) ? 0 : (ref[i] > 254.5f) ? 255 : (unsigned char)(ref[i] + 0.5f);
        unsigned char t8 = (tst[i] < 0.5f) ? 0 : (tst[i] > 254.5f) ? 255 : (unsigned char)(tst[i] + 0.5f);
        if (r8 != t8) ndiff++;
    }
    mse /= (double)npix;
    mae /= (double)npix;
    double psnr = (mse < 1e-15) ? 999.0 : 10.0 * log10(255.0 * 255.0 / mse);

    FILE *fp = fopen(txt_path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", txt_path); free(ref); free(tst); return; }
    fprintf(fp, "Bilateral Filter Quality Metrics\n");
    fprintf(fp, "================================\n");
    fprintf(fp, "Version   : %s\n", label);
    fprintf(fp, "Reference : out_serial.jpg (serial baseline)\n");
    fprintf(fp, "Test      : %s\n", tst_path);
    fprintf(fp, "Pixels    : %zu (%dx%d)\n", npix, n, n);
    fprintf(fp, "MSE       : %.6f\n", mse);
    fprintf(fp, "PSNR      : %.4f dB\n", psnr);
    fprintf(fp, "MAE       : %.6f\n", mae);
    fprintf(fp, "Diff px   : %ld (%.4f%%)\n",
            ndiff, 100.0 * (double)ndiff / (double)npix);
    fprintf(fp, "\n");
    fprintf(fp, "Interpretation:\n");
    fprintf(fp, "  MSE  ~ 0      : pixel-perfect match\n");
    fprintf(fp, "  PSNR >= 40 dB : visually identical\n");
    fprintf(fp, "  MAE  ~ 0      : mean per-pixel error near zero\n");
    fclose(fp);

    printf("Quality metrics saved to %s\n", txt_path);
    printf("  MSE=%.6f  PSNR=%.4f dB  MAE=%.6f\n", mse, psnr, mae);
    free(ref); free(tst);
}

static void bilateral_filter(const float *src, float *dst, int rows, int cols)
{
    const float inv2sd2 = 1.0f / (2.0f * SIGMA_D * SIGMA_D);
    const float inv2sr2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);
    const int   wsize   = 2 * RADIUS + 1;

    float slut[wsize * wsize];
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
            slut[(dy + RADIUS) * wsize + (dx + RADIUS)] =
                expf(-(float)(dx * dx + dy * dy) * inv2sd2);

    /* ---- one-line change: parallelize outer loop across rows ---- */
#pragma omp parallel for schedule(static)
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float cval  = src[y * cols + x];
            float sum_w = 0.0f, sum_v = 0.0f;

            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= rows) continue;
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= cols) continue;
                    float nb   = src[ny * cols + nx];
                    float diff = cval - nb;
                    float w    = slut[(dy + RADIUS) * wsize + (dx + RADIUS)]
                               * expf(-(diff * diff) * inv2sr2);
                    sum_w += w;
                    sum_v += w * nb;
                }
            }
            dst[y * cols + x] = sum_v / sum_w;
        }
    }
}

int main(void)
{
    size_t sz  = (size_t)N * N * sizeof(float);
    float *img = (float *)malloc(sz);
    float *out = (float *)malloc(sz);
    if (!img || !out) { fputs("malloc failed\n", stderr); return 1; }

    if (load_jpg("dog_gray.jpg", img, N) != 0) return 1;

    double t0 = omp_get_wtime();
    bilateral_filter(img, out, N, N);
    double t1 = omp_get_wtime();

    printf("OpenMP,%d,%d,%d,%.4f\n", omp_get_max_threads(), N, RADIUS, t1 - t0);
    printf("CPU computation time: %.6f seconds\n", t1 - t0);
    printf("Communication time: 0.000000 seconds\n");
    printf("Total wall time: %.6f seconds\n", t1 - t0);

    save_jpg("out_openmp.jpg", out, N);
    printf("Output saved to out_openmp.jpg\n");

    char txt_path[64];
    snprintf(txt_path, sizeof(txt_path), "quality_openmp_T%d.txt", omp_get_max_threads());
    char label[32];
    snprintf(label, sizeof(label), "OpenMP T=%d", omp_get_max_threads());
    write_quality_report(txt_path, label, "out_openmp.jpg", N);

    free(img); free(out);
    return 0;
}
