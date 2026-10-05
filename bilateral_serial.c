/*
 * bilateral_serial.c — Sequential baseline
 * Compile: icc -O2 -xHost -o bilateral_serial bilateral_serial.c -lm -ljpeg
 * Run:     ./bilateral_serial
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

static void bilateral_filter(const float *src, float *dst, int rows, int cols)
{
    const float inv2sd2 = 1.0f / (2.0f * SIGMA_D * SIGMA_D);
    const float inv2sr2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);
    const int   wsize   = 2 * RADIUS + 1;

    /* precompute spatial Gaussian weights — avoids repeated exp() in inner loop */
    float slut[wsize * wsize];
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
            slut[(dy + RADIUS) * wsize + (dx + RADIUS)] =
                expf(-(float)(dx * dx + dy * dy) * inv2sd2);

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

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bilateral_filter(img, out, N, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("Serial,1,%d,%d,%.4f\n", N, RADIUS, elapsed);
    printf("CPU computation time: %.6f seconds\n", elapsed);
    printf("Communication time: 0.000000 seconds\n");
    printf("Total wall time: %.6f seconds\n", elapsed);

    save_jpg("out_serial.jpg", out, N);
    printf("Output saved to out_serial.jpg\n");

    free(img); free(out);
    return 0;
}
