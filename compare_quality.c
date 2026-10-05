/*
 * compare_quality.c — MSE / PSNR / MAE between two grayscale JPEG images
 * Compile: icc -O2 -o compare_quality compare_quality.c -lm -ljpeg
 * Usage:   ./compare_quality <reference.jpg> <test.jpg>
 *
 * Typical workflow (run after all four bilateral programs):
 *   ./compare_quality out_serial.jpg out_openmp.jpg
 *   ./compare_quality out_serial.jpg out_mpi_static.jpg
 *   ./compare_quality out_serial.jpg out_mpi_dynamic.jpg
 *
 * Interpretation:
 *   MSE  ~ 0       — pixel-perfect match
 *   PSNR >= 40 dB  — visually identical (JPEG codec differences only)
 *   MAE  ~ 0       — mean per-pixel error near zero
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>

static int load_jpg(const char *path, unsigned char **dst, int *width, int *height)
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

    *width  = (int)cinfo.output_width;
    *height = (int)cinfo.output_height;

    *dst = (unsigned char *)malloc((size_t)(*width) * (size_t)(*height));
    if (!*dst) {
        jpeg_destroy_decompress(&cinfo); fclose(fp); return -1;
    }

    unsigned char *row = (unsigned char *)malloc((size_t)*width);
    for (int y = 0; (int)cinfo.output_scanline < *height; y++) {
        jpeg_read_scanlines(&cinfo, &row, 1);
        for (int x = 0; x < *width; x++)
            (*dst)[(size_t)y * (size_t)(*width) + x] = row[x];
    }
    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <reference.jpg> <test.jpg>\n", argv[0]);
        return 1;
    }

    unsigned char *ref = NULL, *tst = NULL;
    int rw, rh, tw, th;

    if (load_jpg(argv[1], &ref, &rw, &rh) != 0) return 1;
    if (load_jpg(argv[2], &tst, &tw, &th) != 0) { free(ref); return 1; }

    if (rw != tw || rh != th) {
        fprintf(stderr, "Dimension mismatch: %dx%d vs %dx%d\n", rw, rh, tw, th);
        free(ref); free(tst);
        return 1;
    }

    size_t  npix = (size_t)rw * (size_t)rh;
    double  mse  = 0.0;
    double  mae  = 0.0;
    long    ndiff = 0;         /* pixels that differ at all */

    for (size_t i = 0; i < npix; i++) {
        double d = (double)ref[i] - (double)tst[i];
        mse  += d * d;
        mae  += (d < 0.0 ? -d : d);
        if (ref[i] != tst[i]) ndiff++;
    }
    mse /= (double)npix;
    mae /= (double)npix;

    double psnr = (mse < 1e-15) ? 999.0 : 10.0 * log10(255.0 * 255.0 / mse);

    printf("Reference : %s\n",    argv[1]);
    printf("Test      : %s\n",    argv[2]);
    printf("Size      : %dx%d (%zu pixels)\n", rw, rh, npix);
    printf("MSE       : %.6f\n",  mse);
    printf("PSNR      : %.4f dB\n", psnr);
    printf("MAE       : %.6f\n",  mae);
    printf("Diff px   : %ld (%.4f%%)\n",
           ndiff, 100.0 * (double)ndiff / (double)npix);

    free(ref);
    free(tst);
    return 0;
}
