/*
 * bilateral_mpi.c — MPI parallel version (single-node and multi-node)
 * Compile: mpiicc -O2 -xHost -o bilateral_mpi_static bilateral_mpi.c -lm -ljpeg
 * Run (single node):  mpirun -n 8  ./bilateral_mpi
 * Run (multi  node):  mpirun -n 8 --hostfile hostfile ./bilateral_mpi
 *
 * Strategy: 1-D row decomposition with ghost-row exchange via MPI_Sendrecv.
 *   Rank 0 generates the full image and scatters rows to all ranks.
 *   Each rank exchanges RADIUS boundary rows with its neighbours.
 *   After filtering, results are gathered back to Rank 0.
 *
 * Output (CSV):  MPI_<type>,<procs>,<N>,<radius>,<time_s>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <jpeglib.h>

#define N       4096
#define RADIUS  5
#define SIGMA_D 3.0f
#define SIGMA_R 20.0f

#define TAG_DOWN 10   /* data moving to higher-rank neighbour */
#define TAG_UP   20   /* data moving to lower-rank  neighbour */

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

/* Filter local_rows rows stored in src.
 * src layout: [top_gh rows | local_rows rows | bot_gh rows] × N
 * dst layout: [local_rows rows] × N
 */
static void bilateral_filter_local(
    const float *src, float *dst,
    int top_gh, int bot_gh, int local_rows, int cols)
{
    const float inv2sd2 = 1.0f / (2.0f * SIGMA_D * SIGMA_D);
    const float inv2sr2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);
    const int   wsize   = 2 * RADIUS + 1;
    const int   tot     = top_gh + local_rows + bot_gh;

    float slut[wsize * wsize];
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
            slut[(dy + RADIUS) * wsize + (dx + RADIUS)] =
                expf(-(float)(dx * dx + dy * dy) * inv2sd2);

    for (int ly = top_gh; ly < top_gh + local_rows; ly++) {
        for (int x = 0; x < cols; x++) {
            float cval  = src[(size_t)ly * cols + x];
            float sum_w = 0.0f, sum_v = 0.0f;

            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                int ny = ly + dy;
                if (ny < 0 || ny >= tot) continue;
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= cols) continue;
                    float nb   = src[(size_t)ny * cols + nx];
                    float diff = cval - nb;
                    float w    = slut[(dy + RADIUS) * wsize + (dx + RADIUS)]
                               * expf(-(diff * diff) * inv2sr2);
                    sum_w += w;
                    sum_v += w * nb;
                }
            }
            dst[(size_t)(ly - top_gh) * cols + x] = sum_v / sum_w;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- row distribution (handles N % size != 0) ---- */
    int base     = N / size;
    int rem      = N % size;
    int my_rows  = base + (rank < rem ? 1 : 0);

    /* Check the smallest partition (base) so every rank takes the same branch
     * and rank 0 — which may own base+1 rows — still reports the error.     */
    if (base < RADIUS) {
        /* MPI_Abort (not MPI_Finalize) so that ranks with more rows do not
         * hang waiting at the subsequent collectives (Scatterv/Gatherv).   */
        if (rank == 0)
            fprintf(stderr, "Error: rows per process (%d) < RADIUS (%d). "
                    "Reduce process count.\n", base, RADIUS);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* All ranks precompute scatter/gather counts — avoids NULL-pointer issues
     * with MPI_Scatterv/Gatherv on non-root ranks.                           */
    int *scnt = (int *)malloc(size * sizeof(int));
    int *sdsp = (int *)malloc(size * sizeof(int));
    {
        int off = 0;
        for (int r = 0; r < size; r++) {
            scnt[r] = (base + (r < rem ? 1 : 0)) * N;
            sdsp[r] = off;
            off    += scnt[r];
        }
    }

    /* ---- rank 0 generates the full image ---- */
    float *full_img = NULL, *full_out = NULL;
    if (rank == 0) {
        full_img = (float *)malloc((size_t)N * N * sizeof(float));
        full_out = (float *)malloc((size_t)N * N * sizeof(float));
        if (!full_img || !full_out) {
            fputs("malloc failed\n", stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (load_jpg("dog_gray.jpg", full_img, N) != 0)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- scatter core data ---- */
    float *local_core = (float *)malloc((size_t)my_rows * N * sizeof(float));
    MPI_Scatterv(full_img, scnt, sdsp, MPI_FLOAT,
                 local_core, my_rows * N, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* ---- start timing: includes ghost exchange + filter ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* ---- build padded buffer with ghost-row placeholders ---- */
    int top_gh = (rank > 0)        ? RADIUS : 0;
    int bot_gh = (rank < size - 1) ? RADIUS : 0;
    int tot    = top_gh + my_rows + bot_gh;
    float *padded = (float *)calloc((size_t)tot * N, sizeof(float));
    memcpy(padded + (size_t)top_gh * N, local_core, (size_t)my_rows * N * sizeof(float));

    /* ---- ghost-row exchange with MPI_Sendrecv ----
     *
     * Two phases, every rank participates in both (MPI_PROC_NULL at the
     * image boundary turns the missing side into a no-op):
     *   phase 1: send first RADIUS rows up,  receive bottom ghost from below
     *   phase 2: send last  RADIUS rows down, receive top ghost from above
     *
     * Doing "talk to rank-1, then talk to rank+1" instead makes rank r wait
     * for rank r-1 to finish its own first exchange, so the exchanges form a
     * chain 0→1→…→P-1 and communication time grows linearly with P.
     */
    int up   = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    MPI_Status status;
    double comm_start = MPI_Wtime();
    MPI_Sendrecv(
        local_core,                                RADIUS * N, MPI_FLOAT, up,   TAG_UP,
        padded + (size_t)(top_gh + my_rows) * N,   RADIUS * N, MPI_FLOAT, down, TAG_UP,
        MPI_COMM_WORLD, &status);
    MPI_Sendrecv(
        local_core + (size_t)(my_rows - RADIUS) * N, RADIUS * N, MPI_FLOAT, down, TAG_DOWN,
        padded,                                      RADIUS * N, MPI_FLOAT, up,   TAG_DOWN,
        MPI_COMM_WORLD, &status);
    double comm_time_local = MPI_Wtime() - comm_start;

    /* ---- bilateral filter on local partition ---- */
    float *local_out = (float *)malloc((size_t)my_rows * N * sizeof(float));
    double comp_start = MPI_Wtime();
    bilateral_filter_local(padded, local_out, top_gh, bot_gh, my_rows, N);
    double comp_time_local = MPI_Wtime() - comp_start;

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* ---- gather results ---- */
    MPI_Gatherv(local_out, my_rows * N, MPI_FLOAT,
                full_out, scnt, sdsp, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* reduce timing: take max across all ranks (bottleneck determines wall time) */
    double max_comm, max_comp;
    MPI_Reduce(&comm_time_local, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time_local, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* detect single-node vs multi-node from environment (set by Slurm) */
        const char *nodes_env = getenv("SLURM_NNODES");
        int nodes = nodes_env ? atoi(nodes_env) : 1;
        const char *type = (nodes > 1) ? "MPI_static_multi" : "MPI_static_single";
        printf("%s,%d,%d,%d,%.4f\n", type, size, N, RADIUS, t1 - t0);
        printf("CPU computation time: %.6f seconds\n", max_comp);
        printf("Communication time: %.6f seconds\n", max_comm);
        printf("Total wall time: %.6f seconds\n", t1 - t0);

        save_jpg("out_mpi_static.jpg", full_out, N);
        printf("Output saved to out_mpi_static.jpg\n");

        char txt_path[64];
        snprintf(txt_path, sizeof(txt_path), "quality_mpi_static_P%d.txt", size);
        char label[32];
        snprintf(label, sizeof(label), "MPI_static P=%d", size);
        write_quality_report(txt_path, label, "out_mpi_static.jpg", N);

        free(full_img); free(full_out);
    }

    free(local_core); free(local_out); free(padded);
    free(scnt); free(sdsp);
    MPI_Finalize();
    return 0;
}
