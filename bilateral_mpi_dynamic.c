/*
 * bilateral_mpi_dynamic.c — MPI Dynamic Master-Worker version
 * Compile: mpiicc -O2 -xHost -o bilateral_mpi_dynamic bilateral_mpi_dynamic.c -lm -ljpeg
 * Run:     mpirun -n 8 ./bilateral_mpi_dynamic    (requires >= 2 processes)
 *
 * Strategy:
 *   Rank 0 (Master): owns the full image and output buffer.
 *     Dispatches CHUNK_SIZE-row work units to idle workers.
 *     Each unit carries the core rows plus RADIUS ghost rows (above/below)
 *     sourced directly from the full image.
 *   Rank 1..P-1 (Workers): receive padded chunks, run the bilateral filter,
 *     return core results, then ask for the next chunk.
 *
 * Advantage vs static: naturally load-balanced; workers never idle waiting
 *   for a slow peer to finish a large partition.
 * Trade-off: O(N / CHUNK_SIZE) round-trip messages instead of one Scatter.
 *
 * Output (CSV): MPI_dynamic_<single|multi>,<procs>,<N>,<radius>,<time_s>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <jpeglib.h>

#define N          4096
#define RADIUS     5
#define SIGMA_D    3.0f
#define SIGMA_R    20.0f
#define CHUNK_SIZE 32    /* rows per work unit — smaller = finer granularity */

/* Message tags */
#define TAG_WORK   10   /* master → worker: work header + data   */
#define TAG_RESULT 20   /* worker → master: result header + data */

/* Work assignment header (4 ints sent before the data payload) */
typedef struct {
    int start;   /* first core row in the full image; -1 = terminate */
    int rows;    /* number of core rows in this chunk                 */
    int top_pad; /* ghost rows prepended above the core data          */
    int bot_pad; /* ghost rows appended  below the core data          */
} WorkHdr;

/* Result header sent before the result payload */
typedef struct {
    int    start;     /* first core row (so master knows where to store)   */
    int    rows;      /* number of rows in the result payload               */
    double comp_time; /* time spent in bilateral_filter_local on this chunk */
} ResHdr;

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

/* ── Bilateral filter (identical core to static version) ────────────────
 * src layout: [ top_pad | core rows | bot_pad ] × N
 * dst layout: [ core rows ]                      × N
 */
static void bilateral_filter_local(
    const float *src, float *dst,
    int top_pad, int bot_pad, int core_rows, int cols)
{
    const float inv2sd2 = 1.0f / (2.0f * SIGMA_D * SIGMA_D);
    const float inv2sr2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);
    const int   wsize   = 2 * RADIUS + 1;
    const int   tot     = top_pad + core_rows + bot_pad;

    float slut[wsize * wsize];
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
            slut[(dy + RADIUS) * wsize + (dx + RADIUS)] =
                expf(-(float)(dx * dx + dy * dy) * inv2sd2);

    for (int ly = top_pad; ly < top_pad + core_rows; ly++) {
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
            dst[(size_t)(ly - top_pad) * cols + x] = sum_v / sum_w;
        }
    }
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

/* ── Helper: build and dispatch one chunk to a worker ───────────────── */
static void dispatch_chunk(float *full_img, int *next_row_ptr,
                            int worker, MPI_Comm comm)
{
    int rows    = (*next_row_ptr + CHUNK_SIZE <= N)
                  ? CHUNK_SIZE : N - *next_row_ptr;

    /* Ghost rows: clamp at image boundary */
    int top_pad = (*next_row_ptr >= RADIUS)
                  ? RADIUS : *next_row_ptr;
    int remain  = N - (*next_row_ptr + rows);
    int bot_pad = (remain >= RADIUS) ? RADIUS : remain;

    int send_start = *next_row_ptr - top_pad;
    int send_rows  = top_pad + rows + bot_pad;

    WorkHdr hdr = {*next_row_ptr, rows, top_pad, bot_pad};
    MPI_Send(&hdr, (int)sizeof(WorkHdr), MPI_BYTE, worker, TAG_WORK, comm);
    MPI_Send(full_img + (size_t)send_start * N,
             send_rows * N, MPI_FLOAT, worker, TAG_WORK, comm);

    *next_row_ptr += rows;
}

/* ── Master ──────────────────────────────────────────────────────────── */
static int run_master(int size)
{
    float *full_img = (float *)malloc((size_t)N * N * sizeof(float));
    float *full_out = (float *)malloc((size_t)N * N * sizeof(float));
    if (!full_img || !full_out) {
        fputs("malloc failed\n", stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (load_jpg("dog_gray.jpg", full_img, N) != 0)
        MPI_Abort(MPI_COMM_WORLD, 1);

    int next_row = 0;   /* next row to dispatch */
    int pending  = 0;   /* workers currently processing a chunk */
    double *worker_comp = (double *)calloc((size_t)size, sizeof(double));

    double t0 = MPI_Wtime();

    /* ── Initial dispatch ────────────────────────────────────────────
     * Send one chunk to each worker. Workers with no initial work get
     * an immediate termination signal.                                */
    for (int w = 1; w < size; w++) {
        if (next_row < N) {
            dispatch_chunk(full_img, &next_row, w, MPI_COMM_WORLD);
            /* Note: dispatch_chunk advances next_row internally */
            pending++;
        } else {
            WorkHdr term = {-1, 0, 0, 0};
            MPI_Send(&term, (int)sizeof(WorkHdr), MPI_BYTE, w, TAG_WORK,
                     MPI_COMM_WORLD);
        }
    }

    /* ── Dynamic dispatch loop ───────────────────────────────────────
     * Each time a worker finishes, either assign the next chunk or
     * terminate it.                                                    */
    MPI_Status status;
    while (pending > 0) {
        /* Receive result header */
        ResHdr res;
        MPI_Recv(&res, (int)sizeof(ResHdr), MPI_BYTE,
                 MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
        int worker = status.MPI_SOURCE;

        /* Receive result data */
        MPI_Recv(full_out + (size_t)res.start * N,
                 res.rows * N, MPI_FLOAT,
                 worker, TAG_RESULT, MPI_COMM_WORLD, &status);
        pending--;
        worker_comp[worker] += res.comp_time;

        if (next_row < N) {
            dispatch_chunk(full_img, &next_row, worker, MPI_COMM_WORLD);
            pending++;
        } else {
            WorkHdr term = {-1, 0, 0, 0};
            MPI_Send(&term, (int)sizeof(WorkHdr), MPI_BYTE, worker,
                     TAG_WORK, MPI_COMM_WORLD);
        }
    }

    double t1 = MPI_Wtime();

    const char *nodes_env = getenv("SLURM_NNODES");
    int nodes = nodes_env ? atoi(nodes_env) : 1;
    const char *type = (nodes > 1) ? "MPI_dynamic_multi" : "MPI_dynamic_single";
    printf("%s,%d,%d,%d,%.4f\n", type, size, N, RADIUS, t1 - t0);

    double max_comp = 0.0;
    for (int w = 1; w < size; w++)
        if (worker_comp[w] > max_comp) max_comp = worker_comp[w];
    double comm_time = t1 - t0 - max_comp;
    if (comm_time < 0.0) comm_time = 0.0;
    printf("CPU computation time: %.6f seconds\n", max_comp);
    printf("Communication time: %.6f seconds\n", comm_time);
    printf("Total wall time: %.6f seconds\n", t1 - t0);

    save_jpg("out_mpi_dynamic.jpg", full_out, N);
    printf("Output saved to out_mpi_dynamic.jpg\n");

    char txt_path[64];
    snprintf(txt_path, sizeof(txt_path), "quality_mpi_dynamic_P%d.txt", size);
    char label[32];
    snprintf(label, sizeof(label), "MPI_dynamic P=%d", size);
    write_quality_report(txt_path, label, "out_mpi_dynamic.jpg", N);

    free(worker_comp);
    free(full_img);
    free(full_out);
    return 0;
}

/* ── Worker ──────────────────────────────────────────────────────────── */
static int run_worker(void)
{
    MPI_Status status;

    while (1) {
        /* Receive work header */
        WorkHdr hdr;
        MPI_Recv(&hdr, (int)sizeof(WorkHdr), MPI_BYTE,
                 0, TAG_WORK, MPI_COMM_WORLD, &status);

        if (hdr.start < 0) break;   /* termination signal */

        int buf_rows = hdr.top_pad + hdr.rows + hdr.bot_pad;
        float *buf = (float *)malloc((size_t)buf_rows * N * sizeof(float));
        float *out = (float *)malloc((size_t)hdr.rows  * N * sizeof(float));
        if (!buf || !out) { fputs("malloc failed\n", stderr); MPI_Abort(MPI_COMM_WORLD, 1); }

        /* Receive padded chunk data */
        MPI_Recv(buf, buf_rows * N, MPI_FLOAT,
                 0, TAG_WORK, MPI_COMM_WORLD, &status);

        /* Filter */
        double comp_start = MPI_Wtime();
        bilateral_filter_local(buf, out, hdr.top_pad, hdr.bot_pad, hdr.rows, N);
        double comp_time = MPI_Wtime() - comp_start;

        /* Return result */
        ResHdr res = {hdr.start, hdr.rows, comp_time};
        MPI_Send(&res, (int)sizeof(ResHdr), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Send(out, hdr.rows * N, MPI_FLOAT, 0, TAG_RESULT, MPI_COMM_WORLD);

        free(buf);
        free(out);
    }
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr,
                    "Dynamic version requires >= 2 processes "
                    "(1 master + at least 1 worker).\n");
        MPI_Finalize();
        return 1;
    }

    int rc = (rank == 0) ? run_master(size) : run_worker();

    MPI_Finalize();
    return rc;
}
