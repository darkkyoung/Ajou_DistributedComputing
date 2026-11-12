/*
 * HW2 Parallel PageRank — V2 (Ring, Star, PRCSR1 file)
 * AI disclosure: ChatGPT (GPT-6) drafted V1 (2026-11-11) and added
 * PRCSR1 input, local CSR loading, and validation (2026-11-13).
 * Student validation on actual AWS uk-2007-05 is still required.
 * Code generated/significantly revised by AI: this entire file.
 *
 * File format: little-endian PRCSR1 header(64), int64 row_ptr[N+1],
 * int32 outdeg[N], int32 incoming col[M].
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <mpi.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

_Static_assert(sizeof(int) == 4, "This implementation requires 32-bit int");
_Static_assert(sizeof(int32_t) == 4 && sizeof(int64_t) == 8, "Unexpected integer sizes");
_Static_assert(sizeof(double) == 8, "PRCSR1 dump requires 64-bit double");

enum { GRAPH_RING = 1, GRAPH_STAR = 2, GRAPH_FILE = 3 };

static int parse_int(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    long result = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        result < INT_MIN || result > INT_MAX) return 0;
    *value = (int)result;
    return 1;
}

static int parse_double(const char *text, double *value)
{
    char *end = NULL;
    errno = 0;
    double result = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(result)) return 0;
    *value = result;
    return 1;
}

/* Rank k owns [start, start + count). Zero-count blocks are supported. */
static void block_for_rank(int n, int p, int k, int *start, int *count)
{
    int q = n / p;
    int rem = n % p;
    *count = q + (k < rem ? 1 : 0);
    *start = k * q + (k < rem ? k : rem);
}

static void *checked_calloc(size_t count, size_t size, MPI_Comm comm)
{
    if (size != 0 && count > SIZE_MAX / size) {
        fprintf(stderr, "Memory allocation size overflow\n");
        MPI_Abort(comm, 3);
    }
    void *ptr = calloc(count ? count : 1, size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed (count=%zu, size=%zu)\n", count, size);
        MPI_Abort(comm, 3);
    }
    return ptr;
}


/* Read in bounded chunks, using 64-bit offsets for the >16 GB graph file. */
static int read_at(FILE *fp, int64_t offset, void *dst, size_t bytes)
{
    if (offset < 0 || fseeko(fp, (off_t)offset, SEEK_SET) != 0) return 0;
    unsigned char *p = dst;
    while (bytes > 0) {
        size_t chunk = bytes < 8U * 1024U * 1024U ? bytes : 8U * 1024U * 1024U;
        if (fread(p, 1, chunk, fp) != chunk) return 0;
        p += chunk;
        bytes -= chunk;
    }
    return 1;
}

/* Validate on-disk header and exact size before any multi-rank allocation. */
static int read_prcsr_header(const char *path, int64_t h[3], int rank)
{
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "[Rank %d] Cannot stat PRCSR1 file: %s\n", rank, path);
        return 0;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Rank %d] Cannot open PRCSR1 file: %s\n", rank, path);
        return 0;
    }
    unsigned char bytes[64];
    int ok = read_at(fp, 0, bytes, sizeof(bytes));
    fclose(fp);
    if (!ok || memcmp(bytes, "PRCSR1\0\0", 8) != 0) {
        fprintf(stderr, "[Rank %d] Invalid or truncated PRCSR1 header: %s\n", rank, path);
        return 0;
    }

    int64_t n = 0, m = 0, d = 0;
    int32_t version = 0;
    memcpy(&n, bytes + 8, sizeof(n));
    memcpy(&m, bytes + 16, sizeof(m));
    memcpy(&d, bytes + 24, sizeof(d));
    memcpy(&version, bytes + 32, sizeof(version));
    int padding_ok = 1;
    for (int i = 36; i < 64; ++i)
        if (bytes[i] != 0) padding_ok = 0;

    /* INT_MAX bound is needed for MPI_Allgatherv count/displacement API. */
    if (version != 1 || n <= 0 || n > INT_MAX || m < 0 || d < 0 || d > n ||
        !padding_ok) {
        fprintf(stderr, "[Rank %d] Invalid PRCSR1 dimensions/version/padding\n", rank);
        return 0;
    }

    uint64_t prefix = 64ULL + 8ULL * ((uint64_t)n + 1) + 4ULL * (uint64_t)n;
    if ((uint64_t)m > ((uint64_t)INT64_MAX - prefix) / 4ULL) {
        fprintf(stderr, "[Rank %d] PRCSR1 file size overflow\n", rank);
        return 0;
    }
    uint64_t expected = prefix + 4ULL * (uint64_t)m;
    if (st.st_size < 0 || (uint64_t)st.st_size != expected) {
        fprintf(stderr, "[Rank %d] PRCSR1 size mismatch: expected %" PRIu64
                ", found %" PRId64 " bytes\n", rank, expected, (int64_t)st.st_size);
        return 0;
    }
    h[0] = n;
    h[1] = m;
    h[2] = d;
    return 1;
}

/*
 * PRCSR1 layout: header(64), row_ptr(8*(N+1)), outdeg(4*N), col(4*M).
 * Each rank reads only row_ptr[start..start+local_n] and its local col slice.
 * row_ptr is rebased to the local col buffer; outdeg is global.
 * All checks occur before the timed iteration loop.
 */
static int read_file_block(const char *path, const int64_t header[3],
                           int start, int local_n, int *outdeg,
                           int64_t **row_ptr_out, int32_t **col_out,
                           int64_t *local_edges_out, int rank, MPI_Comm comm)
{
    int64_t local_header[3] = {0};
    if (!read_prcsr_header(path, local_header, rank)) return 0;
    for (int i = 0; i < 3; ++i) {
        if (local_header[i] != header[i]) {
            fprintf(stderr, "[Rank %d] Local graph header disagrees with rank 0\n", rank);
            return 0;
        }
    }

    int64_t n = header[0], m = header[1], d = header[2];
    const int64_t row_offset = 64;
    const int64_t outdeg_offset = row_offset + 8 * (n + 1);
    const int64_t col_offset = outdeg_offset + 4 * n;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Rank %d] Cannot reopen PRCSR1 file\n", rank);
        return 0;
    }
    int64_t *rp = checked_calloc((size_t)local_n + 1, sizeof(*rp), comm);
    *row_ptr_out = rp;
    if (!read_at(fp, row_offset + 8 * (int64_t)start, rp,
                 ((size_t)local_n + 1) * sizeof(*rp)) ||
        !read_at(fp, outdeg_offset, outdeg, (size_t)n * sizeof(*outdeg))) {
        fprintf(stderr, "[Rank %d] Failed reading PRCSR1 row_ptr/outdeg\n", rank);
        fclose(fp);
        return 0;
    }

    if (rp[0] < 0 || rp[local_n] < rp[0] || rp[local_n] > m ||
        (start == 0 && rp[0] != 0) ||
        (start + local_n == n && rp[local_n] != m)) {
        fprintf(stderr, "[Rank %d] Invalid CSR block boundaries\n", rank);
        fclose(fp);
        return 0;
    }
    for (int i = 0; i < local_n; ++i) {
        if (rp[i] < 0 || rp[i] > rp[i + 1] || rp[i + 1] > m) {
            fprintf(stderr, "[Rank %d] Nonmonotonic CSR row_ptr\n", rank);
            fclose(fp);
            return 0;
        }
    }

    int64_t sum_outdeg = 0, dangling = 0;
    for (int64_t v = 0; v < n; ++v) {
        if (outdeg[v] < 0) {
            fprintf(stderr, "[Rank %d] Negative outdegree\n", rank);
            fclose(fp);
            return 0;
        }
        sum_outdeg += outdeg[v];
        if (outdeg[v] == 0) ++dangling;
    }
    if (sum_outdeg != m || dangling != d) {
        fprintf(stderr, "[Rank %d] Outdegree sum/dangling count mismatch\n", rank);
        fclose(fp);
        return 0;
    }

    int64_t edge_start = rp[0];
    int64_t local_edges = rp[local_n] - edge_start;
    if ((uint64_t)local_edges > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "[Rank %d] CSR edge allocation too large\n", rank);
        fclose(fp);
        return 0;
    }
    int32_t *col = checked_calloc((size_t)local_edges, sizeof(*col), comm);
    *col_out = col;
    if (!read_at(fp, col_offset + 4 * edge_start, col,
                 (size_t)local_edges * sizeof(*col))) {
        fprintf(stderr, "[Rank %d] Failed reading local in-edge col slice\n", rank);
        fclose(fp);
        return 0;
    }
    fclose(fp);

    for (int64_t e = 0; e < local_edges; ++e) {
        if (col[e] < 0 || col[e] >= n || outdeg[col[e]] <= 0) {
            fprintf(stderr, "[Rank %d] Invalid in-neighbor/outdegree at edge %" PRId64 "\n",
                    rank, e + edge_start);
            return 0;
        }
    }
    for (int i = 0; i <= local_n; ++i) rp[i] -= edge_start;
    *local_edges_out = local_edges;
    return 1;
}

/* Build only this rank's incoming-edge CSR block (not the full graph). */
static void build_local_graph(int graph, int n, int start, int local_n,
                              int64_t **row_ptr, int32_t **col,
                              int64_t *local_edges, MPI_Comm comm)
{
    int64_t *rp = checked_calloc((size_t)local_n + 1, sizeof(*rp), comm);
    for (int i = 0; i < local_n; ++i) {
        int v = start + i;
        rp[i + 1] = rp[i] + (graph == GRAPH_STAR && v == 0 ? n - 1 : 1);
    }

    int64_t edges = rp[local_n];
    if ((uint64_t)edges > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "Edge buffer too large\n");
        MPI_Abort(comm, 3);
    }
    int32_t *cols = checked_calloc((size_t)edges, sizeof(*cols), comm);
    for (int i = 0; i < local_n; ++i) {
        int v = start + i;
        int64_t base = rp[i];
        if (graph == GRAPH_RING) {
            cols[base] = (v == 0 ? n - 1 : v - 1);
        } else if (v == 0) {
            for (int u = 1; u < n; ++u) cols[base + u - 1] = u;
        } else {
            cols[base] = 0;
        }
    }
    *row_ptr = rp;
    *col = cols;
    *local_edges = edges;
}

static int count_nodes(MPI_Comm comm, int procs)
{
    char mine[MPI_MAX_PROCESSOR_NAME] = {0};
    int name_len = 0;
    MPI_Get_processor_name(mine, &name_len);
    char *names = checked_calloc((size_t)procs, MPI_MAX_PROCESSOR_NAME, comm);
    MPI_Allgather(mine, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    int unique = 0;
    for (int i = 0; i < procs; ++i) {
        const char *current = names + (size_t)i * MPI_MAX_PROCESSOR_NAME;
        int found = 0;
        for (int j = 0; j < i; ++j) {
            if (strcmp(current, names + (size_t)j * MPI_MAX_PROCESSOR_NAME) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) ++unique;
    }
    free(names);
    return unique;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    /* Rank 0 parses settings and file header. All ranks receive identical values. */
    int valid = 1;
    int settings[5] = {0}; /* graph, N, max_iters, top_k, benchmark_iters */
    int64_t header[3] = {0}; /* N, M, D */
    double parameters[2] = {0}; /* damping, eps */
    const char *dump_path = NULL;
    if (rank == 0) {
        if (argc < 6 || argc > 9) {
            valid = 0;
        } else {
            if (strcmp(argv[1], "ring") == 0) settings[0] = GRAPH_RING;
            else if (strcmp(argv[1], "star") == 0) settings[0] = GRAPH_STAR;
            else if (strcmp(argv[1], "file") == 0) settings[0] = GRAPH_FILE;
            else valid = 0;

            if (settings[0] != GRAPH_FILE) {
                valid &= parse_int(argv[2], &settings[1]);
                valid &= settings[1] >= (settings[0] == GRAPH_RING ? 2 : 3);
                if (valid) {
                    header[0] = settings[1];
                    header[1] = settings[0] == GRAPH_RING ? settings[1]
                                                         : 2LL * (settings[1] - 1);
                    header[2] = 0;
                }
            } else {
                /* The input file is present on every EC2 node's local EBS. */
                valid &= (argv[2][0] != '\0');
                if (valid) valid &= read_prcsr_header(argv[2], header, rank);
                if (valid) settings[1] = (int)header[0];
            }
            valid &= parse_double(argv[3], &parameters[0]);
            valid &= parse_double(argv[4], &parameters[1]);
            valid &= parse_int(argv[5], &settings[2]);
            settings[3] = 10;
            if (argc >= 7) valid &= parse_int(argv[6], &settings[3]);
            if (argc >= 8) valid &= parse_int(argv[7], &settings[4]);
            if (argc == 9) dump_path = argv[8];
            valid &= settings[2] >= 1 && settings[3] >= 1 && settings[4] >= 0;
            valid &= parameters[0] > 0.0 && parameters[0] < 1.0;
            valid &= parameters[1] > 0.0;
            if (dump_path && settings[0] == GRAPH_FILE &&
                strcmp(dump_path, argv[2]) == 0) {
                fprintf(stderr, "Dump path must differ from input graph path.\n");
                valid = 0;
            }

            const uint16_t endian_check = 1;
            if (*(const unsigned char *)&endian_check != 1) {
                fprintf(stderr, "PRCSR1 requires a little-endian host.\n");
                valid = 0;
            }
        }
        if (!valid) {
            fprintf(stderr, "Invalid input, file, or unsupported PRCSR1 header.\n");
            fprintf(stderr, "Usage: pagerank {ring|star|file} {N|PRCSR1_path} "
                            "damp eps max_iters [top_k] [benchmark_iters] [dump_path]\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid) {
        MPI_Finalize();
        return 2;
    }
    MPI_Bcast(settings, 5, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(parameters, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(header, 3, MPI_INT64_T, 0, MPI_COMM_WORLD);

    int graph = settings[0], n = settings[1], max_iters = settings[2];
    int top_k = settings[3], benchmark_iters = settings[4];
    double damp = parameters[0], eps = parameters[1];
    if (top_k > n) top_k = n;

        double load_start = MPI_Wtime();
    int *recvcounts = checked_calloc((size_t)procs, sizeof(int), MPI_COMM_WORLD);
    int *displs = checked_calloc((size_t)procs, sizeof(int), MPI_COMM_WORLD);
    for (int k = 0; k < procs; ++k)
        block_for_rank(n, procs, k, &displs[k], &recvcounts[k]);
    int start = displs[rank], local_n = recvcounts[rank];

    int *outdeg = checked_calloc((size_t)n, sizeof(int), MPI_COMM_WORLD);
    int64_t *row_ptr = NULL, local_edges = 0;
    int32_t *col = NULL;
    if (graph == GRAPH_FILE) {
        /* A missing or invalid file on any node must stop every rank safely. */
        int local_ok = read_file_block(argv[2], header, start, local_n, outdeg,
                                       &row_ptr, &col, &local_edges, rank,
                                       MPI_COMM_WORLD);
        int all_ok = 0;
        MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!all_ok) {
            if (rank == 0) fprintf(stderr, "PRCSR1 load failed on at least one rank.\n");
            free(recvcounts);
            free(displs);
            free(outdeg);
            free(row_ptr);
            free(col);
            MPI_Finalize();
            return 3;
        }
    } else {
        for (int v = 0; v < n; ++v)
            outdeg[v] = (graph == GRAPH_STAR && v == 0 ? n - 1 : 1);
        build_local_graph(graph, n, start, local_n,
                          &row_ptr, &col, &local_edges, MPI_COMM_WORLD);
    }

    double *old = checked_calloc((size_t)n, sizeof(double), MPI_COMM_WORLD);
    double *next_full = checked_calloc((size_t)n, sizeof(double), MPI_COMM_WORLD);
    double *next_local = checked_calloc((size_t)local_n, sizeof(double), MPI_COMM_WORLD);
    for (int v = 0; v < n; ++v) old[v] = 1.0 / n;
    double load_local = MPI_Wtime() - load_start;

    double dangling_mass = (double)header[2] / n;
    double residual = 0.0;
    double compute_local = 0.0, reduce_local = 0.0, gather_local = 0.0;
    int iters = 0;
    int limit = benchmark_iters > 0 ? benchmark_iters : max_iters;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();
    for (int iter = 0; iter < limit; ++iter) {
        double t = MPI_Wtime();
        double local_stats[2] = {0.0, 0.0};
        for (int i = 0; i < local_n; ++i) {
            double incoming = 0.0;
            for (int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                int32_t u = col[j];
                incoming += old[u] / outdeg[u];
            }
            int v = start + i;
            double value = (1.0 - damp) / n + damp * (incoming + dangling_mass / n);
            next_local[i] = value;
            local_stats[0] += fabs(value - old[v]);
            if (outdeg[v] == 0) local_stats[1] += value;
        }
        compute_local += MPI_Wtime() - t;

        double global_stats[2] = {0.0, 0.0};
        t = MPI_Wtime();
        MPI_Allreduce(local_stats, global_stats, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        reduce_local += MPI_Wtime() - t;

        t = MPI_Wtime();
        MPI_Allgatherv(next_local, local_n, MPI_DOUBLE,
                       next_full, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        gather_local += MPI_Wtime() - t;
        double *temp = old;
        old = next_full;
        next_full = temp;
        residual = global_stats[0];
        dangling_mass = global_stats[1];
        ++iters;
        if (benchmark_iters == 0 && residual < eps) break;
    }
    double total_local = MPI_Wtime() - total_start;

    /* Measurements/printing/dump happen outside the timed iteration loop. */
    double local_times[5] = {total_local, compute_local,
                             reduce_local + gather_local, gather_local, reduce_local};
    double max_times[5] = {0};
    MPI_Allreduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double compute_min = 0.0, load_max = 0.0;
    MPI_Allreduce(&compute_local, &compute_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&load_local, &load_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t edges_min = 0, edges_max = 0;
    MPI_Allreduce(&local_edges, &edges_min, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_edges, &edges_max, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    int nodes = count_nodes(MPI_COMM_WORLD, procs);

    int write_ok = 1;
    if (rank == 0) {
        double sum = 0.0, min_rank = old[0], max_rank = old[0];
        for (int v = 0; v < n; ++v) {
            sum += old[v];
            if (old[v] < min_rank) min_rank = old[v];
            if (old[v] > max_rank) max_rank = old[v];
        }
        printf("[Result] Graph=%s N=%d M=%" PRId64 " Procs=%d Nodes=%d Iters=%d "
               "Residual=%.12g Damp=%.6g Converged=%d Mode=%s Dangling=%.12g "
               "Total=%.6f Compute=%.6f Comm=%.6f Allgatherv=%.6f Allreduce=%.6f Load=%.6f\n",
               graph == GRAPH_RING ? "ring" : (graph == GRAPH_STAR ? "star" : "file"),
               n, header[1], procs, nodes, iters,
               residual, damp, residual < eps, benchmark_iters > 0 ? "benchmark" : "normal",
               dangling_mass, max_times[0], max_times[1], max_times[2],
               max_times[3], max_times[4], load_max);
        printf("[Stats] Sum=%.15g SumError=%.12g Min=%.15g Max=%.15g\n",
               sum, fabs(sum - 1.0), min_rank, max_rank);
        printf("[Balance] InEdgesMin=%" PRId64 " InEdgesMax=%" PRId64
               " ComputeMin=%.6f ComputeMax=%.6f\n",
               edges_min, edges_max, compute_min, max_times[1]);
        unsigned char *picked = checked_calloc((size_t)n, 1, MPI_COMM_WORLD);
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int v = 0; v < n; ++v) {
                if (picked[v]) continue;
                if (best < 0 || old[v] > old[best] ||
                    (old[v] == old[best] && v < best)) best = v;
            }
            if (best < 0) break;
            picked[best] = 1;
            printf("[Top] Rank=%d Node=%d Score=%.15g\n", k + 1, best, old[best]);
        }
        free(picked);
        if (dump_path) {
            FILE *fp = fopen(dump_path, "wb");
            if (!fp || fwrite(old, sizeof(double), (size_t)n, fp) != (size_t)n) {
                fprintf(stderr, "Could not write vector dump: %s\n", dump_path);
                write_ok = 0;
            }
            if (fp && fclose(fp) != 0) write_ok = 0;
        }
    }
    MPI_Bcast(&write_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(recvcounts);
    free(displs);
    free(outdeg);
    free(row_ptr);
    free(col);
    free(old);
    free(next_full);
    free(next_local);
    MPI_Finalize();
    return write_ok ? 0 : 4;
}
