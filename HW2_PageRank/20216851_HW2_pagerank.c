/*
 * HW2 Parallel PageRank (V1: synthetic Ring / Star graphs)
 * AI disclosure: This V1 source was drafted with ChatGPT (GPT-6) on
 * 2026-11-11. The student will validate and revise it through MPI tests.
 *
 * V1 scope: synthetic graph construction, contiguous block decomposition,
 * MPI_Bcast / MPI_Allreduce / MPI_Allgatherv, convergence, and measurements.
 * TODO V2: PRCSR1 file loading and full uk-2007-05 correctness testing.
 */
#include <mpi.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GRAPH_RING = 1, GRAPH_STAR = 2 };

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
    void *ptr = calloc(count ? count : 1, size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed (count=%zu, size=%zu)\n", count, size);
        MPI_Abort(comm, 3);
    }
    return ptr;
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

    /* Root parses and validates settings, then broadcasts them. */
    int valid = 1;
    int settings[5] = {0}; /* graph, N, max_iters, top_k, benchmark_iters */
    double parameters[2] = {0}; /* damping, eps */
    const char *dump_path = NULL;
    if (rank == 0) {
        if (argc < 6 || argc > 9) {
            valid = 0;
        } else {
            if (strcmp(argv[1], "ring") == 0) settings[0] = GRAPH_RING;
            else if (strcmp(argv[1], "star") == 0) settings[0] = GRAPH_STAR;
            else valid = 0; /* file mode to be implemented in V2 */

            valid &= parse_int(argv[2], &settings[1]);
            valid &= parse_double(argv[3], &parameters[0]);
            valid &= parse_double(argv[4], &parameters[1]);
            valid &= parse_int(argv[5], &settings[2]);
            settings[3] = 10;
            if (argc >= 7) valid &= parse_int(argv[6], &settings[3]);
            if (argc >= 8) valid &= parse_int(argv[7], &settings[4]);
            if (argc == 9) dump_path = argv[8];

            valid &= (settings[1] >= (settings[0] == GRAPH_RING ? 2 : 3));
            valid &= (settings[2] >= 1 && settings[3] >= 1 && settings[4] >= 0);
            valid &= (parameters[0] > 0.0 && parameters[0] < 1.0);
            valid &= (parameters[1] > 0.0);
        }
        if (!valid) {
            fprintf(stderr, "Invalid input or unsupported graph type (V1: ring/star only).\n");
            fprintf(stderr, "Usage: pagerank {ring|star} N damp eps max_iters [top_k] [benchmark_iters] [dump_path]\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid) {
        MPI_Finalize();
        return 2;
    }
    MPI_Bcast(settings, 5, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(parameters, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int graph = settings[0], n = settings[1], max_iters = settings[2];
    int top_k = settings[3], benchmark_iters = settings[4];
    double damp = parameters[0], eps = parameters[1];
    if (top_k > n) top_k = n;

    int64_t header[3] = {n, graph == GRAPH_RING ? n : 2LL * (n - 1), 0};
    MPI_Bcast(header, 3, MPI_INT64_T, 0, MPI_COMM_WORLD);

    double load_start = MPI_Wtime();
    int *recvcounts = checked_calloc((size_t)procs, sizeof(int), MPI_COMM_WORLD);
    int *displs = checked_calloc((size_t)procs, sizeof(int), MPI_COMM_WORLD);
    for (int k = 0; k < procs; ++k)
        block_for_rank(n, procs, k, &displs[k], &recvcounts[k]);
    int start = displs[rank], local_n = recvcounts[rank];

    int *outdeg = checked_calloc((size_t)n, sizeof(int), MPI_COMM_WORLD);
    for (int v = 0; v < n; ++v)
        outdeg[v] = (graph == GRAPH_STAR && v == 0 ? n - 1 : 1);

    int64_t *row_ptr = NULL, local_edges = 0;
    int32_t *col = NULL;
    build_local_graph(graph, n, start, local_n,
                      &row_ptr, &col, &local_edges, MPI_COMM_WORLD);

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
               graph == GRAPH_RING ? "ring" : "star", n, header[1], procs, nodes, iters,
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
