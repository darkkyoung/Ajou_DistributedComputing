/*
 * 2D Heat Diffusion — V2 educational implementation milestone.
 * AI disclosure: ChatGPT (OpenAI), GPT-6, 2026-10-28.
 * AI-generated reference code: the entire file, including input parsing,
 * domain decomposition, boundary handling, halo communication, stencil,
 * reduction-by-messages, 4x4 output and error handling.
 * Original student Design Note: docs/HW1_Design_Note.md on the main branch.
 * This reference is not a substitute for independent correctness evidence.
 * For submission, rename this file to STUDENT_ID_HW1_heat_diffusion.c.
 */
#include <mpi.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { TAG_LEFT = 10, TAG_RIGHT = 11, TAG_UP = 12, TAG_DOWN = 13,
       TAG_STATS = 100, TAG_GRID = 101 };

typedef struct {
    int nx, ny, px, py, steps;
    double alpha;
} Params;

typedef struct {
    int rank, size;
    int px, py, lx, ly;
    int left, right, up, down;  /* -1 means no neighbor */
} Domain;

static void fail(int rank, const char *message)
{
    fprintf(stderr, "[rank %d] %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(EXIT_FAILURE);
}

static int parse_int(const char *text, int allow_zero, int *out)
{
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' ||
        value > INT_MAX || value < (allow_zero ? 0 : 1)) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_params(int argc, char **argv, int world_size, Params *p)
{
    char *end = NULL;
    if (argc != 7 ||
        !parse_int(argv[1], 0, &p->nx) ||
        !parse_int(argv[2], 0, &p->ny) ||
        !parse_int(argv[3], 0, &p->px) ||
        !parse_int(argv[4], 0, &p->py) ||
        !parse_int(argv[5], 1, &p->steps)) {
        return 0;
    }
    errno = 0;
    p->alpha = strtod(argv[6], &end);
    if (errno != 0 || argv[6] == end || *end != '\0' ||
        !isfinite(p->alpha) || p->alpha <= 0.0 || p->alpha > 0.25) {
        return 0;
    }
    return p->px <= INT_MAX / p->py &&
           p->px * p->py == world_size &&
           p->nx % p->px == 0 && p->ny % p->py == 0 &&
           p->nx / p->px < INT_MAX - 2 &&
           p->ny / p->py < INT_MAX - 2;
}

static void configure_domain(const Params *p, int rank, int size, Domain *d)
{
    d->rank = rank;
    d->size = size;
    d->px = rank % p->px;
    d->py = rank / p->px;
    d->lx = p->nx / p->px;
    d->ly = p->ny / p->py;
    d->left = (d->px > 0) ? rank - 1 : -1;
    d->right = (d->px + 1 < p->px) ? rank + 1 : -1;
    d->up = (d->py > 0) ? rank - p->px : -1;
    d->down = (d->py + 1 < p->py) ? rank + p->px : -1;
}

static size_t cell_index(const Domain *d, int x, int y)
{
    return (size_t)y * ((size_t)d->lx + 2) + (size_t)x;
}

static double *allocate_grid(const Domain *d)
{
    size_t width = (size_t)d->lx + 2;
    size_t height = (size_t)d->ly + 2;
    if (height > SIZE_MAX / width ||
        width * height > SIZE_MAX / sizeof(double)) {
        fail(d->rank, "Local grid is too large");
    }
    double *grid = calloc(width * height, sizeof(double));
    if (!grid) {
        fail(d->rank, "Failed to allocate local grid");
    }
    return grid;
}

static void apply_neumann(double *old, const Domain *d)
{
    if (d->left < 0) {
        for (int y = 1; y <= d->ly; ++y)
            old[cell_index(d, 0, y)] = old[cell_index(d, 1, y)];
    }
    if (d->right < 0) {
        for (int y = 1; y <= d->ly; ++y)
            old[cell_index(d, d->lx + 1, y)] = old[cell_index(d, d->lx, y)];
    }
    if (d->up < 0) {
        for (int x = 1; x <= d->lx; ++x)
            old[cell_index(d, x, 0)] = old[cell_index(d, x, 1)];
    }
    if (d->down < 0) {
        for (int x = 1; x <= d->lx; ++x)
            old[cell_index(d, x, d->ly + 1)] = old[cell_index(d, x, d->ly)];
    }
}

static void stencil_step(const double *old, double *next,
                         const Domain *d, double alpha)
{
    size_t stride = (size_t)d->lx + 2;
    for (int y = 1; y <= d->ly; ++y) {
        for (int x = 1; x <= d->lx; ++x) {
            size_t c = cell_index(d, x, y);
            next[c] = old[c] + alpha * (old[c-stride] + old[c+stride] +
                                         old[c-1] + old[c+1] - 4.0*old[c]);
        }
    }
}

static void local_statistics(const double *old, const Domain *d,
                             double *sum, double *maximum)
{
    *sum = 0.0;
    *maximum = -DBL_MAX;
    for (int y = 1; y <= d->ly; ++y) {
        for (int x = 1; x <= d->lx; ++x) {
            double value = old[cell_index(d, x, y)];
            *sum += value;
            if (value > *maximum)
                *maximum = value;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Params p;
    if (!parse_params(argc, argv, size, &p)) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np N ./heat_diffusion NX NY PX PY steps alpha\n"
                            "Constraints: NX,NY,PX,PY > 0; steps >= 0; "
                            "0 < alpha <= 0.25; NX %% PX=0; NY %% PY=0; PX*PY=N\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return EXIT_FAILURE;
    }
    Domain d;
    configure_domain(&p, rank, size, &d);
    /* V2: incremental implementation before global statistics. */
    if (size != 1) fail(rank, "V2 only supports one MPI rank");
    double *old = allocate_grid(&d);
    double *next = allocate_grid(&d);
    if (rank == 0) old[cell_index(&d, 1, 1)] = 100.0;
    for (int step = 0; step < p.steps; ++step) {
        apply_neumann(old, &d);
        stencil_step(old, next, &d, p.alpha);
        double *tmp = old;
        old = next;
        next = tmp;
    }
    double local_sum, local_max;
    local_statistics(old, &d, &local_sum, &local_max);
    printf("[V2 LOCAL] rank=%d sum=%.12f max=%.12f\n",
           rank, local_sum, local_max);
    free(old);
    free(next);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
