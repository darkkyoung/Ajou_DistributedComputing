
/*
 * HW1 - 2D Heat Diffusion
 * V0: MPI Initialization and Input Validation
 *
 * AI assistance: ChatGPT (OpenAI), GPT-6
 * Date: 2026-10-28
 * AI-generated: V0 program structure, input parsing,
 * MPI initialization, and rank mapping.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

/* Parse an integer with a specified minimum value. */
int parse_int(const char *text, int min_value, int *result)
{
    char *end;
    errno = 0;

    long value = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' ||
        value < min_value || value > INT_MAX) {
        return 0;
    }

    *result = (int)value;
    return 1;
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int NX, NY, PX, PY, steps;
    double alpha;

    int valid = (argc == 7);

    if (valid) {
        valid =
            parse_int(argv[1], 1, &NX) &&
            parse_int(argv[2], 1, &NY) &&
            parse_int(argv[3], 1, &PX) &&
            parse_int(argv[4], 1, &PY) &&
            parse_int(argv[5], 0, &steps);

        if (valid) {
            char *end;
            errno = 0;
            alpha = strtod(argv[6], &end);

            valid = (errno == 0 && end != argv[6] &&
                     *end == '\0' && isfinite(alpha) &&
                     alpha > 0.0 && alpha <= 0.25);
        }
    }

    if (valid) {
        valid = ((long long)PX * PY == size &&
                 NX % PX == 0 &&
                 NY % PY == 0);
    }

    if (!valid) {
        if (rank == 0) {
            fprintf(stderr,
                "Invalid input!\n"
                "Usage: NX NY PX PY steps alpha\n"
                "PX*PY must equal the MPI process count.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }


    int px = rank % PX;
    int py = rank / PX;


    int localNX = NX / PX;
    int localNY = NY / PY;

    
    /*
     * V1: Allocate local grids and initialize temperature.
     * AI-generated: ChatGPT GPT-6, 2026-10-28.
     */

    /* V1 has no timestep update yet. */
    if (steps != 0) {
        if (rank == 0)
            fprintf(stderr, "V1 supports steps=0 only.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* Step 1: Calculate allocated grid dimensions. */
    size_t width  = (size_t)localNX + 2;
    size_t height = (size_t)localNY + 2;

    /* Check for possible memory-size overflow. */
    if (height > SIZE_MAX / width ||
        width * height > SIZE_MAX / sizeof(double)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    size_t totalCells = width * height;

    /* Step 2: Allocate two temperature grids. */
    double *T_old = calloc(totalCells, sizeof(double));
    double *T_new = calloc(totalCells, sizeof(double));

    if (T_old == NULL || T_new == NULL) {
        fprintf(stderr, "[Rank %d] Memory allocation failed.\n", rank);
        free(T_old);
        free(T_new);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* Step 3: Initialize the global top-left cell. */
    if (rank == 0) {
        T_old[1 * width + 1] = 100.0;
    }

    /* Step 4: Check the initial local temperature sum. */
    double localSum = 0.0;

    for (int y = 1; y <= localNY; y++) {
        for (int x = 1; x <= localNX; x++) {
            localSum += T_old[(size_t)y * width + x];
        }
    }

    printf("[V1] Rank=%d Position=(%d,%d) "
           "Interior=%dx%d Allocated=%zux%zu "
           "LocalSum=%.2f\n",
           rank, px, py,
           localNX, localNY,
           width, height, localSum);

    /* Step 5: Release allocated memory. */
    free(T_old);
    free(T_new);



    MPI_Finalize();

    return 0;
}
