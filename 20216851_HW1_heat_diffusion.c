
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
     * V3: Global coordinates and neighbor rank mapping.
     * AI-generated: ChatGPT GPT-6, 2026-10-28.
     */

    /* Global coordinates of this rank's interior */
    int startX = px * localNX;
    int startY = py * localNY;

    int endX = startX + localNX - 1;
    int endY = startY + localNY - 1;

    /* Neighbor ranks (-1 means no neighbor) */
    int left  = (px > 0) ? rank - 1 : -1;
    int right = (px < PX - 1) ? rank + 1 : -1;

    int up    = (py > 0) ? rank - PX : -1;
    int down  = (py < PY - 1) ? rank + PX : -1;

    
    /*
     * V1: Allocate local grids and initialize temperature.
     * AI-generated: ChatGPT GPT-6, 2026-10-28.
     */

    
    /* V3: Multi-rank heat updates require halo exchange (V4+). */
    if (size != 1 && steps > 0) {
        if (rank == 0)
            fprintf(stderr,
                    "V3: Multi-rank heat updates require halo exchange.\n");

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    
    /* V3: Print process ownership and neighbors. */
    printf("[V3-Map] Rank=%d Position=(%d,%d) "
           "GlobalX=%d..%d GlobalY=%d..%d "
           "Neighbors(L,R,U,D)=(%d,%d,%d,%d)\n",
           rank, px, py,
           startX, endX, startY, endY,
           left, right, up, down);



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
    
    /*
    * V2: Neumann boundary, 5-point stencil,
    * and double buffering.
    * AI-generated: ChatGPT GPT-6, 2026-10-28.
    */

    for (int step = 0; step < steps; step++) {

        /* 1. Left and right Neumann boundaries */
        for (int y = 1; y <= localNY; y++) {
            size_t row = (size_t)y * width;

            T_old[row] = T_old[row + 1];
            T_old[row + localNX + 1] =
                T_old[row + localNX];
        }

        /* 2. Top and bottom Neumann boundaries */
        for (int x = 1; x <= localNX; x++) {
            T_old[x] = T_old[width + x];

            T_old[(size_t)(localNY + 1) * width + x] =
                T_old[(size_t)localNY * width + x];
        }

        /* 3. Compute new temperatures */
        for (int y = 1; y <= localNY; y++) {
            for (int x = 1; x <= localNX; x++) {

                size_t idx = (size_t)y * width + x;

                double C = T_old[idx];
                double N = T_old[idx - width];
                double S = T_old[idx + width];
                double W = T_old[idx - 1];
                double E = T_old[idx + 1];

                T_new[idx] =
                    C + alpha * (N + S + W + E - 4.0 * C);
            }
        }

        /* 4. Swap old and new buffers */
        double *temp = T_old;
        T_old = T_new;
        T_new = temp;
    }


    /* Step 4: Check the initial local temperature sum. */
    
    /*
    * V2: Calculate local sum and maximum.
    * AI-generated: ChatGPT GPT-6, 2026-10-28.
    */

    double localSum = 0.0;
    double localMax = -INFINITY;

    for (int y = 1; y <= localNY; y++) {
        for (int x = 1; x <= localNX; x++) {

            double value =
                T_old[(size_t)y * width + x];

            localSum += value;

            if (value > localMax) {
                localMax = value;
            }
        }
    }

    
    printf("[V3] Rank=%d Position=(%d,%d) "
        "Steps=%d Sum=%.12f Max=%.12f\n",
        rank, px, py, steps, localSum, localMax);


    /* Print the final 4x4 temperature distribution. */
    if (rank == 0 && NX == 4 && NY == 4 &&
        PX == 1 && PY == 1) {

        for (int y = 1; y <= localNY; y++) {
            printf("[Grid] y=%d:", y - 1);

            for (int x = 1; x <= localNX; x++) {
                printf(" %.2f",
                    T_old[(size_t)y * width + x]);
            }

            printf("\n");
        }
    }

    /* Step 5: Release allocated memory. */
    free(T_old);
    free(T_new);

    MPI_Finalize();

    return 0;
}
