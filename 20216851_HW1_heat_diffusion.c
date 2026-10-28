
/*
 * HW1 - 2D Heat Diffusion
 *
 * AI assistance: ChatGPT (OpenAI), GPT-6
 * Dates: 2026-10-28 to 2026-10-29
 *
 * AI-generated/modified code:
 * V0: MPI initialization and input validation
 * V1: Local grid allocation and initialization
 * V2: Stencil computation and double buffering
 * V3: Domain decomposition and neighbor mapping
 * V4: Vertical halo exchange
 * V5: Horizontal halo exchange
 * V6: Full 2D communication integration
 * V7: Global statistics, timing, and grid reconstruction
 * V8: Optional debug output for performance experiments
 *
 * Development and correctness tests were performed
 * incrementally. Performance measurements are reported
 * separately using actual MPI execution results.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

/*
 * V7: Additional message tags for statistics and grid gathering.
 * AI-generated: ChatGPT GPT-6, 2026-10-29.
 */
enum {
    TAG_LEFT = 10,
    TAG_RIGHT = 11,
    TAG_UP = 12,
    TAG_DOWN = 13,
    TAG_STATS = 100,
    TAG_GRID = 101
};


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
    #ifdef HW1_DEBUG_MAP
        /* V8: Global coordinates used for debug output. */
        int startX = px * localNX;
        int startY = py * localNY;

        int endX = startX + localNX - 1;
        int endY = startY + localNY - 1;
    #endif

    /* Neighbor ranks (-1 means no neighbor) */
    int left  = (px > 0) ? rank - 1 : -1;
    int right = (px < PX - 1) ? rank + 1 : -1;

    int up    = (py > 0) ? rank - PX : -1;
    int down  = (py < PY - 1) ? rank + PX : -1;

    
    /*
     * V1: Allocate local grids and initialize temperature.
     * AI-generated: ChatGPT GPT-6, 2026-10-28.
     */

    
    /*
    * V6: Full 2D process-grid support.
    *
    * Horizontal and vertical halo exchanges are both enabled.
    * Valid PX x PY decompositions are accepted.
    *
    * AI-assisted integration: ChatGPT GPT-6, 2026-10-29.
    */

    
    /* V3: Print process ownership and neighbors. */
    #ifdef HW1_DEBUG_MAP
        /* V8: Optional process-grid debug output. */
        printf("[V3-Map] Rank=%d Position=(%d,%d) "
            "GlobalX=%d..%d GlobalY=%d..%d "
            "Neighbors(L,R,U,D)=(%d,%d,%d,%d)\n",
            rank, px, py,
            startX, endX, startY, endY,
            left, right, up, down);
    #endif

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

    
    /*
     * V5: Temporary column buffers.
     * AI-generated: ChatGPT GPT-6, 2026-10-29.
     */

    double *sendColumn = NULL;
    double *recvColumn = NULL;

    if (PX > 1) {
        sendColumn = malloc((size_t)localNY * sizeof(double));
        recvColumn = malloc((size_t)localNY * sizeof(double));

        if (sendColumn == NULL || recvColumn == NULL) {
            fprintf(stderr,
                    "[Rank %d] Column buffer allocation failed.\n",
                    rank);

            free(sendColumn);
            free(recvColumn);
            free(T_old);
            free(T_new);

            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
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
    /* V7: Start measuring the simulation loop. */
    /* AI-generated: ChatGPT GPT-6, 2026-10-29. */
    double simulationStart = MPI_Wtime();
    for (int step = 0; step < steps; step++) {

        
        /*
         * V4: Apply Neumann condition only at global boundaries.
         * AI-generated: ChatGPT GPT-6, 2026-10-28.
         */

        /* Left global boundary */
        if (left == -1) {
            for (int y = 1; y <= localNY; y++) {
                size_t row = (size_t)y * width;
                T_old[row] = T_old[row + 1];
            }
        }

        /* Right global boundary */
        if (right == -1) {
            for (int y = 1; y <= localNY; y++) {
                size_t row = (size_t)y * width;
                T_old[row + localNX + 1] =
                    T_old[row + localNX];
            }
        }

        /* Top global boundary */
        if (up == -1) {
            for (int x = 1; x <= localNX; x++) {
                T_old[x] = T_old[width + x];
            }
        }

        /* Bottom global boundary */
        if (down == -1) {
            for (int x = 1; x <= localNX; x++) {
                T_old[(size_t)(localNY + 1) * width + x] =
                    T_old[(size_t)localNY * width + x];
            }
        }

        /*
         * V5: Horizontal Halo Exchange.
         * Left: Recv -> Send
         * Right: Send -> Recv
         * AI-generated: ChatGPT GPT-6, 2026-10-29.
         */

        /* Exchange with the left neighbor */
        if (left != -1) {

            /* 1. Receive right boundary of left neighbor */
            MPI_Recv(
                recvColumn, localNY, MPI_DOUBLE,
                left, TAG_RIGHT,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            /* 2. Unpack into left halo column (x=0) */
            for (int y = 1; y <= localNY; y++) {
                T_old[(size_t)y * width] =
                    recvColumn[y - 1];
            }

            /* 3. Pack our first interior column (x=1) */
            for (int y = 1; y <= localNY; y++) {
                sendColumn[y - 1] =
                    T_old[(size_t)y * width + 1];
            }

            /* 4. Send to left neighbor */
            MPI_Send(
                sendColumn, localNY, MPI_DOUBLE,
                left, TAG_LEFT, MPI_COMM_WORLD
            );
        }

        /* Exchange with the right neighbor */
        if (right != -1) {

            /* 1. Pack our last interior column */
            for (int y = 1; y <= localNY; y++) {
                sendColumn[y - 1] =
                    T_old[(size_t)y * width + localNX];
            }

            /* 2. Send to right neighbor */
            MPI_Send(
                sendColumn, localNY, MPI_DOUBLE,
                right, TAG_RIGHT, MPI_COMM_WORLD
            );

            /* 3. Receive first column of right neighbor */
            MPI_Recv(
                recvColumn, localNY, MPI_DOUBLE,
                right, TAG_LEFT,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            /* 4. Unpack into right halo column */
            for (int y = 1; y <= localNY; y++) {
                T_old[(size_t)y * width + localNX + 1] =
                    recvColumn[y - 1];
            }
        }


        /*
         * V4: Vertical Halo Exchange.
         * Up: Recv -> Send
         * Down: Send -> Recv
         * AI-generated: ChatGPT GPT-6, 2026-10-28.
         */

        if (up != -1) {
            MPI_Recv(
                &T_old[1],
                localNX, MPI_DOUBLE,
                up, TAG_DOWN,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            MPI_Send(
                &T_old[width + 1],
                localNX, MPI_DOUBLE,
                up, TAG_UP,
                MPI_COMM_WORLD
            );
        }

        if (down != -1) {
            MPI_Send(
                &T_old[(size_t)localNY * width + 1],
                localNX, MPI_DOUBLE,
                down, TAG_DOWN,
                MPI_COMM_WORLD
            );

            MPI_Recv(
                &T_old[(size_t)(localNY + 1) * width + 1],
                localNX, MPI_DOUBLE,
                down, TAG_UP,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
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

    /* V7: End timing before gathering statistics. */
    double localElapsed = MPI_Wtime() - simulationStart;


    /* V6: Calculate final local statistics (interior only). */
    
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

    /*
     * V7: Collect global statistics using MPI_Send/MPI_Recv.
     * AI-generated: ChatGPT GPT-6, 2026-10-29.
     */

    double stats[3] = {
        localSum,
        localMax,
        localElapsed
    };

    if (rank == 0) {

        double globalSum = localSum;
        double globalMax = localMax;
        double maxElapsed = localElapsed;

        /* Receive results from all other ranks. */
        for (int src = 1; src < size; src++) {

            double received[3];

            MPI_Recv(
                received, 3, MPI_DOUBLE,
                src, TAG_STATS,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            globalSum += received[0];

            if (received[1] > globalMax) {
                globalMax = received[1];
            }

            if (received[2] > maxElapsed) {
                maxElapsed = received[2];
            }
        }

        printf("[Result] NX=%d NY=%d Grid=%dx%d "
               "Procs=%d Steps=%d "
               "Sum=%.12f Max=%.12f Time=%.9f\n",
               NX, NY, PX, PY,
               size, steps,
               globalSum, globalMax, maxElapsed);

    } else {

        /* Every non-root rank sends three doubles to rank 0. */
        MPI_Send(
            stats, 3, MPI_DOUBLE,
            0, TAG_STATS, MPI_COMM_WORLD
        );
    }



    
    /*
     * V7: Reconstruct final global 4x4 temperature grid.
     * AI-generated: ChatGPT GPT-6, 2026-10-29.
     *
     * Diagnostic only: excluded from measured simulation time.
     */

    if (NX == 4 && NY == 4) {

        int localCount = localNX * localNY;
        double localBlock[16] = {0};

        /* Step 1: Pack local interior into a contiguous array. */
        int k = 0;

        for (int y = 1; y <= localNY; y++) {
            for (int x = 1; x <= localNX; x++) {
                localBlock[k++] =
                    T_old[(size_t)y * width + x];
            }
        }

        if (rank == 0) {

            double globalGrid[16] = {0};

            /* Step 2: Collect each rank's local block. */
            for (int src = 0; src < size; src++) {

                double received[16];
                const double *block = localBlock;

                if (src != 0) {
                    MPI_Recv(
                        received, localCount, MPI_DOUBLE,
                        src, TAG_GRID,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    );

                    block = received;
                }

                /* Step 3: Find the block's global origin. */
                int originX = (src % PX) * localNX;
                int originY = (src / PX) * localNY;

                /* Step 4: Copy block values to global coordinates. */
                for (int y = 0; y < localNY; y++) {
                    for (int x = 0; x < localNX; x++) {

                        int globalIndex =
                            (originY + y) * NX + (originX + x);

                        int localIndex = y * localNX + x;

                        globalGrid[globalIndex] =
                            block[localIndex];
                    }
                }
            }

            /* Step 5: Print the final global grid. */
            for (int y = 0; y < NY; y++) {

                printf("[Grid] y=%d:", y);

                for (int x = 0; x < NX; x++) {
                    printf(" %.12f",
                           globalGrid[y * NX + x]);
                }

                printf("\n");
            }

        } else {

            /* Non-root ranks send their interior values. */
            MPI_Send(
                localBlock, localCount, MPI_DOUBLE,
                0, TAG_GRID, MPI_COMM_WORLD
            );
        }
    }


    /* Release all allocated buffers. */
    free(sendColumn);
    free(recvColumn);
    free(T_old);
    free(T_new);

    MPI_Finalize();

    return 0;
}
