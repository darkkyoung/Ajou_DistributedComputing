/*
 * HW1 - 2D Heat Diffusion
 * AI tool/model: ChatGPT (OpenAI), GPT-6
 * AI use dates: 2026-10-28 to 2026-10-29
 * AI-generated/modified: entire C implementation, including input parsing,
 * process-grid mapping, Neumann boundaries, halo exchange, stencil and
 * double buffering, global statistics, timing, and 4x4 reconstruction.
 * Core AI-assisted code regions are marked below.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

enum {
    TAG_LEFT = 10,
    TAG_RIGHT = 11,
    TAG_UP = 12,
    TAG_DOWN = 13,
    TAG_STATS = 100,
    TAG_GRID = 101
};

/* AI-generated/modified: integer input validation. */
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

    /* AI-generated/modified: process-grid coordinates and neighbor mapping. */

    #ifdef HW1_DEBUG_MAP

        int startX = px * localNX;
        int startY = py * localNY;

        int endX = startX + localNX - 1;
        int endY = startY + localNY - 1;
    #endif

    int left  = (px > 0) ? rank - 1 : -1;
    int right = (px < PX - 1) ? rank + 1 : -1;

    int up    = (py > 0) ? rank - PX : -1;
    int down  = (py < PY - 1) ? rank + PX : -1;

    #ifdef HW1_DEBUG_MAP

        printf("[V3-Map] Rank=%d Position=(%d,%d) "
            "GlobalX=%d..%d GlobalY=%d..%d "
            "Neighbors(L,R,U,D)=(%d,%d,%d,%d)\n",
            rank, px, py,
            startX, endX, startY, endY,
            left, right, up, down);
    #endif

    /* AI-generated/modified: local grid allocation and initial temperature. */
    size_t width  = (size_t)localNX + 2;
    size_t height = (size_t)localNY + 2;

    if (height > SIZE_MAX / width ||
        width * height > SIZE_MAX / sizeof(double)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    size_t totalCells = width * height;

    double *T_old = calloc(totalCells, sizeof(double));
    double *T_new = calloc(totalCells, sizeof(double));

    if (T_old == NULL || T_new == NULL) {
        fprintf(stderr, "[Rank %d] Memory allocation failed.\n", rank);
        free(T_old);
        free(T_new);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

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

    if (rank == 0) {
        T_old[1 * width + 1] = 100.0;
    }

    /* AI-generated/modified: simulation loop and timing. */

    double simulationStart = MPI_Wtime();
    for (int step = 0; step < steps; step++) {

        /* AI-generated/modified: Neumann boundary values on global edges. */

        if (left == -1) {
            for (int y = 1; y <= localNY; y++) {
                size_t row = (size_t)y * width;
                T_old[row] = T_old[row + 1];
            }
        }

        if (right == -1) {
            for (int y = 1; y <= localNY; y++) {
                size_t row = (size_t)y * width;
                T_old[row + localNX + 1] =
                    T_old[row + localNX];
            }
        }

        if (up == -1) {
            for (int x = 1; x <= localNX; x++) {
                T_old[x] = T_old[width + x];
            }
        }

        if (down == -1) {
            for (int x = 1; x <= localNX; x++) {
                T_old[(size_t)(localNY + 1) * width + x] =
                    T_old[(size_t)localNY * width + x];
            }
        }

        /* AI-generated/modified: horizontal halo (Left Recv->Send, Right Send->Recv). */

        if (left != -1) {

            MPI_Recv(
                recvColumn, localNY, MPI_DOUBLE,
                left, TAG_RIGHT,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            for (int y = 1; y <= localNY; y++) {
                T_old[(size_t)y * width] =
                    recvColumn[y - 1];
            }

            for (int y = 1; y <= localNY; y++) {
                sendColumn[y - 1] =
                    T_old[(size_t)y * width + 1];
            }

            MPI_Send(
                sendColumn, localNY, MPI_DOUBLE,
                left, TAG_LEFT, MPI_COMM_WORLD
            );
        }

        if (right != -1) {

            for (int y = 1; y <= localNY; y++) {
                sendColumn[y - 1] =
                    T_old[(size_t)y * width + localNX];
            }

            MPI_Send(
                sendColumn, localNY, MPI_DOUBLE,
                right, TAG_RIGHT, MPI_COMM_WORLD
            );

            MPI_Recv(
                recvColumn, localNY, MPI_DOUBLE,
                right, TAG_LEFT,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            for (int y = 1; y <= localNY; y++) {
                T_old[(size_t)y * width + localNX + 1] =
                    recvColumn[y - 1];
            }
        }

        /* AI-generated/modified: vertical halo (Up Recv->Send, Down Send->Recv). */

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

        /* AI-generated/modified: 5-point stencil computation. */
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

        double *temp = T_old;
        T_old = T_new;
        T_new = temp;
    }

    double localElapsed = MPI_Wtime() - simulationStart;

    /* AI-generated/modified: local interior sum and maximum. */

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

    /* AI-generated/modified: global statistics gathering via Send/Recv. */

    double stats[3] = {
        localSum,
        localMax,
        localElapsed
    };

    if (rank == 0) {

        double globalSum = localSum;
        double globalMax = localMax;
        double maxElapsed = localElapsed;

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

        MPI_Send(
            stats, 3, MPI_DOUBLE,
            0, TAG_STATS, MPI_COMM_WORLD
        );
    }

    /* AI-generated/modified: 4x4 global-grid reconstruction after timing. */

    if (NX == 4 && NY == 4) {

        int localCount = localNX * localNY;
        double localBlock[16] = {0};

        int k = 0;

        for (int y = 1; y <= localNY; y++) {
            for (int x = 1; x <= localNX; x++) {
                localBlock[k++] =
                    T_old[(size_t)y * width + x];
            }
        }

        if (rank == 0) {

            double globalGrid[16] = {0};

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

                int originX = (src % PX) * localNX;
                int originY = (src / PX) * localNY;

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

            for (int y = 0; y < NY; y++) {

                printf("[Grid] y=%d:", y);

                for (int x = 0; x < NX; x++) {
                    printf(" %.12f",
                           globalGrid[y * NX + x]);
                }

                printf("\n");
            }

        } else {

            MPI_Send(
                localBlock, localCount, MPI_DOUBLE,
                0, TAG_GRID, MPI_COMM_WORLD
            );
        }
    }

    free(sendColumn);
    free(recvColumn);
    free(T_old);
    free(T_new);

    MPI_Finalize();

    return 0;
}
