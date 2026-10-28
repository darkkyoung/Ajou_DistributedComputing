
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

    /* Step 4: Calculate process coordinates */
    int px = rank % PX;
    int py = rank / PX;

    /* Step 5: Calculate local grid size */
    int localNX = NX / PX;
    int localNY = NY / PY;

    printf("[V0] Rank=%d/%d Position=(%d,%d) "
           "Local=%dx%d Steps=%d Alpha=%.2f\n",
           rank, size, px, py,
           localNX, localNY, steps, alpha);

    /* Step 6: Finalize MPI */
    MPI_Finalize();

    return 0;
}
