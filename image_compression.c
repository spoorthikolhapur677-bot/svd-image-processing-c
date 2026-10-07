/*
 * compression.c
 * -------------
 * SVD-based image compression.
 *
 * CORE IDEA:
 * ----------
 * Every matrix A can be written as a sum of rank-1 matrices:
 *
 *   A = sigma_1 * u_1 * v_1^T
 *     + sigma_2 * u_2 * v_2^T
 *     + ...
 *     + sigma_r * u_r * v_r^T
 *
 * where sigma_1 >= sigma_2 >= ... >= sigma_r >= 0 are the singular values,
 * u_i are columns of U, and v_i^T are rows of V^T.
 *
 * The ECKART-YOUNG THEOREM says: the best rank-k approximation to A
 * (in the Frobenius norm sense) is obtained by keeping only the FIRST k terms:
 *
 *   A_k = sigma_1 * u_1 * v_1^T + ... + sigma_k * u_k * v_k^T
 *
 * Equivalently:  A_k = U_k * S_k * Vt_k
 *
 * The reconstruction error is:
 *   ||A - A_k||_F = sqrt( sigma_{k+1}^2 + ... + sigma_r^2 )
 *
 * COMPRESSION RATIO:
 * ------------------
 * Original:   m * n  floating point numbers
 * Compressed: k*m (U_k) + k (S_k diagonal) + k*n (Vt_k)
 *           = k * (m + 1 + n)  numbers
 * Ratio = m*n / (k*(m+n+1))
 *
 * For a 64x64 image with k=10:
 *   Original  = 4096 values
 *   Compressed= 10*(64+1+64) = 1290 values
 *   Ratio     ≈ 3.2x
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compression.h"
#include "svd.h"
#include "image.h"
#include "matrix.h"

/* ================================================================== */
/*  compress_image                                                     */
/* ================================================================== */

/*
 * compress_image(A, k, ratio_out)
 * --------------------------------
 * Returns the rank-k SVD approximation of matrix A.
 *
 * Steps:
 *   1. Compute full SVD: A = U * S * V^T
 *   2. Keep only first k singular values (zero the rest)
 *   3. Reconstruct: A_k = U * S_k * V^T
 *   4. (Optional) Return compression ratio
 */
Matrix *compress_image(Matrix *A, int k, double *ratio_out)
{
    if (A == NULL || k <= 0) return NULL;

    int m = A->rows;
    int n = A->cols;
    int min_mn = (m < n) ? m : n;

    /* Clamp k to the rank */
    if (k > min_mn) k = min_mn;

    /* ---- Step 1: Full SVD ----------------------------------------- */
    Matrix *U  = NULL;
    Matrix *S  = NULL;
    Matrix *Vt = NULL;

    int ok = svd(A, &U, &S, &Vt);
    if (ok != 0)
    {
        printf("compress_image: SVD failed\n");
        return NULL;
    }

    /* ---- Step 2: Zero out singular values beyond k ---------------- */
    /*
     * S is m x n diagonal.  We zero out S[i][i] for i >= k.
     * This is equivalent to keeping only the top-k terms in the sum.
     */
    for (int i = k; i < min_mn; i++)
        S->data[i][i] = 0.0;

    /* ---- Step 3: Reconstruct A_k = U * S_k * V^T ----------------- */
    Matrix *US = multiply(U, S);
    if (US == NULL)
    {
        free_matrix(U); free_matrix(S); free_matrix(Vt);
        return NULL;
    }

    Matrix *A_k = multiply(US, Vt);
    free_matrix(US);

    if (A_k == NULL)
    {
        free_matrix(U); free_matrix(S); free_matrix(Vt);
        return NULL;
    }

    /* ---- Step 4: Compute compression ratio ----------------------- */
    if (ratio_out != NULL)
    {
        /*
         * Compression ratio: how many times smaller is the compressed
         * representation compared to the original?
         * Compressed storage = k*(m + 1 + n)  values
         * Original storage   = m * n           values
         */
        double original   = (double)(m * n);
        double compressed = (double)(k * (m + 1 + n));
        *ratio_out = original / compressed;
    }

    free_matrix(U);
    free_matrix(S);
    free_matrix(Vt);

    return A_k;
}

/* ================================================================== */
/*  run_compression_experiment                                         */
/* ================================================================== */

/*
 * run_compression_experiment(A, output_dir)
 * ------------------------------------------
 * Runs compression for several values of k and prints a table showing:
 *   k | compression ratio | MSE | PSNR | Frobenius error
 */
void run_compression_experiment(Matrix *A, const char *output_dir)
{
    if (A == NULL) return;

    int m = A->rows;
    int n = A->cols;
    int min_mn = (m < n) ? m : n;

    /* Choose k values to test */
    int k_values[] = {1, 5, 10, 20, 50, min_mn};
    int num_k = 6;

    printf("\n");
    printf("==========================================================\n");
    printf(" Image Compression Experiment\n");
    printf(" Image size: %d x %d\n", m, n);
    printf("==========================================================\n");
    printf("%-6s %-12s %-12s %-10s %-12s\n",
           "k", "Ratio", "MSE", "PSNR(dB)", "Frob.Error");
    printf("----------------------------------------------------------\n");

    double original_frob = matrix_frobenius_norm(A);

    for (int t = 0; t < num_k; t++)
    {
        int k = k_values[t];
        if (k > min_mn) k = min_mn;

        double ratio = 0.0;
        Matrix *compressed = compress_image(A, k, &ratio);
        if (compressed == NULL) continue;

        /* Clamp pixel values before computing error */
        clamp_matrix(compressed, 0.0, 255.0);

        double mse  = matrix_mse(A, compressed);
        double psnr = matrix_psnr(A, compressed, 255.0);

        /* Frobenius reconstruction error */
        Matrix *diff = subtract_matrices(A, compressed);
        double frob_err = 0.0;
        if (diff != NULL)
        {
            frob_err = matrix_frobenius_norm(diff);
            free_matrix(diff);
        }

        printf("%-6d %-12.2f %-12.4f %-10.2f %-12.4f\n",
               k, ratio, mse, psnr, frob_err);

        /* Save compressed image if output directory provided */
        if (output_dir != NULL)
        {
            char filename[512];
            snprintf(filename, sizeof(filename),
                     "%s/compressed_k%d.pgm", output_dir, k);
            save_pgm(filename, compressed, 255);
        }

        free_matrix(compressed);
        if (k == min_mn) break; /* full rank — no need to go beyond */
    }

    printf("----------------------------------------------------------\n");
    printf("k=%d is full rank (no compression, perfect reconstruction)\n",
           min_mn);
    printf("\n");
    printf("OBSERVATIONS:\n");
    printf("  - As k increases, MSE decreases and PSNR increases.\n");
    printf("  - Low k = high compression but visible quality loss.\n");
    printf("  - High k = low compression but near-perfect quality.\n");
    printf("  - The optimal k depends on the image and acceptable quality.\n");
    printf("  - Frobenius reconstruction error = sqrt(sigma_{k+1}^2 + ...)\n");
    (void)original_frob; /* suppress unused warning */
}
