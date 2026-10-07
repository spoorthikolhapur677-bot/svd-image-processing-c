/*
 * svd.c
 * -----
 * Singular Value Decomposition  A = U * S * V^T
 * using the ONE-SIDED JACOBI algorithm.
 *
 * =====================================================================
 * ALGORITHM EXPLANATION (for a 2nd-year student)
 * =====================================================================
 *
 * GOAL: factorise matrix A (m x n) into U, S, V^T so that
 *       U * S * V^T = A.
 *
 * KEY IDEA — orthogonalise the columns of A:
 *
 *   Start with a working copy  B = A  and an accumulator  V = I_n.
 *
 *   A "Jacobi rotation" is a 2x2 rotation that we can apply to two
 *   columns of B to make those two columns orthogonal (perpendicular).
 *   We sweep through ALL pairs of columns, applying rotations until
 *   EVERY pair of columns is orthogonal.
 *
 *   Each rotation we apply to B, we also apply to V.
 *   So V keeps track of all the rotations we used.
 *
 * AFTER CONVERGENCE:
 *   - Columns of B are mutually orthogonal.
 *   - ||B[:,k]||  =  sigma_k  (the k-th singular value).
 *   - U[:,k]  =  B[:,k] / sigma_k  (normalise the columns).
 *   - V holds the right singular vectors.
 *   - We return V^T (the transpose of V).
 *
 * THE JACOBI ROTATION ANGLE:
 *   For columns p and q of B, let:
 *     alpha = B[:,p] . B[:,p]     (squared norm of column p)
 *     beta  = B[:,q] . B[:,q]     (squared norm of column q)
 *     gamma = B[:,p] . B[:,q]     (inner product = "how non-orthogonal")
 *
 *   We want to find angle theta such that after rotation,
 *   the new columns p' and q' satisfy  B[:,p'] . B[:,q'] = 0.
 *
 *   The stable Golub-Van Loan formula:
 *     tau = (beta - alpha) / (2 * gamma)
 *     t   = sign(tau) / (|tau| + sqrt(1 + tau^2))    <- tan(theta)
 *     c   = 1 / sqrt(1 + t^2)                         <- cos(theta)
 *     s   = c * t                                      <- sin(theta)
 *
 * WHY ONE-SIDED?
 *   Classic Jacobi applies rotations on BOTH sides (to diagonalise A^T A).
 *   One-sided applies rotations only on the RIGHT (to columns of B),
 *   which is simpler to implement and understand.
 *
 * =====================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svd.h"
#include "matrix.h"

/* Maximum number of full sweeps over all column pairs */
#define JACOBI_MAX_SWEEPS 150

/*
 * Convergence tolerance.
 * We stop rotating a pair (p,q) when |gamma| / sqrt(alpha*beta) < TOL.
 * This means the columns are "orthogonal enough".
 */
#define JACOBI_TOL 1e-12

/* ------------------------------------------------------------------ */
/*  Internal helper: dot product of two columns of a matrix           */
/* ------------------------------------------------------------------ */

static double col_dot(Matrix *M, int col1, int col2)
{
    double s = 0.0;
    for (int i = 0; i < M->rows; i++)
        s += M->data[i][col1] * M->data[i][col2];
    return s;
}

/* ------------------------------------------------------------------ */
/*  Internal helper: apply Jacobi rotation to columns p and q of M   */
/*                                                                     */
/*  The rotation replaces:                                             */
/*    col_p  <--  c * col_p + s * col_q                               */
/*    col_q  <-- -s * col_p + c * col_q                               */
/*                                                                     */
/*  This is a right-multiplication by the 2x2 rotation matrix         */
/*    J = [ c   -s ]                                                   */
/*        [ s    c ]                                                   */
/*  applied to columns p and q.                                        */
/* ------------------------------------------------------------------ */

static void apply_jacobi_rotation(Matrix *M, int p, int q, double c, double s)
{
    for (int i = 0; i < M->rows; i++)
    {
        double mp = M->data[i][p];
        double mq = M->data[i][q];
        M->data[i][p] =  c * mp + s * mq;
        M->data[i][q] = -s * mp + c * mq;
    }
}

/* ------------------------------------------------------------------ */
/*  Comparison function for qsort — sort SigmaEntry by descending val */
/* ------------------------------------------------------------------ */

typedef struct { int index; double value; } SigmaEntry;

static int compare_descending(const void *a, const void *b)
{
    const SigmaEntry *ea = (const SigmaEntry *)a;
    const SigmaEntry *eb = (const SigmaEntry *)b;
    if (eb->value > ea->value) return  1;
    if (eb->value < ea->value) return -1;
    return 0;
}

/* ================================================================== */
/*  MAIN SVD FUNCTION                                                  */
/* ================================================================== */

int svd(Matrix *A, Matrix **U_out, Matrix **S_out, Matrix **Vt_out)
{
    if (A == NULL || U_out == NULL || S_out == NULL || Vt_out == NULL)
        return -1;

    int m = A->rows;
    int n = A->cols;

    /* ----------------------------------------------------------------
     * STEP 1: Initialise working matrices
     *   B = copy of A   (we will modify B in place)
     *   V = I_n         (accumulates the right rotations)
     * ---------------------------------------------------------------- */
    Matrix *B = copy_matrix(A);
    if (B == NULL) return -1;

    Matrix *V = identity_matrix(n);
    if (V == NULL) { free_matrix(B); return -1; }

    /* ----------------------------------------------------------------
     * STEP 2: Jacobi sweeps
     *
     * Repeat until all column pairs are orthogonal (or max sweeps hit).
     * Each "sweep" visits every pair (p, q) with p < q.
     * ---------------------------------------------------------------- */
    int converged = 0;

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS && !converged; sweep++)
    {
        int num_rotations = 0; /* count rotations in this sweep */

        for (int p = 0; p < n - 1; p++)
        {
            for (int q = p + 1; q < n; q++)
            {
                /* Compute the three inner products needed */
                double alpha = col_dot(B, p, p); /* ||B[:,p]||^2   */
                double beta  = col_dot(B, q, q); /* ||B[:,q]||^2   */
                double gamma = col_dot(B, p, q); /* B[:,p].B[:,q]  */

                /* Skip if either column is essentially zero */
                if (alpha * beta < 1e-300) continue;

                /*
                 * Measure how far from orthogonal the columns are.
                 * off = |cos(angle between columns)|
                 * If off is tiny, columns are already orthogonal — skip.
                 */
                double off = fabs(gamma) / sqrt(alpha * beta);
                if (off < JACOBI_TOL) continue;

                num_rotations++;

                /*
                 * Compute the Jacobi rotation angle that zeroes
                 * the off-diagonal of the 2x2 Gram submatrix
                 *   G_pq = [alpha  gamma]
                 *          [gamma  beta ]
                 *
                 * We need tan(2θ) = 2*gamma / (alpha - beta).
                 * The stable formula avoids catastrophic cancellation:
                 *   tau = (alpha - beta) / (2 * gamma)
                 *   t   = sign(tau) / (|tau| + sqrt(1 + tau^2))
                 *   c   = 1 / sqrt(1 + t^2),  s = c * t
                 */
                double tau = (alpha - beta) / (2.0 * gamma);
                double t;
                if (tau >= 0.0)
                    t =  1.0 / ( tau + sqrt(1.0 + tau * tau));
                else
                    t = -1.0 / (-tau + sqrt(1.0 + tau * tau));

                double c = 1.0 / sqrt(1.0 + t * t); /* cosine */
                double s = c * t;                     /* sine   */

                /* Apply rotation to B and accumulate in V */
                apply_jacobi_rotation(B, p, q, c, s);
                apply_jacobi_rotation(V, p, q, c, s);
            }
        }

        /* If no rotation was needed this sweep, we have converged */
        if (num_rotations == 0)
            converged = 1;
    }

    /* ----------------------------------------------------------------
     * STEP 3: Extract singular values = column norms of B
     * Sort them in descending order.
     * ---------------------------------------------------------------- */
    SigmaEntry *entries = malloc(n * sizeof(SigmaEntry));
    if (entries == NULL)
    {
        free_matrix(B); free_matrix(V);
        return -1;
    }

    for (int j = 0; j < n; j++)
    {
        entries[j].index = j;
        entries[j].value = sqrt(col_dot(B, j, j));
    }

    qsort(entries, n, sizeof(SigmaEntry), compare_descending);

    /* ----------------------------------------------------------------
     * STEP 4: Build output matrices U, S, Vt
     * ---------------------------------------------------------------- */
    Matrix *U  = create_matrix(m, m); /* m x m orthogonal */
    Matrix *S  = create_matrix(m, n); /* m x n diagonal   */
    Matrix *Vt = create_matrix(n, n); /* n x n orthogonal */

    if (U == NULL || S == NULL || Vt == NULL)
    {
        free(entries); free_matrix(B); free_matrix(V);
        free_matrix(U); free_matrix(S); free_matrix(Vt);
        return -1;
    }

    /*
     * For each singular value k (in sorted order):
     *   sigma_k  = entries[k].value
     *   u_k      = B[:,orig] / sigma_k   (normalised column of B)
     *   v_k      = V[:,orig]             (column of V = row of Vt)
     */
    for (int k = 0; k < n; k++)
    {
        int    orig  = entries[k].index;
        double sigma = entries[k].value;

        S->data[k][k] = sigma; /* fill diagonal of Sigma */

        if (sigma > 1e-14)
        {
            /* u_k = B[:,orig] / sigma */
            for (int i = 0; i < m; i++)
                U->data[i][k] = B->data[i][orig] / sigma;
        }
        else
        {
            /*
             * Zero (or near-zero) singular value.
             * The corresponding U column can be anything orthogonal.
             * We leave it as zero here; it does not affect reconstruction
             * because sigma_k * 0 = 0.
             */
            for (int i = 0; i < m; i++)
                U->data[i][k] = 0.0;
        }

        /* row k of V^T  =  column orig of V */
        for (int i = 0; i < n; i++)
            Vt->data[k][i] = V->data[i][orig];
    }

    /*
     * Gram-Schmidt completion for U.
     *
     * We always run this block because:
     *   - If m > n: the last (m-n) columns of U were never filled.
     *   - If m == n but some sigma_k == 0: those U columns are zero
     *     and need to be replaced with valid orthonormal vectors.
     *
     * Strategy: for each column of U that has near-zero norm,
     * try every standard basis vector e_0, e_1, ..., e_{m-1}
     * as a candidate, Gram-Schmidt orthogonalise against all
     * previously filled columns, then normalise.
     */
    for (int target = 0; target < m; target++)
    {
        /* Check if this column is already a valid unit vector */
        double col_norm_sq = 0.0;
        for (int i = 0; i < m; i++)
            col_norm_sq += U->data[i][target] * U->data[i][target];
        if (col_norm_sq > 0.5) continue; /* already filled */

        /* Try standard basis vectors as candidates */
        int filled = 0;
        for (int candidate = 0; candidate < m && !filled; candidate++)
        {
            double *v = calloc(m, sizeof(double));
            if (v == NULL) break;
            v[candidate] = 1.0;

            /* Gram-Schmidt: subtract projections onto all prior columns */
            for (int k = 0; k < target; k++)
            {
                double proj = 0.0;
                for (int i = 0; i < m; i++)
                    proj += U->data[i][k] * v[i];
                for (int i = 0; i < m; i++)
                    v[i] -= proj * U->data[i][k];
            }

            /* Normalise */
            double norm = 0.0;
            for (int i = 0; i < m; i++) norm += v[i] * v[i];
            norm = sqrt(norm);

            if (norm > 1e-10)
            {
                for (int i = 0; i < m; i++)
                    U->data[i][target] = v[i] / norm;
                filled = 1;
            }
            free(v);
        }
    }

    free(entries);
    free_matrix(B);
    free_matrix(V);

    *U_out  = U;
    *S_out  = S;
    *Vt_out = Vt;

    return 0;
}
