/*
 * main.c
 * ------
 * SVD and Image Compression — Complete Project Demo
 *
 * Menu:
 *   1. SVD of the teacher's 2x2 matrix + verification
 *   2. Unit-circle experiment (outputs CSV files)
 *   3. Image compression experiment
 *   4. Image denoising experiment
 *   5. Run all tests
 *   6. Exit
 *
 * Compile:
 *   gcc src/main.c src/matrix.c src/svd.c src/image.c \
 *       src/compression.c src/denoise.c -o svd_project -lm
 *
 * Or use:  make
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matrix.h"
#include "svd.h"
#include "image.h"
#include "compression.h"
#include "denoise.h"

#define TOL      1e-6
#define PI       3.14159265358979323846
#define N_CIRCLE 360   /* number of unit-circle sample points */

/* ================================================================== */
/*  Helper macros                                                      */
/* ================================================================== */
#define PASS_FAIL(cond) ((cond) ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")

/* ================================================================== */
/*  DEMO 1 — SVD of the teacher's 2x2 matrix                         */
/* ================================================================== */

static void demo_svd_2x2(void)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" DEMO 1: SVD of the Teacher's 2x2 Matrix\n");
    printf("==========================================================\n\n");
    printf("  A = [ 2   1 ]\n");
    printf("      [-1   1 ]\n\n");

    /* Build A */
    Matrix *A = create_matrix(2, 2);
    A->data[0][0] =  2.0;  A->data[0][1] =  1.0;
    A->data[1][0] = -1.0;  A->data[1][1] =  1.0;

    printf("Matrix A:\n");
    print_matrix(A);

    /* Compute SVD */
    Matrix *U = NULL, *S = NULL, *Vt = NULL;
    int ok = svd(A, &U, &S, &Vt);
    if (ok != 0)
    {
        printf("SVD FAILED (code %d)\n", ok);
        free_matrix(A);
        return;
    }

    printf("\nU (left singular vectors — columns):\n");
    print_matrix(U);

    printf("\nSigma (singular values on diagonal):\n");
    print_matrix(S);

    printf("\nV^T (rows are right singular vectors):\n");
    print_matrix(Vt);

    /* Reconstruction */
    Matrix *US   = multiply(U, S);
    Matrix *Arec = multiply(US, Vt);
    printf("\nReconstructed  U * Sigma * V^T  (should match A):\n");
    print_matrix(Arec);

    /* ---- Verification -------------------------------------------- */
    printf("\n");
    printf("----------------------------------------------------------\n");
    printf(" Verification (tolerance = %.0e)\n", TOL);
    printf("----------------------------------------------------------\n");
    int all_ok = 1;

    /* Test 1: U^T * U = I */
    Matrix *Ut  = transpose(U);
    Matrix *UtU = multiply(Ut, U);
    Matrix *Im  = identity_matrix(U->rows);
    double e1   = matrix_max_abs_diff(UtU, Im);
    printf("Test 1  U^T*U = I        error = %.2e  %s\n",
           e1, PASS_FAIL(e1 < TOL));
    if (e1 >= TOL) all_ok = 0;
    free_matrix(Ut); free_matrix(UtU); free_matrix(Im);

    /* Test 2: V^T * V = I  (Vt*Vt^T = I) */
    Matrix *V   = transpose(Vt);
    Matrix *VtV = multiply(Vt, V);
    Matrix *In  = identity_matrix(Vt->rows);
    double e2   = matrix_max_abs_diff(VtV, In);
    printf("Test 2  V^T*V = I        error = %.2e  %s\n",
           e2, PASS_FAIL(e2 < TOL));
    if (e2 >= TOL) all_ok = 0;
    free_matrix(V); free_matrix(VtV); free_matrix(In);

    /* Test 3: U * S * V^T = A */
    double e3 = matrix_max_abs_diff(Arec, A);
    printf("Test 3  U*S*V^T = A      error = %.2e  %s\n",
           e3, PASS_FAIL(e3 < TOL));
    if (e3 >= TOL) all_ok = 0;

    /* Frobenius error */
    Matrix *diff3 = subtract_matrices(A, Arec);
    double frob   = matrix_frobenius_norm(diff3);
    double frobA  = matrix_frobenius_norm(A);
    printf("       Frobenius error = %.2e  (relative = %.2e)\n",
           frob, (frobA > 1e-15) ? frob / frobA : frob);
    free_matrix(diff3);

    /* Test 4: A * v_i = sigma_i * u_i */
    printf("Test 4  A*v_i = sigma_i*u_i  (sign-insensitive):\n");
    int min_mn = 2;
    for (int k = 0; k < min_mn; k++)
    {
        /* Extract v_k as column vector from row k of Vt */
        Matrix *vk = create_matrix(Vt->cols, 1);
        for (int j = 0; j < Vt->cols; j++)
            vk->data[j][0] = Vt->data[k][j];

        /* Extract u_k as column vector */
        Matrix *uk = create_matrix(U->rows, 1);
        for (int i = 0; i < U->rows; i++)
            uk->data[i][0] = U->data[i][k];

        double sigma_k = S->data[k][k];

        Matrix *Avk = multiply(A, vk);       /* A * v_k            */
        Matrix *suk = scale_matrix(uk, sigma_k); /* sigma_k * u_k  */

        /* Accept both +u and -u (sign ambiguity is mathematically valid) */
        double ep = 0.0, em = 0.0;
        for (int i = 0; i < A->rows; i++)
        {
            double dp = Avk->data[i][0] - suk->data[i][0];
            double dm = Avk->data[i][0] + suk->data[i][0];
            ep += dp*dp; em += dm*dm;
        }
        ep = sqrt(ep); em = sqrt(em);
        double e4k = (ep < em) ? ep : em;

        printf("       i=%d  sigma=%.6f  error=%.2e  %s\n",
               k+1, sigma_k, e4k, PASS_FAIL(e4k < TOL));
        if (e4k >= TOL) all_ok = 0;

        free_matrix(vk); free_matrix(uk);
        free_matrix(Avk); free_matrix(suk);
    }

    printf("----------------------------------------------------------\n");
    printf(" Result: %s\n", all_ok ? "\033[32mALL TESTS PASSED\033[0m"
                                   : "\033[31mSOME TESTS FAILED\033[0m");
    printf("----------------------------------------------------------\n");

    free_matrix(A); free_matrix(U); free_matrix(S); free_matrix(Vt);
    free_matrix(US); free_matrix(Arec);
}

/* ================================================================== */
/*  DEMO 2 — Unit-circle experiment                                   */
/* ================================================================== */

static void demo_unit_circle(void)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" DEMO 2: Unit-Circle Experiment\n");
    printf("==========================================================\n\n");
    printf("Applying A = [2 1; -1 1] to the unit circle  x=cos(t), y=sin(t)\n\n");
    printf("Steps:\n");
    printf("  Original X  →  V^T X  →  Sigma * V^T X  →  U * Sigma * V^T X\n\n");

    /* Build A and compute SVD */
    Matrix *A = create_matrix(2, 2);
    A->data[0][0] =  2.0; A->data[0][1] =  1.0;
    A->data[1][0] = -1.0; A->data[1][1] =  1.0;

    Matrix *U = NULL, *S = NULL, *Vt = NULL;
    if (svd(A, &U, &S, &Vt) != 0)
    {
        printf("SVD failed\n");
        free_matrix(A); return;
    }

    printf("Singular values: sigma_1 = %.6f,  sigma_2 = %.6f\n\n",
           S->data[0][0], S->data[1][1]);
    printf("Geometric meaning:\n");
    printf("  sigma_1 = %.4f : semi-major axis of the output ellipse\n",
           S->data[0][0]);
    printf("  sigma_2 = %.4f : semi-minor axis of the output ellipse\n\n",
           S->data[1][1]);

    /* Open separate CSV files for each stage */
    FILE *f_orig  = fopen("data/output/circle_original.csv",   "w");
    FILE *f_vt    = fopen("data/output/circle_after_Vt.csv",   "w");
    FILE *f_sigma = fopen("data/output/circle_after_sigma.csv","w");
    FILE *f_u     = fopen("data/output/circle_after_U.csv",    "w");
    FILE *f_all   = fopen("data/output/circle_all_steps.csv",  "w");

    if (!f_orig || !f_vt || !f_sigma || !f_u || !f_all)
    {
        printf("Warning: could not open output CSV files.\n");
        printf("Make sure data/output/ directory exists.\n");
        /* Try current directory as fallback */
        if (f_orig)  fclose(f_orig);
        if (f_vt)    fclose(f_vt);
        if (f_sigma) fclose(f_sigma);
        if (f_u)     fclose(f_u);
        if (f_all)   fclose(f_all);
        f_orig  = fopen("circle_original.csv",    "w");
        f_vt    = fopen("circle_after_Vt.csv",    "w");
        f_sigma = fopen("circle_after_sigma.csv", "w");
        f_u     = fopen("circle_after_U.csv",     "w");
        f_all   = fopen("circle_all_steps.csv",   "w");
    }

    /* Write CSV headers */
    if (f_orig)  fprintf(f_orig,  "x,y\n");
    if (f_vt)    fprintf(f_vt,    "x,y\n");
    if (f_sigma) fprintf(f_sigma, "x,y\n");
    if (f_u)     fprintf(f_u,     "x,y\n");
    if (f_all)
        fprintf(f_all,
                "t,orig_x,orig_y,"
                "after_Vt_x,after_Vt_y,"
                "after_sigma_x,after_sigma_y,"
                "after_U_x,after_U_y,"
                "direct_Ax,direct_Ay\n");

    Matrix *X  = create_matrix(2, 1);

    for (int k = 0; k < N_CIRCLE; k++)
    {
        double t  = 2.0 * PI * k / N_CIRCLE;
        double ox = cos(t);
        double oy = sin(t);

        X->data[0][0] = ox;
        X->data[1][0] = oy;

        Matrix *X1 = multiply(Vt, X);          /* V^T * X            */
        Matrix *X2 = multiply(S, X1);          /* Sigma * V^T * X    */
        Matrix *X3 = multiply(U, X2);          /* U * Sigma * V^T * X */
        Matrix *X4 = multiply(A, X);           /* direct A * X       */

        if (f_orig)
            fprintf(f_orig,  "%.8f,%.8f\n", ox, oy);
        if (f_vt && X1)
            fprintf(f_vt,    "%.8f,%.8f\n", X1->data[0][0], X1->data[1][0]);
        if (f_sigma && X2)
            fprintf(f_sigma, "%.8f,%.8f\n", X2->data[0][0], X2->data[1][0]);
        if (f_u && X3)
            fprintf(f_u,     "%.8f,%.8f\n", X3->data[0][0], X3->data[1][0]);
        if (f_all && X1 && X2 && X3 && X4)
            fprintf(f_all,
                    "%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    t, ox, oy,
                    X1->data[0][0], X1->data[1][0],
                    X2->data[0][0], X2->data[1][0],
                    X3->data[0][0], X3->data[1][0],
                    X4->data[0][0], X4->data[1][0]);

        free_matrix(X1); free_matrix(X2);
        free_matrix(X3); free_matrix(X4);
    }

    if (f_orig)  fclose(f_orig);
    if (f_vt)    fclose(f_vt);
    if (f_sigma) fclose(f_sigma);
    if (f_u)     fclose(f_u);
    if (f_all)   fclose(f_all);

    free_matrix(X);
    free_matrix(A); free_matrix(U); free_matrix(S); free_matrix(Vt);

    printf("CSV files written:\n");
    printf("  circle_original.csv    — unit circle (input)\n");
    printf("  circle_after_Vt.csv    — after V^T (rotation)\n");
    printf("  circle_after_sigma.csv — after Sigma (scaling → ellipse)\n");
    printf("  circle_after_U.csv     — after U (final ellipse = A*circle)\n");
    printf("  circle_all_steps.csv   — all steps in one file\n\n");
    printf("Use plot_circle.py to visualise:\n");
    printf("  python plot_circle.py\n");
}

/* ================================================================== */
/*  DEMO 3 — Image compression                                        */
/* ================================================================== */

static void demo_compression(void)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" DEMO 3: Image Compression\n");
    printf("==========================================================\n\n");

    /* Try to load a PGM image from data/input/ */
    Matrix *img = load_pgm("data/input/test_image.pgm");

    if (img == NULL)
    {
        printf("No PGM image found. Generating synthetic 64x64 test image...\n");
        img = generate_test_image(64, 64);
        if (img == NULL) { printf("Failed to generate image\n"); return; }
        save_pgm("data/output/test_image_original.pgm", img, 255);
        printf("Saved: data/output/test_image_original.pgm\n");
    }
    else
    {
        save_pgm("data/output/test_image_original.pgm", img, 255);
    }

    run_compression_experiment(img, "data/output");

    free_matrix(img);
}

/* ================================================================== */
/*  DEMO 4 — Image denoising                                          */
/* ================================================================== */

static void demo_denoising(void)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" DEMO 4: Image Denoising\n");
    printf("==========================================================\n\n");

    /* Load or generate clean image */
    Matrix *clean = load_pgm("data/input/test_image.pgm");
    if (clean == NULL)
    {
        printf("No PGM image found. Generating synthetic 64x64 test image...\n");
        clean = generate_test_image(64, 64);
        if (clean == NULL) { printf("Failed to generate image\n"); return; }
    }

    save_pgm("data/output/denoise_clean.pgm", clean, 255);

    /* Add noise with std_dev = 25 (noticeable but not overwhelming) */
    run_denoising_experiment(clean, 25.0, "data/output");

    free_matrix(clean);
}

/* ================================================================== */
/*  DEMO 5 — Matrix unit tests                                        */
/* ================================================================== */

static void demo_tests(void)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" DEMO 5: Matrix and SVD Unit Tests\n");
    printf("==========================================================\n\n");

    int pass = 0, fail = 0;

    /* --- Test: create_matrix initialises to zero ------------------- */
    {
        Matrix *M = create_matrix(3, 4);
        int ok = 1;
        for (int i = 0; i < 3 && ok; i++)
            for (int j = 0; j < 4 && ok; j++)
                if (M->data[i][j] != 0.0) ok = 0;
        printf("create_matrix zeros   : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;
        free_matrix(M);
    }

    /* --- Test: identity_matrix ------------------------------------- */
    {
        Matrix *I = identity_matrix(4);
        int ok = 1;
        for (int i = 0; i < 4 && ok; i++)
            for (int j = 0; j < 4 && ok; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (fabs(I->data[i][j] - expected) > 1e-14) ok = 0;
            }
        printf("identity_matrix       : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;
        free_matrix(I);
    }

    /* --- Test: transpose ------------------------------------------- */
    {
        Matrix *A = create_matrix(2, 3);
        A->data[0][0]=1; A->data[0][1]=2; A->data[0][2]=3;
        A->data[1][0]=4; A->data[1][1]=5; A->data[1][2]=6;
        Matrix *T = transpose(A);
        int ok = (T->rows == 3 && T->cols == 2 &&
                  T->data[0][0]==1 && T->data[1][0]==2 &&
                  T->data[2][0]==3 && T->data[0][1]==4);
        printf("transpose             : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;
        free_matrix(A); free_matrix(T);
    }

    /* --- Test: multiply 2x3 * 3x2 --------------------------------- */
    {
        Matrix *A = create_matrix(2, 3);
        Matrix *B = create_matrix(3, 2);
        A->data[0][0]=1; A->data[0][1]=2; A->data[0][2]=3;
        A->data[1][0]=4; A->data[1][1]=5; A->data[1][2]=6;
        B->data[0][0]=7; B->data[0][1]=8;
        B->data[1][0]=9; B->data[1][1]=10;
        B->data[2][0]=11;B->data[2][1]=12;
        Matrix *C = multiply(A, B);
        /* C[0][0] = 1*7+2*9+3*11 = 7+18+33 = 58 */
        /* C[0][1] = 1*8+2*10+3*12= 8+20+36 = 64 */
        int ok = (C != NULL &&
                  fabs(C->data[0][0] - 58.0) < 1e-10 &&
                  fabs(C->data[0][1] - 64.0) < 1e-10);
        printf("multiply 2x3*3x2      : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;
        free_matrix(A); free_matrix(B); free_matrix(C);
    }

    /* --- Test: SVD of teacher's matrix ----------------------------- */
    {
        Matrix *A = create_matrix(2, 2);
        A->data[0][0]= 2; A->data[0][1]= 1;
        A->data[1][0]=-1; A->data[1][1]= 1;

        Matrix *U=NULL, *S=NULL, *Vt=NULL;
        int ok_svd = (svd(A, &U, &S, &Vt) == 0);

        /* Check reconstruction */
        int ok = 0;
        if (ok_svd)
        {
            Matrix *US   = multiply(U, S);
            Matrix *Arec = multiply(US, Vt);
            ok = (matrix_max_abs_diff(A, Arec) < TOL);
            free_matrix(US); free_matrix(Arec);
        }
        printf("SVD teacher's matrix  : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;

        /* Check orthogonality of U */
        int ok_u = 0;
        if (ok_svd)
        {
            Matrix *Ut  = transpose(U);
            Matrix *UtU = multiply(Ut, U);
            Matrix *Im  = identity_matrix(2);
            ok_u = (matrix_max_abs_diff(UtU, Im) < TOL);
            free_matrix(Ut); free_matrix(UtU); free_matrix(Im);
        }
        printf("SVD U orthogonal      : %s\n", PASS_FAIL(ok_u));
        ok_u ? pass++ : fail++;

        /* Check orthogonality of V */
        int ok_v = 0;
        if (ok_svd)
        {
            Matrix *V   = transpose(Vt);
            Matrix *VtV = multiply(Vt, V);
            Matrix *In  = identity_matrix(2);
            ok_v = (matrix_max_abs_diff(VtV, In) < TOL);
            free_matrix(V); free_matrix(VtV); free_matrix(In);
        }
        printf("SVD V orthogonal      : %s\n", PASS_FAIL(ok_v));
        ok_v ? pass++ : fail++;

        /* Check singular values are non-negative and descending */
        int ok_sigma = 0;
        if (ok_svd)
            ok_sigma = (S->data[0][0] >= S->data[1][1] &&
                        S->data[1][1] >= 0.0);
        printf("SVD singular values   : %s  (s1=%.4f >= s2=%.4f >= 0)\n",
               PASS_FAIL(ok_sigma),
               S ? S->data[0][0] : 0.0,
               S ? S->data[1][1] : 0.0);
        ok_sigma ? pass++ : fail++;

        if (U)  free_matrix(U);
        if (S)  free_matrix(S);
        if (Vt) free_matrix(Vt);
        free_matrix(A);
    }

    /* --- Test: SVD of a 3x2 rectangular matrix --------------------- */
    {
        Matrix *A = create_matrix(3, 2);
        A->data[0][0]=1; A->data[0][1]=2;
        A->data[1][0]=3; A->data[1][1]=4;
        A->data[2][0]=5; A->data[2][1]=6;

        Matrix *U=NULL, *S=NULL, *Vt=NULL;
        int ok = 0;
        if (svd(A, &U, &S, &Vt) == 0)
        {
            Matrix *US   = multiply(U, S);
            Matrix *Arec = multiply(US, Vt);
            ok = (matrix_max_abs_diff(A, Arec) < TOL);
            free_matrix(US); free_matrix(Arec);
            free_matrix(U); free_matrix(S); free_matrix(Vt);
        }
        printf("SVD 3x2 rectangular   : %s\n", PASS_FAIL(ok));
        ok ? pass++ : fail++;
        free_matrix(A);
    }

    /* --- Test: Frobenius norm --------------------------------------- */
    {
        Matrix *A = create_matrix(2, 2);
        A->data[0][0]=3; A->data[0][1]=4;  /* norm = sqrt(9+16+0+0)=5 */
        double frob = matrix_frobenius_norm(A);
        int ok = (fabs(frob - 5.0) < 1e-10);
        printf("Frobenius norm        : %s  (%.6f, expected 5.0)\n",
               PASS_FAIL(ok), frob);
        ok ? pass++ : fail++;
        free_matrix(A);
    }

    printf("\n");
    printf("Results: %d passed, %d failed\n", pass, fail);
    printf("----------------------------------------------------------\n");
}

/* ================================================================== */
/*  MAIN MENU           