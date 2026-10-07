# svd-image-processing-c
Pure C implementation of Singular Value Decomposition (SVD) with application to image compression, denoising, and unit-circle geometric transformations. Built without external linear algebra libraries.
Singular Value Decomposition and Image Compression
A complete college project demonstrating SVD theory, implementation in pure C, and its applications to image compression and denoising.
1. Introduction
Singular Value Decomposition (SVD) is one of the most powerful tools in linear algebra. It breaks any matrix into three simpler matrices that reveal its fundamental structure. This project implements SVD from scratch in C — with no external libraries — and applies it to grayscale image compression and denoising.
2. Problem Statement
Images are large. A 512×512 grayscale image requires 262,144 numbers to store. SVD lets us approximate the image using far fewer numbers while keeping most of the visual information. The key insight is that most images have low effective rank — a small number of SVD components captures most of the content.
3. Objectives
Implement SVD in C without LAPACK, BLAS, or any external library
Verify the decomposition mathematically (orthogonality, reconstruction)
Demonstrate the geometric meaning using the unit-circle experiment
Apply SVD to grayscale image compression
Apply SVD to image denoising
Provide reusable matrix and SVD modules
4. SVD Theory
What is a Matrix?
A matrix is a rectangular grid of numbers. When you multiply a matrix A by a vector x, you get a new vector Ax. The matrix transforms the vector — it can rotate it, stretch it, reflect it, or any combination.
What is SVD?
SVD says: any matrix A can be written as a product of three matrices:
A = U * Sigma * V^T
where:
Matrix
Shape
Meaning
U
m × m
Left singular vectors — orthogonal (rotation/reflection)
Sigma
m × n
Diagonal — singular values (how much stretching)
V^T
n × n
Right singular vectors — orthogonal (rotation/reflection)
"Orthogonal" means the columns are unit vectors, all perpendicular to each other. For such a matrix Q: Q^T * Q = I (the identity matrix).
What do U, Sigma, V^T do geometrically?
Apply A to a vector x in three steps:
x  →  V^T x  →  Sigma * V^T x  →  U * Sigma * V^T x  =  Ax
V^T x — rotates/reflects x into a special coordinate system
Sigma * (V^T x) — stretches each coordinate axis independently
U * (Sigma * V^T x) — rotates/reflects into the final orientation
The singular values σ₁ ≥ σ₂ ≥ ... ≥ 0 on the diagonal of Sigma tell you how much stretching happens along each axis. Large singular values = important directions. Small singular values = less important directions.
The Unit-Circle Experiment
Take every point on the unit circle: x = cos(t), y = sin(t).
After V^T: still a circle (pure rotation/reflection)
After Sigma: an axis-aligned ellipse with semi-axes σ₁ and σ₂
After U: the final ellipse (rotated/reflected)
This equals A applied directly to every point on the circle
This proves that A maps a circle to an ellipse, and the singular values are exactly the lengths of the ellipse's semi-axes.
5. Teacher's 2×2 Matrix
A = [ 2   1 ]
    [-1   1 ]
A^T A = [ 5  1 ]
**       [ 1  2 ]**
Eigenvalues of A^T A: λ = (7 ± √13) / 2
Singular values:
σ₁ = √((7+√13)/2) ≈ 2.3028
σ₂ = √((7−√13)/2) ≈ 1.3028
Verified output:
U * Sigma * V^T = A  (error ≈ 5e-16, machine precision)
U^T * U = I          (error ≈ 2e-17)
V^T * V = I          (error ≈ 2e-17)
A*v_i = sigma_i*u_i  (error = 0.00e+000)
6. SVD Algorithm — One-Sided Jacobi
We use the one-sided Jacobi algorithm:
Start with B = copy of A, V = identity matrix
For each pair of columns (p, q):
Compute α = ‖B[:,p]‖², β = ‖B[:,q]‖², γ = B[:,p]·B[:,q]
If |γ|/√(αβ) < tolerance: skip (already orthogonal)
Else: compute rotation angle using the stable formula: tau = (α−β)/(2γ),  t = sign(tau)/(|tau|+√(1+tau²))
Apply the rotation to columns p,q of B and to V
Repeat until no rotation is needed (convergence)
Singular values = column norms of converged B
U columns = B columns / singular values
Return V^T
This runs in O(n² × sweeps) per matrix. Typically converges in under 10 sweeps.
7. Image Compression
A grayscale image is stored as a matrix A where A[i][j] = pixel intensity (0–255).
The Eckart-Young theorem says the best rank-k approximation to A is:
A_k = sigma_1*u_1*v_1^T + sigma_2*u_2*v_2^T + ... + sigma_k*u_k*v_k^T
    = U_k * Sigma_k * Vt_k
Storage comparison for an m×n image with rank-k approximation:
Original: m × n values
Compressed: k×m + k + k×n = k(m+n+1) values
Compression ratio: (m×n) / (k(m+n+1))
Results on 64×64 test image:
k
Ratio
MSE
PSNR (dB)
1
31.8×
1401.9
16.7
5
6.35×
29.4
33.5
10
3.18×
~0
~293
20
1.59×
~0
~293
64
0.5×
0
∞ (exact)
For this synthetic test image, rank 10 already gives near-perfect reconstruction. Real photographic images typically need k=20–50 for visually good quality.
8. Image Denoising
Gaussian noise added to a clean image (std_dev=25) gives a noisy image with PSNR ≈ 20.3 dB. SVD denoising keeps only the top-k singular values:
Results:
k
MSE
PSNR (dB)
Improvement
5
127.6
27.1
+6.8 dB
10
234.4
24.4
+4.1 dB
20
410.6
22.0
+1.7 dB
64
608.7
20.3
0 dB (none)
Low k removes the most noise but also blurs detail. k=5 gives the best noise reduction on this synthetic image. The optimal k depends on image content and noise level.
9. Project Structure
SVD_Project/
├── src/
│   ├── main.c           — menu-driven demo program
│   ├── matrix.c / .h    — matrix library
│   ├── svd.c / .h       — SVD algorithm (one-sided Jacobi)
│   ├── image.c / .h     — PGM image load/save, noise, test image
│   ├── compression.c/.h — SVD image compression
│   └── denoise.c / .h   — SVD image denoising
├── tests/
│   ├── test_matrix.c    — 30 matrix library unit tests
│   └── test_svd.c       — 38 SVD unit tests
├── data/
│   ├── input/           — input PGM images
│   └── output/          — generated PGM images and CSV files
├── results/             — place experiment results here
├── docs/                — place report/slides here
├── plot_circle.py       — Python script to plot CSV files (optional)
├── generate_test_pgm.c  — standalone tool to create test PGM
├── Makefile
└── README.md
10. Compilation
Compile everything:
gcc src/main.c src/matrix.c src/svd.c src/image.c src/compression.c src/denoise.c \
    -o svd_project -Wall -std=c99 -Isrc -lm
Matrix tests:
gcc tests/test_matrix.c src/matrix.c -o test_matrix -Wall -std=c99 -Isrc -lm
SVD tests:
gcc tests/test_svd.c src/matrix.c src/svd.c -o test_svd -Wall -std=c99 -Isrc -lm
Generate test PGM image (run once):
gcc generate_test_pgm.c -o gen_pgm -lm
./gen_pgm
On Windows/MinGW, replace ./ with the executable name directly, e.g. gen_pgm.exe.
11. Running
# Main interactive program
./svd_project

# Menu options:
#   1 — SVD of teacher's 2x2 matrix + full verification
#   2 — Unit-circle experiment (writes CSV files to data/output/)
#   3 — Image compression experiment
#   4 — Image denoising experiment
#   5 — Run all unit tests
#   6 — Run everything
#   0 — Exit

# Standalone tests
./test_matrix
./test_svd

# Visualise unit-circle results (requires matplotlib)
python plot_circle.py
12. Makefile Note
The Makefile uses Unix rm -f for the clean target, which does not work on Windows CMD. Use the explicit gcc commands above instead, or run make from Git Bash / WSL on Windows.
13. Verification Results
All tests pass with errors at or below machine precision (~1e-16):
Test
Result
Max Error
U^T U = I
PASS
2.18e-17
V^T V = I
PASS
1.95e-17
U Σ V^T = A
PASS
4.44e-16
A vᵢ = σᵢ uᵢ
PASS
0.00e+00
Matrix tests
30/30 PASS
—
SVD tests
38/38 PASS
—
14. Numerical Notes
Floating-point errors: operations like multiply and sqrt accumulate small rounding errors (~1e-15). This is normal and does not affect results.
Sign ambiguity: if (U, Σ, V^T) is a valid SVD, so is (−U[:,k], Σ, −V^T[k,:]). The reconstruction A = UΣV^T is the same either way. Verification checks both +uᵢ and −uᵢ to handle this correctly.
Convergence: the Jacobi algorithm converges for all real matrices. We run up to 150 sweeps with tolerance 1e-12.
Zero singular values: handled by Gram-Schmidt completion of U.
15. Limitations
Performance: O(n³) per SVD. Suitable for images up to ~200×200. For larger images, use LAPACK in a production environment.
The SVD module stores full m×m U and n×n V^T matrices. A thin-SVD variant would be more memory-efficient.
Denoising assumption: small singular values ≈ noise. This is an approximation; fine image detail also contributes to small singular values.
16. Future Scope
Implement thin/economy SVD for rectangular images
Add colour image support (process each R/G/B channel separately)
Implement iterative/randomised SVD for very large images
Add JPEG-style block SVD (process 8×8 or 16×16 tiles)
Provide a Python wrapper around the C SVD core
17. References
Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.
Strang, G. (2016). Introduction to Linear Algebra (5th ed.). Wellesley-Cambridge Press.
Forsythe, G. E., Malcolm, M. A., & Moler, C. B. (1977). Computer Methods for Mathematical Computations. Prentice-Hall.
Press, W. H., et al. (2007). Numerical Recipes in C (3rd ed.). Cambridge University Press.
Wikipedia — Singular value decomposition: https://en.wikipedia.org/wiki/Singular_value_decomposition
Wikipedia — Jacobi eigenvalue algorithm: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
PGM format specification: http://netpbm.sourceforge.net/doc/pgm.html
18. Conclusion
This project implements SVD entirely in C using the one-sided Jacobi algorithm, verifies it at machine precision, and applies it to image compression and denoising. All 68 tests pass. The code is modular, well-commented, and suitable for use in a college viva or presentation.