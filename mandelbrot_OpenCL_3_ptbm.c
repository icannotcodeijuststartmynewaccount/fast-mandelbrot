/**
 * Ultimate Mandelbrot Set Renderer
 * ================================
 * Features:
 * 1. Multi-architecture: Scalar, SSE4, AVX2
 * 2. GPU offloading via OpenCL (optional)
 * 3. Automatic CPU feature detection
 * 4. Grayscale and color modes
 * 5. Configurable from command line
 * 6. Performance benchmarking
 * 
 * Build: see compile_instructions.txt
 * Usage: mandelbrot [options]
 */
#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <cpuid.h>
#else
#include <intrin.h>
#include <windows.h>
#endif
#define CL_TARGET_OPENCL_VERSION 300
/* ===========================================
   Configuration System
   =========================================== */

typedef struct {
    // Image settings
    int width;
    int height;
    int max_iter;
    
    // Viewport
    long double x_min;
    long double x_max;
    long double y_min;
    long double y_max;
    
    // Performance settings
    int use_gpu;
    int use_avx2;
    int use_sse4;
    int num_threads;
    
    // Output settings
    int grayscale;
    int benchmark;
    const char* output_file;

    // Color settings
    float gamma;
    float brightness;

    // Internal
    int cpu_features;
} Config;

/* ===========================================
   Perturbation Math Configuration
   =========================================== */

typedef enum {
    PTBM_PRECISION_AUTO = 0,    // Auto-select based on zoom
    PTBM_PRECISION_SINGLE = 1,   // Force single precision (float)
    PTBM_PRECISION_DOUBLE = 2,   // Force double precision
    PTBM_PRECISION_MIXED = 3,    // Mixed precision (double for ref, float for delta)
    PTBM_PRECISION_HIGH = 4      // Use long double for extreme zooms
} PtbmPrecision;

typedef struct {
    int enabled;                  // Enable perturbation math
    PtbmPrecision precision;      // Precision mode
    long double reference_x;           // Reference point X
    long double reference_y;           // Reference point Y
    int reference_iters;          // Reference iterations
    double zoom_level;            // Current zoom level
    double error_bound;           // Error bound for approximation
    int use_glitch_correction;    // Enable glitch detection/correction
    int max_refinement_iter;      // Max refinement iterations
} PerturbationConfig;

/* ===========================================
   Function Prototypes - ADD THIS SECTION
   =========================================== */

// Forward declarations for perturbation functions
static void render_mandelbrot_perturbation(Config* cfg, PerturbationConfig* ptbm, uint8_t* image);
static void calculate_reference_point(Config* cfg, PerturbationConfig* ptbm);
static int calculate_reference_iterations(double cx, double cy, int max_iter);
static void detect_and_correct_glitches(Config* cfg, PerturbationConfig* ptbm, uint8_t* image, int* iteration_counts);

// Perturbation math functions
static inline int mandelbrot_perturbation_single(double cx, double cy, double ref_x, double ref_y, int ref_iters, int max_iter, double error_bound);
static inline int mandelbrot_perturbation_double(double cx, double cy, double ref_x, double ref_y, int ref_iters, int max_iter, double error_bound);
static inline int mandelbrot_perturbation_mixed(double cx, double cy, double ref_x, double ref_y, int ref_iters, int max_iter, double error_bound);
static inline int mandelbrot_perturbation_high(double cx, double cy, double ref_x, double ref_y, int ref_iters, int max_iter, double error_bound);

/* ===========================================
   CPU Feature Detection
   =========================================== */

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

typedef enum {
    CPU_FEATURE_NONE   = 0,
    CPU_FEATURE_SSE    = 1 << 0,
    CPU_FEATURE_SSE2   = 1 << 1,
    CPU_FEATURE_SSE3   = 1 << 2,
    CPU_FEATURE_SSSE3  = 1 << 3,
    CPU_FEATURE_SSE4_1 = 1 << 4,
    CPU_FEATURE_SSE4_2 = 1 << 5,
    CPU_FEATURE_AVX    = 1 << 6,
    CPU_FEATURE_AVX2   = 1 << 7,
    CPU_FEATURE_FMA    = 1 << 8
} CPUFeature;

static int detect_cpu_features(void) {
    int features = CPU_FEATURE_NONE;
    
#ifdef _WIN32
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    
    if (cpu_info[2] & (1 << 20)) features |= CPU_FEATURE_SSE4_2;
    if (cpu_info[2] & (1 << 19)) features |= CPU_FEATURE_SSE4_1;
    if (cpu_info[2] & (1 << 9))  features |= CPU_FEATURE_SSSE3;
    if (cpu_info[2] & (1 << 0))  features |= CPU_FEATURE_SSE3;
    if (cpu_info[3] & (1 << 26)) features |= CPU_FEATURE_SSE2;
    if (cpu_info[3] & (1 << 25)) features |= CPU_FEATURE_SSE;
    if (cpu_info[2] & (1 << 28)) features |= CPU_FEATURE_AVX;
    if (cpu_info[2] & (1 << 12)) features |= CPU_FEATURE_FMA;
    
    __cpuidex(cpu_info, 7, 0);
    if (cpu_info[1] & (1 << 5)) features |= CPU_FEATURE_AVX2;
#else
    unsigned int eax, ebx, ecx, edx;
    
    __cpuid(1, eax, ebx, ecx, edx);
    if (ecx & (1 << 20)) features |= CPU_FEATURE_SSE4_2;
    if (ecx & (1 << 19)) features |= CPU_FEATURE_SSE4_1;
    if (ecx & (1 << 9))  features |= CPU_FEATURE_SSSE3;
    if (ecx & (1 << 0))  features |= CPU_FEATURE_SSE3;
    if (edx & (1 << 26)) features |= CPU_FEATURE_SSE2;
    if (edx & (1 << 25)) features |= CPU_FEATURE_SSE;
    if (ecx & (1 << 28)) features |= CPU_FEATURE_AVX;
    if (ecx & (1 << 12)) features |= CPU_FEATURE_FMA;
    
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 5)) features |= CPU_FEATURE_AVX2;
#endif
    
    return features;
}

/* ===========================================
   SIMD Headers (Conditional)
   =========================================== */

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ===========================================
   Scalar Implementation (Always Available)
   =========================================== */

static inline int mandelbrot_scalar(double x0, double y0, int max_iter) {
    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    int iter = 0;
    
    // Pixels escape when r² >= 4.0, so continue while r² < 4.0
    while (x2 + y2 < 4.0 && iter < max_iter) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }
    
    return iter;
}

/* ===========================================
   SSE4 Implementation (if available)
   =========================================== */

#ifdef __SSE4_2__
static inline void mandelbrot_sse4(__m128d x0, __m128d y0, int max_iter, int results[2]) {
    __m128d x = _mm_setzero_pd();
    __m128d y = _mm_setzero_pd();
    __m128d x2 = _mm_setzero_pd();
    __m128d y2 = _mm_setzero_pd();
    
    __m128d two = _mm_set1_pd(2.0);
    __m128d four = _mm_set1_pd(4.0);
    __m128d iter_counts = _mm_setzero_pd();
    __m128d increment = _mm_set1_pd(1.0);
    __m128d active = _mm_set1_pd(1.0);
    
    int iter = 0;
    
    while (iter < max_iter) {
        int active_mask = _mm_movemask_pd(active);
        if (!active_mask) break;
        
        // Calculate r² from previous iteration
        __m128d r2 = _mm_add_pd(x2, y2);
        
        // Update active mask based on OLD r²
        __m128d still_active = _mm_cmplt_pd(r2, four);
        active = _mm_and_pd(active, still_active);
        
        // Increment counters for active pixels
        iter_counts = _mm_add_pd(iter_counts, _mm_and_pd(active, increment));
        
        // Calculate new values
        __m128d xy = _mm_mul_pd(x, y);
        y = _mm_add_pd(_mm_mul_pd(two, xy), y0);
        x = _mm_add_pd(_mm_sub_pd(x2, y2), x0);
        
        x2 = _mm_mul_pd(x, x);
        y2 = _mm_mul_pd(y, y);
        
        iter++;
    }
    
    double iter_array[2];
    _mm_store_pd(iter_array, iter_counts);
    
    results[0] = (int)iter_array[0];
    results[1] = (int)iter_array[1];
    if (results[0] > max_iter) results[0] = max_iter;
    if (results[1] > max_iter) results[1] = max_iter;
}
#endif

/* ===========================================
   AVX2 Implementation (if available)
   =========================================== */
#ifdef __AVX2__
#include <immintrin.h>

static inline void mandelbrot_avx2(
    __m256d x0,
    __m256d y0,
    int max_iter,
    int results[4]
) {
    __m256d x  = _mm256_setzero_pd();
    __m256d y  = _mm256_setzero_pd();
    __m256d x2 = _mm256_setzero_pd();
    __m256d y2 = _mm256_setzero_pd();

    __m256d two  = _mm256_set1_pd(2.0);
    __m256d four = _mm256_set1_pd(4.0);

    __m256i counters = _mm256_setzero_si256();   // 4 x 64-bit counters
    __m256i one      = _mm256_set1_epi64x(1);

    for (int iter = 0; iter < max_iter; iter++) {

        __m256d r2 = _mm256_add_pd(x2, y2);

        // mask = r2 < 4.0
        __m256d mask_pd = _mm256_cmp_pd(r2, four, _CMP_LT_OQ);

        int active_mask = _mm256_movemask_pd(mask_pd);
        if (active_mask == 0)
            break;

        // convert mask to integer mask for counter increment
        __m256i mask_i = _mm256_castpd_si256(mask_pd);

        // increment counters only for active lanes
        counters = _mm256_add_epi64(
            counters,
            _mm256_and_si256(mask_i, one)
        );

        // Compute next iteration
        __m256d xy = _mm256_mul_pd(x, y);

        // y = 2xy + y0
        y = _mm256_fmadd_pd(two, xy, y0);

        // x = x² - y² + x0
        x = _mm256_add_pd(_mm256_sub_pd(x2, y2), x0);

        x2 = _mm256_mul_pd(x, x);
        y2 = _mm256_mul_pd(y, y);
    }

    // Store final results
    long long tmp[4];
    _mm256_storeu_si256((__m256i*)tmp, counters);

    for (int i = 0; i < 4; i++)
        results[i] = (int)tmp[i];
}
#endif

/* ===========================================
   Color Management
   =========================================== */
static inline void get_color(int iter, int max_iter, int grayscale, 
                            float gamma, float brightness,
                            uint8_t* r, uint8_t* g, uint8_t* b) {
    if (iter == max_iter) {
        *r = *g = *b = 0;  // Inside points are black
        return;
    }
    
    // Normalize iteration count
    double t = (double)iter / max_iter;
    
    if (grayscale) {
        // Use configurable gamma and brightness
        double value = pow(t, gamma) * brightness;
        if (value > 1.0) value = 1.0;
        if (value < 0.0) value = 0.0;
        
        uint8_t grey = (uint8_t)(value * 255.0);
        *r = *g = *b = grey;
    } else {
        // Enhanced color gradient with configurable brightness
        double sqrt_t = sqrt(t);
        
        // Apply brightness to colors
        double r_val = (0.5 + 0.5 * sin(sqrt_t * 2.0 * 3.141592653589793 + 0.0)) * brightness;
        double g_val = (0.5 + 0.5 * sin(sqrt_t * 2.0 * 3.141592653589793 + 2.0 * 3.141592653589793 / 3.0)) * brightness;
        double b_val = (0.5 + 0.5 * sin(sqrt_t * 2.0 * 3.141592653589793 + 4.0 * 3.141592653589793 / 3.0)) * brightness;
        
        // Clamp to [0, 1]
        if (r_val > 1.0) r_val = 1.0;
        if (g_val > 1.0) g_val = 1.0;
        if (b_val > 1.0) b_val = 1.0;
        if (r_val < 0.0) r_val = 0.0;
        if (g_val < 0.0) g_val = 0.0;
        if (b_val < 0.0) b_val = 0.0;
        
        *r = (uint8_t)(r_val * 255);
        *g = (uint8_t)(g_val * 255);
        *b = (uint8_t)(b_val * 255);
    }
}

/* ===========================================
   CORRECTED PERTURBATION MATH - TESTED VERSION
   =========================================== */

// Calculate reference point iterations (standard Mandelbrot)
static int calculate_reference_iterations(double cx, double cy, int max_iter) {
    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    int iter = 0;
    
    while (x2 + y2 < 4.0 && iter < max_iter) {
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }
    
    return iter;
}

// Calculate reference point (find a good escaping point) - CLEAN VERSION
static void calculate_reference_point(Config* cfg, PerturbationConfig* ptbm) {
    // Calculate zoom level (needed for both auto and manual)
    double viewport_width = (double)(cfg->x_max - cfg->x_min);
    ptbm->zoom_level = 4.0 / viewport_width;
    
    // Check if user provided a reference point (manual mode)
    if (ptbm->reference_x != 0.0 || ptbm->reference_y != 0.0) {
        printf("Manual reference point mode\n");
        printf("User reference: (%.21Lf, %.21Lf)\n", ptbm->reference_x, ptbm->reference_y);
        
        // Calculate iterations for user reference
        ptbm->reference_iters = calculate_reference_iterations(
            (double)ptbm->reference_x, (double)ptbm->reference_y, cfg->max_iter);
        
        printf("Reference iterations: %d\n", ptbm->reference_iters);
        
        // Set precision mode based on zoom level if auto
        if (ptbm->precision == PTBM_PRECISION_AUTO) {
            if (ptbm->zoom_level > 1e15) {
                ptbm->precision = PTBM_PRECISION_HIGH;
                ptbm->use_glitch_correction = 1;
            } else if (ptbm->zoom_level > 1e10) {
                ptbm->precision = PTBM_PRECISION_MIXED;
                ptbm->use_glitch_correction = 1;
            } else if (ptbm->zoom_level > 1e6) {
                ptbm->precision = PTBM_PRECISION_DOUBLE;
            } else {
                ptbm->precision = PTBM_PRECISION_SINGLE;
            }
        }
        return;
    }
    
    // AUTO MODE - find a good reference point
    printf("Auto reference point mode\n");
    
    // Start with center
    long double test_x = (cfg->x_min + cfg->x_max) / 2.0L;
    long double test_y = (cfg->y_min + cfg->y_max) / 2.0L;
    
    int iter = calculate_reference_iterations((double)test_x, (double)test_y, cfg->max_iter);
    
    if (iter == cfg->max_iter) {
        printf("Center is inside the set, searching for better reference...\n");
        
        // Try points at the corners and edges
        long double candidates[][2] = {
    {cfg->x_min, cfg->y_min},
    {cfg->x_min, cfg->y_max},
    {cfg->x_max, cfg->y_min},
    {cfg->x_max, cfg->y_max},
    {cfg->x_min, (cfg->y_min + cfg->y_max) / 2.0L},
    {cfg->x_max, (cfg->y_min + cfg->y_max) / 2.0L},
    {(cfg->x_min + cfg->x_max) / 2.0L, cfg->y_min},
    {(cfg->x_min + cfg->x_max) / 2.0L, cfg->y_max}
};

int best_iter = -1;  // Start with -1 (no valid reference yet)
long double best_x = test_x;
long double best_y = test_y;

for (int i = 0; i < 8; i++) {
    int titer = calculate_reference_iterations((double)candidates[i][0], (double)candidates[i][1], cfg->max_iter);
    // Look for iterations that escape (titer < max_iter) and are HIGH
    if (titer < cfg->max_iter && titer > best_iter) {
        best_iter = titer;
        best_x = candidates[i][0];
        best_y = candidates[i][1];
    }
}

// If still no good reference found (all points inside set), use center
if (best_iter == -1) {
    best_iter = calculate_reference_iterations((double)test_x, (double)test_y, cfg->max_iter);
    best_x = test_x;
    best_y = test_y;
    printf("Warning: No escaping reference found, using center (may cause issues)\n");
}

ptbm->reference_x = best_x;
ptbm->reference_y = best_y;
ptbm->reference_iters = best_iter;
printf("Auto reference point: (%.21Lf, %.21Lf) with %d iterations\n", 
       best_x, best_y, best_iter);
    } else {
        ptbm->reference_x = test_x;
        ptbm->reference_y = test_y;
        ptbm->reference_iters = iter;
        printf("Center escapes: using (%.21Lf, %.21Lf) with %d iterations\n", 
               test_x, test_y, iter);
    }
    
    // Set precision mode based on zoom level if auto
    if (ptbm->precision == PTBM_PRECISION_AUTO) {
        if (ptbm->zoom_level > 1e15) {
            ptbm->precision = PTBM_PRECISION_HIGH;
            ptbm->use_glitch_correction = 1;
        } else if (ptbm->zoom_level > 1e10) {
            ptbm->precision = PTBM_PRECISION_MIXED;
            ptbm->use_glitch_correction = 1;
        } else if (ptbm->zoom_level > 1e6) {
            ptbm->precision = PTBM_PRECISION_DOUBLE;
        } else {
            ptbm->precision = PTBM_PRECISION_SINGLE;
        }
    }
}

// Double precision perturbation - CORRECT VERSION
static inline int mandelbrot_perturbation_double(
    double cx, double cy,
    double ref_x, double ref_y, int ref_iters,
    int max_iter, double error_bound) {
    
    (void)error_bound;  // Not needed for basic implementation
    
    // Delta from reference
    double dx = cx - ref_x;
    double dy = cy - ref_y;
    
    // Reference orbit
    double X = 0.0, Y = 0.0;
    double X2 = 0.0, Y2 = 0.0;
    
    int iter = 0;
    
    while (iter < ref_iters && iter < max_iter) {
        // Calculate actual point
        double x = X + dx;
        double y = Y + dy;
        double r2 = x*x + y*y;
        
        if (r2 >= 4.0) {
            break;
        }
        
        // Update delta using perturbation formula
        // Δz_{n+1} = 2*Z_n*Δz_n + Δz_n^2 + Δc
        double new_dx = 2.0 * X * dx - 2.0 * Y * dy + dx*dx - dy*dy + (cx - ref_x);
        double new_dy = 2.0 * X * dy + 2.0 * Y * dx + 2.0 * dx * dy + (cy - ref_y);
        
        // Update reference
        double new_X = X2 - Y2 + ref_x;
        double new_Y = 2.0 * X * Y + ref_y;
        
        dx = new_dx;
        dy = new_dy;
        X = new_X;
        Y = new_Y;
        X2 = X*X;
        Y2 = Y*Y;
        
        iter++;
    }
    
    return iter;
}

// High precision perturbation - CORRECT VERSION
static inline int mandelbrot_perturbation_high(
    double cx, double cy,
    double ref_x, double ref_y, int ref_iters,
    int max_iter, double error_bound) {
    
    (void)error_bound;
    
    // Delta from reference
    long double dx = (long double)cx - (long double)ref_x;
    long double dy = (long double)cy - (long double)ref_y;
    long double dcx = dx;
    long double dcy = dy;
    
    // Reference orbit
    long double X = 0.0L, Y = 0.0L;
    long double X2 = 0.0L, Y2 = 0.0L;
    long double ref_xl = (long double)ref_x;
    long double ref_yl = (long double)ref_y;
    
    int iter = 0;
    
    while (iter < ref_iters && iter < max_iter) {
        // Calculate actual point
        long double x = X + dx;
        long double y = Y + dy;
        long double r2 = x*x + y*y;
        
        if (r2 >= 4.0L) {
            break;
        }
        
        // Update delta
        long double new_dx = 2.0L * X * dx - 2.0L * Y * dy + dx*dx - dy*dy + dcx;
        long double new_dy = 2.0L * X * dy + 2.0L * Y * dx + 2.0L * dx * dy + dcy;
        
        // Update reference
        long double new_X = X2 - Y2 + ref_xl;
        long double new_Y = 2.0L * X * Y + ref_yl;
        
        dx = new_dx;
        dy = new_dy;
        X = new_X;
        Y = new_Y;
        X2 = X*X;
        Y2 = Y*Y;
        
        iter++;
    }
    
    return iter;
}

// Mixed precision - CORRECT VERSION
static inline int mandelbrot_perturbation_mixed(
    double cx, double cy,
    double ref_x, double ref_y, int ref_iters,
    int max_iter, double error_bound) {
    
    (void)error_bound;
    
    // Delta in float
    float dx = (float)(cx - ref_x);
    float dy = (float)(cy - ref_y);
    float dcx = dx;
    float dcy = dy;
    
    // Reference in double
    double X = 0.0, Y = 0.0;
    double X2 = 0.0, Y2 = 0.0;
    
    int iter = 0;
    
    while (iter < ref_iters && iter < max_iter) {
        // Calculate actual point (mix double and float)
        double x = X + (double)dx;
        double y = Y + (double)dy;
        double r2 = x*x + y*y;
        
        if (r2 >= 4.0) {
            break;
        }
        
        // Update delta in float
        float new_dx = 2.0f * (float)X * dx - 2.0f * (float)Y * dy + dx*dx - dy*dy + dcx;
        float new_dy = 2.0f * (float)X * dy + 2.0f * (float)Y * dx + 2.0f * dx * dy + dcy;
        
        // Update reference in double
        double new_X = X2 - Y2 + ref_x;
        double new_Y = 2.0 * X * Y + ref_y;
        
        dx = new_dx;
        dy = new_dy;
        X = new_X;
        Y = new_Y;
        X2 = X*X;
        Y2 = Y*Y;
        
        iter++;
    }
    
    return iter;
}

// Single precision - CORRECT VERSION
static inline int mandelbrot_perturbation_single(
    double cx, double cy,
    double ref_x, double ref_y, int ref_iters,
    int max_iter, double error_bound) {
    
    (void)error_bound;
    
    // Everything in float
    float dx = (float)(cx - ref_x);
    float dy = (float)(cy - ref_y);
    float dcx = dx;
    float dcy = dy;
    
    float X = 0.0f, Y = 0.0f;
    float X2 = 0.0f, Y2 = 0.0f;
    float ref_xf = (float)ref_x;
    float ref_yf = (float)ref_y;
    
    int iter = 0;
    
    while (iter < ref_iters && iter < max_iter) {
        float x = X + dx;
        float y = Y + dy;
        float r2 = x*x + y*y;
        
        if (r2 >= 4.0f) {
            break;
        }
        
        float new_dx = 2.0f * X * dx - 2.0f * Y * dy + dx*dx - dy*dy + dcx;
        float new_dy = 2.0f * X * dy + 2.0f * Y * dx + 2.0f * dx * dy + dcy;
        
        float new_X = X2 - Y2 + ref_xf;
        float new_Y = 2.0f * X * Y + ref_yf;
        
        dx = new_dx;
        dy = new_dy;
        X = new_X;
        Y = new_Y;
        X2 = X*X;
        Y2 = Y*Y;
        
        iter++;
    }
    
    return iter;
}

/* ===========================================
   Glitch Detection and Correction
   =========================================== */

typedef struct {
    int x, y;
    int iter;
    double cx, cy;
} GlitchSample;

static void detect_and_correct_glitches(
    Config* cfg, 
    PerturbationConfig* ptbm,
    uint8_t* image,
    int* iteration_counts) {
    
    (void)ptbm;  // Suppress unused parameter warning
    
    // Simple glitch detection: check for discontinuities
    int glitch_count = 0;
    GlitchSample* glitches = malloc(cfg->width * cfg->height * sizeof(GlitchSample));
    
    for (int y = 1; y < cfg->height - 1; y++) {
        for (int x = 1; x < cfg->width - 1; x++) {
            int idx = y * cfg->width + x;
            int iter = iteration_counts[idx];
            
            // Check neighbors for large differences
            int neighbor_sum = iteration_counts[(y-1)*cfg->width + x] +
                               iteration_counts[(y+1)*cfg->width + x] +
                               iteration_counts[y*cfg->width + (x-1)] +
                               iteration_counts[y*cfg->width + (x+1)];
            
            int avg_neighbor = neighbor_sum / 4;
            
            if (abs(iter - avg_neighbor) > cfg->max_iter / 10) {
                // Potential glitch, mark for recalculation
                glitches[glitch_count].x = x;
                glitches[glitch_count].y = y;
                glitches[glitch_count].iter = iter;
                
                double dx = (cfg->x_max - cfg->x_min) / cfg->width;
                double dy = (cfg->y_max - cfg->y_min) / cfg->height;
                glitches[glitch_count].cx = cfg->x_min + x * dx;
                glitches[glitch_count].cy = cfg->y_min + y * dy;
                
                glitch_count++;
            }
        }
    }
    
    // Recalculate glitched pixels with higher precision
    if (glitch_count > 0) {
        printf("Detected %d potential glitches, recalculating...\n", glitch_count);
        
        #pragma omp parallel for
        for (int g = 0; g < glitch_count; g++) {
            int x = glitches[g].x;
            int y = glitches[g].y;
            double cx = glitches[g].cx;
            double cy = glitches[g].cy;
            
            // Force direct calculation for glitched pixels
            int new_iter = mandelbrot_scalar(cx, cy, cfg->max_iter);
            
            iteration_counts[y * cfg->width + x] = new_iter;
            
            // Update image
            int idx = (y * cfg->width + x) * 3;
            uint8_t r, g, b;
            get_color(new_iter, cfg->max_iter, cfg->grayscale,
                      cfg->gamma, cfg->brightness, &r, &g, &b);
            image[idx] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
    
    free(glitches);
}

/* ===========================================
   GPU OpenCL Support (Optional)
   =========================================== */

#ifdef USE_OPENCL
#include <CL/cl.h>

/* ===========================================
   OpenCL GPU Kernel Source Code
   =========================================== */
static const char* mandelbrot_kernel_source = 
"// Mandelbrot GPU Kernel\n"
"// Optimized for OpenCL 1.2+\n"
"\n"
"#ifndef M_PI\n"
"#define M_PI 3.14159265358979323846\n"
"#endif\n"
"\n"
"// Main kernel\n"
"#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
"\n"
"__attribute__((intel_reqd_sub_group_size(32)))\n"
"__kernel void mandelbrot_gpu(\n"
"    __global uchar* output,\n"
"    const double x_min,\n"
"    const double x_max,\n"
"    const double y_min,\n"
"    const double y_max,\n"
"    const int width,\n"
"    const int height,\n"
"    const int max_iter,\n"
"    const int grayscale,\n"
"    const float gamma,\n"
"    const float brightness)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    \n"
"    if (x >= width || y >= height) return;\n"
"    \n"
"    double x0 = x_min + (x_max - x_min) * x / width;\n"
"    double y0 = y_min + (y_max - y_min) * y / height;\n"
"    \n"
"    // Fast bailout for main cardioid and period-2 bulb\n"
"    double q = (x0 - 0.25) * (x0 - 0.25) + y0 * y0;\n"
"    if (q * (q + (x0 - 0.25)) < 0.25 * y0 * y0) {\n"
"        int idx = (y * width + x) * 3;\n"
"        output[idx] = 0;\n"
"        output[idx + 1] = 0;\n"
"        output[idx + 2] = 0;\n"
"        return;\n"
"    }\n"
"    \n"
"    double x_plus1 = x0 + 1.0;\n"
"    if (x_plus1 * x_plus1 + y0 * y0 < 0.0625) {\n"
"        int idx = (y * width + x) * 3;\n"
"        output[idx] = 0;\n"
"        output[idx + 1] = 0;\n"
"        output[idx + 2] = 0;\n"
"        return;\n"
"    }\n"
"    \n"
"    double zx = 0.0, zy = 0.0;\n"
"    double zx2 = 0.0, zy2 = 0.0;\n"
"    int iter = 0;\n"
"    \n"
"    while (zx2 + zy2 <= 4.0 && iter < max_iter) {\n"
"        zy = 2.0 * zx * zy + y0;\n"
"        zx = zx2 - zy2 + x0;\n"
"        zx2 = zx * zx;\n"
"        zy2 = zy * zy;\n"
"        iter++;\n"
"    }\n"
"    \n"
"    float t = (float)iter / max_iter;\n"
"    int idx = (y * width + x) * 3;\n"
"    \n"
"    if (iter == max_iter) {\n"
"        output[idx] = 0;\n"
"        output[idx + 1] = 0;\n"
"        output[idx + 2] = 0;\n"
"    } else if (grayscale) {\n"
"        // Use configurable gamma and brightness\n"
"        float value = pow(t, gamma) * brightness;\n"
"        if (value > 1.0f) value = 1.0f;\n"
"        if (value < 0.0f) value = 0.0f;\n"
"        \n"
"        uchar grey = (uchar)(value * 255.0f);\n"
"        output[idx] = grey;\n"
"        output[idx + 1] = grey;\n"
"        output[idx + 2] = grey;\n"
"    } else {\n"
"        // Color mode with brightness control\n"
"        float sqrt_t = sqrt(t);\n"
"        \n"
"        // Apply brightness to colors\n"
"        float r_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI + 0.0f)) * brightness;\n"
"        float g_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI + 2.0f * M_PI / 3.0f)) * brightness;\n"
"        float b_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI + 4.0f * M_PI / 3.0f)) * brightness;\n"
"        \n"
"        // Clamp to [0, 1]\n"
"        if (r_val > 1.0f) r_val = 1.0f;\n"
"        if (g_val > 1.0f) g_val = 1.0f;\n"
"        if (b_val > 1.0f) b_val = 1.0f;\n"
"        if (r_val < 0.0f) r_val = 0.0f;\n"
"        if (g_val < 0.0f) g_val = 0.0f;\n"
"        if (b_val < 0.0f) b_val = 0.0f;\n"
"        \n"
"        output[idx] = (uchar)(r_val * 255.0f);\n"
"        output[idx + 1] = (uchar)(g_val * 255.0f);\n"
"        output[idx + 2] = (uchar)(b_val * 255.0f);\n"
"    }\n"
"}";

/* ===========================================
   FIXED GPU Perturbation Kernel - Continue after reference escapes
   =========================================== */
static const char* mandelbrot_perturbation_kernel_source = 
"// Mandelbrot GPU Perturbation Kernel - FIXED\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"\n"
"__kernel void mandelbrot_perturbation_gpu(\n"
"    __global uchar* output,\n"
"    const double ref_x,\n"
"    const double ref_y,\n"
"    const int ref_iters,\n"
"    const double x_min,\n"
"    const double x_max,\n"
"    const double y_min,\n"
"    const double y_max,\n"
"    const int width,\n"
"    const int height,\n"
"    const int max_iter,\n"
"    const int grayscale,\n"
"    const float gamma,\n"
"    const float brightness)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"    \n"
"    double cx = x_min + (x_max - x_min) * x / width;\n"
"    double cy = y_min + (y_max - y_min) * y / height;\n"
"    \n"
"    // Delta from reference\n"
"    double dx = cx - ref_x;\n"
"    double dy = cy - ref_y;\n"
"    double dcx = dx;\n"
"    double dcy = dy;\n"
"    \n"
"    // Reference orbit\n"
"    double X = 0.0, Y = 0.0;\n"
"    double X2 = 0.0, Y2 = 0.0;\n"
"    \n"
"    int iter = 0;\n"
"    \n"
"    // IMPORTANT: Continue to max_iter, not ref_iters!\n"
"    // Reference may escape early, but perturbation continues\n"
"    while (iter < max_iter) {\n"
"        // Calculate actual point\n"
"        double zx = X + dx;\n"
"        double zy = Y + dy;\n"
"        double r2 = zx*zx + zy*zy;\n"
"        \n"
"        if (r2 >= 4.0) {\n"
"            break;\n"
"        }\n"
"        \n"
"        // Update delta (perturbation formula)\n"
"        double new_dx = 2.0 * X * dx - 2.0 * Y * dy + dx*dx - dy*dy + dcx;\n"
"        double new_dy = 2.0 * X * dy + 2.0 * Y * dx + 2.0 * dx * dy + dcy;\n"
"        \n"
"        // Update reference (even if it escaped, continue the math)\n"
"        double new_X = X2 - Y2 + ref_x;\n"
"        double new_Y = 2.0 * X * Y + ref_y;\n"
"        \n"
"        dx = new_dx;\n"
"        dy = new_dy;\n"
"        X = new_X;\n"
"        Y = new_Y;\n"
"        X2 = X*X;\n"
"        Y2 = Y*Y;\n"
"        \n"
"        iter++;\n"
"    }\n"
"    \n"
"    // Color calculation\n"
"    float t = (float)iter / max_iter;\n"
"    int idx = (y * width + x) * 3;\n"
"    \n"
"    if (iter == max_iter) {\n"
"        output[idx] = 0;\n"
"        output[idx+1] = 0;\n"
"        output[idx+2] = 0;\n"
"    } else if (grayscale) {\n"
"        float value = pow(t, gamma) * brightness;\n"
"        if (value > 1.0f) value = 1.0f;\n"
"        uchar grey = (uchar)(value * 255.0f);\n"
"        output[idx] = output[idx+1] = output[idx+2] = grey;\n"
"    } else {\n"
"        float sqrt_t = sqrt(t);\n"
"        float r_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI)) * brightness;\n"
"        float g_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI + 2.0f*M_PI/3.0f)) * brightness;\n"
"        float b_val = (0.5f + 0.5f * sin(sqrt_t * 2.0f * M_PI + 4.0f*M_PI/3.0f)) * brightness;\n"
"        output[idx] = (uchar)(r_val * 255.0f);\n"
"        output[idx+1] = (uchar)(g_val * 255.0f);\n"
"        output[idx+2] = (uchar)(b_val * 255.0f);\n"
"    }\n"
"}";

/* ===========================================
   Enhanced OpenCL GPU Management
   =========================================== */

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_output;
    size_t max_work_group_size;
    int supports_double;
    int initialized;
} GPUContext;

static GPUContext g_ctx = {0};

static void gpu_print_info(void) {
    char device_name[256];
    char vendor_name[256];
    cl_device_type device_type;
    cl_ulong global_mem;
    cl_uint compute_units;
    
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    
    printf("GPU Device: %s\n", device_name);
    printf("Vendor: %s\n", vendor_name);
    printf("Type: %s\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : 
                         device_type == CL_DEVICE_TYPE_CPU ? "CPU" : "Other");
    printf("Global Memory: %.1f MB\n", global_mem / (1024.0 * 1024.0));
    printf("Compute Units: %u\n", compute_units);
    printf("Max Work Group: %zu\n", g_ctx.max_work_group_size);
}

static int gpu_init(void) {
    cl_int err;
    
    // Get platform
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &g_ctx.platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 0;
    }
    
    // Get GPU device (prefer discrete GPU)
    cl_device_id devices[16];
    cl_uint num_devices;
    
    // First try NVIDIA
    err = clGetDeviceIDs(g_ctx.platform, CL_DEVICE_TYPE_GPU, 16, devices, &num_devices);
    
    // Then try AMD/Intel
    if (err != CL_SUCCESS || num_devices == 0) {
        err = clGetDeviceIDs(g_ctx.platform, CL_DEVICE_TYPE_ALL, 16, devices, &num_devices);
    }
    
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "No OpenCL devices found\n");
        return 0;
    }
    
    // Choose the best device (prefer discrete GPU with more memory)
    g_ctx.device = devices[0];
    for (cl_uint i = 0; i < num_devices; i++) {
        cl_device_type type;
        cl_ulong mem_size;
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
        
        // Prefer discrete GPU
        if (type == CL_DEVICE_TYPE_GPU) {
            g_ctx.device = devices[i];
            // If it's NVIDIA/AMD and has good memory, use it
            char vendor[256];
            clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
            if (strstr(vendor, "NVIDIA") || strstr(vendor, "AMD")) {
                break;
            }
        }
    }
    
    // Check double precision support
    char extensions[1024];
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    g_ctx.supports_double = strstr(extensions, "cl_khr_fp64") != NULL;
    
    if (!g_ctx.supports_double) {
        printf("Warning: GPU doesn't support double precision. Using float.\n");
    }
    
    // Get work group size
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                   sizeof(g_ctx.max_work_group_size), &g_ctx.max_work_group_size, NULL);
    
    // Create context
    g_ctx.context = clCreateContext(NULL, 1, &g_ctx.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context: %d\n", err);
        return 0;
    }
    
    // Create command queue with profiling (OpenCL 2.0+ style)
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
    0
};
g_ctx.queue = clCreateCommandQueueWithProperties(g_ctx.context, 
                                                 g_ctx.device, 
                                                 props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %d\n", err);
        return 0;
    }
    
    // Create program
    g_ctx.program = clCreateProgramWithSource(g_ctx.context, 1, 
                                             &mandelbrot_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %d\n", err);
        return 0;
    }
    
    // Build program with optimization flags
    const char* options = g_ctx.supports_double ? 
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros" :
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -DUSE_FLOAT";
    
    err = clBuildProgram(g_ctx.program, 1, &g_ctx.device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[8192];
        clGetProgramBuildInfo(g_ctx.program, g_ctx.device, CL_PROGRAM_BUILD_LOG,
                             sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", build_log);
        return 0;
    }
    
    // Create kernel
    g_ctx.kernel = clCreateKernel(g_ctx.program, "mandelbrot_gpu", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel: %d\n", err);
        return 0;
    }
    
    g_ctx.initialized = 1;
    gpu_print_info();
    return 1;
}

static void gpu_cleanup(void) {
    if (g_ctx.buffer_output) clReleaseMemObject(g_ctx.buffer_output);
    if (g_ctx.kernel) clReleaseKernel(g_ctx.kernel);
    if (g_ctx.program) clReleaseProgram(g_ctx.program);
    if (g_ctx.queue) clReleaseCommandQueue(g_ctx.queue);
    if (g_ctx.context) clReleaseContext(g_ctx.context);
    memset(&g_ctx, 0, sizeof(g_ctx));
}

static double gpu_render(Config* cfg, uint8_t* image) {
    if (!g_ctx.initialized && !gpu_init()) {
        return -1.0;  // Indicate failure
    }
    
    cl_int err;
    cl_event event;
    double render_time = 0.0;
    
    // Create output buffer
    size_t image_size = cfg->width * cfg->height * 3;
    g_ctx.buffer_output = clCreateBuffer(g_ctx.context, CL_MEM_WRITE_ONLY,
                                        image_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer: %d\n", err);
        return -1.0;
    }
    
    // Set kernel arguments
    err = 0;
    err |= clSetKernelArg(g_ctx.kernel, 0, sizeof(cl_mem), &g_ctx.buffer_output);
    
    if (g_ctx.supports_double) {
        double x_min = cfg->x_min;
        double x_max = cfg->x_max;
        double y_min = cfg->y_min;
        double y_max = cfg->y_max;
        
        err |= clSetKernelArg(g_ctx.kernel, 1, sizeof(double), &x_min);
        err |= clSetKernelArg(g_ctx.kernel, 2, sizeof(double), &x_max);
        err |= clSetKernelArg(g_ctx.kernel, 3, sizeof(double), &y_min);
        err |= clSetKernelArg(g_ctx.kernel, 4, sizeof(double), &y_max);
    } else {
        // Convert to float if no double support
        float x_min = (float)cfg->x_min;
        float x_max = (float)cfg->x_max;
        float y_min = (float)cfg->y_min;
        float y_max = (float)cfg->y_max;
        
        err |= clSetKernelArg(g_ctx.kernel, 1, sizeof(float), &x_min);
        err |= clSetKernelArg(g_ctx.kernel, 2, sizeof(float), &x_max);
        err |= clSetKernelArg(g_ctx.kernel, 3, sizeof(float), &y_min);
        err |= clSetKernelArg(g_ctx.kernel, 4, sizeof(float), &y_max);
    }
    
    err |= clSetKernelArg(g_ctx.kernel, 5, sizeof(int), &cfg->width);
    err |= clSetKernelArg(g_ctx.kernel, 6, sizeof(int), &cfg->height);
    err |= clSetKernelArg(g_ctx.kernel, 7, sizeof(int), &cfg->max_iter);
    err |= clSetKernelArg(g_ctx.kernel, 8, sizeof(int), &cfg->grayscale);
    err |= clSetKernelArg(g_ctx.kernel, 9, sizeof(float), &cfg->gamma);
    err |= clSetKernelArg(g_ctx.kernel, 10, sizeof(float), &cfg->brightness);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments: %d\n", err);
        clReleaseMemObject(g_ctx.buffer_output);
        g_ctx.buffer_output = 0;
        return -1.0;
    }
    
    // Execute kernel
    size_t global_work_size[2] = {cfg->width, cfg->height};
    size_t local_work_size[2] = {16, 16};  // Good default for most GPUs
    
    // Adjust local work size if needed
    if (global_work_size[0] % local_work_size[0] != 0) {
        local_work_size[0] = 1;
    }
    if (global_work_size[1] % local_work_size[1] != 0) {
        local_work_size[1] = 1;
    }
    
    err = clEnqueueNDRangeKernel(g_ctx.queue, g_ctx.kernel, 2, NULL,
                                global_work_size, local_work_size,
                                0, NULL, &event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute kernel: %d\n", err);
        clReleaseMemObject(g_ctx.buffer_output);
        g_ctx.buffer_output = 0;
        return -1.0;
    }
    
    // Wait for completion
    clWaitForEvents(1, &event);
    
    // Read results
    err = clEnqueueReadBuffer(g_ctx.queue, g_ctx.buffer_output, CL_TRUE, 0,
                             image_size, image, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read buffer: %d\n", err);
    }
    
    // Get profiling info
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    render_time = (end - start) * 1e-9;  // Convert to seconds
    
    // Cleanup
    clReleaseEvent(event);
    clReleaseMemObject(g_ctx.buffer_output);
    g_ctx.buffer_output = 0;
    
    return render_time;
}

static double gpu_perturbation_render(Config* cfg, PerturbationConfig* ptbm, uint8_t* image) {
    if (!g_ctx.initialized && !gpu_init()) {
        return -1.0;
    }

    printf("DEBUG: Entering gpu_perturbation_render\n");
    printf("DEBUG: ptbm pointer = %p\n", ptbm);
    printf("DEBUG: ptbm->reference_x = %.21Lf\n", ptbm->reference_x);
    printf("DEBUG: ptbm->reference_y = %.21Lf\n", ptbm->reference_y);
    printf("DEBUG: ptbm->reference_iters = %d\n", ptbm->reference_iters);
    
    // Rest of your function...


    cl_int err;
    cl_event event;
    
    // Create perturbation program
    cl_program ptbm_program = clCreateProgramWithSource(g_ctx.context, 1, 
                                 &mandelbrot_perturbation_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create perturbation program: %d\n", err);
        return -1.0;
    }
    
    // Build with optimization flags
    const char* options = g_ctx.supports_double ? 
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros" :
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -DUSE_FLOAT";
    
    err = clBuildProgram(ptbm_program, 1, &g_ctx.device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[8192];
        clGetProgramBuildInfo(ptbm_program, g_ctx.device, CL_PROGRAM_BUILD_LOG,
                             sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Perturbation kernel build failed:\n%s\n", build_log);
        clReleaseProgram(ptbm_program);
        return -1.0;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(ptbm_program, "mandelbrot_perturbation_gpu", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create perturbation kernel: %d\n", err);
        clReleaseProgram(ptbm_program);
        return -1.0;
    }
    
    // Create output buffer
    size_t image_size = cfg->width * cfg->height * 3;
    cl_mem buffer = clCreateBuffer(g_ctx.context, CL_MEM_WRITE_ONLY,
                                   image_size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer: %d\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(ptbm_program);
        return -1.0;
    }
    
    // Set kernel arguments
    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(double), &ptbm->reference_x);
    err |= clSetKernelArg(kernel, 2, sizeof(double), &ptbm->reference_y);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &ptbm->reference_iters);
    double x_min = (double)cfg->x_min;
    double x_max = (double)cfg->x_max;
    double y_min = (double)cfg->y_min;
    double y_max = (double)cfg->y_max;
    err |= clSetKernelArg(kernel, 4, sizeof(double), &x_min);
    err |= clSetKernelArg(kernel, 5, sizeof(double), &x_max);
    err |= clSetKernelArg(kernel, 6, sizeof(double), &y_min);
    err |= clSetKernelArg(kernel, 7, sizeof(double), &y_max);
    err |= clSetKernelArg(kernel, 4, sizeof(double), &cfg->x_min);
    err |= clSetKernelArg(kernel, 5, sizeof(double), &cfg->x_max);
    err |= clSetKernelArg(kernel, 6, sizeof(double), &cfg->y_min);
    err |= clSetKernelArg(kernel, 7, sizeof(double), &cfg->y_max);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &cfg->width);
    err |= clSetKernelArg(kernel, 9, sizeof(int), &cfg->height);
    err |= clSetKernelArg(kernel, 10, sizeof(int), &cfg->max_iter);
    err |= clSetKernelArg(kernel, 11, sizeof(int), &cfg->grayscale);
    err |= clSetKernelArg(kernel, 12, sizeof(float), &cfg->gamma);
    err |= clSetKernelArg(kernel, 13, sizeof(float), &cfg->brightness);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments\n");
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(ptbm_program);
        return -1.0;
    }
    
    // Execute kernel
    size_t global_work_size[2] = {cfg->width, cfg->height};
    err = clEnqueueNDRangeKernel(g_ctx.queue, kernel, 2, NULL,
                                global_work_size, NULL, 0, NULL, &event);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute kernel: %d\n", err);
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(ptbm_program);
        return -1.0;
    }
    
    clWaitForEvents(1, &event);
    
    // Read results
    err = clEnqueueReadBuffer(g_ctx.queue, buffer, CL_TRUE, 0,
                             image_size, image, 0, NULL, NULL);
    
    // Get timing
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double render_time = (end - start) * 1e-9;
    
    // DEBUG: Print reference info
    printf("\nGPU Perturbation Debug Info:\n");
    printf("  Reference point: (%0.21Lf, %0.21Lf)\n", ptbm->reference_x, ptbm->reference_y);
    printf("  Reference iterations: %d\n", ptbm->reference_iters);
    printf("  Max iterations: %d\n", cfg->max_iter);
    printf("  Image size: %dx%d\n", cfg->width, cfg->height);
    
    // DEBUG: Check a few pixels and show their RGB values and estimated iterations
    printf("\nGPU Sample pixels (center region):\n");
    int center_x = cfg->width / 2;
    int center_y = cfg->height / 2;
    int all_black = 1;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int x = center_x + dx;
            int y = center_y + dy;
            if (x >= 0 && x < cfg->width && y >= 0 && y < cfg->height) {
                int idx = (y * cfg->width + x) * 3;
                int r = image[idx];
                int g = image[idx+1];
                int b = image[idx+2];
                printf("  (%+d,%+d): RGB(%3d,%3d,%3d)", dx, dy, r, g, b);
                
                // Estimate iteration from color
                if (r == 0 && g == 0 && b == 0) {
                    printf(" (inside set)\n");
                } else {
                    all_black = 0;
                    // Rough estimate based on color formula
                    float t = (r / 255.0f) / cfg->brightness;
                    int est_iter = (int)(t * cfg->max_iter);
                    printf(" (~%d iterations)\n", est_iter);
                }
            }
        }
    }
    if (all_black) {
        printf("WARNING: All sampled pixels are black! All points are being marked as inside the set.\n");
    }
    
    // Cleanup
    clReleaseEvent(event);
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(ptbm_program);
    
    return render_time;
}

#else  // USE_OPENCL not defined

// Stub functions when OpenCL is not available
static int gpu_init(void) { return 0; }
static void gpu_cleanup(void) {}
static void gpu_print_info(void) {
    printf("GPU support not compiled (use -DUSE_OPENCL)\n");
}
static double gpu_render(Config* cfg, uint8_t* image) {
    (void)cfg; (void)image;
    printf("GPU rendering not available. Compile with -DUSE_OPENCL and link with OpenCL library.\n");
    return -1.0;
}

static double gpu_perturbation_render(Config* cfg, PerturbationConfig* ptbm, uint8_t* image) {
    (void)cfg; (void)ptbm; (void)image;
    printf("GPU perturbation not available. Compile with -DUSE_OPENCL.\n");
    return -1.0;
}

#endif  // USE_OPENCL

/* ===========================================
   Enhanced render_mandelbrot function with GPU and Perturbation support
   =========================================== */

static void render_mandelbrot(Config* cfg, PerturbationConfig* ptbm, uint8_t* image) {
    // If perturbation is enabled, use that (with GPU if requested)
    if (ptbm && ptbm->enabled) {
        // CRITICAL FIX: Calculate reference iterations BEFORE GPU call!
        if (ptbm->reference_iters == 0) {
            ptbm->reference_iters = calculate_reference_iterations(
                ptbm->reference_x, ptbm->reference_y, cfg->max_iter);
        }
        
        if (cfg->use_gpu) {
            printf("\nGPU Perturbation Rendering...\n");
            
            // Print GPU info before rendering
            if (!g_ctx.initialized) {
                if (gpu_init()) {
                    gpu_print_info();
                }
            } else {
                gpu_print_info();
            }
            
            double gpu_time = gpu_perturbation_render(cfg, ptbm, image);
            if (gpu_time > 0) {
                printf("GPU Perturbation: %.3f seconds\n", gpu_time);
                return;
            }
            printf("GPU perturbation failed, falling back to CPU perturbation\n");
        }
        
        // Make sure reference_iters is set for CPU too
        if (ptbm->reference_iters == 0) {
            ptbm->reference_iters = calculate_reference_iterations(
                ptbm->reference_x, ptbm->reference_y, cfg->max_iter);
        }
        render_mandelbrot_perturbation(cfg, ptbm, image);
        return;
    }
    
    // Original GPU/CPU rendering code (for non-perturbation mode)
    clock_t total_start = clock();
    
    // GPU rendering (optional)
    if (cfg->use_gpu) {
        printf("\nGPU Rendering...\n");
        
        // Print GPU info before rendering
        if (!g_ctx.initialized) {
            if (gpu_init()) {
                gpu_print_info();
            }
        } else {
            gpu_print_info();
        }
        
        double gpu_time = gpu_render(cfg, image);
        if (gpu_time > 0) {
            printf("GPU: %.3f seconds\n", gpu_time);
            return;
        }
        printf("GPU failed, using CPU\n");
        cfg->use_gpu = 0;
    }
    
    // Rest of CPU rendering code...
    printf("\nCPU Rendering...\n");
    
    // Check if we should use AVX2
    int use_avx2 = cfg->use_avx2 && (cfg->cpu_features & CPU_FEATURE_AVX2);
    printf("Using: %s\n", use_avx2 ? "AVX2 + OpenMP" : "Scalar + OpenMP");
    
    // Configure threads
    int num_threads = cfg->num_threads;
    if (num_threads <= 0) {
        num_threads = 0;  // 0 = use OpenMP default
    }
    
    int actual_threads = num_threads ? num_threads : omp_get_max_threads();
    printf("Threads: %d\n", actual_threads);
    
    clock_t cpu_start = clock();
    
    double dx = (cfg->x_max - cfg->x_min) / cfg->width;
    double dy = (cfg->y_max - cfg->y_min) / cfg->height;
    
    // PARALLEL LOOP WITH OPENMP
    #pragma omp parallel for num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
    for (int y = 0; y < cfg->height; y++) {
        double y0 = cfg->y_min + y * dy;
        int x = 0;
        
#ifdef __AVX2__
        if (use_avx2) {
            for (; x <= cfg->width - 4; x += 4) {
                // Use aligned memory for AVX2
                double x_coords[4] __attribute__((aligned(32)));
                for (int k = 0; k < 4; k++) {
                    x_coords[k] = cfg->x_min + (x + k) * dx;
                }
                
                // Use aligned load
                __m256d x0_vec = _mm256_load_pd(x_coords);
                __m256d y0_vec = _mm256_set1_pd(y0);
                
                int results[4];
                mandelbrot_avx2(x0_vec, y0_vec, cfg->max_iter, results);
                
                for (int k = 0; k < 4; k++) {
                    int idx = (y * cfg->width + (x + k)) * 3;
                    uint8_t r, g, b;
                    get_color(results[k], cfg->max_iter, cfg->grayscale, 
                              cfg->gamma, cfg->brightness, &r, &g, &b);
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
#endif
        
        // SCALAR PROCESSING
        for (; x < cfg->width; x++) {
            double x0 = cfg->x_min + x * dx;
            int iter = mandelbrot_scalar(x0, y0, cfg->max_iter);
            
            int idx = (y * cfg->width + x) * 3;
            uint8_t r, g, b;
            get_color(iter, cfg->max_iter, cfg->grayscale, 
                      cfg->gamma, cfg->brightness, &r, &g, &b);
            image[idx] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
    
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    printf("\n=== Summary ===\n");
    printf("CPU (%d threads, %s): %.3f seconds\n", 
           actual_threads, use_avx2 ? "AVX2" : "Scalar", cpu_time);
    printf("Performance: %.1f MPixels/second\n", 
           (cfg->width * cfg->height) / (cpu_time * 1000000.0));
    printf("Total: %.3f seconds\n", total_time);
}

/* ===========================================
   Perturbation Rendering Function
   =========================================== */

static void render_mandelbrot_perturbation(
    Config* cfg, 
    PerturbationConfig* ptbm,
    uint8_t* image) {
    
    printf("\nPerturbation Math Enabled\n");
    printf("Precision mode: ");
    switch (ptbm->precision) {
        case PTBM_PRECISION_AUTO: printf("Auto"); break;
        case PTBM_PRECISION_SINGLE: printf("Single (float)"); break;
        case PTBM_PRECISION_DOUBLE: printf("Double"); break;
        case PTBM_PRECISION_MIXED: printf("Mixed (double ref + float delta)"); break;
        case PTBM_PRECISION_HIGH: printf("High (long double)"); break;
    }
    printf("\n");
    printf("Zoom level: %.2e\n", ptbm->zoom_level);
    printf("Error bound: %.2e\n", ptbm->error_bound);
    
    // Calculate reference point iterations
    printf("Calculating reference point...\n");
    ptbm->reference_iters = calculate_reference_iterations(
        ptbm->reference_x, ptbm->reference_y, cfg->max_iter);
    printf("Reference iterations: %d\n", ptbm->reference_iters);
    
    // Allocate iteration count array for glitch detection
    int* iteration_counts = malloc(cfg->width * cfg->height * sizeof(int));
    
    double dx = (cfg->x_max - cfg->x_min) / cfg->width;
    double dy = (cfg->y_max - cfg->y_min) / cfg->height;
    
    clock_t start = clock();
    
    // Parallel rendering with perturbation
    #pragma omp parallel for num_threads(cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads())
    for (int y = 0; y < cfg->height; y++) {
        double y0 = cfg->y_min + y * dy;
        
        for (int x = 0; x < cfg->width; x++) {
            double x0 = cfg->x_min + x * dx;
            
            int iter;
            
            // Use appropriate precision mode
            switch (ptbm->precision) {
                case PTBM_PRECISION_SINGLE:
                    iter = mandelbrot_perturbation_single(
                        x0, y0, 
                        ptbm->reference_x, ptbm->reference_y, 
                        ptbm->reference_iters, cfg->max_iter,
                        ptbm->error_bound);
                    break;
                    
                case PTBM_PRECISION_DOUBLE:
                    iter = mandelbrot_perturbation_double(
                        x0, y0,
                        ptbm->reference_x, ptbm->reference_y,
                        ptbm->reference_iters, cfg->max_iter,
                        ptbm->error_bound);
                    break;
                    
                case PTBM_PRECISION_MIXED:
                    iter = mandelbrot_perturbation_mixed(
                        x0, y0,
                        ptbm->reference_x, ptbm->reference_y,
                        ptbm->reference_iters, cfg->max_iter,
                        ptbm->error_bound);
                    break;
                    
                case PTBM_PRECISION_HIGH:
                    iter = mandelbrot_perturbation_high(
                        x0, y0,
                        ptbm->reference_x, ptbm->reference_y,
                        ptbm->reference_iters, cfg->max_iter,
                        ptbm->error_bound);
                    break;
                    
                default:
                    iter = mandelbrot_scalar(x0, y0, cfg->max_iter);
            }
            
            iteration_counts[y * cfg->width + x] = iter;
            
            int idx = (y * cfg->width + x) * 3;
            uint8_t r, g, b;
            get_color(iter, cfg->max_iter, cfg->grayscale,
                      cfg->gamma, cfg->brightness, &r, &g, &b);
            image[idx] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
    
    clock_t end = clock();
    double render_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Glitch detection and correction
    if (ptbm->use_glitch_correction) {
        detect_and_correct_glitches(cfg, ptbm, image, iteration_counts);
    }
    
    printf("Perturbation rendering time: %.3f seconds\n", render_time);
    printf("Performance: %.1f MPixels/second\n", 
           (cfg->width * cfg->height) / (render_time * 1000000.0));
    
    free(iteration_counts);
}

/* ===========================================
   PNG File I/O
   =========================================== */

static int save_png(const char* filename, uint8_t* image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return 0;
    }
    
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 
                                                  NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return 0;
    }
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return 0;
    }
    
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return 0;
    }
    
    png_init_io(png_ptr, fp);
    
    // Set image properties
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    
    // Write header
    png_write_info(png_ptr, info_ptr);
    
    // Prepare row pointers
    png_bytep row_pointers[height];
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)(image + y * width * 3);
    }
    
    // Write image data
    png_write_image(png_ptr, row_pointers);
    
    // Write end
    png_write_end(png_ptr, NULL);
    
    // Cleanup
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    
    return 1;
}

/* ===========================================
   PPM File I/O
   =========================================== */

static int save_ppm(const char* filename, uint8_t* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return 0;
    }
    
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    size_t written = fwrite(image, 1, width * height * 3, fp);
    fclose(fp);
    
    if (written != width * height * 3) {
        fprintf(stderr, "Failed to write all data to %s\n", filename);
        return 0;
    }
    
    return 1;
}

/* ===========================================
   Enhanced Command Line Parsing with Perturbation
   =========================================== */

static void print_help(void) {
    printf("Ultimate Mandelbrot Renderer\n");
    printf("Usage: mandelbrot [options]\n\n");
    printf("Options:\n");
    printf("  -w, --width WIDTH     Image width (default: 1920)\n");
    printf("  -h, --height HEIGHT   Image height (default: 1080)\n");
    printf("  -i, --iter ITER       Max iterations (default: 1000)\n");
    printf("  -x, --xmin X          Left bound (default: -2.5)\n");
    printf("  -X, --xmax X          Right bound (default: 1.5)\n");
    printf("  -y, --ymin Y          Bottom bound (default: -1.5)\n");
    printf("  -Y, --ymax Y          Top bound (default: 1.5)\n");
    printf("  -g, --grayscale       Output grayscale image\n");
    printf("  -o, --output FILE     Output file (default: mandelbrot.ppm)\n");
    printf("  -t, --threads N       Number of threads (0=auto)\n");
    printf("  --no-avx2             Disable AVX2 even if available\n");
    printf("  --no-sse4             Disable SSE4 even if available\n");
#ifdef USE_OPENCL
    printf("  --gpu                 Use GPU acceleration\n");
#endif
    printf("  --benchmark           Run performance benchmark\n");
    printf("  --gamma VALUE         Gamma correction (default: 0.3)\n");
    printf("  --brightness VALUE    Brightness multiplier (default: 1.0)\n");
    
    printf("\nPerturbation Math Options (for deep zoom):\n");
    printf("  --ptbm                  Enable perturbation math\n");
    printf("  --ptbm-precision MODE    Set precision mode:\n");
    printf("                            auto    - Auto-select based on zoom\n");
    printf("                            single  - Single precision (float)\n");
    printf("                            double  - Double precision\n");
    printf("                            mixed   - Mixed precision\n");
    printf("                            high    - High precision (long double)\n");
    printf("  --ptbm-error BOUND       Set error bound (default: 1e-6)\n");
    printf("  --ptbm-no-glitch         Disable glitch detection/correction\n");
    printf("  --ptbm-ref X Y           Set custom reference point\n");
    
    printf("  --help                Show this help\n\n");
    
    printf("Examples:\n");
    printf("  mandelbrot -w 1280 -h 720 -i 5000 -o output.ppm\n");
    printf("  mandelbrot -x -0.748 -X -0.744 -y 0.1 -Y 0.104 -i 10000\n");
    printf("  mandelbrot --ptbm -x -1.749999 -X -1.749998 -y 0.000001 -Y 0.000002 -i 50000\n");
    printf("  mandelbrot --ptbm --ptbm-precision mixed --ptbm-error 1e-8 [deep zoom coords]\n");
}

static void parse_args(int argc, char** argv, Config* cfg, PerturbationConfig* ptbm) {
    // Defaults for Config
    cfg->width = 1920;
    cfg->height = 1080;
    cfg->max_iter = 1000;
    cfg->x_min = -2.5;
    cfg->x_max = 1.5;
    cfg->y_min = -1.5;
    cfg->y_max = 1.5;
    cfg->use_gpu = 0;
    cfg->use_avx2 = 1;
    cfg->use_sse4 = 1;
    cfg->num_threads = 0;
    cfg->grayscale = 0;
    cfg->benchmark = 0;
    cfg->output_file = "mandelbrot.ppm";
    cfg->cpu_features = detect_cpu_features();
    cfg->gamma = 0.3f;
    cfg->brightness = 1.0f;
    
    // Defaults for PerturbationConfig
    memset(ptbm, 0, sizeof(PerturbationConfig));
    ptbm->enabled = 0;
    ptbm->precision = PTBM_PRECISION_AUTO;
    ptbm->error_bound = 1e-6;
    ptbm->use_glitch_correction = 1;
    ptbm->max_refinement_iter = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help();
            exit(0);
        } else if (strcmp(argv[i], "--width") == 0 || strcmp(argv[i], "-w") == 0) {
            if (++i < argc) cfg->width = atoi(argv[i]);
        } else if (strcmp(argv[i], "--height") == 0 || strcmp(argv[i], "-h") == 0) {
            if (++i < argc) cfg->height = atoi(argv[i]);
        } else if (strcmp(argv[i], "--iter") == 0 || strcmp(argv[i], "-i") == 0) {
            if (++i < argc) cfg->max_iter = atoi(argv[i]);
        } else if (strcmp(argv[i], "--xmin") == 0 || strcmp(argv[i], "-x") == 0) {
            if (++i < argc) {
                sscanf(argv[i], "%Lf", &cfg->x_min);
                printf("DEBUG: Set x_min = %.21Lf from '%s'\n", cfg->x_min, argv[i]);
            }
        } else if (strcmp(argv[i], "--xmax") == 0 || strcmp(argv[i], "-X") == 0) {
            if (++i < argc) {
                sscanf(argv[i], "%Lf", &cfg->x_max);
                printf("DEBUG: Set x_max = %.21Lf from '%s'\n", cfg->x_max, argv[i]);
            }
        } else if (strcmp(argv[i], "--ymin") == 0 || strcmp(argv[i], "-y") == 0) {
            if (++i < argc) {
                sscanf(argv[i], "%Lf", &cfg->y_min);
                printf("DEBUG: Set y_min = %.21Lf from '%s'\n", cfg->y_min, argv[i]);
            }
        } else if (strcmp(argv[i], "--ymax") == 0 || strcmp(argv[i], "-Y") == 0) {
            if (++i < argc) {
                sscanf(argv[i], "%Lf", &cfg->y_max);
                printf("DEBUG: Set y_max = %.21Lf from '%s'\n", cfg->y_max, argv[i]);
            }
        } else if (strcmp(argv[i], "--grayscale") == 0 || strcmp(argv[i], "-g") == 0) {
            cfg->grayscale = 1;
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            if (++i < argc) cfg->output_file = argv[i];
        } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
            if (++i < argc) cfg->num_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--no-avx2") == 0) {
            cfg->use_avx2 = 0;
        } else if (strcmp(argv[i], "--no-sse4") == 0) {
            cfg->use_sse4 = 0;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            cfg->use_gpu = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            cfg->benchmark = 1;
        } else if (strcmp(argv[i], "--gamma") == 0) {
            if (++i < argc) cfg->gamma = atof(argv[i]);
        } else if (strcmp(argv[i], "--brightness") == 0) {
            if (++i < argc) cfg->brightness = atof(argv[i]);
        }
        // Perturbation options
        else if (strcmp(argv[i], "--ptbm") == 0) {
            ptbm->enabled = 1;
        } else if (strcmp(argv[i], "--ptbm-precision") == 0 && i+1 < argc) {
            i++;
            if (strcmp(argv[i], "auto") == 0) ptbm->precision = PTBM_PRECISION_AUTO;
            else if (strcmp(argv[i], "single") == 0) ptbm->precision = PTBM_PRECISION_SINGLE;
            else if (strcmp(argv[i], "double") == 0) ptbm->precision = PTBM_PRECISION_DOUBLE;
            else if (strcmp(argv[i], "mixed") == 0) ptbm->precision = PTBM_PRECISION_MIXED;
            else if (strcmp(argv[i], "high") == 0) ptbm->precision = PTBM_PRECISION_HIGH;
        } else if (strcmp(argv[i], "--ptbm-error") == 0 && i+1 < argc) {
            i++;
            ptbm->error_bound = atof(argv[i]);
        } else if (strcmp(argv[i], "--ptbm-no-glitch") == 0) {
            ptbm->use_glitch_correction = 0;
        } else if (strcmp(argv[i], "--ptbm-ref") == 0 && i+2 < argc) {
            i++;
            sscanf(argv[i], "%Lf", &ptbm->reference_x);
            printf("DEBUG: Set ref_x = %.21Lf\n", ptbm->reference_x);
            i++;
            sscanf(argv[i], "%Lf", &ptbm->reference_y);
            printf("DEBUG: Set ref_y = %.21Lf\n", ptbm->reference_y);
        }
    }
}

/* ===========================================
   Benchmark Function
   =========================================== */

static void run_benchmark(Config* cfg) {
    printf("\n=== Performance Benchmark ===\n");
    
    // Test different iteration counts
    int test_iterations[] = {256, 1000, 4000, 16000, 32000};
    int num_tests = sizeof(test_iterations) / sizeof(test_iterations[0]);
    
    // Small test image for quick benchmarking
    int orig_width = cfg->width;
    int orig_height = cfg->height;
    cfg->width = 640;
    cfg->height = 480;
    
    uint8_t* test_image = malloc(cfg->width * cfg->height * 3);
    
    for (int i = 0; i < num_tests; i++) {
        cfg->max_iter = test_iterations[i];
        
        clock_t start = clock();
        render_mandelbrot(cfg, NULL, test_image);
        clock_t end = clock();
        
        double time = (double)(end - start) / CLOCKS_PER_SEC;
        printf("  %5d iterations: %.3f seconds (%.1f iterations/ms)\n",
               test_iterations[i], time, test_iterations[i] / (time * 1000.0));
    }
    
    // Restore original size
    cfg->width = orig_width;
    cfg->height = orig_height;
    free(test_image);
}

/* ===========================================
   Updated Main Function
   =========================================== */

int main(int argc, char** argv) {
    Config cfg;
    PerturbationConfig ptbm;
    
    // Parse arguments (now includes perturbation config)
    parse_args(argc, argv, &cfg, &ptbm);
    
    // Calculate reference point if perturbation enabled
    if (ptbm.enabled) {
        calculate_reference_point(&cfg, &ptbm);
    }

    printf("=== Ultimate Mandelbrot Renderer ===\n");
    printf("Resolution: %dx%d\n", cfg.width, cfg.height);
    printf("Iterations: %d\n", cfg.max_iter);
    printf("Viewport: [%.21Lf, %.21Lf] x [%.21Lf, %.21Lf]\n", 
           cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);
    if (ptbm.enabled) {
        printf("Perturbation: Enabled\n");
        printf("Reference point: (%.21Lf, %.21Lf)\n", ptbm.reference_x, ptbm.reference_y);
    } else {
        printf("Perturbation: Disabled\n");
    }
    printf("Color: %s\n", cfg.grayscale ? "Grayscale" : "RGB");
    printf("Gamma: %.2f, Brightness: %.2f\n", cfg.gamma, cfg.brightness);
    printf("CPU Features: ");
    if (cfg.cpu_features & CPU_FEATURE_AVX2) printf("AVX2 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE4_2) printf("SSE4.2 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE4_1) printf("SSE4.1 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE3) printf("SSE3 ");
    if (cfg.cpu_features == CPU_FEATURE_NONE) printf("Scalar only");
    printf("\n");
    
    if (cfg.benchmark) {
        run_benchmark(&cfg);
    }
    
    printf("Allocating image buffer...\n");
    fflush(stdout);
    
    // Allocate image
    uint8_t* image = malloc(cfg.width * cfg.height * 3);
    if (!image) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    printf("Starting render_mandelbrot()...\n");
    fflush(stdout);
    
    // Render (now passes perturbation config)
    render_mandelbrot(&cfg, ptbm.enabled ? &ptbm : NULL, image);
    
    printf("render_mandelbrot() completed!\n");
    fflush(stdout);
    
    // Save image - check extension
    const char* ext = strrchr(cfg.output_file, '.');
    int success = 0;
    
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0)) {
        success = save_png(cfg.output_file, image, cfg.width, cfg.height);
    } else {
        // Default to PPM
        success = save_ppm(cfg.output_file, image, cfg.width, cfg.height);
    }
    
    if (success) {
        printf("Image saved to: %s\n", cfg.output_file);
    } else {
        fprintf(stderr, "Failed to save image to %s\n", cfg.output_file);
    }
    
    free(image);
    
#ifdef USE_OPENCL
    gpu_cleanup();
#endif
    
    return 0;
}
