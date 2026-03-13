/**
 * Mandelbrot Set Renderer
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

/* ===========================================
   Configuration System
   =========================================== */

typedef struct {
    // Image settings
    int width;
    int height;
    int max_iter;
    
    // Viewport
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    
    // Performance settings
    int use_gpu;
    int use_avx2;
    int use_sse4;
    int num_threads;
    
    // Output settings
    int grayscale;
    int benchmark;
    const char* output_file;

    // Color settings  <-- ADD THIS SECTION
    float gamma;
    float brightness;

    // Internal
    int cpu_features;
} Config;

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
        __m128d still_active = _mm_cmplt_pd(r2, four);  // <-- Changed to cmplt
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
//AVX 2 implementation
#ifdef __AVX2__
static inline void mandelbrot_avx2(__m256d x0, __m256d y0, int max_iter, int results[4]) {
    __m256d x = _mm256_setzero_pd();
    __m256d y = _mm256_setzero_pd();
    __m256d x2 = _mm256_setzero_pd();
    __m256d y2 = _mm256_setzero_pd();
    
    __m256d two = _mm256_set1_pd(2.0);
    __m256d four = _mm256_set1_pd(4.0);
    
    // Counters for each point
    int counters[4] = {0, 0, 0, 0};
    int active[4] = {1, 1, 1, 1};  // 1 = still calculating
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate r²
        __m256d r2 = _mm256_add_pd(x2, y2);
        
        // Check which points have escaped (r² >= 4.0)
        double r2_arr[4];
        _mm256_storeu_pd(r2_arr, r2);
        
        // Update counters for points that haven't escaped
        for (int i = 0; i < 4; i++) {
            if (active[i] && r2_arr[i] < 4.0) {
                counters[i]++;
            } else {
                active[i] = 0;  // Point has escaped
            }
        }
        
        // Check if all points have escaped
        int all_escaped = 1;
        for (int i = 0; i < 4; i++) {
            if (active[i]) {
                all_escaped = 0;
                break;
            }
        }
        if (all_escaped) break;
        
        // Calculate next values for all points
        __m256d xy = _mm256_mul_pd(x, y);
        y = _mm256_fmadd_pd(two, xy, y0);
        x = _mm256_add_pd(_mm256_sub_pd(x2, y2), x0);
        
        // Update squares
        x2 = _mm256_mul_pd(x, x);
        y2 = _mm256_mul_pd(y, y);
    }
    
    // Return results
    for (int i = 0; i < 4; i++) {
        results[i] = counters[i];
        if (results[i] > max_iter) results[i] = max_iter;
    }
}
#endif
/* ===========================================
   Color Management
   =========================================== */
static inline void get_color(int iter, int max_iter, int grayscale, 
                            float gamma, float brightness,  // ADD THESE PARAMS
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
   Thread Management
   =========================================== */
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct {
    Config* config;
    uint8_t* image;
    int start_row;
    int end_row;
    double* timings;
} ThreadData;
#ifdef _WIN32
DWORD WINAPI render_thread(void* arg) {
#else
void* render_thread(void* arg) {
#endif
    ThreadData* data = (ThreadData*)arg;
    Config* cfg = data->config;
    
    clock_t start = clock();
    
    double dx = (cfg->x_max - cfg->x_min) / cfg->width;
    double dy = (cfg->y_max - cfg->y_min) / cfg->height;
    
    // Runtime CPU feature detection - FIXED
    int use_avx2 = cfg->use_avx2 && (cfg->cpu_features & CPU_FEATURE_AVX2);
    int use_sse4 = cfg->use_sse4 && (cfg->cpu_features & CPU_FEATURE_SSE4_2);
    
    for (int y = data->start_row; y < data->end_row; y++) {
        double y0 = cfg->y_min + y * dy;
        
        int x = 0;
        
        // AVX2 - 4 pixels at a time
// In render_thread function, inside the AVX2 section:
#ifdef __AVX2__
if (use_avx2) {
    for (; x <= cfg->width - 4; x += 4) {
        // Aligned array for AVX2
        double x_coords[4] __attribute__((aligned(32)));
        for (int k = 0; k < 4; k++) {
            x_coords[k] = cfg->x_min + (x + k) * dx;
        }
        
        // Aligned load (faster and required for alignment)
        __m256d x0_vec = _mm256_load_pd(x_coords);
        __m256d y0_vec = _mm256_set1_pd(y0);
        
        int results[4];  // <-- This is where results is declared
        mandelbrot_avx2(x0_vec, y0_vec, cfg->max_iter, results);
        
        // DEBUG PRINT - add this here
        if (y == 0 && x < 16) {  // Only print first few pixels of first row
            printf("AVX2 results at (%d,%d): %d %d %d %d (x0: %.3f,%.3f,%.3f,%.3f)\n", 
                   x, y, results[0], results[1], results[2], results[3],
                   x_coords[0], x_coords[1], x_coords[2], x_coords[3]);
        }
        
        for (int k = 0; k < 4; k++) {
            int idx = (y * cfg->width + (x + k)) * 3;
            uint8_t r, g, b;
            get_color(results[k], cfg->max_iter, cfg->grayscale, 
            cfg->gamma, cfg->brightness,&r, &g, &b);
            data->image[idx] = r;
            data->image[idx + 1] = g;
            data->image[idx + 2] = b;
        }
    }
}
#endif

        // SSE4 - 2 pixels at a time
#ifdef __SSE4_2__
        if (use_sse4 && !use_avx2) {  // Don't use both
            for (; x <= cfg->width - 2; x += 2) {
                double x_coords[2];
                for (int k = 0; k < 2; k++) {
                    x_coords[k] = cfg->x_min + (x + k) * dx;
                }
                
                // Use unaligned load - FIXED
                __m128d x0_vec = _mm_loadu_pd(x_coords);
                __m128d y0_vec = _mm_set1_pd(y0);
                
                int results[2];
                mandelbrot_sse4(x0_vec, y0_vec, cfg->max_iter, results);
                
                for (int k = 0; k < 2; k++) {
                    int idx = (y * cfg->width + (x + k)) * 3;
                    uint8_t r, g, b;
                    get_color(results[k], cfg->max_iter, cfg->grayscale, 
                              cfg->gamma, cfg->brightness, &r, &g, &b);
                    data->image[idx] = r;
                    data->image[idx + 1] = g;
                    data->image[idx + 2] = b;
                }
            }
        }
#endif
        
        // Scalar for remainder
        for (; x < cfg->width; x++) {
            double x0_scalar = cfg->x_min + x * dx;
            int iter = mandelbrot_scalar(x0_scalar, y0, cfg->max_iter);
            
            int idx = (y * cfg->width + x) * 3;
            uint8_t r, g, b;
            get_color(iter, cfg->max_iter, cfg->grayscale,
                      cfg->gamma, cfg->brightness, &r, &g, &b);
            data->image[idx] = r;
            data->image[idx + 1] = g;
            data->image[idx + 2] = b;
        }
    }
    
    clock_t end = clock();
    // FIXED: Store timing in thread index instead of row index
    int thread_index = data->start_row / ((cfg->height + cfg->num_threads - 1) / cfg->num_threads);
    if (thread_index >= 0 && thread_index < cfg->num_threads) {
        data->timings[thread_index] = (double)(end - start) / CLOCKS_PER_SEC;
    }
    
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ===========================================
   GPU OpenCL Support (Optional)
   =========================================== */

#ifdef USE_OPENCL
#include <CL/cl.h>

/* ===========================================
   OpenCL GPU Kernel Source Code (MUST be at file scope, not inside function!)
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
    
    // Create command queue with profiling
    g_ctx.queue = clCreateCommandQueue(g_ctx.context, g_ctx.device, 
                                       CL_QUEUE_PROFILING_ENABLE, &err);
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

#endif  // USE_OPENCL

/* ===========================================
   Enhanced render_mandelbrot function with GPU support - AVX2 + OPENMP
   =========================================== */

static void render_mandelbrot(Config* cfg, uint8_t* image) {
    clock_t total_start = clock();
    
    // GPU rendering (optional)
    if (cfg->use_gpu) {
        printf("\nGPU Rendering...\n");
        double gpu_time = gpu_render(cfg, image);
        if (gpu_time > 0) {
            printf("GPU: %.3f seconds\n", gpu_time);
            return;
        }
        printf("GPU failed, using CPU\n");
        cfg->use_gpu = 0;
    }
    
    printf("\nCPU Rendering...\n");
    
    // Check if we should use AVX2
    int use_avx2 = cfg->use_avx2 && (cfg->cpu_features & CPU_FEATURE_AVX2);
    printf("Using: %s\n", use_avx2 ? "AVX2 + OpenMP" : "Scalar + OpenMP");
    
    // Configure threads - FIXED
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
        
// In the AVX2 section of render_mandelbrot:
#ifdef __AVX2__
if (use_avx2) {
    for (; x <= cfg->width - 4; x += 4) {
        // Use aligned memory for AVX2
        double x_coords[4] __attribute__((aligned(32)));
        for (int k = 0; k < 4; k++) {
            x_coords[k] = cfg->x_min + (x + k) * dx;
        }
        
        // Use aligned load
        __m256d x0_vec = _mm256_load_pd(x_coords);  // Changed to load_pd
        __m256d y0_vec = _mm256_set1_pd(y0);
        
        int results[4];
        mandelbrot_avx2(x0_vec, y0_vec, cfg->max_iter, results);
        
        // DEBUG: Print first few results
        if (y < 2 && x < 16) {
            printf("AVX2 at (%d,%d): results=%d,%d,%d,%d\n", 
                   x, y, results[0], results[1], results[2], results[3]);
        }
        
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
   PPM File I/O - ADD THIS COMPLETE FUNCTION!
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
   Command Line Parsing
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
    printf("  --help                Show this help\n\n");
    
    printf("Examples:\n");
    printf("  mandelbrot -w 1280 -h 720 -i 5000 -o output.ppm\n");
    printf("  mandelbrot -x -0.748 -X -0.744 -y 0.1 -Y 0.104 -i 10000\n");
    printf("  mandelbrot --grayscale --benchmark\n");
}

static void parse_args(int argc, char** argv, Config* cfg) {
    // Defaults
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

    cfg->gamma = 0.3f;      // Default gamma for grayscale
    cfg->brightness = 1.0f; // Default brightness

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
            if (++i < argc) cfg->x_min = atof(argv[i]);
        } else if (strcmp(argv[i], "--xmax") == 0 || strcmp(argv[i], "-X") == 0) {
            if (++i < argc) cfg->x_max = atof(argv[i]);
        } else if (strcmp(argv[i], "--ymin") == 0 || strcmp(argv[i], "-y") == 0) {
            if (++i < argc) cfg->y_min = atof(argv[i]);
        } else if (strcmp(argv[i], "--ymax") == 0 || strcmp(argv[i], "-Y") == 0) {
            if (++i < argc) cfg->y_max = atof(argv[i]);
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
        render_mandelbrot(cfg, test_image);
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
   Main Function
   =========================================== */

int main(int argc, char** argv) {
    Config cfg;
    parse_args(argc, argv, &cfg);
    
    printf("=== Ultimate Mandelbrot Renderer ===\n");
    printf("Resolution: %dx%d\n", cfg.width, cfg.height);
    printf("Iterations: %d\n", cfg.max_iter);
    printf("Viewport: [%.6f, %.6f] x [%.6f, %.6f]\n", 
           cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);
    printf("CPU Features: ");
    if (cfg.cpu_features & CPU_FEATURE_AVX2) printf("AVX2 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE4_2) printf("SSE4.2 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE4_1) printf("SSE4.1 ");
    if (cfg.cpu_features & CPU_FEATURE_SSE3) printf("SSE3 ");
    if (cfg.cpu_features == CPU_FEATURE_NONE) printf("Scalar only");
    printf("\n");
    
    // =========== TEST CODE ===========
    printf("\n=== Testing AVX2 vs Scalar ===\n");
    double test_points[4] = {-2.0, -1.0, 0.0, 0.25};
    double test_y = 0.0;
    int test_max_iter = 100;
    
    printf("Testing with max_iter = %d\n", test_max_iter);
    
    // Test scalar for each point
    for (int i = 0; i < 4; i++) {
        int scalar_result = mandelbrot_scalar(test_points[i], test_y, test_max_iter);
        printf("Scalar at (%.2f, %.2f): %d\n", test_points[i], test_y, scalar_result);
    }
    
    // Test AVX2
#ifdef __AVX2__
    if (cfg.cpu_features & CPU_FEATURE_AVX2) {
        // Note: _mm256_set_pd loads in REVERSE order! Last argument goes to first position
        __m256d x0_test = _mm256_set_pd(0.25, 0.0, -1.0, -2.0);  // Reversed!
        __m256d y0_test = _mm256_set1_pd(test_y);
        int avx2_results[4];
        mandelbrot_avx2(x0_test, y0_test, test_max_iter, avx2_results);
        
        printf("AVX2 results: [0]=%d, [1]=%d, [2]=%d, [3]=%d\n", 
               avx2_results[0], avx2_results[1], avx2_results[2], avx2_results[3]);
        printf("Expected order (AVX2): -2.0, -1.0, 0.0, 0.25\n");
        printf("Expected values: ~1, 100, 100, 100\n");
        
        // Check if results make sense
        if (avx2_results[0] <= 5 && avx2_results[1] == 100 && 
            avx2_results[2] == 100 && avx2_results[3] == 100) {
            printf("AVX2 TEST: PASSED! ✓\n");
        } else {
            printf("AVX2 TEST: FAILED! ✗\n");
        }
    } else {
        printf("AVX2 not available on this CPU\n");
    }
#else
    printf("AVX2 not compiled in\n");
#endif
    // =========== END TEST CODE ===========
    
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
    
    // Render
    render_mandelbrot(&cfg, image);
    
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
}  // <-- This closes main()
