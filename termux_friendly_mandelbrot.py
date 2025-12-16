"""
Mandelbrot Termux Edition
Auto-detects environment and uses appropriate backend.
"""

import sys
import os

# Detect if running on Termux
def is_termux():
    """Check if running in Termux environment."""
    return 'com.termux' in os.environ.get('PREFIX', '')

# Try to import Numba, fallback if not available
HAS_NUMBA = False
if not is_termux():
    try:
        from numba import jit, prange
        HAS_NUMBA = True
        print("✅ Numba available - using accelerated backend")
    except ImportError:
        print("⚠️  Numba not found - using NumPy backend")
else:
    print("📱 Termux detected - using optimized NumPy backend")

# Import other dependencies
import numpy as np
from PIL import Image
import time
from tqdm import tqdm


# Conditional function definition
if HAS_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def mandelbrot_kernel(c_real, c_imag, max_iter, progress_counter=None):
        """Numba-accelerated kernel (not available on Termux)."""
        height, width = c_real.shape
        divtime = np.zeros((height, width), dtype=np.int32)
        
        for i in prange(height):
            for j in range(width):
                cr = c_real[i, j]
                ci = c_imag[i, j]
                zr, zi = 0.0, 0.0
                
                for k in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4.0:
                        divtime[i, j] = k
                        break
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
                else:
                    divtime[i, j] = max_iter - 1
        
        return divtime
else:
    def mandelbrot_kernel(c_real, c_imag, max_iter, progress_counter=None):
        """NumPy-based kernel (works everywhere, including Termux)."""
        height, width = c_real.shape
        divtime = np.zeros((height, width), dtype=np.int32)
        threshold = 4.0
        
        # Tiled processing for memory efficiency
        tile_size = min(256, height)  # Smaller tiles for Termux
        
        for ty in range(0, height, tile_size):
            tile_end = min(ty + tile_size, height)
            tile_height = tile_end - ty
            
            for tx in range(0, width, tile_size):
                tile_end_x = min(tx + tile_size, width)
                tile_width = tile_end_x - tx
                
                # Process tile
                c_real_tile = c_real[ty:tile_end, tx:tile_end_x]
                c_imag_tile = c_imag[ty:tile_end, tx:tile_end_x]
                
                z_real = np.zeros_like(c_real_tile, dtype=np.float64)
                z_imag = np.zeros_like(c_imag_tile, dtype=np.float64)
                tile_result = np.zeros((tile_height, tile_width), dtype=np.int32)
                mask = np.ones((tile_height, tile_width), dtype=bool)
                
                for k in range(max_iter):
                    if not np.any(mask):
                        break
                    
                    zr2 = z_real[mask] * z_real[mask]
                    zi2 = z_imag[mask] * z_imag[mask]
                    
                    z_imag[mask] = 2.0 * z_real[mask] * z_imag[mask] + c_imag_tile[mask]
                    z_real[mask] = zr2 - zi2 + c_real_tile[mask]
                    
                    diverged = (zr2 + zi2) > threshold
                    if np.any(diverged):
                        div_indices = np.where(mask)
                        tile_result[div_indices[0][diverged], div_indices[1][diverged]] = k
                        mask[div_indices[0][diverged], div_indices[1][diverged]] = False
                
                tile_result[mask] = max_iter - 1
                divtime[ty:tile_end, tx:tile_end_x] = tile_result
        
        return divtime


def render_mandelbrot(width=1280, height=720, max_iter=4096,
                      x_min=-2.0, x_max=0.5, y_min=-1.25, y_max=1.25):
    """Main render function - auto-selects best backend."""
    
    print(f"{'='*60}")
    print(f"MANDELBROT RENDERER")
    print(f"{'='*60}")
    print(f"Backend: {'Numba' if HAS_NUMBA else 'NumPy (Termux Optimized)'}")
    print(f"Resolution: {width}x{height}")
    print(f"Iterations: {max_iter}")
    print(f"{'='*60}")
    
    # Create coordinate grid
    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # Render with timing
    start_time = time.time()
    mandelbrot_set = mandelbrot_kernel(X, Y, max_iter)
    render_time = time.time() - start_time
    
    # Apply coloring
    mandelbrot_set = np.log(mandelbrot_set + 1)
    mandelbrot_set = (mandelbrot_set / np.max(mandelbrot_set)) * 255
    mandelbrot_set = np.uint8(mandelbrot_set)
    
    # Create image
    img = Image.fromarray(mandelbrot_set, mode='L')
    
    # Performance stats
    megapixels = (width * height) / 1e6
    speed = megapixels / render_time if render_time > 0 else 0
    
    print(f"\nRENDER COMPLETE!")
    print(f"Time: {render_time:.2f} seconds")
    print(f"Speed: {speed:.2f} MPix/sec")
    print(f"{'='*60}")
    
    return img, render_time


if __name__ == "__main__":
    # Auto-adjust for Termux
    if is_termux():
        # Smaller settings for Termux
        width, height = 800, 600
        max_iter = 1024
        print("📱 Termux detected - using mobile-optimized settings")
    else:
        # Full settings for desktop
        width, height = 1280, 720
        max_iter = 4096
    
    img, time_taken = render_mandelbrot(
        width=width,
        height=height,
        max_iter=max_iter,
        x_min=-0.178,
        x_max=-0.148,
        y_min=-1.0409375,
        y_max=-1.0240625
    )
    
    # Save
    filename = f"mandelbrot_{'termux' if is_termux() else 'desktop'}.png"
    img.save(filename)
    print(f"Saved: {filename}")