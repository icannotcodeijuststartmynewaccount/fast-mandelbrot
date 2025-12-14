"""
Fast Mandelbrot Set Renderer using Numba and NumPy
Author: [Your Name]
License: MIT
"""

import numpy as np
from PIL import Image
from numba import jit, prange
import time
import os
import threading
from tqdm import tqdm


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def mandelbrot_kernel(c_real, c_imag, max_iter, progress_counter):
    """
    Core Mandelbrot calculation kernel.
    """
    height, width = c_real.shape
    divtime = np.zeros((height, width), dtype=np.int32)
    
    for i in prange(height):
        for j in range(width):
            cr = c_real[i, j]
            ci = c_imag[i, j]
            zr, zi = 0.0, 0.0

            # Optimized escape loop
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
        
        # Progress tracking
        progress_counter[i] = 1
    
    return divtime


def monitor_progress(progress_counter, total_rows):
    """
    Background thread for tqdm progress bar.
    """
    pbar = tqdm(total=total_rows, desc="Rendering Mandelbrot", unit="rows",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}]')
    
    last_completed = 0
    while last_completed < total_rows:
        current_completed = np.sum(progress_counter)
        if current_completed > last_completed:
            pbar.update(current_completed - last_completed)
            last_completed = current_completed
        time.sleep(0.05)
    
    pbar.close()


def render_mandelbrot(width=1920, height=1080, max_iter=32768,
                      x_min=-2.0, x_max=0.5, y_min=-1.25, y_max=1.25):
    """
    Render the Mandelbrot set with progress bar.
    
    Parameters:
    -----------
    width, height : int
        Output image dimensions
    max_iter : int
        Maximum iterations per pixel
    x_min, x_max : float
        X-axis bounds in complex plane
    y_min, y_max : float
        Y-axis bounds in complex plane
    
    Returns:
    --------
    PIL.Image
        Grayscale image of the Mandelbrot set
    """
    print(f"Rendering {width}x{height} with {max_iter} iterations...")
    
    # Create coordinate grid
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    C = x + y[:, None] * 1j
    
    # Progress tracking setup
    progress_counter = np.zeros(height, dtype=np.int32)
    
    # Start progress monitor
    monitor_thread = threading.Thread(target=monitor_progress,
                                      args=(progress_counter, height))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Render
    start_time = time.time()
    mandelbrot_set = mandelbrot_kernel(C.real, C.imag, max_iter, progress_counter)
    render_time = time.time() - start_time
    
    # Wait for progress thread
    monitor_thread.join(timeout=1.0)
    
    # Convert to image
    mandelbrot_set = np.log(mandelbrot_set + 1)
    mandelbrot_set = (mandelbrot_set / np.max(mandelbrot_set)) * 255
    mandelbrot_set = np.uint8(mandelbrot_set)
    
    img = Image.fromarray(mandelbrot_set, mode='L')
    
    # Performance stats
    megapixels = (width * height) / 1e6
    print(f"\n{'='*60}")
    print(f"RENDER COMPLETE!")
    print(f"Resolution: {width}x{height}")
    print(f"Iterations: {max_iter}")
    print(f"Render time: {render_time:.2f} seconds")
    print(f"Speed: {megapixels/render_time:.2f} megapixels/sec")
    print(f"{'='*60}")
    
    return img


if __name__ == "__main__":
    # Example: Render a zoomed-in region
    img = render_mandelbrot(
        width=1920,
        height=1080,
        max_iter=32768,
        x_min=-0.178,
        x_max=-0.148,
        y_min=-1.0409375,
        y_max=-1.0240625
    )
    
    # Save to Downloads folder
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'mandelbrot_fast.png')
    img.save(downloads_path)
    print(f"\nImage saved to: {downloads_path}")
    img.show()
