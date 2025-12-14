#!/usr/bin/env python3
"""
Benchmark script for mandelbrot_fast
"""

import time
from mandelbrot_fast import render_mandelbrot

resolutions = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (3840, 2160)
]

iterations = [256, 1000, 4096, 32768]

print("Mandelbrot Render Benchmark")
print("=" * 50)

for w, h in resolutions:
    for max_iter in iterations:
        print(f"\n{w}x{h}, {max_iter} iterations:")
        
        start = time.time()
        img = render_mandelbrot(width=w, height=h, max_iter=max_iter)
        elapsed = time.time() - start
        
        megapixels = (w * h) / 1_000_000
        speed = megapixels / elapsed if elapsed > 0 else 0
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {speed:.2f} MP/s")
        
        # Save small samples for verification
        if w <= 1920 and max_iter <= 4096:
            img.save(f"bench_{w}x{h}_iter{max_iter}.png")

print("\n" + "=" * 50)
print("Benchmark complete! Sample images saved.")
