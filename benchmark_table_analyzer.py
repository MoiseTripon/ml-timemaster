# benchmark_table_analyzer.py
import glob
import time
import csv
from statistics import mean, median
import cv2

from src.table_analyzer import TableAnalyzer

def run_once(img, analyzer):
    t0 = time.perf_counter()
    tb = analyzer.detect_table_borders(img)
    t1 = time.perf_counter()
    cells = analyzer.detect_cells(img, tb)
    cells = analyzer.remove_duplicate_cells(cells)
    t2 = time.perf_counter()
    grid, r, c = analyzer.organize_cells_into_grid(cells)
    t3 = time.perf_counter()

    return {
        "t_borders_ms": (t1 - t0) * 1000,
        "t_cells_ms": (t2 - t1) * 1000,
        "t_grid_ms": (t3 - t2) * 1000,
        "t_total_ms": (t3 - t0) * 1000,
        "cells_after_dedupe": len(cells),
        "rows": r,
        "cols": c,
    }

def main():
    paths = sorted(glob.glob("test_images/*.*"))
    if not paths:
        print("No images found in test_images/. Add some PNG/JPG files.")
        return

    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable: {p}")
            continue
        imgs.append((p, img))

    analyzer = TableAnalyzer()

    # Warm-up (JIT caches, OpenCV lazy inits)
    for _, img in imgs[:2]:
        _ = run_once(img, analyzer)

    rows = []
    for path, img in imgs:
        metrics = run_once(img, analyzer)
        rows.append({"path": path, **metrics})
        print(f"{path}: {metrics['t_total_ms']:.1f} ms, "
              f"borders {metrics['t_borders_ms']:.1f} ms, "
              f"cells {metrics['t_cells_ms']:.1f} ms, "
              f"grid {metrics['t_grid_ms']:.1f} ms, "
              f"cells {metrics['cells_after_dedupe']}, "
              f"{metrics['rows']}x{metrics['cols']}")

    totals = [r["t_total_ms"] for r in rows]
    print("\n=== Summary ===")
    print(f"images: {len(rows)}")
    print(f"avg:    {mean(totals):.1f} ms/img")
    print(f"median: {median(totals):.1f} ms/img")
    print(f"min:    {min(totals):.1f} ms   max: {max(totals):.1f} ms")

    # Write CSV for later comparisons
    out_csv = "/tmp/table_bench.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote results to {out_csv}")

if __name__ == "__main__":
    main()