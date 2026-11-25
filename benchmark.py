import argparse
import platform
import subprocess
import sys
from statistics import median
from time import perf_counter

import numpy as np

from bonus_a import check

LEFT = "(A|B)&(C|D)"
RIGHT = "(A&C)|(A&D)|(B&C)|(B&D)"


def measure(n, fast, runs, repeats):
    expected = 16 if fast else 1 << (4 * n)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(runs):
            result = check(LEFT, RIGHT, n, fast)
            if result != (True, expected, None):
                raise RuntimeError(f"Unexpected result at n={n}: {result}")
        times.append((perf_counter() - start) / runs)
    return median(times), expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    cpu = platform.processor()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        cpu = result.stdout.strip() or cpu
    print(f"{platform.platform()}; {cpu}; Python {platform.python_version()}; "
          f"NumPy {np.__version__}", file=sys.stderr)
    print("n,old_seconds,new_seconds,old_checks,new_checks,old_repeats,new_repeats,new_runs",
          flush=True)
    for n in range(1, args.max_n + 1):
        old_repeats = 5 if n <= 6 else 1
        new_repeats, new_runs = 7, 500
        check(LEFT, RIGHT, n, True)
        fast, fast_checks = measure(n, True, new_runs, new_repeats)
        slow, slow_checks = measure(n, False, 1, old_repeats)
        print(n, f"{slow:.12g}", f"{fast:.12g}", slow_checks, fast_checks,
              old_repeats, new_repeats, new_runs, sep=",", flush=True)


if __name__ == "__main__":
    main()
