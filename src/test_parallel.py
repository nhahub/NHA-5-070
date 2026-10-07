import time
from concurrent.futures import ThreadPoolExecutor


def fake_tool(name, seconds):
    print(f"{name} started")

    time.sleep(seconds)

    print(f"{name} finished")

    return name


def run_sequential():
    start = time.perf_counter()

    fake_tool("notes", 1)
    fake_tool("style", 1)
    fake_tool("profile", 1)

    return time.perf_counter() - start


def run_parallel():
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fake_tool, "notes", 1),
            executor.submit(fake_tool, "style", 1),
            executor.submit(fake_tool, "profile", 1),
        ]

        for future in futures:
            future.result()

    return time.perf_counter() - start


print("===== SEQUENTIAL =====")

sequential_time = run_sequential()

print(f"Sequential time: {sequential_time:.2f} seconds")


print("\n===== PARALLEL =====")

parallel_time = run_parallel()

print(f"Parallel time: {parallel_time:.2f} seconds")


print("\n===== COMPARISON =====")

print(
    f"Speedup: "
    f"{sequential_time / parallel_time:.2f}x"
)