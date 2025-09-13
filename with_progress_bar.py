
import os, time, functools, threading, sys
from contextlib import contextmanager
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

PROGRESS_ENABLED = True#os.environ.get("MIKA_PROGRESS", "1") != "0"

def _animate_bar(desc: str, stop_evt: threading.Event, interval: float = 0.05):
    if not PROGRESS_ENABLED:
        return
    if tqdm:
        total = 200
        with tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True,
                  bar_format="{l_bar}{bar} {elapsed}") as bar:
            i = 0
            while not stop_evt.is_set():
                bar.update(1)
                i = (i + 1) % total
                if i == 0:
                    bar.n = 0
                    bar.last_print_n = 0
                    bar.refresh()
                time.sleep(interval)
    else:
        spin = "|/-\\"
        i = 0
        while not stop_evt.is_set():
            print(f"\r[{desc}] {spin[i % len(spin)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)
        if not tqdm:
            print("\r" + " " * (len(desc) + 6) + "\r", end="")

@contextmanager
def progress(desc: str):
    stop_evt = threading.Event()
    t = threading.Thread(target=_animate_bar, args=(desc, stop_evt), daemon=True)
    t.start()
    start = time.time()
    exc = None
    try:
        yield
    except Exception as e:
        exc = e
        raise
    finally:
        stop_evt.set()
        t.join()
        elapsed = time.time() - start
        # final status line
        status = "✓" if exc is None else "✗"
        print(f"{desc}: {status} ({elapsed:.2f}s)")

def with_progress(desc: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not PROGRESS_ENABLED:
                return fn(*args, **kwargs)
            with progress(desc):
                result = fn(*args, **kwargs)
            # ensure the next prints start on a fresh line after the bar
            print()
            #sys.stdout.flush()
            return result
        return wrapper
    return deco

def wrap_progress(ns: dict, mapping: dict[str, str]):
    """Wrap existing callables by name with progress/timing."""
    for name, desc in mapping.items():
        fn = ns.get(name)
        if callable(fn):
            fancy = f" ~ | [{desc}] "
            ns[name] = with_progress(desc)(fn)
        else:
            print(f"[wrap_progress] Skipped: {name} not found or not callable")
