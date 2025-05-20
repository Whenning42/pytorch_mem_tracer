import os
import re as _re
import sys

import torch
from cffi import FFI

ffi = FFI()
ffi.cdef(
    """
void push_line(const char *line);
void report_stats();
"""
)
lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), "allocator.so"), ffi.RTLD_LAZY)


def make_tracer(patterns: list[str] | None = None):
    if patterns is None:
        patterns = [".*"]

    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        filename = frame.f_code.co_filename
        for pat in patterns:
            try:
                if _re.search(pat, filename):
                    loc = f"{os.path.abspath(filename)}:{frame.f_lineno} ({frame.f_code.co_name})"
                    lib.push_line(loc.encode("utf-8"))
                    break
            except AttributeError as e:
                pass
        return tracer

    return tracer


def switch_allocator():
    """Needs to be called before any torch allocations are performed."""
    allocator_path = os.path.join(os.path.dirname(__file__), "allocator.so")
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        allocator_path,
        "malloc_impl",
        "free_impl",
    )
    torch.cuda.memory.change_current_allocator(new_alloc)


def start_tracing(*patterns):
    sys.settrace(make_tracer(*patterns))


def stop_tracing():
    sys.settrace(None)


def report_stats():
    lib.report_stats()
