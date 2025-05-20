# Pytorch Mem Tracer 

A simple Pytorch memory tracer.

This memory tracer allows you to trace a section of pytorch code and get a Python
line level breakdown of where device memory was allocated when the section hit its
peak memory usage level.


# Usage

To get started, you'll need to build the allocator's dynamic library. For this step,
you'll need to have cuda installed on your machine and you'll probably need to update
the makefile to point the cuda include path to your system's cuda include directory.

Then, to use the tracer, run you script with this directory in the PYTHONPATH and use
this python code:

``` python
# Before allocating any tensors.
import pytorch_mem_tracer as mem_tracer
mem_tracer.switch_allocator()

# Around the function you want to trace.
mem_tracer.start_tracing()
model(args) # Example function to trace
mem_tracer.stop_tracing()
mem_tracer.report_stats()

```


# Example Output 

Here's an example run of the profiler from a forward pass with Gemma 3 4B bf16 inference
using Google's [reference pytorch implementation](https://github.com/google/gemma_pytorch)
with an 8192 token context window:

Note: Allocations that are performed between the .switch_allocator() call and the
.start_tracing() call are reported with a Location of "untraced".

```
Max mem: 13.54 GB
Total mem allocated: 27.16 GB
Mem, Cumulative Mem, Median, Count, Location
8.040 GB  8.040 GB  5.000 KB  818  untraced
2.000 GB  10.04 GB  2.000 GB  1  .../gemma_pytorch/gemma/model.py:336 (forward)
1.000 GB  11.04 GB  1.000 GB  1  ../.venv/lib/python3.10/site-packages/torch/nn/functional.py:2143 (softmax)
544.1 MB  11.57 GB  16.00 MB  34  .../gemma_pytorch/gemma/gemma3_model.py:324 (generate)
544.1 MB  12.10 GB  16.00 MB  34  .../gemma_pytorch/gemma/gemma3_model.py:325 (generate)
256.1 MB  12.35 GB  256.1 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:307 (generate)
256.1 MB  12.60 GB  256.1 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:310 (generate)
256.0 MB  12.85 GB  256.0 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:343 (generate)
256.0 MB  13.10 GB  256.0 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:344 (generate)
72.12 MB  13.17 GB  64.00 MB  2  .../gemma_pytorch/gemma/model.py:133 (forward)
64.02 MB  13.24 GB  64.02 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:261 (create_attention_mask)
64.02 MB  13.30 GB  64.02 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:265 (create_attention_mask)
48.00 MB  13.35 GB  32.00 MB  2  .../gemma_pytorch/gemma/model.py:105 (apply_rotary_emb)
40.00 MB  13.39 GB  40.00 MB  1  .../gemma_pytorch/gemma/gemma3_model.py:141 (forward)
40.00 MB  13.42 GB  40.00 MB  1  .../gemma_pytorch/gemma/model.py:462 (forward)
40.00 MB  13.46 GB  40.00 MB  1  .../gemma_pytorch/gemma/model.py:186 (forward)
32.00 MB  13.49 GB  32.00 MB  1  .../gemma_pytorch/gemma/model.py:300 (forward)
32.00 MB  13.53 GB  32.00 MB  1  .../gemma_pytorch/gemma/model.py:299 (forward)
16.00 MB  13.54 GB  8.000 MB  2  .../.venv/lib/python3.10/site-packages/torch/nn/modules/module.py:1923 (__getattr__)
...
```

With this output, and some light investigation one can easily breakdown the model's memory
usage as:
- 8.04 GB for the model parameters. (output line 1)
- 3.00 GB for the full attention matrix. (output lines 2 and 3)
- 1.08 GB for the KV cache. (output lines 4 and 5)
- 1.54 GB other.


# Limitations

The tracer is somewhat slow when stepping through Python code, so try not trace too many
lines of Python code.

The tracer currently doesn't support any configurability as to how stats are reported
or which devices' memory usages are tracked. With that said, the tracer is super simple,
so if you want those features, it's simple to fork this repo and add them.
