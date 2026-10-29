# NPU Benchmark Report
Generated: 2026-10-16 20:13:35

## Environment

- **Torch version**: 2.10.0+cpu
- **torch_npu version**: 2.10.0.post5.dev20260815
- **NPU model**: Ascend950PR_9579
- **NPU count**: 2
- **Profiling mode**: msprof

## LogSumExpFwdOp

### tileops

| Label | Latency (us) | tflops | bandwidth_tbs | msprof | prof_mode | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn-weights-4k | 53.0042 | 0.3165 | 0.1583 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': -1} | {'block_m': 16, 'tile_n': 0} | float16 | [32, 32, 4096] |
| attn-weights-4k | 52.9656 | 0.3168 | 0.1584 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': -1} | {'block_m': 16, 'tile_n': 0} | bfloat16 | [32, 32, 4096] |
| attn-weights-32k | 110.3945 | 1.2158 | 0.6079 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': -1} | {'block_m': 1, 'tile_n': 16384} | bfloat16 | [32, 32, 32768] |
| lm-head-logits | 20.3497 | 0.0805 | 0.0403 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': -1} | {'block_m': 1, 'tile_n': 25600} | float16 | [4, 102400] |
| lm-head-logits | 20.3773 | 0.0804 | 0.0402 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': -1} | {'block_m': 1, 'tile_n': 25600} | bfloat16 | [4, 102400] |
| 3d-multidim-reduce | 24.8058 | 0.3382 | 0.1691 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | {'dim': [0, 2]} | {'block_m': 4, 'tile_n': 0} | float16 | [4, 128, 4096] |

### torch

| Label | Latency (us) | tflops | bandwidth_tbs | msprof | prof_mode | dim | keepdim | op_params | result_bl | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn-weights-4k | 49.4815 | 0.3391 | 0.1696 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | -1 | False | {'dim': -1} | {'bandwidth_tbs': 0.16957169892377968, 'latency_us': 49.481464497041415, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.339060619376032} | {'block_m': 16, 'tile_n': 0} | float16 | [32, 32, 4096] |
| attn-weights-4k | 49.6129 | 0.3382 | 0.1691 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | -1 | False | {'dim': -1} | {'bandwidth_tbs': 0.16912260042056543, 'latency_us': 49.612860606060615, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.3381626416024584} | {'block_m': 16, 'tile_n': 0} | bfloat16 | [32, 32, 4096] |
| attn-weights-32k | 400.9695 | 0.3347 | 0.1674 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | -1 | False | {'dim': -1} | {'bandwidth_tbs': 0.1673716238948721, 'latency_us': 400.969474025974, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.33473303254827236} | {'block_m': 1, 'tile_n': 16384} | bfloat16 | [32, 32, 32768] |
| lm-head-logits | 29.5971 | 0.0554 | 0.0277 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | -1 | False | {'dim': -1} | {'bandwidth_tbs': 0.0276786230801222, 'latency_us': 29.597137026239068, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.0553567055674166} | {'block_m': 1, 'tile_n': 25600} | float16 | [4, 102400] |
| lm-head-logits | 29.6354 | 0.0553 | 0.0276 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | -1 | False | {'dim': -1} | {'bandwidth_tbs': 0.02764287368211611, 'latency_us': 29.63541379310345, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.05528520746962802} | {'block_m': 1, 'tile_n': 25600} | bfloat16 | [4, 102400] |
| 3d-multidim-reduce | 38.7017 | 0.2168 | 0.1084 | {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'} | msprof | [0, 2] | False | {'dim': [0, 2]} | {'bandwidth_tbs': 0.10838181005740212, 'latency_us': 38.70169724770643, 'msprof': {'backend': 'msprof', 'latency_source': 'do_bench(msprof)'}, 'prof_mode': 'msprof', 'tflops': 0.21675039072083938} | {'block_m': 4, 'tile_n': 0} | float16 | [4, 128, 4096] |
