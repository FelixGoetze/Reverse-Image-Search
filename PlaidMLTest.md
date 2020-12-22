# Testing PlaidML

Mobilenet inference time for 1024 examples
GPU: 9.268s
CPU: 129.153s
Intel GPU: 25.425s

## GPU:

╰─❯ plaidbench keras mobilenet
Running 1024 examples with mobilenet, batch size 1, on backend plaid
INFO:plaidml:Opening device "metal_amd_radeon_pro_460.0"
Compiling network... Warming up... Running...
Example finished, elapsed: 3.112s (compile), 9.268s (execution)

---

## Network Name Inference Latency Time / FPS

mobilenet 9.05 ms 0.00 ms / 1000000000.00 fps
Correctness: PASS, max_error: 1.675534622336272e-05, max_abs_error: 7.674098014831543e-07, fail_ratio: 0.0

## CPU:

╰─❯ plaidbench keras mobilenet
Running 1024 examples with mobilenet, batch size 1, on backend plaid
INFO:plaidml:Opening device "llvm_cpu.0"
Compiling network... Warming up... Running...
Example finished, elapsed: 3.659s (compile), 129.153s (execution)

---

## Network Name Inference Latency Time / FPS

mobilenet 126.13 ms 122.86 ms / 8.14 fps
Correctness: PASS, max_error: 1.7640328223933466e-05, max_abs_error: 7.040798664093018e-07, fail_ratio: 0.0

## Intel GPU

╰─❯ plaidbench keras mobilenet
Running 1024 examples with mobilenet, batch size 1, on backend plaid
INFO:plaidml:Opening device "metal_intel(r)\_hd_graphics_530.0"
Compiling network... Warming up... Running...
Example finished, elapsed: 6.070s (compile), 25.425s (execution)

---

## Network Name Inference Latency Time / FPS

mobilenet 24.83 ms 0.00 ms / 1000000000.00 fps
Correctness: PASS, max_error: 6.440454399125883e-06, max_abs_error: 5.811452865600586e-07, fail_ratio: 0.0
