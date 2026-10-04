# GPU backends

*What a device backend is in Vx, which ones exist, and how a new one gets in.*

This is for contributors who want to bring Vx to a new device, and for users
who want to know what runs where. Related:
[`adding_a_topology.md`](adding_a_topology.md) (declaring the hardware),
[`spawn_on.md`](spawn_on.md) (the execution model),
[`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) (building and testing).

______________________________________________________________________

## What a backend is

A backend is three pieces. The NVIDIA backend is the worked example for each.

1. **A machine file.** The hardware is declared in a `.vx` file: its memory
   spaces and their capacities, the transfer edges between them, the `arch:`
   it executes, and `dtypes:` — the element types it supports. No compiler
   change is needed for this part; see [`adding_a_topology.md`](adding_a_topology.md)
   and the examples in [`fleet/`](../fleet/).
1. **A device image compiler.** `spawn on(Topology::X) { ... }` is outlined
   into a kernel, and eligible kernels are cloned into a standard MLIR
   `gpu.module`, which no vendor owns. From there, `deviceImageOf()` in
   `src/dialect/VxLowering.cpp` runs the NVVM passes and produces PTX. A new
   backend adds a sibling of that function producing its own image format
   (for example SPIR-V, through MLIR's `convert-gpu-to-spirv` or the XeVM
   target).
1. **A runtime dispatch library.** Compiled programs call a small C interface
   (`include/vx_hardware_runtime.h`): allocate and copy in, launch a kernel,
   wait, copy out, free. `runtime/cuda_dispatch.cpp` is the CUDA
   implementation; a new backend adds its own file next to it and an arm in
   `build.rs`. The argument marshalling in `runtime/vx_kernel_launch.h` is
   vendor-free, unit-tested on the host, and meant to be reused.

## What a backend does not do

The language, the type checker, and the placement rules (`transfer`, memory
spaces, capacity checks) are the same for every backend. Two decisions of
record:

- **Matmul goes to a vendor library** (cuBLAS today; oneMKL would be the
  Intel parallel). A hand-tiled GEMM loses to the vendor one, so generated
  kernels are for everything else.
- **A device limitation is a compile error, never a silent change.** A device
  that lacks f64 declares `dtypes:` without it, and placing an f64 tensor
  there is error E6026 before anything runs (`fleet/m4-uma.vx` already does
  this for a GPU without f64). We do not demote f64 to f32 behind the
  programmer's back.

## Status

| Target | Route | Status |
|---|---|---|
| CPU (x86-64, arm64) | native, through LLVM | working, tested in CI |
| NVIDIA | `gpu` dialect → NVVM → PTX; cuBLAS for matmul | working for placed kernels; completion tracked in #251 |
| Apple GPU / ANE | library routing (MPS, CoreML) | working for the routed patterns |
| Intel XPU | SPIR-V, Level Zero runtime | offered by a contributor; planning in #1137 |
| AMD | ROCDL (when it has an owner) | machine file exists (`fleet/mi300x.vx`); no backend, no owner |
| TPU | — | out of scope: the vendor compiler stack is closed, so a backend cannot be built outside Google |

SYCL as a compile target is declined: Vx's checker already does the job
SYCL's C++ layer does. What the SYCL stack offers Vx is its runtime (Level
Zero) and its libraries (oneMKL), and those are used directly.

## Support tiers

CI runs on Ubuntu with no GPU, so "merged" has to mean something a machine
without the hardware can check. The tiers, modeled on Rust's target tiers:

- **Tier 1 — CPU paths.** CI runs the tests; a regression blocks the merge.
- **Tier 2 — NVIDIA.** Maintainer-owned. CI proves it builds; correctness is
  shown by parity runs on rented hardware before any release or public claim.
- **Tier 3 — community backends.** Must build in CI with no vendor SDK
  installed: the device-image half compiles and its output is checked with
  FileCheck, and the vendor-free runtime pieces have unit tests in
  `tests/runtime/`. Correctness is shown by parity runs the hardware owner
  records in each PR, with the test marked `REQUIRES: gpu`. A named owner is
  a requirement for merging; a backend that loses its owner is marked
  unmaintained here, not reverted. A Tier 3 regression never blocks `main`.

## Shared work before a second backend

These are the places where the compiler currently assumes NVIDIA is the only
device target. Each gets its own issue; they are vendor-neutral, so they keep
their value whichever backend lands first.

1. **The kernel eligibility gate accepts one arch.** `materializeGpuKernels`
   in `src/dialect/VxLowering.cpp` only clones kernels whose `arch:` is
   `nvptx64` into the `gpu.module`. It needs to become a choice keyed on the
   declared `arch:`.
1. **The dispatch payload cannot carry a binary image.** The payload is a
   sequence of NUL-terminated `key=value` entries, which works because PTX is
   text; `deviceImageOf()` rejects images containing a NUL byte. SPIR-V is
   binary, so the image needs an encoding (or the payload format changes),
   and the payload should carry a version key (`abi=1`) so a dispatch library
   that sees a format it does not know refuses instead of guessing. The
   `vx_plugin_*` interface is not frozen yet, and the version key is what
   makes changing it safe.
1. **Address spaces are mapped for NVVM only.** `AddressSpace` in
   `src/arch.rs` knows the NVPTX numbering, and one place in
   `VxLowering.cpp` hardcodes NVVM's shared-memory address space 3. Each
   target needs its own mapping through the same table.
1. **Scalar element types escape the `dtypes:` check.** E6026 covers tensors
   that are placed or transferred. An f64 *scalar* inside a `spawn` body
   passes the checker today and would only fail on the device. The body of a
   `spawn` has to be checked against the target topology's `dtypes:` list.
1. **One dispatch library per build.** `build.rs` builds exactly one runtime
   dispatch library and programs load that one (`VX_DISPATCH_LIB` overrides
   it by hand). Running two device kinds from one program needs dispatch
   routed by topology.
1. **A conformance test set.** A vendor-neutral set of placed-kernel programs
   whose results are compared against the CPU path.
   `tests/backend/pass/placed_kernel_four_operands.vx` is the reference
   shape.

## The bar for being listed as working

The conformance set passes on real hardware, with the runs recorded in the
PR. The bar is the CPU path's result: same program, same numbers.

## How to start

1. **Talk first.** Open an issue (or use the one you have) and agree on the
   slicing before writing much code.
1. **Machine file.** Declare the device per
   [`adding_a_topology.md`](adding_a_topology.md). This needs no compiler
   change and immediately exercises the placement checks, including
   `dtypes:`.
1. **Device image compiler.** The sibling of `deviceImageOf()`. This half is
   testable in CI with FileCheck, so it can merge before any runtime exists.
1. **Runtime dispatch library.** Implement the `vx_plugin_*` entry points,
   reuse `runtime/vx_kernel_launch.h`, add the `build.rs` arm.

Keep PRs small and in that order; dependent PRs go in a GitHub stacked PR.
Signing the CLA (`docs/CLA.md`) is checked by CI on the first PR.

One thing about parallelism so it does not surprise you: a kernel body runs
parallel only when the compiler can prove the outer `for` loop safe to split
across threads (the proof is conservative and syntactic), and it uses one
grid dimension today. #251 shows what completing a backend looks like for
NVIDIA.
