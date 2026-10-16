// A CPU-only stand-in for CUDA's <cuda_runtime.h>: the two cudaError_t values the checkpoint paths branch on
// and the types a header-only helper needs.  The build of memory_guard_test has no CUDA toolkit (CMakeLists:
// "CPU only, no GPU"), and a native <cuda_runtime.h> would not compile on this machine anyway.
#pragma once

enum cudaError {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorOutOfMemory = 2,        // the historical alias, CUDA's own value
    cudaErrorIllegalAddress = 700,
    cudaErrorIllegalInstruction = 701,
    cudaErrorDeviceUninitialized = 704,   // cudaErrorUnknown's value in CUDA 13 and later
    cudaErrorUnknown = 999,
};
using cudaError_t = cudaError;
