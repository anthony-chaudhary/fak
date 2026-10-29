#pragma once

// fcuda_alloc_status wraps every raw cudaMalloc/cudaMallocManaged whose caller reports the miss
// itself (nullptr or a typed status). A capacity miss is not sticky, but the runtime keeps it as
// this thread's last error until something reads it; the next kernel's CK(cudaGetLastError()) or a
// GDN/KV preflight then reports "out of memory" at an unrelated line and refuses a healthy context
// as a stale execution fault. Consuming it here, where it was raised, keeps one OOM from poisoning
// every later request. Any other status (a sticky fault seen by the allocator) is left for the
// next status reader, so a genuinely dead context still refuses.
static inline cudaError_t fcuda_alloc_status(cudaError_t e) {
  if (e == cudaErrorMemoryAllocation) (void)cudaGetLastError();
  return e;
}
