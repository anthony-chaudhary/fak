// doorbell_ar.metal — Stream-Async GPU-Doorbell Direct RDMA all-reduce over USB4/Thunderbolt.
//
// Translates XINNOV-03: direct RDMA / RoCE collective all-reduce over USB4 / Thunderbolt 4
// interconnects for Apple Silicon Mac clusters and unified memory architectures.
//
// GPU thread 0 in each threadgroup acquire-spins on the peer arrival flag in
// MTLResourceStorageModeShared memory, followed by element-wise addition into the
// destination buffer with zero CPU intervention.

#include <metal_stdlib>
using namespace metal;

// DoorbellControl defines the synchronization header mapped in MTLResourceStorageModeShared memory.
struct DoorbellControl {
    atomic_uint arrival_flag; // atomic arrival indicator (written by peer over direct RDMA)
    uint32_t    sequence;     // transfer sequence counter
    uint32_t    count;        // float32 element count
    uint32_t    reserved;     // 16-byte alignment padding
};

// tb_doorbell_wait_add implements stream-async GPU-doorbell direct RDMA all-reduce.
// Thread 0 in each threadgroup acquire-spins on peer arrival flag,
// followed by element-wise addition into destination buffer.
//
// Buffers:
//   0: doorbell   - DoorbellControl header (in MTLResourceStorageModeShared memory)
//   1: peer_data  - remote peer partial vector transferred via direct RDMA
//   2: local_data - local rank partial vector
//   3: dst        - destination output buffer for reduced result
//   4: count      - number of float32 elements
//   5: target_seq - sequence number to wait for (non-zero; defaults to 1 if 0)
kernel void tb_doorbell_wait_add(
    device const DoorbellControl* doorbell   [[buffer(0)]],
    device const float*           peer_data  [[buffer(1)]],
    device const float*           local_data [[buffer(2)]],
    device float*                 dst        [[buffer(3)]],
    constant uint&                count      [[buffer(4)]],
    constant uint&                target_seq [[buffer(5)]],
    uint                          tid        [[thread_position_in_grid]],
    uint                          lid        [[thread_position_in_threadgroup]]
) {
    // Thread 0 in threadgroup acquire-spins on peer arrival flag.
    // Memory order acquire ensures peer data payload writes across Thunderbolt/USB4
    // are fully visible before reduction begins.
    if (lid == 0) {
        uint expected = target_seq > 0 ? target_seq : 1;
        while (atomic_load_explicit((device const atomic_uint*)&(doorbell->arrival_flag), memory_order_acquire) < expected) {
            // Spin-waiting on peer arrival flag over USB4/Thunderbolt RDMA
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Vectorized element-wise reduction into destination buffer
    if (tid < count) {
        dst[tid] = local_data[tid] + peer_data[tid];
    }
}

// tb_doorbell_wait_add4 implements vectorized 4-way element-wise addition for float4 SIMD lanes.
kernel void tb_doorbell_wait_add4(
    device const DoorbellControl* doorbell   [[buffer(0)]],
    device const float4*          peer_data  [[buffer(1)]],
    device const float4*          local_data [[buffer(2)]],
    device float4*                dst        [[buffer(3)]],
    constant uint&                count4     [[buffer(4)]],
    constant uint&                target_seq [[buffer(5)]],
    uint                          tid        [[thread_position_in_grid]],
    uint                          lid        [[thread_position_in_threadgroup]]
) {
    if (lid == 0) {
        uint expected = target_seq > 0 ? target_seq : 1;
        while (atomic_load_explicit((device const atomic_uint*)&(doorbell->arrival_flag), memory_order_acquire) < expected) {
            // Spin-waiting on peer arrival flag over USB4/Thunderbolt RDMA
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid < count4) {
        dst[tid] = local_data[tid] + peer_data[tid];
    }
}
