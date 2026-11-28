// megakernel_taskgraph.cuh - structure-only skeleton of a decode megakernel task graph.
//
// fixed-width int32 descriptors, dependency counters held in global memory, and a warp-role split
// (8 consumers, 1 controller, 1 producer, 1 storer). No upstream code is copied; only the ABI
// shape is re-expressed. Reference envelope upstream: H100 sm_90a BF16.
//
// SKELETON ONLY: constants and a descriptor layout. It declares no kernel, is included only under
// FAK_CUDA_MEGAKERNEL_TASKGRAPH, and carries no tuning, loader change, or speed claim. The Go
// oracle in megakernel_taskgraph_test.go reads these #defines so the two cannot drift.
#ifndef FAK_MEGAKERNEL_TASKGRAPH_CUH
#define FAK_MEGAKERNEL_TASKGRAPH_CUH

#define FAK_MK_TASK_INTS 32
#define FAK_MK_CONSUMER_WARPS 8
#define FAK_MK_CONTROLLER_WARPS 1
#define FAK_MK_PRODUCER_WARPS 1
#define FAK_MK_STORER_WARPS 1

// Descriptor int32 slots the oracle pins; the remaining slots are reserved (zero).
#define FAK_MK_SLOT_OP 0
#define FAK_MK_SLOT_IN_A 1
#define FAK_MK_SLOT_IN_B 2
#define FAK_MK_SLOT_OUT 3
#define FAK_MK_SLOT_WAIT_COUNTER 4
#define FAK_MK_SLOT_WAIT_TARGET 5
#define FAK_MK_SLOT_SIGNAL_COUNTER 6
#define FAK_MK_SLOT_ARG 7

typedef struct fak_mk_task {
  int slots[FAK_MK_TASK_INTS];
} fak_mk_task;

#endif  // FAK_MEGAKERNEL_TASKGRAPH_CUH
