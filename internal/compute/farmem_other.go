//go:build !linux

package compute

// The far-memory probes read the kernel's NUMA topology, which only linux exposes in
// a stable form (/sys/devices/system/node). Every other platform fails open: no far
// tier is confirmed, so the placement plane keeps today's behavior (the #1470 fence —
// never fabricate a pressure for a tier the box cannot prove).

func numaFarMemory() (total, free int64, known bool) { return 0, FreeUnknown, false }

func cxlMemory() (total, free int64, known bool) { return 0, FreeUnknown, false }
