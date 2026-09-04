//go:build !linux

package alloc

import "time"

// HugepageStatus captures the kernel's hugepage configuration at startup.
type HugepageStatus struct {
	ExplicitTotal      int
	ExplicitFree       int
	ExplicitPageSizeKB int
	ExplicitAvailMB    int
	THPMode            string

	// 1 GB hugepage pool
	Has1GB   bool
	Total1GB int
	Free1GB  int
}

// SetTHPMode is a no-op on non-Linux platforms.
func SetTHPMode(mode string) {}

// GetTHPMode returns empty string on non-Linux platforms.
func GetTHPMode() string { return "" }

// CheckHugepages returns an empty status on non-Linux platforms.
func CheckHugepages() HugepageStatus {
	return HugepageStatus{}
}

// SufficientExplicitHugepages always returns false on non-Linux platforms.
func SufficientExplicitHugepages(st HugepageStatus, needBytes uint64) bool {
	return false
}

// THPAvailable always returns false on non-Linux platforms.
func THPAvailable(st HugepageStatus) bool {
	return false
}

// HugepageAllocResult reports the outcome of an EnsureHugepages attempt.
type HugepageAllocResult struct {
	Attempted      bool
	RequestedPages int
	PreviousPages  int
	CurrentPages   int
	FreePages      int
	PageSizeKB     int
	Err            error
}

// EnsureHugepages is a no-op on non-Linux platforms.
func EnsureHugepages(needBytes uint64, pageSizeKB int) HugepageAllocResult {
	return HugepageAllocResult{}
}

// ResolveHugepageSize is a no-op on non-Linux platforms.
func ResolveHugepageSize(hugepageSizeCfg string, useHugePages bool, st HugepageStatus, needBytes uint64) (int, error) {
	if !useHugePages {
		return 0, nil
	}
	return 2048, nil
}

// ReleaseHugepages is a no-op on non-Linux platforms.
func ReleaseHugepages(previousCount int, pageSizeKB ...int) error { return nil }

// FreeHugepageBytes returns 0 on non-Linux platforms.
func FreeHugepageBytes() uint64 { return 0 }

// madviseHugepage is a no-op on non-Linux platforms.
func madviseHugepage(data []byte) {}

// NodeHugepageInfo holds per-NUMA-node hugepage state.
type NodeHugepageInfo struct {
	NodeID     int
	Total      int
	Free       int
	PageSizeKB int
}

// NodeBudget specifies how many hugepages a NUMA node needs.
type NodeBudget struct {
	NodeID    int
	NeedBytes uint64
	NeedPages int
}

// ReadNodeHugepages is a no-op on non-Linux platforms.
func ReadNodeHugepages(node int, pageSizeKB int) (total, free int, err error) { return 0, 0, nil }

// ReleaseNodeHugepages is a no-op on non-Linux platforms.
func ReleaseNodeHugepages(targets map[int]int, pageSizeKB int) error { return nil }

// DistributeNodeHugepages is a no-op on non-Linux platforms.
func DistributeNodeHugepages(nodeBudgets []NodeBudget, pageSizeKB int) []NodeHugepageInfo {
	return nil
}

// DefaultHugepageStatePath is used when no explicit path is configured.
const DefaultHugepageStatePath = "/tmp/cama-hugepages.state"

// HugepageState records allocation state so hugepages can be released after a crash.
type HugepageState struct {
	PID             int         `json:"pid"`
	PageSizeKB      int         `json:"page_size_kb"`
	PreviousGlobal  int         `json:"previous_global"`
	PerNodePrevious map[int]int `json:"per_node_previous"`
	AllocatedAt     time.Time   `json:"allocated_at"`
}

// WriteHugepageState is a no-op on non-Linux platforms.
func WriteHugepageState(path string, state HugepageState) error { return nil }

// ReadHugepageState is a no-op on non-Linux platforms.
func ReadHugepageState(path string) (HugepageState, error) { return HugepageState{}, nil }

// RemoveHugepageState is a no-op on non-Linux platforms.
func RemoveHugepageState(path string) {}

// ReleaseFromState is a no-op on non-Linux platforms.
func ReleaseFromState(path string) (bool, error) { return false, nil }
