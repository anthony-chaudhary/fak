//go:build linux

package alloc

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

// HugepageStatus captures the kernel's hugepage configuration at startup.
type HugepageStatus struct {
	ExplicitTotal      int    // HugePages_Total from /proc/meminfo
	ExplicitFree       int    // HugePages_Free from /proc/meminfo
	ExplicitPageSizeKB int    // Hugepagesize from /proc/meminfo (typically 2048)
	ExplicitAvailMB    int    // Free * PageSizeKB / 1024
	THPMode            string // "always", "madvise", or "never" (from sysfs)

	// 1 GB hugepage pool (detected from sysfs independently of /proc/meminfo)
	Has1GB   bool // true if /sys/kernel/mm/hugepages/hugepages-1048576kB/ exists
	Total1GB int  // nr_hugepages for 1 GB pool
	Free1GB  int  // free_hugepages for 1 GB pool
}

// thpMode is the THP mode detected at startup, used by allocate() for madvise hints.
var thpMode string

// SetTHPMode stores the detected THP mode for use by region allocation.
func SetTHPMode(mode string) {
	thpMode = mode
}

// GetTHPMode returns the current THP mode setting.
func GetTHPMode() string {
	return thpMode
}

// CheckHugepages reads /proc/meminfo and THP sysfs to determine hugepage availability.
// Returns a zero-value status on read errors (containers, unusual kernels).
func CheckHugepages() HugepageStatus {
	var st HugepageStatus

	// Parse /proc/meminfo for explicit hugepage info
	data, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			key := strings.TrimSuffix(fields[0], ":")
			val, err := strconv.Atoi(fields[1])
			if err != nil {
				continue
			}
			switch key {
			case "HugePages_Total":
				st.ExplicitTotal = val
			case "HugePages_Free":
				st.ExplicitFree = val
			case "Hugepagesize":
				st.ExplicitPageSizeKB = val
			}
		}
	}
	if st.ExplicitPageSizeKB > 0 {
		st.ExplicitAvailMB = st.ExplicitFree * st.ExplicitPageSizeKB / 1024
	}

	// Parse THP mode from sysfs: format is "always [madvise] never"
	// The active mode is enclosed in brackets.
	thpData, err := os.ReadFile("/sys/kernel/mm/transparent_hugepage/enabled")
	if err == nil {
		s := strings.TrimSpace(string(thpData))
		if idx := strings.Index(s, "["); idx >= 0 {
			if end := strings.Index(s[idx:], "]"); end >= 0 {
				st.THPMode = s[idx+1 : idx+end]
			}
		}
	}
	if st.THPMode == "" {
		st.THPMode = "unknown"
	}

	// Detect 1 GB hugepage pool from sysfs (independent of /proc/meminfo default page size)
	const sysfs1GB = "/sys/kernel/mm/hugepages/hugepages-1048576kB"
	if tData, err := os.ReadFile(sysfs1GB + "/nr_hugepages"); err == nil {
		st.Has1GB = true
		st.Total1GB, _ = strconv.Atoi(strings.TrimSpace(string(tData)))
		if fData, err := os.ReadFile(sysfs1GB + "/free_hugepages"); err == nil {
			st.Free1GB, _ = strconv.Atoi(strings.TrimSpace(string(fData)))
		}
	}

	return st
}

// ResolveHugepageSize determines the effective hugepage size in KB based on
// the user's config string and detected system state.
//
//   - "auto":  prefer 1 GB if available and sufficient for needBytes, else 2 MB
//   - "2mb":   force 2 MB hugepages
//   - "1gb":   force 1 GB hugepages (returns error if unavailable)
//   - "":      same as "auto"
//
// Returns 0 if hugepages are disabled (useHugePages=false), or the resolved
// page size in KB (2048 or 1048576).
func ResolveHugepageSize(hugepageSizeCfg string, useHugePages bool, st HugepageStatus, needBytes uint64) (int, error) {
	if !useHugePages {
		return 0, nil
	}

	switch hugepageSizeCfg {
	case "", "auto":
		// Prefer 1 GB if the pool exists AND has enough free pages for the workload.
		// Otherwise fall back to 2 MB (the proven default).
		if st.Has1GB && st.Free1GB > 0 {
			const pageSize1GB uint64 = 1048576 * 1024 // 1 GB in bytes
			pagesNeeded := (needBytes + pageSize1GB - 1) / pageSize1GB
			if uint64(st.Free1GB) >= pagesNeeded {
				return 1048576, nil
			}
			// Not enough 1 GB pages — fall through to 2 MB
			log.Printf("[hugepages] 1GB pool has %d free pages but need %d — falling back to 2MB",
				st.Free1GB, pagesNeeded)
		} else if !st.Has1GB {
			log.Printf("[hugepages] auto: 1GB pool not detected — using 2MB pages")
		}
		return 2048, nil

	case "2mb":
		return 2048, nil

	case "1gb":
		if !st.Has1GB {
			return 0, fmt.Errorf("hugepage_size=%q but 1GB hugepages are not available on this system "+
				"(no /sys/kernel/mm/hugepages/hugepages-1048576kB/ — add 'hugepagesz=1G hugepages=N' to kernel cmdline)", hugepageSizeCfg)
		}
		return 1048576, nil

	default:
		return 0, fmt.Errorf("unknown hugepage_size %q (valid: auto, 2mb, 1gb)", hugepageSizeCfg)
	}
}

// SufficientExplicitHugepages returns true if free explicit hugepages
// can cover the requested memory size in bytes.
func SufficientExplicitHugepages(st HugepageStatus, needBytes uint64) bool {
	if st.ExplicitPageSizeKB == 0 || st.ExplicitFree == 0 {
		return false
	}
	pageSizeBytes := uint64(st.ExplicitPageSizeKB) * 1024
	pagesNeeded := (needBytes + pageSizeBytes - 1) / pageSizeBytes
	return uint64(st.ExplicitFree) >= pagesNeeded
}

// THPAvailable returns true if THP is "always" or "madvise".
func THPAvailable(st HugepageStatus) bool {
	return st.THPMode == "always" || st.THPMode == "madvise"
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

// hugepageSysfsPath returns the global sysfs path for a hugepage pool of the given size.
// pageSizeKB: 2048 → /proc/sys/vm/nr_hugepages (default pool)
// pageSizeKB: 1048576 → /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
func hugepageSysfsPath(pageSizeKB int) string {
	if pageSizeKB == 1048576 {
		return "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
	}
	// For 2 MB (and any other default), use the traditional /proc path
	return "/proc/sys/vm/nr_hugepages"
}

// readNrHugepages reads the current nr_hugepages from /proc/sys/vm/nr_hugepages.
func readNrHugepages() (int, error) {
	return readNrHugepagesForSize(2048)
}

// readNrHugepagesForSize reads nr_hugepages for the specified page size pool.
func readNrHugepagesForSize(pageSizeKB int) (int, error) {
	data, err := os.ReadFile(hugepageSysfsPath(pageSizeKB))
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(data)))
}

// writeNrHugepages writes a new value to /proc/sys/vm/nr_hugepages.
func writeNrHugepages(n int) error {
	return writeNrHugepagesForSize(2048, n)
}

// writeNrHugepagesForSize writes nr_hugepages for the specified page size pool.
func writeNrHugepagesForSize(pageSizeKB int, n int) error {
	return os.WriteFile(hugepageSysfsPath(pageSizeKB), []byte(strconv.Itoa(n)), 0644)
}

// EnsureHugepages reserves exactly the right number of explicit hugepages for needBytes.
// If pageSizeKB is 0, it defaults to 2048 (2 MB standard hugepage).
// Both increases AND decreases nr_hugepages to match the target — stale hugepages
// from a previous run are released so they don't starve normal page allocations
// (critical when mlockall is active). Returns gracefully on EACCES/EROFS.
func EnsureHugepages(needBytes uint64, pageSizeKB int) HugepageAllocResult {
	if pageSizeKB <= 0 {
		pageSizeKB = 2048
	}
	pageSizeBytes := uint64(pageSizeKB) * 1024
	pagesNeeded := int((needBytes + pageSizeBytes - 1) / pageSizeBytes)

	r := HugepageAllocResult{PageSizeKB: pageSizeKB}

	current, err := readNrHugepagesForSize(pageSizeKB)
	if err != nil {
		r.Err = fmt.Errorf("reading nr_hugepages (page_size=%dkB): %w", pageSizeKB, err)
		return r
	}
	r.PreviousPages = current

	if current == pagesNeeded {
		// Exact match — no write needed
		r.CurrentPages = current
		st := CheckHugepages()
		r.FreePages = st.ExplicitFree
		return r
	}

	r.Attempted = true
	r.RequestedPages = pagesNeeded

	if current > pagesNeeded {
		log.Printf("[hugepages] releasing stale hugepages: %d → %d (freeing %d pages / %.1f GB back to normal pool)",
			current, pagesNeeded, current-pagesNeeded,
			float64(int64(current-pagesNeeded)*int64(pageSizeKB))/(1024*1024))
	}

	if err := writeNrHugepagesForSize(pageSizeKB, pagesNeeded); err != nil {
		r.Err = fmt.Errorf("writing nr_hugepages (page_size=%dkB): %w", pageSizeKB, err)
		r.CurrentPages = current
		return r
	}

	// Re-read to verify actual allocation (kernel may allocate fewer due to fragmentation)
	actual, err := readNrHugepagesForSize(pageSizeKB)
	if err != nil {
		r.Err = fmt.Errorf("re-reading nr_hugepages (page_size=%dkB): %w", pageSizeKB, err)
		r.CurrentPages = current
		return r
	}
	r.CurrentPages = actual

	st := CheckHugepages()
	r.FreePages = st.ExplicitFree

	return r
}

// ReleaseHugepages restores nr_hugepages to previousCount (from before auto-allocation).
// No-op if current count is already at or below previousCount.
// pageSizeKB selects which pool to release (2048 for 2MB, 1048576 for 1GB).
// For backward compatibility, 0 defaults to 2048.
func ReleaseHugepages(previousCount int, pageSizeKB ...int) error {
	psk := 2048
	if len(pageSizeKB) > 0 && pageSizeKB[0] > 0 {
		psk = pageSizeKB[0]
	}
	current, err := readNrHugepagesForSize(psk)
	if err != nil {
		return fmt.Errorf("reading current nr_hugepages (page_size=%dkB): %w", psk, err)
	}
	if current <= previousCount {
		return nil
	}
	return writeNrHugepagesForSize(psk, previousCount)
}

// FreeHugepageBytes returns the number of free explicit hugepage bytes available.
// Returns 0 on read errors or if no explicit hugepages are configured.
func FreeHugepageBytes() uint64 {
	st := CheckHugepages()
	if st.ExplicitPageSizeKB == 0 || st.ExplicitFree == 0 {
		return 0
	}
	return uint64(st.ExplicitFree) * uint64(st.ExplicitPageSizeKB) * 1024
}

// NodeHugepageInfo holds per-NUMA-node hugepage state.
type NodeHugepageInfo struct {
	NodeID     int
	Total      int
	Free       int
	PageSizeKB int
}

// readNodeHugepages reads hugepage counts for a specific NUMA node.
// Path: /sys/devices/system/node/nodeN/hugepages/hugepages-{size}kB/
func readNodeHugepages(node int, pageSizeKB int) (total, free int, err error) {
	base := fmt.Sprintf("/sys/devices/system/node/node%d/hugepages/hugepages-%dkB", node, pageSizeKB)
	tData, err := os.ReadFile(base + "/nr_hugepages")
	if err != nil {
		return 0, 0, err
	}
	total, err = strconv.Atoi(strings.TrimSpace(string(tData)))
	if err != nil {
		return 0, 0, err
	}
	fData, err := os.ReadFile(base + "/free_hugepages")
	if err != nil {
		return total, 0, err
	}
	free, _ = strconv.Atoi(strings.TrimSpace(string(fData)))
	return total, free, nil
}

// writeNodeHugepages sets the hugepage count for a specific NUMA node.
func writeNodeHugepages(node int, pageSizeKB int, pages int) error {
	path := fmt.Sprintf("/sys/devices/system/node/node%d/hugepages/hugepages-%dkB/nr_hugepages", node, pageSizeKB)
	return os.WriteFile(path, []byte(strconv.Itoa(pages)), 0644)
}

// ReadNodeHugepages reads hugepage counts for a specific NUMA node.
// Returns (total, free, err). The total is the current nr_hugepages setting;
// free is how many of those are unused. Useful for capturing pre-allocation
// state so shutdown can restore it.
func ReadNodeHugepages(node int, pageSizeKB int) (total, free int, err error) {
	return readNodeHugepages(node, pageSizeKB)
}

// ReleaseNodeHugepages restores per-node hugepage counts to their previous
// values. Each entry in targets maps nodeID → previous nr_hugepages count.
// This is the per-node counterpart to ReleaseHugepages — used on shutdown
// to cleanly undo per-node allocations instead of relying on the kernel's
// proportional distribution via the global nr_hugepages.
func ReleaseNodeHugepages(targets map[int]int, pageSizeKB int) error {
	if pageSizeKB <= 0 {
		pageSizeKB = 2048
	}
	var firstErr error
	for nodeID, target := range targets {
		current, _, err := readNodeHugepages(nodeID, pageSizeKB)
		if err != nil {
			log.Printf("[hugepages] node %d: cannot read current count for release: %v", nodeID, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if current <= target {
			continue // already at or below target
		}
		if err := writeNodeHugepages(nodeID, pageSizeKB, target); err != nil {
			log.Printf("[hugepages] node %d: release failed (target %d): %v", nodeID, target, err)
			if firstErr == nil {
				firstErr = err
			}
		} else {
			log.Printf("[hugepages] node %d: released %d → %d pages", nodeID, current, target)
		}
	}
	return firstErr
}

// NodeBudget specifies how many hugepages a NUMA node needs.
type NodeBudget struct {
	NodeID    int
	NeedBytes uint64
	NeedPages int
}

// DistributeNodeHugepages sets per-node hugepage counts to match the per-node
// slab budget (from NUMA shard assignment). This prevents the kernel's default
// proportional distribution from starving nodes that need more hugepages than
// their share of total RAM would suggest.
//
// Safe for single-node: if nodeBudgets has one entry, this is equivalent to
// the global EnsureHugepages path. For memory-only nodes (CXL, HBM), the
// caller decides how much budget each node gets — this function just sets
// the hugepage counts.
//
// Returns per-node results for logging. Errors are non-fatal — the caller
// should fall back to global hugepage allocation on failure.
func DistributeNodeHugepages(nodeBudgets []NodeBudget, pageSizeKB int) []NodeHugepageInfo {
	if pageSizeKB <= 0 {
		pageSizeKB = 2048
	}
	pageSizeBytes := uint64(pageSizeKB) * 1024

	var results []NodeHugepageInfo
	for _, nb := range nodeBudgets {
		pagesNeeded := int((nb.NeedBytes + pageSizeBytes - 1) / pageSizeBytes)

		total, free, err := readNodeHugepages(nb.NodeID, pageSizeKB)
		if err != nil {
			log.Printf("[hugepages] node %d: cannot read per-node hugepages: %v — skipping per-node distribution", nb.NodeID, err)
			return nil // signal: per-node sysfs not available, fall back to global
		}

		// Release excess hugepages on this node (stale from previous run or
		// misplaced by the global nr_hugepages write distributing across all nodes).
		if total > pagesNeeded {
			log.Printf("[hugepages] node %d: releasing excess hugepages: %d → %d (freeing %.1f GB)",
				nb.NodeID, total, pagesNeeded,
				float64(int64(total-pagesNeeded)*int64(pageSizeKB))/(1024*1024))
			if err := writeNodeHugepages(nb.NodeID, pageSizeKB, pagesNeeded); err != nil {
				log.Printf("[hugepages] node %d: cannot release excess: %v", nb.NodeID, err)
				results = append(results, NodeHugepageInfo{
					NodeID: nb.NodeID, Total: total, Free: free, PageSizeKB: pageSizeKB,
				})
				continue
			}
			newTotal, newFree, _ := readNodeHugepages(nb.NodeID, pageSizeKB)
			results = append(results, NodeHugepageInfo{
				NodeID: nb.NodeID, Total: newTotal, Free: newFree, PageSizeKB: pageSizeKB,
			})
			if pagesNeeded == 0 {
				log.Printf("[hugepages] node %d: released all hugepages (no shards assigned)", nb.NodeID)
			} else {
				log.Printf("[hugepages] node %d: reduced to %d pages (%.1f GB) for shard budget",
					nb.NodeID, newTotal, float64(nb.NeedBytes)/float64(1<<30))
			}
			continue
		}

		if free >= pagesNeeded {
			// Already sufficient
			results = append(results, NodeHugepageInfo{
				NodeID: nb.NodeID, Total: total, Free: free, PageSizeKB: pageSizeKB,
			})
			log.Printf("[hugepages] node %d: %d free pages sufficient for %d needed (%.1f GB)",
				nb.NodeID, free, pagesNeeded, float64(nb.NeedBytes)/float64(1<<30))
			continue
		}

		// Need more — set the per-node count
		if err := writeNodeHugepages(nb.NodeID, pageSizeKB, pagesNeeded); err != nil {
			log.Printf("[hugepages] node %d: cannot set per-node nr_hugepages=%d: %v — will rely on global pool",
				nb.NodeID, pagesNeeded, err)
			results = append(results, NodeHugepageInfo{
				NodeID: nb.NodeID, Total: total, Free: free, PageSizeKB: pageSizeKB,
			})
			continue
		}

		// Re-read to verify
		newTotal, newFree, _ := readNodeHugepages(nb.NodeID, pageSizeKB)
		results = append(results, NodeHugepageInfo{
			NodeID: nb.NodeID, Total: newTotal, Free: newFree, PageSizeKB: pageSizeKB,
		})

		if newTotal < pagesNeeded {
			log.Printf("[hugepages] WARNING: node %d: requested %d pages but only got %d (%.1f GB short) — may fall back to 4KB pages",
				nb.NodeID, pagesNeeded, newTotal,
				float64(int64(pagesNeeded-newTotal)*int64(pageSizeKB))/(1024*1024))
		} else {
			log.Printf("[hugepages] node %d: allocated %d pages (%.1f GB) for shard budget",
				nb.NodeID, newTotal, float64(nb.NeedBytes)/float64(1<<30))
		}
	}

	return results
}

// DefaultHugepageStatePath is used when no explicit path is configured.
const DefaultHugepageStatePath = "/tmp/cama-hugepages.state"

// HugepageState records allocation state so hugepages can be released after a crash.
type HugepageState struct {
	PID             int         `json:"pid"`
	PageSizeKB      int         `json:"page_size_kb"`
	PreviousGlobal  int         `json:"previous_global"`   // -1 if per-node path was used
	PerNodePrevious map[int]int `json:"per_node_previous"` // nil for single-NUMA
	AllocatedAt     time.Time   `json:"allocated_at"`
}

// WriteHugepageState persists the allocation state to disk.
func WriteHugepageState(path string, state HugepageState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal hugepage state: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return fmt.Errorf("write hugepage state tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("rename hugepage state: %w", err)
	}
	return nil
}

// ReadHugepageState reads a previously-written state file.
func ReadHugepageState(path string) (HugepageState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return HugepageState{}, err
	}
	var st HugepageState
	if err := json.Unmarshal(data, &st); err != nil {
		return HugepageState{}, fmt.Errorf("parse hugepage state: %w", err)
	}
	return st, nil
}

// RemoveHugepageState deletes the state file.
func RemoveHugepageState(path string) {
	os.Remove(path)
}

// ReleaseFromState reads the state file, releases hugepages to their
// pre-allocation values, and removes the state file.
func ReleaseFromState(path string) (bool, error) {
	state, err := ReadHugepageState(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("read hugepage state: %w", err)
	}

	pageSizeKB := state.PageSizeKB
	if pageSizeKB <= 0 {
		pageSizeKB = 2048
	}

	released := false

	if len(state.PerNodePrevious) > 0 {
		// Per-node release
		log.Printf("[hugepages] crash recovery: releasing per-node hugepages from crashed PID %d (allocated %s ago)",
			state.PID, time.Since(state.AllocatedAt).Round(time.Second))
		for nodeID, target := range state.PerNodePrevious {
			current, _, err := readNodeHugepages(nodeID, pageSizeKB)
			if err != nil {
				log.Printf("[hugepages] crash recovery: node %d: cannot read: %v", nodeID, err)
				continue
			}
			if current <= target {
				continue
			}
			if err := writeNodeHugepages(nodeID, pageSizeKB, target); err != nil {
				log.Printf("[hugepages] crash recovery: node %d: release failed (%d → %d): %v", nodeID, current, target, err)
			} else {
				freedGB := float64(int64(current-target)*int64(pageSizeKB)) / (1024 * 1024)
				log.Printf("[hugepages] crash recovery: node %d: released %d → %d pages (freed %.1f GB)", nodeID, current, target, freedGB)
				released = true
			}
		}
	} else if state.PreviousGlobal >= 0 {
		// Global release
		current, err := readNrHugepagesForSize(pageSizeKB)
		if err != nil {
			return false, fmt.Errorf("read current nr_hugepages: %w", err)
		}
		if current > state.PreviousGlobal {
			freedGB := float64(int64(current-state.PreviousGlobal)*int64(pageSizeKB)) / (1024 * 1024)
			log.Printf("[hugepages] crash recovery: releasing %d → %d pages (freeing %.1f GB) from crashed PID %d (allocated %s ago)",
				current, state.PreviousGlobal, freedGB, state.PID, time.Since(state.AllocatedAt).Round(time.Second))
			if err := writeNrHugepagesForSize(pageSizeKB, state.PreviousGlobal); err != nil {
				return false, fmt.Errorf("release hugepages: %w", err)
			}
			released = true
		}
	}

	RemoveHugepageState(path)
	return released, nil
}

// madviseHugepage applies MADV_HUGEPAGE to the given byte slice.
func madviseHugepage(data []byte) {
	if len(data) == 0 {
		return
	}
	const MADV_HUGEPAGE = 14
	ptr := unsafe.Pointer(&data[0])
	syscall.Syscall(syscall.SYS_MADVISE, uintptr(ptr), uintptr(len(data)), MADV_HUGEPAGE)
}
