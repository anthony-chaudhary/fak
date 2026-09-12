package ctxmmu

import (
	"errors"
	"fmt"
	"sync"
)

// cow.go — Zero-copy subagent prefix branching via copy-on-write page tables on UMA (#11618).
//
// When a coordinator or parent agent fans out to N subagents sharing a large prefix
// (e.g. 32k-64k tokens of repo index, tool schemas, goal), physical DRAM pages are not duplicated.
// Instead, page block pointers are shallow copied in O(1) time (<1 ms latency), reference counts
// are incremented, and child sessions operate over shared immutable blocks. Any subsequent token
// appends or mutations to shared blocks trigger fine-grained Copy-On-Write (COW) at block granularity,
// allocating private physical pages only for modified blocks while preserving parent and sibling blocks.

const (
	// DefaultCOWBlockCapacity is the default token capacity per page block (64 tokens/block).
	DefaultCOWBlockCapacity = 64

	// DefaultCOWBytesPerToken is the default KV cache byte footprint per token in UMA DRAM (128 bytes).
	DefaultCOWBytesPerToken = 128

	// MaxCandidateTreeDepth is the maximum allowable speculative candidate tree recursion depth (#12319).
	MaxCandidateTreeDepth = 8
)

// Sentinel errors for candidate tree speculation (#12319).
var (
	ErrCandidateNotFound = errors.New("ctxmmu: candidate not found")
	ErrCandidateExists   = errors.New("ctxmmu: candidate already exists")
	ErrMaxTreeDepth      = errors.New("ctxmmu: max candidate tree depth exceeded")
	ErrCandidateReleased = errors.New("ctxmmu: candidate has already been released")
)

// FormatSavedMemory formats an avoided physical memory allocation byte count into a canonical summary string:
// e.g. "Saved 1.8 GB GPU VRAM / host DRAM via zero-copy subagent KV sharing".
func FormatSavedMemory(bytes int64) string {
	return fmt.Sprintf("Saved %s GPU VRAM / host DRAM via zero-copy subagent KV sharing", FormatBytes(bytes))
}

// FormatBytes formats byte counts into human-readable representation (B, KB, MB, GB, TB).
func FormatBytes(bytes int64) string {
	const (
		kb = 1024
		mb = 1024 * kb
		gb = 1024 * mb
		tb = 1024 * gb
	)
	switch {
	case bytes >= tb:
		return fmt.Sprintf("%.1f TB", float64(bytes)/float64(tb))
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	case bytes >= kb:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(kb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// -----------------------------------------------------------------------------
// MMU Integration & Default Instance
// -----------------------------------------------------------------------------

var (
	defaultCOWPageTable = NewCOWPageTable()
	mmuCOWMu            sync.Mutex
	mmuCOWPageTables    = make(map[*MMU]*COWPageTable)
)

// COWPageTable returns the COWPageTable instance associated with this MMU.
func (m *MMU) COWPageTable() *COWPageTable {
	if m == nil {
		return defaultCOWPageTable
	}
	mmuCOWMu.Lock()
	defer mmuCOWMu.Unlock()
	tbl := mmuCOWPageTables[m]
	if tbl == nil {
		tbl = NewCOWPageTable()
		mmuCOWPageTables[m] = tbl
	}
	return tbl
}

// ForkCOWSession forks a session branch via the default COWPageTable.
func ForkCOWSession(parentID, childID string) (*SessionBranch, error) {
	return defaultCOWPageTable.ForkSession(parentID, childID)
}

// ReleaseCOWSession releases a session branch via the default COWPageTable.
func ReleaseCOWSession(sessionID string) error {
	return defaultCOWPageTable.ReleaseSession(sessionID)
}

// AppendCOWTokens appends tokens into a session branch via the default COWPageTable.
func AppendCOWTokens(sessionID string, tokens []int) error {
	return defaultCOWPageTable.AppendTokens(sessionID, tokens)
}
