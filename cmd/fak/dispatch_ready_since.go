package main

import (
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatchIssueSourceRow is the command-side GitHub issue row. The router's pure Issue type
// deliberately carries only routing facts, so the impure shell keeps created/updated provenance
// alongside it and preserves the timestamps inside the opaque backlog-cache row.
type dispatchIssueSourceRow struct {
	dispatchtick.Issue
	CreatedAt   time.Time `json:"createdAt,omitempty"`
	UpdatedAt   time.Time `json:"updatedAt,omitempty"`
	CreatedUnix int64     `json:"created_unix,omitempty"`
	UpdatedUnix int64     `json:"updated_unix,omitempty"`
}

type dispatchIssueProvenance struct {
	CreatedUnix int64
	UpdatedUnix int64
}

type dispatchIssueProvenanceKey struct {
	Root   string
	Number int
}

var dispatchIssueProvenanceStore = struct {
	sync.RWMutex
	byIssue map[dispatchIssueProvenanceKey]dispatchIssueProvenance
}{byIssue: map[dispatchIssueProvenanceKey]dispatchIssueProvenance{}}

func (r dispatchIssueSourceRow) provenance() dispatchIssueProvenance {
	created, updated := r.CreatedUnix, r.UpdatedUnix
	if !r.CreatedAt.IsZero() {
		created = r.CreatedAt.Unix()
	}
	if !r.UpdatedAt.IsZero() {
		updated = r.UpdatedAt.Unix()
	}
	return dispatchIssueProvenance{CreatedUnix: created, UpdatedUnix: updated}
}

func rememberDispatchIssueProvenance(root string, row dispatchIssueSourceRow) {
	if row.Number <= 0 {
		return
	}
	p := row.provenance()
	dispatchIssueProvenanceStore.Lock()
	dispatchIssueProvenanceStore.byIssue[dispatchIssueProvenanceKey{
		Root: filepath.Clean(root), Number: row.Number,
	}] = p
	dispatchIssueProvenanceStore.Unlock()
}

func dispatchIssueProvenanceFor(root string, number int) dispatchIssueProvenance {
	dispatchIssueProvenanceStore.RLock()
	p := dispatchIssueProvenanceStore.byIssue[dispatchIssueProvenanceKey{
		Root: filepath.Clean(root), Number: number,
	}]
	dispatchIssueProvenanceStore.RUnlock()
	return p
}

// dispatchReadySince is the authoritative candidate-builder precedence for #3589:
// an observed eligibility transition (for example a prerequisite release / entry into the
// ready set) wins; otherwise use GitHub updatedAt, then createdAt; with no trustworthy stamp
// return zero. Zero is intentionally fail-closed in dispatchaging: wait remains zero and the
// candidate never earns an aging boost or trips the starvation deadline.
func dispatchReadySince(eligibilityUnix, updatedUnix, createdUnix int64) int64 {
	switch {
	case eligibilityUnix > 0:
		return eligibilityUnix
	case updatedUnix > 0:
		return updatedUnix
	case createdUnix > 0:
		return createdUnix
	default:
		return 0
	}
}

func dispatchIssueReadySinceStamp(root string, state dispatchPrereqState, number int) int64 {
	eligibilityUnix := int64(0)
	if state.ReadySince != nil {
		eligibilityUnix = state.ReadySince[strconv.Itoa(number)]
	}
	p := dispatchIssueProvenanceFor(root, number)
	return dispatchReadySince(eligibilityUnix, p.UpdatedUnix, p.CreatedUnix)
}

func dispatchIssueSourceRows(rows []dispatchIssueSourceRow, root string) []dispatchtick.Issue {
	issues := make([]dispatchtick.Issue, 0, len(rows))
	for _, row := range rows {
		rememberDispatchIssueProvenance(root, row)
		issues = append(issues, row.Issue)
	}
	return issues
}
