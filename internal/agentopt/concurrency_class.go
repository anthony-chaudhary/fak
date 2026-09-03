package agentopt

import (
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Structured refusal reasons for lane lease arbitration.
const (
	RefuseBudgetExhausted = "REFUSE_BUDGET_EXHAUSTED"
	RefuseTreeCollision   = "REFUSE_TREE_COLLISION"
	RefuseInvalidRequest  = "REFUSE_INVALID_REQUEST"
)

// LaneLeaseRequest models a request to acquire a lane lease.
type LaneLeaseRequest struct {
	LaneKind     string   `json:"lane_kind"`
	LaneName     string   `json:"lane_name"`
	TreePatterns []string `json:"tree_patterns"`
	WorkerID     string   `json:"worker_id"`
}

// LaneLease represents an active lane lease held by a worker.
type LaneLease struct {
	LaneKind     string    `json:"lane_kind"`
	LaneName     string    `json:"lane_name"`
	TreePatterns []string  `json:"tree_patterns"`
	WorkerID     string    `json:"worker_id"`
	AcquiredAt   time.Time `json:"acquired_at,omitempty"`
}

// ArbitrationResult records the outcome of a lane lease acquisition attempt.
type ArbitrationResult struct {
	Granted  bool       `json:"granted"`
	Acquired bool       `json:"acquired"`
	Reason   string     `json:"reason,omitempty"`
	Lease    *LaneLease `json:"lease,omitempty"`
}

// ConcurrencyClassArbiter arbitrates lane lease requests across peer subagents,
// enforcing per-lane-kind concurrency budgets and file-tree disjointness.
type ConcurrencyClassArbiter struct {
	mu      sync.RWMutex
	budgets map[string]int
	leases  []LaneLease
}

// NewConcurrencyClassArbiter creates an arbiter with the specified concurrency-class budgets.
func NewConcurrencyClassArbiter(budgets map[string]int) *ConcurrencyClassArbiter {
	b := make(map[string]int)
	for k, v := range budgets {
		b[k] = v
	}
	return &ConcurrencyClassArbiter{
		budgets: b,
		leases:  make([]LaneLease, 0),
	}
}

// SetBudget configures the maximum active leases for a given lane kind.
func (a *ConcurrencyClassArbiter) SetBudget(laneKind string, maxActive int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.budgets == nil {
		a.budgets = make(map[string]int)
	}
	a.budgets[laneKind] = maxActive
}

// GetBudget returns the maximum active leases for a lane kind, if configured.
func (a *ConcurrencyClassArbiter) GetBudget(laneKind string) (int, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.budgets == nil {
		return 0, false
	}
	v, ok := a.budgets[laneKind]
	return v, ok
}

// ActiveCount returns the number of active leases, optionally filtered by lane kind.
func (a *ConcurrencyClassArbiter) ActiveCount(laneKind string) int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := 0
	for _, l := range a.leases {
		if laneKind == "" || l.LaneKind == laneKind {
			count++
		}
	}
	return count
}

// AcquireLease attempts to acquire a lane lease for a worker.
// It checks concurrency-class budgets first, then file-tree disjointness.
func (a *ConcurrencyClassArbiter) AcquireLease(req LaneLeaseRequest) ArbitrationResult {
	a.mu.Lock()
	defer a.mu.Unlock()

	laneName := strings.TrimSpace(req.LaneName)
	workerID := strings.TrimSpace(req.WorkerID)
	laneKind := strings.TrimSpace(req.LaneKind)

	if workerID == "" || laneName == "" {
		return ArbitrationResult{
			Granted:  false,
			Acquired: false,
			Reason:   RefuseInvalidRequest,
		}
	}

	// 1. Enforce per-lane-kind concurrency budget.
	if a.budgets != nil && laneKind != "" {
		if limit, ok := a.budgets[laneKind]; ok {
			count := 0
			for _, l := range a.leases {
				if l.LaneKind == laneKind {
					count++
				}
			}
			if count >= limit {
				return ArbitrationResult{
					Granted:  false,
					Acquired: false,
					Reason:   RefuseBudgetExhausted,
				}
			}
		}
	}

	// 2. Refuse lane name collisions or overlapping file trees.
	for _, l := range a.leases {
		if l.LaneName == laneName {
			return ArbitrationResult{
				Granted:  false,
				Acquired: false,
				Reason:   RefuseTreeCollision,
			}
		}
		if treesCollide(req.TreePatterns, l.TreePatterns) {
			return ArbitrationResult{
				Granted:  false,
				Acquired: false,
				Reason:   RefuseTreeCollision,
			}
		}
	}

	// 3. Grant and store the lease.
	lease := LaneLease{
		LaneKind:     laneKind,
		LaneName:     laneName,
		TreePatterns: append([]string(nil), req.TreePatterns...),
		WorkerID:     workerID,
		AcquiredAt:   time.Now().UTC(),
	}
	a.leases = append(a.leases, lease)

	return ArbitrationResult{
		Granted:  true,
		Acquired: true,
		Lease:    &lease,
	}
}

// ReleaseLease releases an active lease for workerID on laneName.
// If workerID is non-empty, it must match the lease's worker.
func (a *ConcurrencyClassArbiter) ReleaseLease(workerID, laneName string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	laneName = strings.TrimSpace(laneName)
	workerID = strings.TrimSpace(workerID)

	for i, l := range a.leases {
		if l.LaneName == laneName {
			if workerID != "" && l.WorkerID != workerID {
				return false
			}
			a.leases = append(a.leases[:i], a.leases[i+1:]...)
			return true
		}
	}
	return false
}

// ListLeases returns a copy of all currently active leases.
func (a *ConcurrencyClassArbiter) ListLeases() []LaneLease {
	a.mu.RLock()
	defer a.mu.RUnlock()

	res := make([]LaneLease, len(a.leases))
	for i, l := range a.leases {
		res[i] = LaneLease{
			LaneKind:     l.LaneKind,
			LaneName:     l.LaneName,
			TreePatterns: append([]string(nil), l.TreePatterns...),
			WorkerID:     l.WorkerID,
			AcquiredAt:   l.AcquiredAt,
		}
	}
	return res
}

// isUniversalPattern reports whether p represents the entire file tree.
func isUniversalPattern(p string) bool {
	p = strings.TrimSpace(filepath.ToSlash(p))
	return p == "**" || p == "*" || p == "." || p == "..." || p == "./"
}

// normalizePattern sanitizes a tree pattern for comparison.
func normalizePattern(p string) string {
	p = strings.TrimSpace(p)
	p = filepath.ToSlash(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	p = filepath.Clean(p)
	p = filepath.ToSlash(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	return p
}

// extractBase strips trailing wildcards and reports whether a wildcard was present.
func extractBase(p string) (string, bool) {
	if strings.HasSuffix(p, "/**") {
		return strings.TrimSuffix(p, "/**"), true
	}
	if strings.HasSuffix(p, "/*") {
		return strings.TrimSuffix(p, "/*"), true
	}
	if strings.HasSuffix(p, "/...") {
		return strings.TrimSuffix(p, "/..."), true
	}
	if strings.HasSuffix(p, "...") {
		return strings.TrimSuffix(p, "..."), true
	}
	if strings.Contains(p, "*") {
		return p, true
	}
	return p, false
}

// patternsCollide reports whether two file-tree patterns overlap.
func patternsCollide(p1, p2 string) bool {
	if isUniversalPattern(p1) || isUniversalPattern(p2) {
		return true
	}

	n1 := normalizePattern(p1)
	n2 := normalizePattern(p2)

	if n1 == "" || n2 == "" {
		return false
	}
	if n1 == n2 {
		return true
	}

	base1, _ := extractBase(n1)
	base2, _ := extractBase(n2)

	if base1 == base2 {
		return true
	}
	if base1 != "" && strings.HasPrefix(base2, base1+"/") {
		return true
	}
	if base2 != "" && strings.HasPrefix(base1, base2+"/") {
		return true
	}

	if matched, err := filepath.Match(n1, n2); err == nil && matched {
		return true
	}
	if matched, err := filepath.Match(n2, n1); err == nil && matched {
		return true
	}

	return segmentsCollide(strings.Split(n1, "/"), strings.Split(n2, "/"))
}

// segmentsCollide checks for segment-level collision including wildcards.
func segmentsCollide(s1, s2 []string) bool {
	if len(s1) == 0 && len(s2) == 0 {
		return true
	}
	if len(s1) == 0 || len(s2) == 0 {
		return false
	}

	head1 := s1[0]
	head2 := s2[0]

	if head1 == "**" || head1 == "..." {
		for i := 0; i <= len(s2); i++ {
			if segmentsCollide(s1[1:], s2[i:]) {
				return true
			}
		}
		return false
	}
	if head2 == "**" || head2 == "..." {
		for i := 0; i <= len(s1); i++ {
			if segmentsCollide(s1[i:], s2[1:]) {
				return true
			}
		}
		return false
	}

	if head1 == head2 || head1 == "*" || head2 == "*" {
		return segmentsCollide(s1[1:], s2[1:])
	}
	if matched, err := filepath.Match(head1, head2); err == nil && matched {
		return segmentsCollide(s1[1:], s2[1:])
	}
	if matched, err := filepath.Match(head2, head1); err == nil && matched {
		return segmentsCollide(s1[1:], s2[1:])
	}

	return false
}

// treesCollide reports whether any pattern in pats1 overlaps with any pattern in pats2.
func treesCollide(pats1, pats2 []string) bool {
	if len(pats1) == 0 || len(pats2) == 0 {
		return false
	}
	for _, p1 := range pats1 {
		p1 = strings.TrimSpace(p1)
		if p1 == "" {
			continue
		}
		for _, p2 := range pats2 {
			p2 = strings.TrimSpace(p2)
			if p2 == "" {
				continue
			}
			if patternsCollide(p1, p2) {
				return true
			}
		}
	}
	return false
}
