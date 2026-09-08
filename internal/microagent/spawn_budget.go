package microagent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// SpawnBudget bounds one host-mediated recursive task tree.
type SpawnBudget struct {
	// RootID and RootGoal anchor the host-derived ancestry. Both are required
	// for child admission and immutable after the first admitted child.
	RootID           string
	RootGoal         string
	MaxDepth         int
	MaxChildren      int
	MaxDescendants   int
	MaxTokens        int64
	MaxOutputTokens  int64
	MaxCostMicrosUSD int64

	mu           sync.Mutex
	descendants  int
	children     map[string]int
	reserved     LineageBudget
	spent        LineageBudget
	reservations map[string]LineageBudget
	rootID       string
	rootGoal     string
	goals        map[string]goalLineage
	goalOwners   map[string]string
	store        SpawnBudgetLogger
}

// SpawnBudgetLogger defines the persistence seam for child admissions and reconciliations.
type SpawnBudgetLogger interface {
	LogRoot(rootID, rootGoal string) error
	LogAdmit(request SpawnRequest) error
	LogReconcile(childID string, actual LineageBudget) error
	LogRelease(request SpawnRequest) error
}

type goalLineage struct {
	goalFingerprint string
	pathFingerprint string
	ancestors       map[string]struct{}
	depth           int
}

// LineageBudget is one host-authored resource envelope. Tokens includes output
// tokens; OutputTokens carries the stricter decode-only ceiling. Cost is stored
// in micro-USD so admission never depends on floating-point arithmetic.
type LineageBudget struct {
	Tokens        int64
	OutputTokens  int64
	CostMicrosUSD int64
}

// SpawnRequest carries lineage metadata the host, not the child, adjudicates.
type SpawnRequest struct {
	ParentID     string
	ChildID      string
	Goal         string
	Depth        int
	Budget       LineageBudget
	Capabilities CapabilityEnvelope
	// Context carries the parent cancellation/deadline. Nil uses the host root context.
	Context context.Context
}

var (
	ErrSpawnBudget     = errors.New("microagent: recursive spawn budget refused")
	ErrDuplicateGoal   = errors.New("microagent: duplicate child goal")
	ErrCyclicGoal      = errors.New("microagent: cyclic child goal")
	ErrInvalidAncestry = errors.New("microagent: invalid child ancestry")
)

// Admit reserves one child slot. A refusal never consumes aggregate capacity.
func (b *SpawnBudget) Admit(request SpawnRequest) error {
	if b == nil {
		return fmt.Errorf("%w: missing host budget", ErrSpawnBudget)
	}
	if request.ParentID == "" || request.ChildID == "" || request.Depth < 1 {
		return fmt.Errorf("%w: %w: parent id, child id, and positive depth are required", ErrSpawnBudget, ErrInvalidAncestry)
	}
	goalFingerprint, err := fingerprintGoal(request.Goal)
	if err != nil {
		return err
	}
	if err := validateLineageBudget("reservation", request.Budget); err != nil {
		return err
	}
	limits := LineageBudget{Tokens: b.MaxTokens, OutputTokens: b.MaxOutputTokens, CostMicrosUSD: b.MaxCostMicrosUSD}
	if err := validateLineageLimits(limits); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.ensureGoalRoot(); err != nil {
		return err
	}
	parent, ok := b.goals[request.ParentID]
	if !ok {
		return fmt.Errorf("%w: %w: parent %q is not in the admitted goal lineage", ErrSpawnBudget, ErrInvalidAncestry, request.ParentID)
	}
	if request.Depth != parent.depth+1 {
		return fmt.Errorf("%w: %w: depth %d does not extend parent %q at depth %d", ErrSpawnBudget, ErrInvalidAncestry, request.Depth, request.ParentID, parent.depth)
	}
	if _, exists := b.goals[request.ChildID]; exists {
		return fmt.Errorf("%w: %w: child %q already names admitted work", ErrSpawnBudget, ErrDuplicateGoal, request.ChildID)
	}
	if _, cyclic := parent.ancestors[goalFingerprint]; cyclic || parent.goalFingerprint == goalFingerprint {
		return fmt.Errorf("%w: %w: goal for child %q repeats an ancestor", ErrSpawnBudget, ErrCyclicGoal, request.ChildID)
	}
	if owner, duplicate := b.goalOwners[goalFingerprint]; duplicate {
		return fmt.Errorf("%w: %w: goal for child %q duplicates %q", ErrSpawnBudget, ErrDuplicateGoal, request.ChildID, owner)
	}
	if b.MaxDepth > 0 && request.Depth > b.MaxDepth {
		return fmt.Errorf("%w: depth %d exceeds %d", ErrSpawnBudget, request.Depth, b.MaxDepth)
	}
	if b.MaxChildren > 0 && b.children[request.ParentID] >= b.MaxChildren {
		return fmt.Errorf("%w: parent %q exhausted child fanout %d", ErrSpawnBudget, request.ParentID, b.MaxChildren)
	}
	if b.MaxDescendants > 0 && b.descendants >= b.MaxDescendants {
		return fmt.Errorf("%w: lineage exhausted aggregate descendants %d", ErrSpawnBudget, b.MaxDescendants)
	}
	if _, exists := b.reservations[request.ChildID]; exists {
		return fmt.Errorf("%w: child %q already has a reservation", ErrSpawnBudget, request.ChildID)
	}
	if err := b.canReserve(request.Budget); err != nil {
		return err
	}
	if b.children == nil {
		b.children = make(map[string]int)
	}
	if b.reservations == nil {
		b.reservations = make(map[string]LineageBudget)
	}
	b.children[request.ParentID]++
	b.descendants++
	b.reserved = addLineageBudget(b.reserved, request.Budget)
	b.reservations[request.ChildID] = request.Budget
	ancestors := cloneFingerprints(parent.ancestors)
	ancestors[parent.goalFingerprint] = struct{}{}
	b.goals[request.ChildID] = goalLineage{
		goalFingerprint: goalFingerprint,
		pathFingerprint: fingerprintGoalPath(parent.pathFingerprint, goalFingerprint),
		ancestors:       ancestors,
		depth:           request.Depth,
	}
	b.goalOwners[goalFingerprint] = request.ChildID
	if b.store != nil {
		if err := b.store.LogAdmit(request); err != nil {
			b.children[request.ParentID]--
			b.descendants--
			b.reserved = subtractLineageBudget(b.reserved, request.Budget)
			delete(b.reservations, request.ChildID)
			delete(b.goals, request.ChildID)
			delete(b.goalOwners, goalFingerprint)
			return fmt.Errorf("%w: persist child admission: %w", ErrSpawnBudget, err)
		}
	}
	return nil
}

// reconcile replaces one completed child's conservative reservation with
// host-observed usage. Missing reconciliation stays fully charged. Usage above
// the reservation is refused without releasing capacity.
func (b *SpawnBudget) reconcile(childID string, actual LineageBudget) error {
	if b == nil {
		return fmt.Errorf("%w: missing host budget", ErrSpawnBudget)
	}
	if childID == "" {
		return fmt.Errorf("%w: child id is required for reconciliation", ErrSpawnBudget)
	}
	if err := validateLineageBudget("actual usage", actual); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	reservation, ok := b.reservations[childID]
	if !ok {
		return fmt.Errorf("%w: child %q has no active reservation", ErrSpawnBudget, childID)
	}
	if actual.Tokens > reservation.Tokens || actual.OutputTokens > reservation.OutputTokens || actual.CostMicrosUSD > reservation.CostMicrosUSD {
		return fmt.Errorf("%w: child %q usage exceeds its reservation", ErrSpawnBudget, childID)
	}
	if b.store != nil {
		if err := b.store.LogReconcile(childID, actual); err != nil {
			return fmt.Errorf("%w: persist child reconciliation: %w", ErrSpawnBudget, err)
		}
	}
	b.reserved = subtractLineageBudget(b.reserved, reservation)
	b.spent = addLineageBudget(b.spent, actual)
	delete(b.reservations, childID)
	return nil
}

// Reconcile replaces one completed child's conservative reservation with
// host-observed usage.
func (b *SpawnBudget) Reconcile(childID string, actual LineageBudget) error {
	return b.reconcile(childID, actual)
}

// Settle is an alias for Reconcile.
func (b *SpawnBudget) Settle(childID string, actual LineageBudget) error {
	return b.reconcile(childID, actual)
}

// Reserved reports active conservative reservations across all unreconciled children.
func (b *SpawnBudget) Reserved() LineageBudget {
	if b == nil {
		return LineageBudget{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.reserved
}

// Spent reports settled consumption across all completed children.
func (b *SpawnBudget) Spent() LineageBudget {
	if b == nil {
		return LineageBudget{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.spent
}

// Remaining reports remaining unreserved capacity for the lineage.
func (b *SpawnBudget) Remaining() LineageBudget {
	if b == nil {
		return LineageBudget{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	used := addLineageBudget(b.spent, b.reserved)
	rem := LineageBudget{}
	if b.MaxTokens > 0 {
		rem.Tokens = b.MaxTokens - used.Tokens
		if rem.Tokens < 0 {
			rem.Tokens = 0
		}
	}
	if b.MaxOutputTokens > 0 {
		rem.OutputTokens = b.MaxOutputTokens - used.OutputTokens
		if rem.OutputTokens < 0 {
			rem.OutputTokens = 0
		}
	}
	if b.MaxCostMicrosUSD > 0 {
		rem.CostMicrosUSD = b.MaxCostMicrosUSD - used.CostMicrosUSD
		if rem.CostMicrosUSD < 0 {
			rem.CostMicrosUSD = 0
		}
	}
	return rem
}

// Allowance is an alias for Remaining.
func (b *SpawnBudget) Allowance() LineageBudget {
	return b.Remaining()
}

// Reservation reports the active reservation for a child, if any.
func (b *SpawnBudget) Reservation(childID string) (LineageBudget, bool) {
	if b == nil {
		return LineageBudget{}, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	res, ok := b.reservations[childID]
	return res, ok
}

// Reservations returns a snapshot of active reservations.
func (b *SpawnBudget) Reservations() map[string]LineageBudget {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	res := make(map[string]LineageBudget, len(b.reservations))
	for k, v := range b.reservations {
		res[k] = v
	}
	return res
}

// Descendants reports host-admitted children across the entire lineage.
func (b *SpawnBudget) Descendants() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.descendants
}

func (b *SpawnBudget) release(request SpawnRequest) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.children[request.ParentID] > 0 {
		b.children[request.ParentID]--
		b.descendants--
	}
	if reservation, ok := b.reservations[request.ChildID]; ok {
		b.reserved = subtractLineageBudget(b.reserved, reservation)
		delete(b.reservations, request.ChildID)
	}
	if goal, ok := b.goals[request.ChildID]; ok {
		delete(b.goals, request.ChildID)
		delete(b.goalOwners, goal.goalFingerprint)
	}
	if b.store != nil {
		_ = b.store.LogRelease(request)
	}
}

func (b *SpawnBudget) ensureGoalRoot() error {
	rootID := strings.TrimSpace(b.RootID)
	rootFingerprint, err := fingerprintGoal(b.RootGoal)
	if rootID == "" || err != nil {
		return fmt.Errorf("%w: %w: root id and root goal are required", ErrSpawnBudget, ErrInvalidAncestry)
	}
	if b.rootID != "" {
		if b.rootID != rootID || b.rootGoal != rootFingerprint {
			return fmt.Errorf("%w: %w: root identity changed after admission", ErrSpawnBudget, ErrInvalidAncestry)
		}
		return nil
	}
	b.rootID, b.rootGoal = rootID, rootFingerprint
	if b.goals == nil {
		b.goals = make(map[string]goalLineage)
	}
	if b.goalOwners == nil {
		b.goalOwners = make(map[string]string)
	}
	b.goals[rootID] = goalLineage{
		goalFingerprint: rootFingerprint,
		pathFingerprint: rootFingerprint,
		ancestors:       map[string]struct{}{},
		depth:           0,
	}
	b.goalOwners[rootFingerprint] = rootID
	if b.store != nil {
		if err := b.store.LogRoot(rootID, b.RootGoal); err != nil {
			return fmt.Errorf("%w: persist root identity: %w", ErrSpawnBudget, err)
		}
	}
	return nil
}

func fingerprintGoal(goal string) (string, error) {
	normalized := strings.ToLower(strings.Join(strings.Fields(goal), " "))
	if normalized == "" {
		return "", fmt.Errorf("%w: goal is required", ErrSpawnBudget)
	}
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:]), nil
}

func fingerprintGoalPath(parentPath, goalFingerprint string) string {
	sum := sha256.Sum256([]byte(parentPath + "\x00" + goalFingerprint))
	return hex.EncodeToString(sum[:])
}

func cloneFingerprints(src map[string]struct{}) map[string]struct{} {
	dst := make(map[string]struct{}, len(src)+1)
	for fingerprint := range src {
		dst[fingerprint] = struct{}{}
	}
	return dst
}

// ReconcileChild accepts usage only after the child has retired, keeping live
// work from returning capacity before it can spend the reservation.
func (h *Host) ReconcileChild(childID string, actual LineageBudget) error {
	if h == nil || h.spawnBudget == nil {
		return fmt.Errorf("%w: host has no recursive spawn budget", ErrSpawnBudget)
	}
	h.mu.Lock()
	_, live := h.live[childID]
	h.mu.Unlock()
	if live {
		return fmt.Errorf("%w: child %q is still live", ErrSpawnBudget, childID)
	}
	return h.spawnBudget.reconcile(childID, actual)
}

func (b *SpawnBudget) canReserve(request LineageBudget) error {
	checks := []struct {
		name      string
		limit     int64
		spent     int64
		reserved  int64
		requested int64
	}{
		{name: "tokens", limit: b.MaxTokens, spent: b.spent.Tokens, reserved: b.reserved.Tokens, requested: request.Tokens},
		{name: "output tokens", limit: b.MaxOutputTokens, spent: b.spent.OutputTokens, reserved: b.reserved.OutputTokens, requested: request.OutputTokens},
		{name: "cost micro-USD", limit: b.MaxCostMicrosUSD, spent: b.spent.CostMicrosUSD, reserved: b.reserved.CostMicrosUSD, requested: request.CostMicrosUSD},
	}
	for _, check := range checks {
		if !lineageBudgetFits(check.limit, check.spent, check.reserved, check.requested) {
			return fmt.Errorf("%w: lineage aggregate %s budget exhausted: used=%d requested=%d limit=%d", ErrSpawnBudget, check.name, check.spent+check.reserved, check.requested, check.limit)
		}
	}
	return nil
}

func validateLineageBudget(kind string, budget LineageBudget) error {
	if budget.Tokens < 0 || budget.OutputTokens < 0 || budget.CostMicrosUSD < 0 {
		return fmt.Errorf("%w: %s must be nonnegative", ErrSpawnBudget, kind)
	}
	if budget.OutputTokens > budget.Tokens {
		return fmt.Errorf("%w: %s output tokens exceed total tokens", ErrSpawnBudget, kind)
	}
	return nil
}

func validateLineageLimits(limits LineageBudget) error {
	if limits.Tokens < 0 || limits.OutputTokens < 0 || limits.CostMicrosUSD < 0 {
		return fmt.Errorf("%w: aggregate limits must be nonnegative", ErrSpawnBudget)
	}
	if limits.Tokens > 0 && limits.OutputTokens > limits.Tokens {
		return fmt.Errorf("%w: aggregate output-token limit exceeds total-token limit", ErrSpawnBudget)
	}
	return nil
}

func lineageBudgetFits(limit, spent, reserved, requested int64) bool {
	const maxInt64 = int64(^uint64(0) >> 1)
	if reserved > maxInt64-spent || requested > maxInt64-spent-reserved {
		return false
	}
	if limit == 0 {
		return true
	}
	used := spent + reserved
	return used <= limit && requested <= limit-used
}

func addLineageBudget(a, b LineageBudget) LineageBudget {
	return LineageBudget{Tokens: a.Tokens + b.Tokens, OutputTokens: a.OutputTokens + b.OutputTokens, CostMicrosUSD: a.CostMicrosUSD + b.CostMicrosUSD}
}

func subtractLineageBudget(a, b LineageBudget) LineageBudget {
	return LineageBudget{Tokens: a.Tokens - b.Tokens, OutputTokens: a.OutputTokens - b.OutputTokens, CostMicrosUSD: a.CostMicrosUSD - b.CostMicrosUSD}
}
