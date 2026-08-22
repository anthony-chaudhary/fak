package microagent

import (
	"errors"
	"fmt"
	"sync"
)

// SpawnBudget bounds one host-mediated recursive task tree.
type SpawnBudget struct {
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
	Depth        int
	Budget       LineageBudget
	Capabilities CapabilityEnvelope
}

var ErrSpawnBudget = errors.New("microagent: recursive spawn budget refused")

// Admit reserves one child slot. A refusal never consumes aggregate capacity.
func (b *SpawnBudget) Admit(request SpawnRequest) error {
	if b == nil {
		return fmt.Errorf("%w: missing host budget", ErrSpawnBudget)
	}
	if request.ParentID == "" || request.ChildID == "" || request.Depth < 1 {
		return fmt.Errorf("%w: incomplete lineage", ErrSpawnBudget)
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
	b.reserved = subtractLineageBudget(b.reserved, reservation)
	b.spent = addLineageBudget(b.spent, actual)
	delete(b.reservations, childID)
	return nil
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
