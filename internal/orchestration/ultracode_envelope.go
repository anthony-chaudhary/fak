package orchestration

import (
	"fmt"
	"math"
	"time"
)

const UltracodeEnvelopeReceiptSchema = "fak.ultracode_budget_receipt.v1"

const (
	UltracodeBudgetAuthorityProvider   = "provider-reported"
	UltracodeBudgetAuthorityIncomplete = "incomplete"

	UltracodeBudgetReasonIncomplete   = "BUDGET_RECEIPT_INCOMPLETE"
	UltracodeBudgetReasonTokenOverrun = "PARENT_TOKEN_BUDGET_EXCEEDED"
	UltracodeBudgetReasonWallOverrun  = "PARENT_WALL_DEADLINE_EXCEEDED"
)

// UltracodeChildBudget binds one launched child to a conserved slice of the
// parent token envelope. ProviderTokens is trusted only when Covered is true.
type UltracodeChildBudget struct {
	ChildID        string `json:"child_id"`
	ReservedTokens int64  `json:"reserved_tokens"`
	ProviderTokens int64  `json:"provider_tokens"`
	Authority      string `json:"authority"`
	Covered        bool   `json:"covered"`
	Overrun        bool   `json:"overrun"`
}

// UltracodeEnvelopeReceipt is the one launch/status/benchmark budget artifact.
// Admitted stays false until every child has provider-authoritative usage and
// neither the aggregate token ceiling nor the shared wall deadline was crossed.
type UltracodeEnvelopeReceipt struct {
	Schema          string                 `json:"schema"`
	DeclaredTokens  int64                  `json:"declared_tokens"`
	WallBudgetMS    int64                  `json:"wall_budget_ms"`
	StartedAt       time.Time              `json:"started_at"`
	DeadlineAt      time.Time              `json:"deadline_at"`
	Authority       string                 `json:"authority"`
	CoveredChildren int                    `json:"covered_children"`
	TotalChildren   int                    `json:"total_children"`
	ConsumedTokens  int64                  `json:"consumed_tokens"`
	RemainingTokens int64                  `json:"remaining_tokens"`
	ConsumedWallMS  int64                  `json:"consumed_wall_ms"`
	RemainingWallMS int64                  `json:"remaining_wall_ms"`
	TokenOverrun    bool                   `json:"token_overrun"`
	WallOverrun     bool                   `json:"wall_overrun"`
	Overrun         bool                   `json:"overrun"`
	Complete        bool                   `json:"complete"`
	Admitted        bool                   `json:"admitted"`
	Reason          string                 `json:"reason,omitempty"`
	Children        []UltracodeChildBudget `json:"children"`
}

type UltracodeChildUsage struct {
	ChildID        string
	ProviderTokens int64
	Authority      string
}

// NewUltracodeEnvelopeReceipt reserves the parent token budget exactly once.
// Remainders are distributed deterministically, so the child slices sum to the
// declared cap instead of giving every child the full cap.
func NewUltracodeEnvelopeReceipt(tokens int64, wall time.Duration, started time.Time, childIDs []string) (UltracodeEnvelopeReceipt, error) {
	if tokens <= 0 || wall <= 0 || started.IsZero() || len(childIDs) == 0 {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode envelope requires positive token/wall limits, a start time, and children")
	}
	events := make([]BudgetEvent, 0, len(childIDs))
	seen := make(map[string]struct{}, len(childIDs))
	base, extra := tokens/int64(len(childIDs)), tokens%int64(len(childIDs))
	if base <= 0 {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode token envelope %d cannot cover %d children", tokens, len(childIDs))
	}
	for i, id := range childIDs {
		if id == "" {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode child id is empty")
		}
		if _, ok := seen[id]; ok {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("duplicate ultracode child %q", id)
		}
		seen[id] = struct{}{}
		allocation := base
		if int64(i) < extra {
			allocation++
		}
		events = append(events, BudgetEvent{Kind: BudgetReserve, NodeID: id, ParentID: RootBudgetNodeID, Workers: 1, Tokens: allocation})
	}
	ledger, err := FoldBudgetEvents(Budget{MaxWorkers: len(childIDs) + 1, MaxTokens: tokens}, events)
	if err != nil {
		return UltracodeEnvelopeReceipt{}, err
	}
	children := make([]UltracodeChildBudget, 0, len(childIDs))
	for _, id := range childIDs {
		children = append(children, UltracodeChildBudget{
			ChildID: id, ReservedTokens: ledger.Nodes[id].Allocation.MaxTokens,
			Authority: UltracodeBudgetAuthorityIncomplete,
		})
	}
	wallMS := wall.Milliseconds()
	if wallMS <= 0 {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode wall envelope must be at least one millisecond")
	}
	started = started.UTC()
	return UltracodeEnvelopeReceipt{
		Schema: UltracodeEnvelopeReceiptSchema, DeclaredTokens: tokens,
		WallBudgetMS: wallMS, StartedAt: started, DeadlineAt: started.Add(time.Duration(wallMS) * time.Millisecond),
		Authority: UltracodeBudgetAuthorityIncomplete, TotalChildren: len(children),
		RemainingTokens: tokens, RemainingWallMS: wallMS,
		Reason: UltracodeBudgetReasonIncomplete, Children: children,
	}, nil
}

// FoldUltracodeEnvelopeReceipt joins provider usage onto the conserved launch
// reservation. Missing or non-authoritative child usage is an invalid receipt,
// never permission to continue into a value verdict.
func FoldUltracodeEnvelopeReceipt(base UltracodeEnvelopeReceipt, usage []UltracodeChildUsage, now time.Time) (UltracodeEnvelopeReceipt, error) {
	if base.Schema != UltracodeEnvelopeReceiptSchema || base.DeclaredTokens <= 0 || base.WallBudgetMS <= 0 || base.StartedAt.IsZero() || base.DeadlineAt.IsZero() || now.IsZero() || len(base.Children) == 0 {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("invalid ultracode envelope receipt")
	}
	if base.WallBudgetMS > int64(time.Duration(math.MaxInt64)/time.Millisecond) || !base.DeadlineAt.Equal(base.StartedAt.Add(time.Duration(base.WallBudgetMS)*time.Millisecond)) {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode envelope wall deadline does not match its declared limit")
	}
	var reserved int64
	baseIDs := make(map[string]struct{}, len(base.Children))
	for _, child := range base.Children {
		if child.ChildID == "" || child.ReservedTokens <= 0 {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("invalid ultracode child reservation")
		}
		if _, exists := baseIDs[child.ChildID]; exists {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("duplicate ultracode child %q", child.ChildID)
		}
		if child.ReservedTokens > math.MaxInt64-reserved {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode child reservations overflow")
		}
		baseIDs[child.ChildID] = struct{}{}
		reserved += child.ReservedTokens
	}
	if reserved != base.DeclaredTokens {
		return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode child reservations total %d, want declared parent %d", reserved, base.DeclaredTokens)
	}
	out := base
	out.Children = append([]UltracodeChildBudget(nil), base.Children...)
	out.Authority = UltracodeBudgetAuthorityIncomplete
	out.CoveredChildren = 0
	out.ConsumedTokens = 0
	out.RemainingTokens = out.DeclaredTokens
	out.TokenOverrun = false
	out.WallOverrun = false
	out.Overrun = false
	out.Complete = false
	out.Admitted = false
	out.Reason = UltracodeBudgetReasonIncomplete
	byID := make(map[string]int, len(out.Children))
	for i := range out.Children {
		out.Children[i].ProviderTokens = 0
		out.Children[i].Authority = UltracodeBudgetAuthorityIncomplete
		out.Children[i].Covered = false
		out.Children[i].Overrun = false
		byID[out.Children[i].ChildID] = i
	}
	seen := make(map[string]struct{}, len(usage))
	for _, sample := range usage {
		idx, ok := byID[sample.ChildID]
		if !ok {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("usage for unknown ultracode child %q", sample.ChildID)
		}
		if _, ok := seen[sample.ChildID]; ok {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("duplicate usage for ultracode child %q", sample.ChildID)
		}
		if sample.ProviderTokens < 0 {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("negative usage for ultracode child %q", sample.ChildID)
		}
		seen[sample.ChildID] = struct{}{}
		child := &out.Children[idx]
		child.ProviderTokens = sample.ProviderTokens
		child.Authority = sample.Authority
		child.Covered = sample.Authority == UltracodeBudgetAuthorityProvider
		child.Overrun = sample.ProviderTokens > child.ReservedTokens
		if sample.ProviderTokens > math.MaxInt64-out.ConsumedTokens {
			return UltracodeEnvelopeReceipt{}, fmt.Errorf("ultracode provider usage overflows aggregate")
		}
		out.ConsumedTokens += sample.ProviderTokens
		if child.Covered {
			out.CoveredChildren++
		}
	}
	if out.ConsumedTokens >= out.DeclaredTokens {
		out.RemainingTokens = 0
	} else {
		out.RemainingTokens = out.DeclaredTokens - out.ConsumedTokens
	}
	elapsed := now.Sub(out.StartedAt)
	if elapsed < 0 {
		elapsed = 0
	}
	out.ConsumedWallMS = elapsed.Milliseconds()
	if out.ConsumedWallMS >= out.WallBudgetMS {
		out.RemainingWallMS = 0
	} else {
		out.RemainingWallMS = out.WallBudgetMS - out.ConsumedWallMS
	}
	out.TokenOverrun = out.ConsumedTokens > out.DeclaredTokens
	out.WallOverrun = !now.Before(out.DeadlineAt)
	out.Overrun = out.TokenOverrun || out.WallOverrun
	out.TotalChildren = len(out.Children)
	out.Complete = out.CoveredChildren == out.TotalChildren
	if out.Complete {
		out.Authority = UltracodeBudgetAuthorityProvider
	} else {
		out.Authority = UltracodeBudgetAuthorityIncomplete
	}
	switch {
	case out.TokenOverrun:
		out.Reason = UltracodeBudgetReasonTokenOverrun
	case out.WallOverrun:
		out.Reason = UltracodeBudgetReasonWallOverrun
	case !out.Complete:
		out.Reason = UltracodeBudgetReasonIncomplete
	default:
		out.Reason = ""
	}
	out.Admitted = out.Complete && !out.Overrun
	return out, nil
}

// ValidateUltracodeEnvelopeReceipt replays a persisted receipt from its child
// evidence. A consumer must not trust precomputed admission or overrun booleans.
func ValidateUltracodeEnvelopeReceipt(receipt UltracodeEnvelopeReceipt) error {
	if receipt.ConsumedWallMS < 0 || receipt.ConsumedWallMS > int64(time.Duration(math.MaxInt64)/time.Millisecond) {
		return fmt.Errorf("invalid ultracode consumed wall time")
	}
	usage := make([]UltracodeChildUsage, 0, len(receipt.Children))
	for _, child := range receipt.Children {
		usage = append(usage, UltracodeChildUsage{
			ChildID: child.ChildID, ProviderTokens: child.ProviderTokens, Authority: child.Authority,
		})
	}
	observedAt := receipt.StartedAt.Add(time.Duration(receipt.ConsumedWallMS) * time.Millisecond)
	replayed, err := FoldUltracodeEnvelopeReceipt(receipt, usage, observedAt)
	if err != nil {
		return err
	}
	if receipt.Authority != replayed.Authority ||
		receipt.CoveredChildren != replayed.CoveredChildren || receipt.TotalChildren != replayed.TotalChildren ||
		receipt.ConsumedTokens != replayed.ConsumedTokens || receipt.RemainingTokens != replayed.RemainingTokens ||
		receipt.RemainingWallMS != replayed.RemainingWallMS || receipt.TokenOverrun != replayed.TokenOverrun ||
		receipt.WallOverrun != replayed.WallOverrun || receipt.Overrun != replayed.Overrun ||
		receipt.Complete != replayed.Complete || receipt.Admitted != replayed.Admitted || receipt.Reason != replayed.Reason {
		return fmt.Errorf("ultracode envelope summary does not match child evidence")
	}
	for i := range receipt.Children {
		if receipt.Children[i] != replayed.Children[i] {
			return fmt.Errorf("ultracode child %q summary does not match provider evidence", receipt.Children[i].ChildID)
		}
	}
	return nil
}
