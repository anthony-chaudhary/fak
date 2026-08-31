package model

import "fmt"

// SpeculativeCostComponent records one wall-time category without pretending
// that an uninstrumented category cost zero. Measured=false is a hard fence
// against treating a partial receipt as end-to-end performance evidence.
type SpeculativeCostComponent struct {
	Nanoseconds int64 `json:"nanoseconds"`
	Measured    bool  `json:"measured"`
}

// SpeculativeCostAccounting names every cost that a speculative-decode
// performance claim must include. A target-verification receipt may populate
// only the categories it directly owns; MissingCostCategories reports the rest.
type SpeculativeCostAccounting struct {
	Setup              SpeculativeCostComponent `json:"setup"`
	Drafting           SpeculativeCostComponent `json:"drafting"`
	TargetVerification SpeculativeCostComponent `json:"target_verification"`
	Rejection          SpeculativeCostComponent `json:"rejection"`
	Rollback           SpeculativeCostComponent `json:"rollback"`
	Synchronization    SpeculativeCostComponent `json:"synchronization"`
	Recovery           SpeculativeCostComponent `json:"recovery"`
	KnownMemoryBytes   int64                    `json:"known_memory_bytes"`
	MemoryMeasured     bool                     `json:"memory_measured"`
}

// TargetVerificationReceipt distinguishes one genuine target verification
// operation from an explicit ordinary-target-decode downgrade. It intentionally
// cannot substantiate an end-to-end speedup while drafting, rejection, memory,
// or recovery costs remain unmeasured.
type TargetVerificationReceipt struct {
	Schema                       string                    `json:"schema"`
	Engine                       string                    `json:"engine"`
	Path                         string                    `json:"path"`
	DowngradeReason              string                    `json:"downgrade_reason,omitempty"`
	DraftTokens                  int                       `json:"draft_tokens"`
	AcceptedTokens               int                       `json:"accepted_tokens"`
	RejectedTokens               int                       `json:"rejected_tokens"`
	TargetVerificationOperations int                       `json:"target_verification_operations"`
	TargetDecodeSteps            int                       `json:"target_decode_steps"`
	OneOperation                 bool                      `json:"one_operation"`
	Accounting                   SpeculativeCostAccounting `json:"accounting"`
}

// MissingCostCategories returns every category that prevents this receipt from
// supporting an end-to-end performance claim.
func (r TargetVerificationReceipt) MissingCostCategories() []string {
	var missing []string
	for _, category := range []struct {
		name string
		cost SpeculativeCostComponent
	}{
		{name: "setup", cost: r.Accounting.Setup},
		{name: "drafting", cost: r.Accounting.Drafting},
		{name: "target_verification", cost: r.Accounting.TargetVerification},
		{name: "rejection", cost: r.Accounting.Rejection},
		{name: "rollback", cost: r.Accounting.Rollback},
		{name: "synchronization", cost: r.Accounting.Synchronization},
		{name: "recovery", cost: r.Accounting.Recovery},
	} {
		if !category.cost.Measured {
			missing = append(missing, category.name)
		}
	}
	if !r.Accounting.MemoryMeasured {
		missing = append(missing, "memory")
	}
	return missing
}

// EndToEndMeasured is true only when every required cost category is present.
func (r TargetVerificationReceipt) EndToEndMeasured() bool {
	return len(r.MissingCostCategories()) == 0
}

// HybridSpecCheckpoint identifies one speculative suffix and its accepted prefix.
// A checkpoint belongs to exactly one state and can be consumed only once.
type HybridSpecCheckpoint struct {
	owner      *HybridSpecState
	generation uint64
	baseCursor int
	cursor     int
	kv         [][]byte
	recurrent  [][]byte
}

// HybridSpecState keeps attention KV and recurrent state at one token cursor.
// Speculative writes and restoration always move both state families together.
// baseCursor permits an accepted prefix to be evicted without losing the
// absolute token position used to validate later speculative transactions.
type HybridSpecState struct {
	kv         [][]byte
	recurrent  [][]byte
	baseCursor int
	cursor     int
	generation uint64
	active     bool
}

// NewHybridSpecState copies an accepted hybrid-state prefix.
func NewHybridSpecState(kv, recurrent [][]byte) (*HybridSpecState, error) {
	if len(kv) != len(recurrent) {
		return nil, fmt.Errorf("hybrid speculative state: KV length %d differs from recurrent length %d", len(kv), len(recurrent))
	}
	return &HybridSpecState{
		kv:        cloneHybridState(kv),
		recurrent: cloneHybridState(recurrent),
		cursor:    len(kv),
	}, nil
}

// Begin snapshots the accepted resident state before speculative writes.
func (s *HybridSpecState) Begin() HybridSpecCheckpoint {
	s.generation++
	s.active = true
	return HybridSpecCheckpoint{
		owner:      s,
		generation: s.generation,
		baseCursor: s.baseCursor,
		cursor:     s.cursor,
		kv:         cloneHybridState(s.kv),
		recurrent:  cloneHybridState(s.recurrent),
	}
}

// Append adds one speculative token to both state families.
func (s *HybridSpecState) Append(kv, recurrent []byte) error {
	if !s.active {
		return fmt.Errorf("hybrid speculative state: append requires an active checkpoint")
	}
	s.kv = append(s.kv, append([]byte(nil), kv...))
	s.recurrent = append(s.recurrent, append([]byte(nil), recurrent...))
	s.cursor++
	return nil
}

// Commit keeps accepted speculative tokens and discards the rejected suffix.
func (s *HybridSpecState) Commit(checkpoint HybridSpecCheckpoint, accepted int) error {
	if err := s.validate(checkpoint); err != nil {
		return err
	}
	speculative := s.cursor - checkpoint.cursor
	if accepted < 0 || accepted > speculative {
		return fmt.Errorf("hybrid speculative state: accepted %d tokens from suffix of %d", accepted, speculative)
	}
	resident := len(checkpoint.kv) + accepted
	s.kv = cloneHybridState(s.kv[:resident])
	s.recurrent = cloneHybridState(s.recurrent[:resident])
	s.cursor = checkpoint.cursor + accepted
	s.active = false
	return nil
}

// Rollback restores both state families to the exact checkpointed prefix.
func (s *HybridSpecState) Rollback(checkpoint HybridSpecCheckpoint) error {
	if err := s.validate(checkpoint); err != nil {
		return err
	}
	s.kv = cloneHybridState(checkpoint.kv)
	s.recurrent = cloneHybridState(checkpoint.recurrent)
	s.baseCursor = checkpoint.baseCursor
	s.cursor = checkpoint.cursor
	s.active = false
	return nil
}

// RetainLast evicts accepted resident state before the newest keep tokens while
// preserving the absolute cursor. Eviction is forbidden during a transaction so
// rollback always restores one coherent KV/recurrent window.
func (s *HybridSpecState) RetainLast(keep int) error {
	if keep < 0 {
		return fmt.Errorf("hybrid speculative state: negative retained token count %d", keep)
	}
	if s.active {
		return fmt.Errorf("hybrid speculative state: cannot evict during an active checkpoint")
	}
	if keep >= len(s.kv) {
		return nil
	}
	drop := len(s.kv) - keep
	s.kv = cloneHybridState(s.kv[drop:])
	s.recurrent = cloneHybridState(s.recurrent[drop:])
	s.baseCursor += drop
	return nil
}

// Cursor returns the absolute accepted-plus-speculative token position.
func (s *HybridSpecState) Cursor() int { return s.cursor }

// KVState returns a copy of the resident attention state.
func (s *HybridSpecState) KVState() [][]byte { return cloneHybridState(s.kv) }

// RecurrentState returns a copy of the resident recurrent GDN state.
func (s *HybridSpecState) RecurrentState() [][]byte { return cloneHybridState(s.recurrent) }

func (s *HybridSpecState) validate(checkpoint HybridSpecCheckpoint) error {
	if checkpoint.owner != s {
		return fmt.Errorf("hybrid speculative state: foreign checkpoint")
	}
	if !s.active || checkpoint.generation != s.generation {
		return fmt.Errorf("hybrid speculative state: stale checkpoint")
	}
	if len(s.kv) != len(s.recurrent) || s.cursor-s.baseCursor != len(s.kv) ||
		checkpoint.baseCursor != s.baseCursor || checkpoint.cursor > s.cursor ||
		checkpoint.cursor-checkpoint.baseCursor != len(checkpoint.kv) {
		return fmt.Errorf("hybrid speculative state: inconsistent paired state")
	}
	return nil
}

func cloneHybridState(state [][]byte) [][]byte {
	if state == nil {
		return nil
	}
	cloned := make([][]byte, len(state))
	for i := range state {
		cloned[i] = append([]byte(nil), state[i]...)
	}
	return cloned
}
