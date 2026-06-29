package memq

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// RecallBackend adapts a loaded recall core image to the memq Backend interface, so
// the SAME algebra an agent authors over the in-memory store runs, unchanged, over the
// real durable store fak already ships. It is the proof that memq is not a toy: recall's
// own Recall/tombstone are re-expressible as memq queries against this backend.
//
// It implements Backend (Cells + trust-gated Materialize) and Tombstoner (recall's
// negative-only RequestContextChange, persisted to dir). It deliberately does NOT
// implement Pruner: a recall image's CAS GC + reseal pass is recall.Dream's job, which
// memq complements — a prune op against this backend is reported as proposal-only with
// a pointer to `fak dream`.
type RecallBackend struct {
	s   *recall.Session
	dir string // where Tombstone persists the mutated manifest
}

// NewRecallBackend wraps a loaded session. dir is the image directory a tombstone is
// persisted back to; pass "" to keep tombstones in memory only (proposal/preview).
func NewRecallBackend(s *recall.Session, dir string) *RecallBackend {
	return &RecallBackend{s: s, dir: dir}
}

// Cells maps the recall page table to memq cells, carrying only safe metadata.
func (r *RecallBackend) Cells(_ context.Context) ([]Cell, error) {
	pages := r.s.Pages()
	out := make([]Cell, 0, len(pages))
	for _, p := range pages {
		kind := "tool_result"
		if p.Quarantined {
			kind = "sealed"
		}
		out = append(out, Cell{
			ID:         "step:" + strconv.Itoa(p.Step),
			Step:       p.Step,
			Role:       p.Role,
			Kind:       kind,
			Descriptor: p.Descriptor,
			Digest:     p.Digest,
			Bytes:      p.Len,
			Durability: NormDurability(p.Durability),
			Sealed:     p.Quarantined,
			Tombstoned: r.s.Tombstoned(p.Step),
			Witness:    p.Witness,
		})
	}
	return out, nil
}

// Materialize pages a recall page in through its trust gate, translating recall's seal
// errors into memq.ErrSealed so the executor labels the refusal without importing the
// recall error surface into memq core.
func (r *RecallBackend) Materialize(ctx context.Context, id string) ([]byte, error) {
	step, err := parseStep(id)
	if err != nil {
		return nil, err
	}
	b, err := r.s.Resolve(ctx, step)
	if err != nil {
		if errors.Is(err, recall.ErrStale) {
			return nil, fmt.Errorf("%w: %v", ErrStale, err)
		}
		if errors.Is(err, recall.ErrSealed) || errors.Is(err, recall.ErrTombstoned) {
			return nil, fmt.Errorf("%w: %v", ErrSealed, err)
		}
		return nil, err
	}
	return b, nil
}

// Tombstone records recall's negative-only context change for the page and persists
// the mutated manifest (the CAS bytes are untouched). With dir=="" it suppresses in
// the loaded session only (a preview that does not survive the process).
func (r *RecallBackend) Tombstone(_ context.Context, id, reason, by string) (bool, error) {
	step, err := parseStep(id)
	if err != nil {
		return false, err
	}
	if r.s.Tombstoned(step) {
		return false, nil
	}
	if _, err := r.s.RequestContextChange(recall.ContextChangeRequest{
		Action: recall.ContextActionTombstone, Step: step, Reason: reason, RequestedBy: by,
	}); err != nil {
		return false, err
	}
	if r.dir != "" {
		if err := r.s.Persist(r.dir); err != nil {
			return false, err
		}
	}
	return true, nil
}

func parseStep(id string) (int, error) {
	rest, ok := strings.CutPrefix(id, "step:")
	if !ok {
		return 0, fmt.Errorf("memq: recall backend id %q is not step:N", id)
	}
	n, err := strconv.Atoi(rest)
	if err != nil {
		return 0, fmt.Errorf("memq: recall backend id %q: %w", id, err)
	}
	return n, nil
}
