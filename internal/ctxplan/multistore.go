package ctxplan

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// The cross-session horizon (#566). The scaling story targets 50 -> 1,000,000 turns, but a
// single agent SESSION is rarely a million turns — that horizon spans MANY sessions. The
// planner already views one session's transcript (the agent seam) or one finished recall
// core image (recall.CtxStore). This file supplies the missing image: a durable,
// cross-session UNION so a fresh session reconstructs its working set as a planned VIEW over
// everything a principal did before — the "1 current turn + a flexible history" bootstrap,
// where the history is durable and cross-session, not per-session.
//
// The union is itself a ctxplan.Store, so it drops straight into Materialize / PlanCells /
// BuildIndex unchanged — nothing in the planner core learns about sessions. Two invariants
// are what the seam adds on top of a plain concatenation, and both are enforced here:
//
//  1. The DURABILITY BOUNDARY at the session seam. Expire-by-default is a temporal posture;
//     across a session boundary it means a PRIOR session contributes only the spans that
//     OUTLIVE their origin session — bounded and durable (preferences, identity, learned
//     utility). Its turn/session spans expired when that session ended. The CURRENT session
//     expires nothing (it has not crossed a boundary yet).
//  2. The TRUST GATE travels across the boundary. A sealed/tombstoned span from a prior
//     session is still surfaced in Spans (so the audit sees it and scoring pins it to 0
//     Benefit), but Materialize routes to the owning sub-store, which refuses its bytes.
//     Poison a prior session quarantined can never re-enter this session's context.

// SessionScope classifies a union source by its relationship to the CURRENT planning
// session — the axis the cross-session durability boundary turns on.
type SessionScope int

const (
	// ScopePrior is a past session's image (or a durable memory tier): a span survives into
	// this session's candidate set only if it OUTLIVES its origin session — bounded or durable.
	// A prior turn/session span expired when that session ended and is dropped from the union.
	ScopePrior SessionScope = iota
	// ScopeCurrent is the live session's image: every span survives, because "expired across a
	// session boundary" is undefined for a boundary this session has not crossed.
	ScopeCurrent
)

// String renders the scope for provenance/audit output.
func (s SessionScope) String() string {
	if s == ScopeCurrent {
		return "current"
	}
	return "prior"
}

// survives reports whether a span of the given durability, drawn from a source of this scope,
// clears the cross-session boundary into the candidate set. The current session keeps all; a
// prior source keeps only spans that outlive a session (rank >= bounded).
func (s SessionScope) survives(durability string) bool {
	if s == ScopeCurrent {
		return true
	}
	return durabilityRank[NormDurability(durability)] >= durabilityRank[DurabilityBounded]
}

// crossSessionSep separates a source name from the sub-store's native span id in a unioned id
// ("prior0#span:3"). Materialize splits on the FIRST separator, so a sub-store that is itself
// a CrossSessionStore composes: the nested union re-splits its own remainder.
const crossSessionSep = "#"

// ErrExpired is returned by CrossSessionStore.Materialize for a prior-session span whose
// durability did not clear the cross-session boundary (a turn/session span from a past
// session). Such a span is never in the candidate set — Spans drops it — so the planner never
// selects it; refusing it at the gate too means a STALE recovery handle held from the prior
// session cannot page it back across the boundary behind the planner's back.
var ErrExpired = errors.New("ctxplan: span expired across the session boundary")

// Source is one image in a cross-session union: a Store, a stable Name (which namespaces its
// span ids so two sources' ids never collide), and its Scope (which sets the durability
// boundary applied to its spans). In NewUnion the slice order is oldest -> newest; that order
// becomes the union's global step order, so a prior session's spans rank older than the
// current session's.
type Source struct {
	Name  string
	Store Store
	Scope SessionScope
}

// CrossSessionStore unions a principal's session images into one ctxplan.Store, applying the
// durability boundary and preserving the trust gate (see the file header). It holds no bytes
// of its own — every page-in routes to the sub-store that owns the span.
type CrossSessionStore struct {
	sources []Source // oldest -> newest; index is the generation used for global step order
}

// NewUnion builds a CrossSessionStore from explicit sources given OLDEST-first (index =
// generation). It errors on an empty, duplicate, or separator-bearing name, since the union's
// id routing depends on names being unique and splittable.
func NewUnion(sources ...Source) (*CrossSessionStore, error) {
	seen := make(map[string]bool, len(sources))
	for _, s := range sources {
		if s.Name == "" || strings.Contains(s.Name, crossSessionSep) {
			return nil, fmt.Errorf("ctxplan: invalid union source name %q (empty or contains %q)", s.Name, crossSessionSep)
		}
		if seen[s.Name] {
			return nil, fmt.Errorf("ctxplan: duplicate union source name %q", s.Name)
		}
		if s.Store == nil {
			return nil, fmt.Errorf("ctxplan: union source %q has a nil Store", s.Name)
		}
		seen[s.Name] = true
	}
	return &CrossSessionStore{sources: append([]Source(nil), sources...)}, nil
}

// NewCrossSessionStore is the common case: the CURRENT session image plus zero or more PRIOR
// images given most-recent-first ("my previous session, the one before that, ..."). Names are
// auto-assigned ("current", "prior0", ...), and the sources are ordered oldest -> newest
// internally so global step order matches real recency. It cannot fail (names are well-formed
// by construction).
func NewCrossSessionStore(current Store, prior ...Store) *CrossSessionStore {
	srcs := make([]Source, 0, len(prior)+1)
	// prior is most-recent-first; lay them oldest-first so steps ascend toward the present.
	for i := len(prior) - 1; i >= 0; i-- {
		srcs = append(srcs, Source{Name: fmt.Sprintf("prior%d", i), Store: prior[i], Scope: ScopePrior})
	}
	srcs = append(srcs, Source{Name: "current", Store: current, Scope: ScopeCurrent})
	c, _ := NewUnion(srcs...) // names are well-formed by construction
	return c
}

// Spans unions every source's SAFE span metadata, applying the durability boundary (prior
// turn/session spans dropped) and namespacing ids by source. Global Step ascends oldest ->
// newest across sources, so a prior session's spans rank older than the current session's,
// and native order is preserved within a source.
func (c *CrossSessionStore) Spans(ctx context.Context) ([]Span, error) {
	var out []Span
	step := 0
	for _, src := range c.sources {
		spans, err := src.Store.Spans(ctx)
		if err != nil {
			return nil, fmt.Errorf("ctxplan: union source %q: %w", src.Name, err)
		}
		for _, sp := range spans {
			if !src.Scope.survives(sp.Durability) {
				continue // expired across the session boundary
			}
			sp.ID = src.Name + crossSessionSep + sp.ID
			sp.Step = step
			step++
			out = append(out, sp)
		}
	}
	return out, nil
}

// Materialize routes a unioned id back to its owning sub-store's trust-gated page-in. It
// re-checks the durability boundary for a prior source so a stale recovery handle cannot page
// an expired span back across the boundary; the sub-store then applies its own seal/tombstone
// gate, so ErrSealed/ErrTombstoned propagate unchanged (materialize.go maps them to their
// canonical refusal reasons).
func (c *CrossSessionStore) Materialize(ctx context.Context, id string) ([]byte, error) {
	name, native, ok := strings.Cut(id, crossSessionSep)
	if !ok {
		return nil, fmt.Errorf("ctxplan: union id %q missing source prefix", id)
	}
	src, ok := c.sourceByName(name)
	if !ok {
		return nil, fmt.Errorf("ctxplan: union id %q names unknown source %q", id, name)
	}
	if src.Scope == ScopePrior {
		expired, err := c.expiredInSource(ctx, src, native)
		if err != nil {
			return nil, err
		}
		if expired {
			return nil, fmt.Errorf("%w: %s", ErrExpired, id)
		}
	}
	return src.Store.Materialize(ctx, native)
}

// sourceByName finds a source by its namespace (linear over the few sources in a union).
func (c *CrossSessionStore) sourceByName(name string) (Source, bool) {
	for _, s := range c.sources {
		if s.Name == name {
			return s, true
		}
	}
	return Source{}, false
}

// expiredInSource reports whether the native id, in a prior source, refers to a span that did
// NOT clear the durability boundary. An id absent from the source is treated as not-expired,
// so a genuine lookup miss surfaces as the sub-store's own "no span" error, not a spurious
// ErrExpired.
func (c *CrossSessionStore) expiredInSource(ctx context.Context, src Source, nativeID string) (bool, error) {
	spans, err := src.Store.Spans(ctx)
	if err != nil {
		return false, fmt.Errorf("ctxplan: union source %q: %w", src.Name, err)
	}
	for _, sp := range spans {
		if sp.ID == nativeID {
			return !src.Scope.survives(sp.Durability), nil
		}
	}
	return false, nil
}

// SourceStat is the per-source provenance of a cross-session union: how many spans the source
// held, how many survived the boundary into the candidate set, how many expired, and how many
// surviving spans the trust gate will refuse. It is the readable witness for the #566
// acceptance — "durable spans surfaced and turn-scoped spans expired" is a diff of Survived
// vs Expired per prior source.
type SourceStat struct {
	Name       string `json:"name"`
	Scope      string `json:"scope"`
	Total      int    `json:"total"`
	Survived   int    `json:"survived"`
	Expired    int    `json:"expired"`
	Sealed     int    `json:"sealed"`     // surviving spans the gate will refuse (still audited)
	Tombstoned int    `json:"tombstoned"` // surviving spans context control suppressed
}

// Provenance reports the per-source accounting of the boundary. It reads only SAFE span
// metadata (never pages any bytes in), so it is cheap to attach to a plan's EXPLAIN.
func (c *CrossSessionStore) Provenance(ctx context.Context) ([]SourceStat, error) {
	out := make([]SourceStat, 0, len(c.sources))
	for _, src := range c.sources {
		spans, err := src.Store.Spans(ctx)
		if err != nil {
			return nil, fmt.Errorf("ctxplan: union source %q: %w", src.Name, err)
		}
		st := SourceStat{Name: src.Name, Scope: src.Scope.String(), Total: len(spans)}
		for _, sp := range spans {
			if !src.Scope.survives(sp.Durability) {
				st.Expired++
				continue
			}
			st.Survived++
			if sp.Sealed {
				st.Sealed++
			}
			if sp.Tombstoned {
				st.Tombstoned++
			}
		}
		out = append(out, st)
	}
	return out, nil
}
