package ctxplan

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// tstore is a tiny in-test Store with per-span control over durability, seal, and tombstone
// (MemStore.Add can set only Sealed). It re-implements the same trust-gated page-in as
// MemStore/recall: a sealed span refuses with ErrSealed, a tombstoned one with ErrTombstoned.
type tstore struct {
	spans []Span
	cas   map[string][]byte
}

func newTStore() *tstore { return &tstore{cas: map[string][]byte{}} }

func (t *tstore) add(id, role, durability, body string, sealed, tombstoned bool) {
	dg := Digest([]byte(body))
	desc := role + ": " + body
	if sealed {
		desc = role + ": [sealed]"
	}
	t.spans = append(t.spans, Span{
		ID: id, Step: len(t.spans), Role: role, Descriptor: desc,
		Digest: dg, Bytes: int64(len(body)), Durability: NormDurability(durability),
		Sealed: sealed, Tombstoned: tombstoned,
	})
	t.cas[dg] = []byte(body)
}

func (t *tstore) Spans(_ context.Context) ([]Span, error) {
	out := make([]Span, len(t.spans))
	copy(out, t.spans)
	return out, nil
}

func (t *tstore) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, s := range t.spans {
		if s.ID != id {
			continue
		}
		if s.Sealed {
			return nil, fmt.Errorf("%w: %s", ErrSealed, id)
		}
		if s.Tombstoned {
			return nil, fmt.Errorf("%w: %s", ErrTombstoned, id)
		}
		b, ok := t.cas[s.Digest]
		if !ok {
			return nil, fmt.Errorf("tstore: bytes absent for %s", id)
		}
		return append([]byte(nil), b...), nil
	}
	return nil, fmt.Errorf("tstore: no span %s", id)
}

func spanIDSet(spans []Span) map[string]bool {
	m := make(map[string]bool, len(spans))
	for _, s := range spans {
		m[s.ID] = true
	}
	return m
}

func spanIDs(spans []Span) []string {
	out := make([]string, len(spans))
	for i, s := range spans {
		out[i] = s.ID
	}
	return out
}

// TestCrossSessionDurabilityBoundary is the core of #566: a PRIOR session's turn/session
// spans expire at the boundary, its bounded/durable spans survive into the new session's
// candidate set, and the CURRENT session keeps everything (turn included).
func TestCrossSessionDurabilityBoundary(t *testing.T) {
	prior := newTStore()
	prior.add("p-turn", "assistant", DurabilityTurn, "ephemeral scratch", false, false)
	prior.add("p-session", "tool", DurabilitySession, "session note", false, false)
	prior.add("p-bounded", "user", DurabilityBounded, "bounded preference", false, false)
	prior.add("p-durable", "system", DurabilityDurable, "durable identity", false, false)

	cur := newTStore()
	cur.add("c-turn", "assistant", DurabilityTurn, "current scratch", false, false)
	cur.add("c-durable", "system", DurabilityDurable, "current system", false, false)

	u := NewCrossSessionStore(cur, prior)
	spans, err := u.Spans(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := spanIDSet(spans)

	if got["prior0#p-turn"] || got["prior0#p-session"] {
		t.Errorf("prior turn/session span survived the session boundary: %v", spanIDs(spans))
	}
	if !got["prior0#p-bounded"] || !got["prior0#p-durable"] {
		t.Errorf("prior bounded/durable span expired wrongly: %v", spanIDs(spans))
	}
	if !got["current#c-turn"] || !got["current#c-durable"] {
		t.Errorf("current-session span dropped (current expires nothing): %v", spanIDs(spans))
	}
}

// TestCrossSessionStepOrderOldestToNewest checks the union lays global steps oldest ->
// newest: every surviving prior span ranks older than every current span.
func TestCrossSessionStepOrderOldestToNewest(t *testing.T) {
	prior := newTStore()
	prior.add("p-a", "user", DurabilityDurable, "older a", false, false)
	prior.add("p-b", "user", DurabilityDurable, "older b", false, false)
	cur := newTStore()
	cur.add("c-a", "user", DurabilityDurable, "newer a", false, false)

	u := NewCrossSessionStore(cur, prior)
	spans, _ := u.Spans(context.Background())
	maxPrior, minCurrent := -1, 1<<30
	for _, s := range spans {
		switch {
		case s.ID == "current#c-a":
			if s.Step < minCurrent {
				minCurrent = s.Step
			}
		default:
			if s.Step > maxPrior {
				maxPrior = s.Step
			}
		}
	}
	if maxPrior >= minCurrent {
		t.Errorf("prior spans should rank older than current: maxPriorStep=%d minCurrentStep=%d", maxPrior, minCurrent)
	}
}

// TestCrossSessionTrustGateTravels: a sealed/tombstoned DURABLE prior span survives the
// durability boundary (so the audit still sees it), but the trust gate refuses its bytes on
// page-in — poison a prior session quarantined never re-enters this session's context.
func TestCrossSessionTrustGateTravels(t *testing.T) {
	prior := newTStore()
	prior.add("p-sealed", "tool", DurabilityDurable, "secret", true, false)
	prior.add("p-tomb", "tool", DurabilityDurable, "suppressed", false, true)
	cur := newTStore()
	cur.add("c-sealed", "tool", DurabilitySession, "current secret", true, false)

	u := NewCrossSessionStore(cur, prior)
	ctx := context.Background()

	spans, _ := u.Spans(ctx)
	got := spanIDSet(spans)
	if !got["prior0#p-sealed"] || !got["prior0#p-tomb"] {
		t.Fatalf("sealed/tombstoned durable prior spans must be surfaced for audit: %v", spanIDs(spans))
	}

	if _, err := u.Materialize(ctx, "prior0#p-sealed"); !errors.Is(err, ErrSealed) {
		t.Errorf("prior sealed page-in: want ErrSealed, got %v", err)
	}
	if _, err := u.Materialize(ctx, "prior0#p-tomb"); !errors.Is(err, ErrTombstoned) {
		t.Errorf("prior tombstoned page-in: want ErrTombstoned, got %v", err)
	}
	if _, err := u.Materialize(ctx, "current#c-sealed"); !errors.Is(err, ErrSealed) {
		t.Errorf("current sealed page-in: want ErrSealed, got %v", err)
	}
}

// TestCrossSessionIDRoutingNoCollision: two prior images that share a native id ("span:0")
// get distinct unioned ids, and each routes back to the correct sub-store's bytes.
func TestCrossSessionIDRoutingNoCollision(t *testing.T) {
	a := newTStore()
	a.add("span:0", "user", DurabilityDurable, "from A", false, false)
	b := newTStore()
	b.add("span:0", "user", DurabilityDurable, "from B", false, false)
	cur := newTStore()
	cur.add("span:0", "user", DurabilityDurable, "from CUR", false, false)

	// prior most-recent-first: prior[0]=a -> "prior0", prior[1]=b -> "prior1".
	u := NewCrossSessionStore(cur, a, b)
	ctx := context.Background()

	spans, _ := u.Spans(ctx)
	if len(spans) != 3 {
		t.Fatalf("want 3 distinct unioned spans, got %d: %v", len(spans), spanIDs(spans))
	}
	for id, want := range map[string]string{
		"prior0#span:0":  "from A",
		"prior1#span:0":  "from B",
		"current#span:0": "from CUR",
	} {
		body, err := u.Materialize(ctx, id)
		if err != nil {
			t.Fatalf("materialize %s: %v", id, err)
		}
		if string(body) != want {
			t.Errorf("materialize %s = %q, want %q", id, body, want)
		}
	}
}

// TestCrossSessionExpiredNotPageable: a stale recovery handle for a prior turn span cannot
// page it back across the boundary — Materialize refuses with ErrExpired even though the
// sub-store still holds the bytes.
func TestCrossSessionExpiredNotPageable(t *testing.T) {
	prior := newTStore()
	prior.add("p-turn", "assistant", DurabilityTurn, "scratch", false, false)
	cur := newTStore()
	cur.add("c-d", "system", DurabilityDurable, "sys", false, false)

	u := NewCrossSessionStore(cur, prior)
	if _, err := u.Materialize(context.Background(), "prior0#p-turn"); !errors.Is(err, ErrExpired) {
		t.Errorf("force-materialize of expired prior turn span: want ErrExpired, got %v", err)
	}
}

// TestCrossSessionFirstTurnMaterializesFaithfulView is the #566 witness: a fresh session's
// first turn materializes a faithful, budget-bounded view over a multi-session store, with a
// durable prior span SURFACED and a turn-scoped prior span EXPIRED.
func TestCrossSessionFirstTurnMaterializesFaithfulView(t *testing.T) {
	prior := newTStore()
	prior.add("p-turn", "assistant", DurabilityTurn, "old refund fee scratch", false, false)
	prior.add("p-pref", "user", DurabilityDurable, "user prefers the refund fee waived", false, false)
	cur := newTStore()
	cur.add("c-goal", "goal", DurabilitySession, "handle the refund fee", false, false)

	u := NewCrossSessionStore(cur, prior)
	ctx := context.Background()
	f := Forecast{Intents: []string{"refund fee"}, Horizon: 1}

	view, err := Materialize(ctx, u, f, Budget{Tokens: 100}, nil)
	if err != nil {
		t.Fatal(err)
	}

	rendered := map[string]bool{}
	for _, r := range view.Rendered {
		rendered[r.ID] = true
	}
	if !rendered["prior0#p-pref"] {
		t.Errorf("durable prior span not surfaced in first-turn view; rendered=%v", view.Rendered)
	}
	if rendered["prior0#p-turn"] {
		t.Errorf("expired prior turn span leaked into the first-turn view")
	}

	spans, _ := u.Spans(ctx)
	if spanIDSet(spans)["prior0#p-turn"] {
		t.Errorf("expired prior turn span present as a candidate")
	}
	if view.RenderedTokens() > 100 {
		t.Errorf("resident view %d tokens exceeds budget 100", view.RenderedTokens())
	}
	if !view.Witness.Faithful {
		t.Errorf("cross-session view not faithful (a candidate was destroyed): %+v", view.Witness)
	}
}

// TestCrossSessionProvenance checks the readable witness: per-source Total/Survived/Expired/
// Sealed counts, so "durable surfaced, turn-scoped expired" is auditable without paging bytes.
func TestCrossSessionProvenance(t *testing.T) {
	prior := newTStore()
	prior.add("p1", "a", DurabilityTurn, "x", false, false)
	prior.add("p2", "a", DurabilitySession, "x", false, false)
	prior.add("p3", "a", DurabilityBounded, "x", false, false)
	prior.add("p4", "a", DurabilityDurable, "x", false, false)
	prior.add("p5", "a", DurabilityDurable, "x", true, false) // sealed durable: survives, gate refuses
	cur := newTStore()
	cur.add("c1", "a", DurabilityTurn, "x", false, false)

	u := NewCrossSessionStore(cur, prior)
	prov, err := u.Provenance(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]SourceStat{}
	for _, s := range prov {
		byName[s.Name] = s
	}
	if p := byName["prior0"]; p.Total != 5 || p.Expired != 2 || p.Survived != 3 || p.Sealed != 1 {
		t.Errorf("prior provenance = %+v, want Total=5 Expired=2 Survived=3 Sealed=1", p)
	}
	if c := byName["current"]; c.Total != 1 || c.Expired != 0 || c.Survived != 1 {
		t.Errorf("current provenance = %+v, want Total=1 Expired=0 Survived=1", c)
	}
}

// TestNewUnionRejectsBadNames guards the id-routing contract: names must be unique, non-empty,
// and free of the separator, and a source's Store must be non-nil.
func TestNewUnionRejectsBadNames(t *testing.T) {
	ok := newTStore()
	cases := []struct {
		name    string
		sources []Source
	}{
		{"empty name", []Source{{Name: "", Store: ok, Scope: ScopeCurrent}}},
		{"separator in name", []Source{{Name: "a#b", Store: ok, Scope: ScopeCurrent}}},
		{"duplicate name", []Source{{Name: "x", Store: ok, Scope: ScopePrior}, {Name: "x", Store: ok, Scope: ScopeCurrent}}},
		{"nil store", []Source{{Name: "x", Store: nil, Scope: ScopeCurrent}}},
	}
	for _, tc := range cases {
		if _, err := NewUnion(tc.sources...); err == nil {
			t.Errorf("%s: NewUnion should have errored", tc.name)
		}
	}
}
