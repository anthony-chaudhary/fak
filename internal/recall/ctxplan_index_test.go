package recall

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// persistImageDir records the airline session, persists its core image (manifest.json +
// cas.json) under a known temp dir, and returns the dir plus the reloaded Session — so a test
// can write a sibling index.json next to a REAL core image, exactly as a resumed session does.
func persistImageDir(t *testing.T) (string, *Session) {
	t.Helper()
	dir := t.TempDir()
	if err := recordAirline(t).Persist(dir); err != nil {
		t.Fatalf("persist core image: %v", err)
	}
	s, err := Load(dir)
	if err != nil {
		t.Fatalf("load core image: %v", err)
	}
	return dir, s
}

// indexProbeForecasts are the forecasts the re-attach==rebuild witnesses compare over: a pure
// pin (the benign pages), a relevance query, and a pin naming a SEALED page (which must be
// probed yet never selected).
func indexProbeForecasts() []ctxplan.Forecast {
	return []ctxplan.Forecast{
		{Pins: []string{"page:0", "page:2"}},
		{Intents: []string{"refund fee tier"}},
		{Intents: []string{"flights cheapest direct"}, Pins: []string{"page:0"}},
		{Pins: []string{"page:0", "page:1", "page:2", "page:3"}}, // includes the sealed pages
	}
}

// TestPersistIndexReattachEqualsRebuild is THE half-a witness (issue #558): an index persisted
// alongside the recall core image and re-attached via LoadIndex makes the EXACT SAME per-turn
// decisions as one rebuilt from the page table with AttachIndex — Probe is byte-identical and
// the index-bounded plan selects the same spans, across several forecasts. So persisting the
// index buys back the per-resume O(N) rebuild without ever changing what the planner does.
func TestPersistIndexReattachEqualsRebuild(t *testing.T) {
	ctx := context.Background()
	dir, s := persistImageDir(t)

	rebuilt, err := AttachIndex(ctx, s)
	if err != nil {
		t.Fatalf("AttachIndex: %v", err)
	}
	if err := PersistIndex(dir, rebuilt); err != nil {
		t.Fatalf("PersistIndex: %v", err)
	}
	reattached, err := LoadIndex(dir)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	for _, f := range indexProbeForecasts() {
		if !reflect.DeepEqual(reattached.Probe(f, ctxplan.ProbeOptions{}), rebuilt.Probe(f, ctxplan.ProbeOptions{})) {
			t.Errorf("re-attached probe != rebuild for forecast %+v", f)
		}
		pa := reattached.PlanCells(f, ctxplan.Budget{Tokens: 1000}, nil, ctxplan.ProbeOptions{})
		pb := rebuilt.PlanCells(f, ctxplan.Budget{Tokens: 1000}, nil, ctxplan.ProbeOptions{})
		if !reflect.DeepEqual(planSelectedIDs(pa), planSelectedIDs(pb)) {
			t.Errorf("re-attached plan != rebuild plan for forecast %+v:\n got %v\nwant %v",
				f, planSelectedIDs(pa), planSelectedIDs(pb))
		}
	}
}

// TestPersistIndexIsSiblingOfCoreImage proves the index lands in the SAME directory as the
// core image — index.json next to manifest.json and cas.json — so the three files travel
// together as one durable session artifact.
func TestPersistIndexIsSiblingOfCoreImage(t *testing.T) {
	ctx := context.Background()
	dir, s := persistImageDir(t)

	ix, err := AttachIndex(ctx, s)
	if err != nil {
		t.Fatalf("AttachIndex: %v", err)
	}
	if err := PersistIndex(dir, ix); err != nil {
		t.Fatalf("PersistIndex: %v", err)
	}
	for _, f := range []string{"manifest.json", "cas.json", IndexFile} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("expected %s alongside the core image: %v", f, err)
		}
	}
}

// TestPersistedIndexKeepsTrustInvariant is the safety witness: the persisted index image
// carries SAFE metadata only — a sealed page persists with its Sealed flag and its
// sealed-safe descriptor, so the re-attached index never selects it (it is elided sealed),
// and the raw poison/secret bytes never appear in index.json.
func TestPersistedIndexKeepsTrustInvariant(t *testing.T) {
	ctx := context.Background()
	dir, s := persistImageDir(t)

	ix, err := AttachIndex(ctx, s)
	if err != nil {
		t.Fatalf("AttachIndex: %v", err)
	}
	if err := PersistIndex(dir, ix); err != nil {
		t.Fatalf("PersistIndex: %v", err)
	}

	// The persisted bytes must never contain the quarantined content.
	raw, err := os.ReadFile(filepath.Join(dir, IndexFile))
	if err != nil {
		t.Fatalf("read index.json: %v", err)
	}
	for _, leak := range []string{"ignore previous instructions", "sk-abcdef", "AKIAIOSFODNN7EXAMPLE"} {
		if strings.Contains(string(raw), leak) {
			t.Fatalf("the persisted index image leaked quarantined content: %q", leak)
		}
	}

	// The re-attached index probes the sealed pages (pinned) yet never selects them.
	reattached, err := LoadIndex(dir)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	f := ctxplan.Forecast{Pins: []string{"page:1", "page:3"}}
	p := reattached.PlanCells(f, ctxplan.Budget{Tokens: 1000}, nil, ctxplan.ProbeOptions{})
	sel := planSelectedIDs(p)
	if sel["page:1"] || sel["page:3"] {
		t.Fatalf("INVARIANT VIOLATED: a sealed page entered a re-attached index's resident view: %v", sel)
	}
	sealedElided := map[string]bool{}
	for _, e := range p.Elided {
		if e.Reason == ctxplan.ElideSealed {
			sealedElided[e.ID] = true
		}
	}
	if !sealedElided["page:1"] || !sealedElided["page:3"] {
		t.Errorf("sealed pages must be elided sealed; elided=%+v", p.Elided)
	}
}

// TestLoadIndexAbsentIsNotExist proves the fallback signal: LoadIndex on a core-image dir with
// no index.json reports os.ErrNotExist, so a caller (LoadOrAttachIndex) knows to rebuild from
// the manifest rather than failing the resume.
func TestLoadIndexAbsentIsNotExist(t *testing.T) {
	dir, _ := persistImageDir(t)
	if _, err := LoadIndex(dir); err == nil {
		t.Fatal("LoadIndex must error when no index.json was persisted")
	} else if !os.IsNotExist(err) {
		t.Errorf("an absent index must report os.ErrNotExist, got %v", err)
	}
}

// TestLoadOrAttachIndexFallsBackThenReattaches witnesses the resume convenience end to end:
// with no index.json it REBUILDS from the manifest (and the rebuilt index plans correctly);
// after a PersistIndex it RE-ATTACHES the persisted one — and both make identical decisions.
func TestLoadOrAttachIndexFallsBackThenReattaches(t *testing.T) {
	ctx := context.Background()
	dir, s := persistImageDir(t)
	f := ctxplan.Forecast{Pins: []string{"page:0", "page:2"}, Intents: []string{"refund fee"}}

	// First resume: no index.json yet -> rebuild from the manifest.
	built, err := LoadOrAttachIndex(ctx, dir, s)
	if err != nil {
		t.Fatalf("LoadOrAttachIndex (rebuild path): %v", err)
	}
	want := planSelectedIDs(built.PlanCells(f, ctxplan.Budget{Tokens: 1000}, nil, ctxplan.ProbeOptions{}))
	if !want["page:0"] || !want["page:2"] {
		t.Fatalf("rebuilt index must select the pinned benign pages, got %v", want)
	}

	// Persist, then resume again: now it re-attaches the persisted index, identical decisions.
	if err := PersistIndex(dir, built); err != nil {
		t.Fatalf("PersistIndex: %v", err)
	}
	reattached, err := LoadOrAttachIndex(ctx, dir, s)
	if err != nil {
		t.Fatalf("LoadOrAttachIndex (re-attach path): %v", err)
	}
	got := planSelectedIDs(reattached.PlanCells(f, ctxplan.Budget{Tokens: 1000}, nil, ctxplan.ProbeOptions{}))
	if !reflect.DeepEqual(got, want) {
		t.Errorf("re-attach plan != rebuild plan: got %v want %v", got, want)
	}
}

// planSelectedIDs is the id set of a plan's resident selection — a local helper for the
// re-attach==rebuild comparisons (the ctxplan package's selectedIDs is unexported to it).
func planSelectedIDs(p ctxplan.Plan) map[string]bool {
	out := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		out[s.ID] = true
	}
	return out
}
