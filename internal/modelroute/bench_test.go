package modelroute

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// ---------------------------------------------------------------------------
// Run — the three-axis aggregate over a manifest, determinism, honesty fences.
// ---------------------------------------------------------------------------

func demoCompare(frontier string) Comparison {
	return DemoCorpus().Compare(DefaultManifest(), SingleModelManifest("frontier"), nil, nil, frontier)
}

// The built-in demo corpus is an HONEST trade, not a rigged win: per-aspect
// routing is cheaper/faster (easy aspects hit the small/mid tier), the two-model
// vote ensemble is a deliberate PREMIUM that rescues one case, and a downgrade
// loses one case the single model got right — so quality ties. These are the
// exact, hand-computed aggregates the CLI prints and the witness pins.
func TestBenchDemoCorpusAggregates(t *testing.T) {
	cmp := demoCompare("frontier")

	// COST (rough $/Mtok-out summed over members). Routed: small/default/mid on
	// the easy aspects, the frontier tier on the hard step, and TWO unpriced
	// guards (charged at the frontier rate) on each write ensemble.
	if !approx(cmp.Routed.Cost, 96.25) {
		t.Fatalf("routed cost = %v, want 96.25", cmp.Routed.Cost)
	}
	if !approx(cmp.Single.Cost, 120) { // 8 cases x frontier out=15
		t.Fatalf("single cost = %v, want 120", cmp.Single.Cost)
	}

	// LATENCY (rough per-member ms summed). Same shape: cheap tiers on easy work,
	// frontier on the hard step, two frontier-latency guards per ensemble.
	if !approx(cmp.Routed.Latency, 860) {
		t.Fatalf("routed latency = %v, want 860", cmp.Routed.Latency)
	}
	if !approx(cmp.Single.Latency, 960) { // 8 cases x frontier 120
		t.Fatalf("single latency = %v, want 960", cmp.Single.Latency)
	}

	// QUALITY — a tie: the ensemble rescues C5 (single model said "deny"), the
	// downgrade loses C2 (default said "41"), 7/8 on both arms.
	if !approx(cmp.Routed.Quality, 0.875) || cmp.Routed.Hits != 7 || cmp.Routed.Misses != 1 {
		t.Fatalf("routed quality = %v hits=%d misses=%d, want 0.875 / 7 / 1", cmp.Routed.Quality, cmp.Routed.Hits, cmp.Routed.Misses)
	}
	if !approx(cmp.Single.Quality, 0.875) || cmp.Single.Misses != 1 {
		t.Fatalf("single quality = %v misses=%d, want 0.875 / 1", cmp.Single.Quality, cmp.Single.Misses)
	}

	// The write-shaped tool calls are the two ensemble cases.
	if cmp.Routed.Ensembles != 2 || cmp.Single.Ensembles != 0 {
		t.Fatalf("ensembles routed=%d single=%d, want 2 / 0", cmp.Routed.Ensembles, cmp.Single.Ensembles)
	}
	if cmp.Routed.Cases != 8 || cmp.Single.Cases != 8 || cmp.Cases != 8 {
		t.Fatalf("case count wrong: routed=%d single=%d cmp=%d", cmp.Routed.Cases, cmp.Single.Cases, cmp.Cases)
	}
}

// The deltas express the trade on each axis: routed is ~20% cheaper and ~10%
// faster on total compute, quality tied (one rescue offsets one downgrade).
func TestBenchDemoCorpusDeltas(t *testing.T) {
	cmp := demoCompare("frontier")
	if c := cmp.CostSavingFrac(); !(c > 0.19 && c < 0.20) {
		t.Fatalf("cost saving frac = %v, want ~0.198 (cheaper)", c)
	}
	if l := cmp.LatencySavingFrac(); !(l > 0.10 && l < 0.11) {
		t.Fatalf("latency saving frac = %v, want ~0.104 (faster)", l)
	}
	if q := cmp.QualityDelta(); !approx(q, 0) {
		t.Fatalf("quality delta = %v, want 0 (tie)", q)
	}
}

// An unpriced ensemble member is charged at the conservative frontier rate in
// BOTH lenses and DISCLOSED in Assumed — the benchmark never invents a cheap
// number, exactly like the cost lens it reuses.
func TestBenchUnpricedEnsembleDisclosed(t *testing.T) {
	cmp := demoCompare("frontier")
	if !reflect.DeepEqual(cmp.Routed.Assumed, []string{"guard-a", "guard-b"}) {
		t.Fatalf("routed assumed = %v, want [guard-a guard-b]", cmp.Routed.Assumed)
	}
	if len(cmp.Single.Assumed) != 0 {
		t.Fatalf("single arm should have no assumed models, got %v", cmp.Single.Assumed)
	}
}

// Run is deterministic: same corpus, manifest, books -> identical Metrics. This
// is the property that makes the benchmark a fair, repeatable harness.
func TestBenchDeterministic(t *testing.T) {
	corpus, routed, single := DemoCorpus(), DefaultManifest(), SingleModelManifest("frontier")
	first := corpus.Compare(routed, single, nil, nil, "frontier")
	for i := 0; i < 30; i++ {
		got := corpus.Compare(routed, single, nil, nil, "frontier")
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("Compare not deterministic at %d:\n first=%+v\n got  =%+v", i, first, got)
		}
	}
}

// ---------------------------------------------------------------------------
// Quality — an ensemble can RESCUE a case a single model gets wrong (the thesis
// the benchmark exists to make measurable), at an honest cost/latency premium.
// ---------------------------------------------------------------------------

func TestBenchEnsembleRescueWinsQualityAtPremium(t *testing.T) {
	// One case: a write call. The single frontier model is WRONG ("deny"); two
	// guards agree on the right "approve", so the vote ensemble folds to it.
	corpus := Corpus{{
		Subject:  Subject{Aspect: AspectToolCall, Tool: "write_file"},
		Outputs:  map[string]string{"frontier": "deny", "guard-a": "approve", "guard-b": "approve"},
		Expected: "approve",
	}}
	cmp := corpus.Compare(DefaultManifest(), SingleModelManifest("frontier"), nil, nil, "frontier")

	// Quality: the ensemble rescues -> routed 1.0, single 0.0.
	if !approx(cmp.Routed.Quality, 1.0) || !approx(cmp.Single.Quality, 0.0) {
		t.Fatalf("rescue quality routed=%v single=%v, want 1.0 / 0.0", cmp.Routed.Quality, cmp.Single.Quality)
	}
	if q := cmp.QualityDelta(); !(q > 0.99) {
		t.Fatalf("quality delta = %v, want ~+1.0", q)
	}
	// But the ensemble is a premium: two frontier-rate members cost/latency MORE
	// than one. The benchmark reports this honestly, not as a saving.
	if cmp.CostSavingFrac() >= 0 {
		t.Fatalf("a 2-member ensemble should be a cost premium (<0 saving), got %v", cmp.CostSavingFrac())
	}
	if cmp.LatencySavingFrac() >= 0 {
		t.Fatalf("a 2-member ensemble should be a latency premium (<0 saving), got %v", cmp.LatencySavingFrac())
	}
}

// A downgrade can LOSE quality the single model had — the benchmark does not hide
// that routing is sometimes worse on accuracy.
func TestBenchDowngradeLosesQuality(t *testing.T) {
	corpus := Corpus{{
		Subject:  Subject{Aspect: AspectRequest, Latency: LatencyInteractive, PromptTokens: 8192}, // -> default
		Outputs:  map[string]string{"frontier": "right", "default": "wrong"},
		Expected: "right",
	}}
	cmp := corpus.Compare(DefaultManifest(), SingleModelManifest("frontier"), nil, nil, "frontier")
	if !approx(cmp.Routed.Quality, 0.0) || !approx(cmp.Single.Quality, 1.0) {
		t.Fatalf("downgrade quality routed=%v single=%v, want 0.0 / 1.0", cmp.Routed.Quality, cmp.Single.Quality)
	}
	if q := cmp.QualityDelta(); !approx(q, -1.0) {
		t.Fatalf("quality delta = %v, want -1.0", q)
	}
}

// A single PICK over a best_of corpus: the judge-scored member wins. This routes
// through Combine's best_of arm, proving the benchmark folds every reduction.
func TestBenchBestOfReduction(t *testing.T) {
	// A manifest whose only plan is a best_of ensemble of two unpriced drafters.
	routed := Manifest{
		Default: Plan{
			Members: []Member{{Model: "drafter-a"}, {Model: "drafter-b"}},
			Reduce:  ReduceBestOf,
		},
	}
	corpus := Corpus{{
		Subject:  Subject{Aspect: AspectRequest},
		Outputs:  map[string]string{"drafter-a": "good", "drafter-b": "best"},
		Scores:   map[string]float64{"drafter-a": 0.4, "drafter-b": 0.9},
		Expected: "best",
	}}
	cmp := corpus.Compare(routed, SingleModelManifest("frontier"), nil, nil, "frontier")
	// best_of picks drafter-b ("best", score 0.9) -> routed correct; single frontier
	// has no recorded output here so it misses (empty != "best").
	if !approx(cmp.Routed.Quality, 1.0) {
		t.Fatalf("best_of routed quality = %v, want 1.0", cmp.Routed.Quality)
	}
}

// ---------------------------------------------------------------------------
// SingleModelManifest — the SOTA baseline shape: one model for every aspect.
// ---------------------------------------------------------------------------

func TestSingleModelManifestRoutesEverythingToOneModel(t *testing.T) {
	m := SingleModelManifest("frontier")
	if err := m.Validate(); err != nil {
		t.Fatalf("single manifest invalid: %v", err)
	}
	if len(m.Rules) != 0 || m.Default.Primary() != "frontier" {
		t.Fatalf("single manifest should have no rules and a frontier default, got rules=%d primary=%q", len(m.Rules), m.Default.Primary())
	}
	// Every aspect of the demo corpus hits the one-model default.
	for _, cs := range DemoCorpus() {
		d := m.Route(cs.Subject)
		if d.Matched || d.Plan.Primary() != "frontier" || d.Plan.IsEnsemble() {
			t.Fatalf("single arm routed %v to %q (matched=%v)", cs.Subject.Aspect, d.Plan.Primary(), d.Matched)
		}
	}
}

// ---------------------------------------------------------------------------
// Corpus round-trip + validation (mirrors the manifest round-trip discipline).
// ---------------------------------------------------------------------------

func TestCorpusRoundTripExact(t *testing.T) {
	c := DemoCorpus()
	re, err := ParseCorpus(c.JSON())
	if err != nil {
		t.Fatalf("re-parse demo corpus: %v", err)
	}
	// JSON() does not carry struct-only Note? It does (Note has a json tag). But
	// DemoCorpus Cases are value-equal after a marshal/parse round trip.
	if !reflect.DeepEqual(re, c) {
		t.Fatalf("corpus round-trip not exact:\n want %+v\n got  %+v", c, re)
	}
}

func TestCorpusParseRejectsBadInput(t *testing.T) {
	for name, bad := range map[string][]byte{
		"empty array":    []byte(`[]`),
		"unknown field":  []byte(`[{"subject":{"aspect":"request"},"outputs":{"a":"x"},"expected":"x","bogus":1}]`),
		"empty expected": []byte(`[{"subject":{"aspect":"request"},"outputs":{"a":"x"},"expected":""}]`),
		"no outputs":     []byte(`[{"subject":{"aspect":"request"},"outputs":{},"expected":"x"}]`),
		"malformed":      []byte(`[{`),
	} {
		if _, err := ParseCorpus(bad); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

// A corpus loaded off disk deserializes a Subject's snake_case fields (the json
// tags added for this purpose), so an operator can author a corpus as JSON.
func TestCorpusSubjectSnakeCase(t *testing.T) {
	raw := []byte(`[{"subject":{"aspect":"tool_call","tool":"search_kb","prompt_tokens":512,"latency":"interactive","complexity":"low","labels":{"domain":"legal"}},"outputs":{"small":"ok"},"expected":"ok"}]`)
	c, err := ParseCorpus(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	s := c[0].Subject
	if s.Aspect != AspectToolCall || s.Tool != "search_kb" || s.PromptTokens != 512 ||
		s.Latency != LatencyInteractive || s.Complexity != ComplexityLow || s.Labels["domain"] != "legal" {
		t.Fatalf("subject snake_case not deserialized: %+v", s)
	}
}

// ---------------------------------------------------------------------------
// LatencyBook — overlay + parse (mirrors the PriceBook tests).
// ---------------------------------------------------------------------------

func TestLatencyBookOverlayAndParse(t *testing.T) {
	base := DefaultLatencies()
	over, err := ParseLatencies("mystery-7b=42, large=999")
	if err != nil {
		t.Fatalf("ParseLatencies: %v", err)
	}
	book := base.Overlay(over)
	if book["mystery-7b"] != 42 || book["large"] != 999 || book["small"] != 20 {
		t.Fatalf("overlay wrong: %v", book)
	}
	for _, bad := range []string{"small", "=5", "small=x"} {
		if _, err := ParseLatencies(bad); err == nil {
			t.Fatalf("ParseLatencies(%q) should error", bad)
		}
	}
	if pb, err := ParseLatencies("a=1, b=2"); err != nil || len(pb) != 2 {
		t.Fatalf("good spec failed: pb=%v err=%v", pb, err)
	}
}

// An unpriced member is charged at the frontier model's latency (or the anchor)
// — never a silent zero — mirroring the cost lens's conservative assumption.
func TestMemberLatencyUnpricedIsFrontier(t *testing.T) {
	lat := DefaultLatencies()
	if ms, priced := memberLatency("small", "frontier", lat); !priced || ms != 20 {
		t.Fatalf("priced small: ms=%v priced=%v", ms, priced)
	}
	if ms, priced := memberLatency("mystery", "frontier", lat); priced || ms != 120 {
		t.Fatalf("unpriced should fall back to frontier 120 unpriced: ms=%v priced=%v", ms, priced)
	}
	// No frontier in book either -> the anchor.
	if ms, priced := memberLatency("mystery", "absent", lat); priced || ms != FrontierLatencyAnchor {
		t.Fatalf("no frontier -> anchor: ms=%v priced=%v", ms, priced)
	}
}

// ---------------------------------------------------------------------------
// Committed fixtures (examples/routing-bench/) — canonical-form + the documented
// aggregates. The routing-bench analogue of TestRoutingPresetsRoundTrip: the
// corpus and the two manifests round-trip byte-exact (a hand-edit that drifts
// from canonical form fails the build), and running them reproduces the demo's
// documented three-axis numbers.
// ---------------------------------------------------------------------------

func TestRoutingBenchFixturesCanonicalAndReproducible(t *testing.T) {
	dir := filepath.Join("..", "..", "examples", "routing-bench")
	for _, name := range []string{"demo-corpus.json", "routed.json", "single-model.json"} {
		name := name
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, name)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			var canon []byte
			switch name {
			case "demo-corpus.json":
				c, err := ParseCorpus(raw)
				if err != nil {
					t.Fatalf("%s fails ParseCorpus: %v", path, err)
				}
				canon = c.JSON()
			default: // a fak-route/v1 manifest
				m, err := ParseManifest(raw)
				if err != nil {
					t.Fatalf("%s fails ParseManifest: %v", path, err)
				}
				canon = m.JSON()
			}
			if string(canon) != string(raw) {
				t.Fatalf("%s is not canonical (round-trip not exact):\n--- canonical ---\n%s\n--- file ---\n%s",
					path, string(canon), string(raw))
			}
		})
	}

	// Running the committed routed + single manifests over the committed corpus
	// reproduces the demo's documented three-axis aggregates.
	corpus, err := LoadCorpus(filepath.Join(dir, "demo-corpus.json"))
	if err != nil {
		t.Fatalf("load corpus: %v", err)
	}
	routed, err := LoadManifest(filepath.Join(dir, "routed.json"))
	if err != nil {
		t.Fatalf("load routed: %v", err)
	}
	single, err := LoadManifest(filepath.Join(dir, "single-model.json"))
	if err != nil {
		t.Fatalf("load single: %v", err)
	}
	cmp := corpus.Compare(routed, single, nil, nil, "frontier")
	if !approx(cmp.Routed.Cost, 96.25) || !approx(cmp.Single.Cost, 120) {
		t.Fatalf("fixture cost routed=%v single=%v, want 96.25 / 120", cmp.Routed.Cost, cmp.Single.Cost)
	}
	if !approx(cmp.QualityDelta(), 0) {
		t.Fatalf("fixture quality delta = %v, want 0", cmp.QualityDelta())
	}
}
