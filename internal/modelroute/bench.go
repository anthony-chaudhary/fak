package modelroute

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// THE OFFLINE ROUTING BENCHMARK — per-aspect + ensemble vs single-model, across
// cost / quality / latency, with NO model in the loop.
// ---------------------------------------------------------------------------
//
// THE GAP IT FILLS. docs/model-routing.md states the thesis as a CATEGORICAL
// capability gap (route at any aspect, first-class ensembles) — and is explicit
// that any "10x" is "a target to be measured, never an inferred or borrowed
// number". This file is the measuring instrument: a deterministic, offline
// benchmark that runs a CORPUS of cases through TWO manifests — a routed policy
// (per-aspect + ensemble) and a single-model baseline (the SOTA shape: one
// frontier model for everything) — and reports the delta on each of three axes.
//
// OFFLINE means OFFLINE. Every case carries the STAND-IN OUTPUT each candidate
// model produces for it (a recorded answer, never a live model call), exactly as
// `fak route --simulate` already does. So the benchmark reuses the two pure,
// already-witnessed halves of this package — Route (the decision) and Combine
// (the fold) — over fixed votes: deterministic end to end, no key, no GPU, no
// network. It measures what the POLICY does to a recorded workload, not what a
// non-bit-exact engine would do live (that is the [STUB] dispatch half).
//
// THE THREE AXES, each an honest rough lens (never a bill, never a measured SLA):
//   - COST    — reuses EstimateSavings: the routed plan's per-member $/Mtok-out,
//     summed over the corpus. A single-model baseline pays the frontier rate on
//     every case; per-aspect routing pays it only on the hard aspects, and an
//     ensemble pays it on every member (a premium). Unpriced members are charged
//     at the conservative frontier rate and DISCLOSED — fak never invents a cheap
//     number.
//   - QUALITY — the fraction of cases whose folded output equals the case's
//     Expected ground truth. This is where an ensemble can WIN (a vote that folds
//     to the right answer where a single model errs) and where a downgrade can
//     LOSE (a cheap model wrong where the frontier was right). The corpus is the
//     stand-in workload; the number is the policy's accuracy on THAT workload,
//     not a generalization claim.
//   - LATENCY — a rough per-call latency summed over members (the latency analogue
//     of the cost lens). An ensemble does N members' work, so its compute latency
//     is the SUM (consistent with cost summing member cost); a parallel
//     dispatch's wall-clock is bounded by the MAX, which this lens deliberately
//     does not assume — it reports total compute, the honest "more work" figure.
//     Rough and overridable, like the price book.
//
// WHAT THE BENCHMARK IS NOT: it is not a live measurement, not a claim that
// routing beats a frontier model on real traffic, and not a substitute for the
// [STUB] dispatch. It is the deterministic harness that turns a recorded corpus
// into a three-axis policy comparison, so an operator can SEE the trade a routing
// manifest makes before wiring it live.

// LatencyBook maps a model id to its rough per-call latency in ms — the offline
// latency lens, mirroring PriceBook for the cost lens. Every figure is a rough
// order-of-magnitude ESTIMATE for the benchmark, never a measured wall-clock.
type LatencyBook map[string]float64

// FrontierLatencyAnchor is the rough per-call latency of the SOTA baseline model
// (one Opus-class frontier call), mirroring FrontierAnchor for cost. Anchored to
// the same tier the cost lens uses so the two lenses describe the same baseline.
const FrontierLatencyAnchor = 120.0

// DefaultLatencies is the built-in rough latency ladder, keyed by the same tier
// names DefaultPrices uses so the two lenses share a vocabulary. Smaller tiers
// are faster; the frontier tier matches FrontierLatencyAnchor. Override any of
// it with `fak routebench --latencies model=ms,...`.
func DefaultLatencies() LatencyBook {
	return LatencyBook{
		"frontier": 120, "large": 120, // Opus-class — the SOTA baseline tier
		"mid": 60, "medium": 60, "default": 60, // balanced mid tier
		"small": 20, "tiny": 20, "mini": 20, "nano": 20, // small/fast tier
		"local": 2, "in-kernel": 2, "on-device": 2, "kernel": 2, // no marginal compute
	}
}

// Overlay returns a copy of book with every entry of over applied on top.
func (book LatencyBook) Overlay(over LatencyBook) LatencyBook {
	merged := make(LatencyBook, len(book)+len(over))
	for k, v := range book {
		merged[k] = v
	}
	for k, v := range over {
		merged[k] = v
	}
	return merged
}

// ParseLatencies reads a --latencies spec into a LatencyBook overlay:
// comma-separated "model=ms" pairs (e.g. "small=20,large=120"). Fails loud on a
// malformed pair, mirroring ParsePrices and the manifest's DisallowUnknownFields.
// The caller layers the result on top of DefaultLatencies.
func ParseLatencies(spec string) (LatencyBook, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return LatencyBook{}, nil
	}
	out := LatencyBook{}
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 || strings.TrimSpace(kv[0]) == "" {
			return nil, fmt.Errorf("modelroute: bad --latencies pair %q (want model=ms)", pair)
		}
		var ms float64
		if _, err := fmt.Sscanf(strings.TrimSpace(kv[1]), "%g", &ms); err != nil {
			return nil, fmt.Errorf("modelroute: --latencies %q: %w", pair, err)
		}
		out[strings.TrimSpace(kv[0])] = ms
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// THE CORPUS — a set of offline cases (Subject + per-model stand-in outputs).
// ---------------------------------------------------------------------------

// Case is one offline benchmark item: the Subject to route, the STAND-IN OUTPUT
// (and optional best_of SCORE) each candidate model produces for it, and the
// Expected ground-truth answer quality is scored against. Outputs are corpus
// fixtures — recorded answers, never live model calls — so the benchmark is
// deterministic end to end.
//
// A model the routed plan selects that has NO entry in Outputs contributes an
// empty output to the fold (and is very likely a quality miss); a single-model
// baseline only ever reads its one member's output. Scores feed ReduceBestOf.
type Case struct {
	Subject  Subject            `json:"subject"`
	Outputs  map[string]string  `json:"outputs"`
	Scores   map[string]float64 `json:"scores,omitempty"`
	Expected string             `json:"expected"`
	Note     string             `json:"note,omitempty"`
}

// Corpus is the offline case set a benchmark runs over.
type Corpus []Case

// ParseCorpus decodes and sanity-checks a corpus. Unknown JSON fields are
// rejected (DisallowUnknownFields) so a typo fails loudly. Every case must carry
// an Expected (the thing quality is scored against); empty Outputs is allowed
// (every case would then miss) but flagged, since it usually means a fixture bug.
func ParseCorpus(b []byte) (Corpus, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var c Corpus
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("modelroute: parse corpus: %w", err)
	}
	if len(c) == 0 {
		return nil, fmt.Errorf("modelroute: corpus has no cases")
	}
	for i, cs := range c {
		if cs.Expected == "" {
			return nil, fmt.Errorf("modelroute: corpus case %d has an empty expected (no ground truth to score)", i)
		}
		if len(cs.Outputs) == 0 {
			return nil, fmt.Errorf("modelroute: corpus case %d has no stand-in outputs", i)
		}
	}
	return c, nil
}

// LoadCorpus reads and parses a corpus from a file path.
func LoadCorpus(path string) (Corpus, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("modelroute: read corpus %s: %w", path, err)
	}
	return ParseCorpus(b)
}

// JSON renders the Corpus as canonical indented JSON (terminated by a newline).
func (c Corpus) JSON() []byte {
	b, _ := json.MarshalIndent(c, "", "  ")
	return append(b, '\n')
}

// ---------------------------------------------------------------------------
// THE METRICS — per-arm aggregates on the three axes.
// ---------------------------------------------------------------------------

// Metrics is one arm's aggregate over a corpus: cost (sum of routed $/Mtok-out
// over members), latency (sum of rough per-member ms), and quality (the fraction
// of cases whose folded output matched Expected). Ensembles counts the cases the
// arm ran as a >1-member plan; Assumed lists models unpriced in EITHER lens
// (charged at the conservative frontier rate and disclosed).
type Metrics struct {
	Name      string   `json:"name"`
	Cases     int      `json:"cases"`
	Ensembles int      `json:"ensembles"`
	Cost      float64  `json:"cost_per_mtok_out"` // rough $/Mtok-out summed over members
	Latency   float64  `json:"latency_ms"`        // rough per-member ms summed
	Quality   float64  `json:"quality"`           // fraction matching Expected [0,1]
	Hits      int      `json:"hits"`
	Misses    int      `json:"misses"`
	Assumed   []string `json:"assumed,omitempty"`
}

// Run executes one Manifest over the Corpus and aggregates the three axes. Pure
// and deterministic: same corpus, manifest, and books -> identical Metrics. For
// each case it Routes, builds the member votes from the case's stand-in Outputs
// (member order preserved — the deterministic-reduce contract), Combines, and
// scores the folded output against Expected. Cost reuses EstimateSavings (the
// documented cost lens); latency sums per-member rough ms (unpriced -> frontier).
func (c Corpus) Run(name string, m Manifest, book PriceBook, lat LatencyBook, frontier string) Metrics {
	met := Metrics{Name: name, Cases: len(c)}
	if book == nil {
		book = DefaultPrices()
	}
	if lat == nil {
		lat = DefaultLatencies()
	}
	assumed := map[string]bool{}
	for _, cs := range c {
		d := m.Route(cs.Subject)
		plan := d.Plan
		if plan.IsEnsemble() {
			met.Ensembles++
		}

		// COST — the documented cost lens (unpriced -> frontier, disclosed).
		sav := EstimateSavings(d, book, frontier)
		met.Cost += sav.RoutedOut
		for _, a := range sav.Assumed {
			assumed[a] = true
		}

		// LATENCY — rough per-member ms (unpriced -> frontier, disclosed).
		for _, mem := range plan.Members {
			ms, priced := memberLatency(mem.Model, frontier, lat)
			met.Latency += ms
			if !priced {
				assumed[mem.Model] = true
			}
		}

		// QUALITY — fold the stand-in member outputs and score vs Expected.
		reduce := plan.Reduce
		if !plan.IsEnsemble() {
			reduce = ReduceFirst // a single pick has nothing to fold
		}
		res, err := Combine(reduce, votesFor(plan, cs))
		if err != nil || res.Output != cs.Expected {
			met.Misses++
			continue
		}
		met.Hits++
	}
	met.Quality = float64(met.Hits) / float64(met.Cases)
	for a := range assumed {
		met.Assumed = append(met.Assumed, a)
	}
	sort.Strings(met.Assumed)
	return met
}

// memberLatency returns a model's rough per-call latency. A model with no entry
// is charged at the frontier model's latency (or the anchor if that too is
// absent) — the conservative assumption, mirroring the cost lens. priced reports
// whether the model had its own entry (false -> assumed).
func memberLatency(model, frontier string, lat LatencyBook) (ms float64, priced bool) {
	if v, ok := lat[model]; ok {
		return v, true
	}
	if frontier != "" {
		if v, ok := lat[frontier]; ok {
			return v, false
		}
	}
	return FrontierLatencyAnchor, false
}

// votesFor builds the Combine input for a plan over a case: one Vote per member
// (in member order), drawing Output/Score from the case's stand-in maps. A model
// with no recorded output contributes "" (an honest, usually-missing vote).
func votesFor(plan Plan, cs Case) []Vote {
	votes := make([]Vote, 0, len(plan.Members))
	for _, mem := range plan.Members {
		v := Vote{Member: mem, Output: cs.Outputs[mem.Model]}
		if s, ok := cs.Scores[mem.Model]; ok {
			v.Score = s
		}
		votes = append(votes, v)
	}
	return votes
}

// ---------------------------------------------------------------------------
// THE COMPARISON — routed (per-aspect + ensemble) vs single-model.
// ---------------------------------------------------------------------------

// Comparison is the routed-vs-single-model result over one corpus: both arms'
// Metrics plus the frontier baseline name. The delta helpers express the trade
// on each axis; positive cost/latency = the routed arm is cheaper/faster,
// positive quality = the routed arm is more accurate.
type Comparison struct {
	Routed   Metrics `json:"routed"`
	Single   Metrics `json:"single"`
	Frontier string  `json:"frontier"`
	Cases    int     `json:"cases"`
}

// Compare runs the routed (per-aspect + ensemble) and single (one-model) Manifests
// over the same Corpus and returns both Metrics plus the frontier baseline name.
// Pure and deterministic. The routed manifest is the policy under test; the
// single manifest is the SOTA baseline shape (one model for everything) — build
// it with SingleModelManifest.
func (c Corpus) Compare(routed, single Manifest, book PriceBook, lat LatencyBook, frontier string) Comparison {
	return Comparison{
		Routed:   c.Run("routed", routed, book, lat, frontier),
		Single:   c.Run("single", single, book, lat, frontier),
		Frontier: frontierName(frontier),
		Cases:    len(c),
	}
}

// CostSavingFrac is (single - routed) / single on cost: positive means the routed
// arm is cheaper. Negative means the routed arm costs more (e.g. an ensemble-heavy
// policy spending more compute than one frontier call per aspect).
func (cmp Comparison) CostSavingFrac() float64 {
	if cmp.Single.Cost == 0 {
		return 0
	}
	return (cmp.Single.Cost - cmp.Routed.Cost) / cmp.Single.Cost
}

// LatencySavingFrac is (single - routed) / single on total compute latency:
// positive means the routed arm does less total work. (A parallel ensemble's
// wall-clock is bounded by the member max, which this lens does not assume.)
func (cmp Comparison) LatencySavingFrac() float64 {
	if cmp.Single.Latency == 0 {
		return 0
	}
	return (cmp.Single.Latency - cmp.Routed.Latency) / cmp.Single.Latency
}

// QualityDelta is routed - single accuracy: positive means the routed arm answered
// more of the corpus correctly; negative means a downgrade lost accuracy the
// single-model baseline had.
func (cmp Comparison) QualityDelta() float64 {
	return cmp.Routed.Quality - cmp.Single.Quality
}

// frontierName resolves the baseline label for display.
func frontierName(frontier string) string {
	if frontier != "" {
		return frontier
	}
	return "frontier"
}

// ---------------------------------------------------------------------------
// THE BASELINE + DEMO CORPUS — runnable with no args, no model, no network.
// ---------------------------------------------------------------------------

// SingleModelManifest is the SOTA baseline shape: one model for EVERY aspect —
// the policy a request-level router (RouteLLM, Martian, …) reduces from, and the
// thing per-aspect routing is measured AGAINST. With no rules, every Subject hits
// the fail-closed Default, so the whole corpus routes to the one frontier model.
func SingleModelManifest(frontier string) Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{
			Members: []Member{{Model: frontier, Role: "primary"}},
			Reason:  "single-model baseline: every aspect routes to one frontier model (the SOTA shape)",
		},
	}
}

// DemoCorpus is the built-in 8-case offline corpus `fak routebench` runs with no
// --corpus flag: it exercises the built-in DefaultManifest's three rules (a short
// interactive request -> small, a hard reasoning step -> large, a write-shaped
// tool call -> a two-model vote ensemble) plus the fail-closed default, against a
// single-model baseline. The stand-in outputs encode an HONEST trade — NOT a
// rigged "routing wins everything": per-aspect routing is cheaper/faster on the
// easy aspects, the ensemble is a deliberate premium that RESCUES one case a
// single model gets wrong, and a downgrade to the default LOSES one case the
// single model got right, so the quality deltas offset. It is a recorded fixture
// to make the benchmark runnable now, not a claim about real traffic.
func DemoCorpus() Corpus {
	return Corpus{
		// C1: short interactive request -> routed: small (rule interactive-short).
		// Both arms correct; routed pays the small tier (cost/latency win).
		{
			Subject:  Subject{Aspect: AspectRequest, Latency: LatencyInteractive, PromptTokens: 1024},
			Outputs:  map[string]string{"frontier": "Paris", "small": "Paris"},
			Expected: "Paris",
			Note:     "short interactive turn -> small model",
		},
		// C2: long interactive request (>4096 tokens) -> routed: default (no rule).
		// The weaker default errs where the frontier was right: an honest quality LOSS.
		{
			Subject:  Subject{Aspect: AspectRequest, Latency: LatencyInteractive, PromptTokens: 8192},
			Outputs:  map[string]string{"frontier": "42", "default": "41"},
			Expected: "42",
			Note:     "downgrade to default loses a case the frontier got right",
		},
		// C3: hard reasoning step -> routed: large (rule hard-reasoning). Both correct;
		// large ties the frontier price/latency (the tier routing can't undercut).
		{
			Subject:  Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			Outputs:  map[string]string{"frontier": "9", "large": "9"},
			Expected: "9",
			Note:     "high-complexity step -> large model (frontier tier, no saving)",
		},
		// C4: easy step -> routed: default. Both correct; routed pays the mid tier.
		{
			Subject:  Subject{Aspect: AspectStep, Complexity: ComplexityLow},
			Outputs:  map[string]string{"frontier": "ok", "default": "ok"},
			Expected: "ok",
			Note:     "easy step -> default (mid tier)",
		},
		// C5: write-shaped tool call -> routed: vote(guard-a, guard-b) (rule guard-writes).
		// The ensemble is a PREMIUM (two models) that RESCUES a case the single
		// frontier model got wrong: two guards agree on "approve" where it said "deny".
		{
			Subject:  Subject{Aspect: AspectToolCall, Tool: "write_file"},
			Outputs:  map[string]string{"frontier": "deny", "guard-a": "approve", "guard-b": "approve"},
			Expected: "approve",
			Note:     "ensemble rescues a case the single model got wrong",
		},
		// C6: another write call -> routed: vote(guard-a, guard-b). The guards split
		// (1-1); the deterministic tie-break (output asc) lands on "approve", matching
		// the single model. Honest: an ensemble is not magic — here it merely ties.
		{
			Subject:  Subject{Aspect: AspectToolCall, Tool: "write_keys"},
			Outputs:  map[string]string{"frontier": "approve", "guard-a": "approve", "guard-b": "deny"},
			Expected: "approve",
			Note:     "split ensemble ties via the deterministic vote tie-break",
		},
		// C7: unmatched query -> routed: default. Both correct; routed pays mid tier.
		{
			Subject:  Subject{Aspect: AspectQuery},
			Outputs:  map[string]string{"frontier": "rome", "default": "rome"},
			Expected: "rome",
			Note:     "unmatched query -> fail-closed default",
		},
		// C8: delete-shaped tool call -> routed: default (guard-writes matches write_*).
		// Both correct; routed pays mid tier.
		{
			Subject:  Subject{Aspect: AspectToolCall, Tool: "delete_file"},
			Outputs:  map[string]string{"frontier": "done", "default": "done"},
			Expected: "done",
			Note:     "delete tool call -> default (only write_* is guarded)",
		},
	}
}
