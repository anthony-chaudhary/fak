// Package nativeperf owns the committed hill-climb graph for fak-native raw-model
// performance. It describes planning and evidence; it does not run a model or
// promote projected throughput into a benchmark witness.
package nativeperf

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const Schema = "fak-native-performance-graph/v2"

type Status string

const (
	StatusPresent Status = "present"
	StatusPartial Status = "partial"
	StatusAbsent  Status = "absent"
)

type EvidenceClass string

const (
	ClassHypothesis EvidenceClass = "hypothesis"
	ClassWitnessed  EvidenceClass = "witnessed"
	ClassComparison EvidenceClass = "comparison"
)

type Comparability string

const (
	ComparabilityExact       Comparability = "exact"
	ComparabilityApproximate Comparability = "approximate"
)

type Envelope struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	ModelRevision     string `json:"model_revision"`
	ArtifactSHA256    string `json:"artifact_sha256"`
	Quantization      string `json:"quantization"`
	Hardware          string `json:"hardware"`
	MemoryGiB         int    `json:"memory_gib"`
	Engine            string `json:"engine"`
	Backend           string `json:"backend"`
	ForwardPath       string `json:"forward_path"`
	PromptTokens      int    `json:"prompt_tokens"`
	DecodeTokens      int    `json:"decode_tokens"`
	Repetitions       int    `json:"repetitions"`
	Temperature       int    `json:"temperature"`
	ThroughputMetric  string `json:"throughput_metric"`
	ComparabilityRule string `json:"comparability_rule"`
}

type ExpectedRange struct {
	FloorTokensPerSecond float64       `json:"floor_tokens_per_second"`
	RoofTokensPerSecond  float64       `json:"roof_tokens_per_second"`
	Classification       EvidenceClass `json:"classification"`
	Provenance           string        `json:"provenance"`
}

type Throughput struct {
	TokensPerSecond float64       `json:"tokens_per_second"`
	Classification  EvidenceClass `json:"classification"`
	Engine          string        `json:"engine"`
	Comparability   Comparability `json:"comparability"`
	ObservedOn      string        `json:"observed_on"`
	Provenance      string        `json:"provenance"`
	Note            string        `json:"note"`
}

type IssueRef struct {
	Number int    `json:"number"`
	URL    string `json:"url"`
}

type Feature struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	RungID     string `json:"rung_id"`
	Enabled    bool   `json:"enabled"`
	Status     Status `json:"status"`
	Observable string `json:"observable"`
}

type Rung struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	DependencyIDs []string      `json:"dependency_ids"`
	Enabled       bool          `json:"enabled"`
	Status        Status        `json:"status"`
	Expected      ExpectedRange `json:"expected"`
	Witnessed     *Throughput   `json:"witnessed"`
	Gap           string        `json:"gap"`
	NextIssue     IssueRef      `json:"next_issue"`
}

// Applicability binds a lever to one pinned execution envelope. The redundant
// backend and platform fields make accidental cross-platform reuse fail closed.
type Applicability struct {
	EnvelopeID string `json:"envelope_id"`
	Platform   string `json:"platform"`
	Backend    string `json:"backend"`
}

// PlanningEffect is a hypothesis or planning target, never a benchmark result.
type PlanningEffect struct {
	Summary        string        `json:"summary"`
	Classification EvidenceClass `json:"classification"`
	Provenance     string        `json:"provenance"`
}

// WitnessedEffect is populated only from an end-to-end fak-native receipt.
type WitnessedEffect struct {
	Summary        string        `json:"summary"`
	Classification EvidenceClass `json:"classification"`
	Provenance     string        `json:"provenance"`
	Receipt        string        `json:"receipt"`
}

type Lever struct {
	ID            string           `json:"id"`
	Name          string           `json:"name"`
	Applicability Applicability    `json:"applicability"`
	Enabled       bool             `json:"enabled"`
	Status        Status           `json:"status"`
	DependencyIDs []string         `json:"dependency_ids"`
	ConflictIDs   []string         `json:"conflict_ids"`
	Expected      PlanningEffect   `json:"expected"`
	Witnessed     *WitnessedEffect `json:"witnessed"`
	OwningIssue   IssueRef         `json:"owning_issue"`
	NextWitness   string           `json:"next_witness"`
}

type Graph struct {
	Schema     string     `json:"schema"`
	Envelope   Envelope   `json:"envelope"`   // Legacy primary-envelope view.
	Comparison Throughput `json:"comparison"` // Legacy Metal comparison view.
	Rungs      []Rung     `json:"rungs"`      // Legacy aggregate-rung view.
	Features   []Feature  `json:"features"`   // Legacy feature view.
	Envelopes  []Envelope `json:"envelopes"`
	Levers     []Lever    `json:"levers"`
}

// ActiveGraph returns the committed, independently attributable Metal and CUDA
// Qwen3.8-27B planning graph. Existing primary-envelope/rung fields remain so v1
// readers retain their previous view.
func ActiveGraph() Graph {
	metal := Envelope{
		ID:                "qwen38-27b-q4km-m3pro-p32-t64",
		Model:             "unsloth/Qwen3.8-27B-GGUF",
		ModelRevision:     "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		ArtifactSHA256:    "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
		Quantization:      "Q4_K_M",
		Hardware:          "Apple M3 Pro, 18-GPU-core",
		MemoryGiB:         36,
		Engine:            "fak-native",
		Backend:           "metal",
		ForwardPath:       "metal/qwen35-hybrid-session-v1",
		PromptTokens:      32,
		DecodeTokens:      64,
		Repetitions:       3,
		Temperature:       0,
		ThroughputMetric:  "end_to_end_decode_tokens_per_second",
		ComparabilityRule: "exact artifact, machine, P/T, sampling, repetitions, and execution identity",
	}
	cuda := Envelope{
		ID:                "qwen38-27b-q4k-a100-p1-decode",
		Model:             "Qwen3.8-27B GGUF pinned by issue #8635",
		ModelRevision:     "issue-8635-pinned-revision",
		ArtifactSHA256:    "issue-8635-pinned-artifact",
		Quantization:      "Q4_K",
		Hardware:          "NVIDIA A100",
		MemoryGiB:         0,
		Engine:            "fak-native",
		Backend:           "cuda",
		ForwardPath:       "cuda/q4k-p1-decode",
		PromptTokens:      1,
		DecodeTokens:      64,
		Repetitions:       3,
		Temperature:       0,
		ThroughputMetric:  "end_to_end_decode_tokens_per_second",
		ComparabilityRule: "same issue-#8635 pinned artifact, A100, P=1 decode path, sampling, repetitions, quality gates, and fak-native identity",
	}
	comparison := Throughput{
		TokensPerSecond: 6.966061,
		Classification:  ClassComparison,
		Engine:          "llama.cpp b9828",
		Comparability:   ComparabilityApproximate,
		ObservedOn:      "2026-08-23",
		Provenance:      "https://github.com/anthony-chaudhary/fak/issues/8697",
		Note:            "same machine, artifact, and P32/T64 target; comparison remains approximate until a joint matched receipt captures both engines",
	}
	rungs := []Rung{
		{
			ID: "resident-q4k-baseline", Name: "Resident Q4_K native baseline", DependencyIDs: []string{}, Enabled: true, Status: StatusPresent,
			Expected:  ExpectedRange{FloorTokensPerSecond: 2.3, RoofTokensPerSecond: 3.3, Classification: ClassHypothesis, Provenance: "retest planning band from docs/benchmarks/QWEN38-27B-LATEST.md and issue #8697; endpoints remain evidence only in their original envelopes"},
			Witnessed: &Throughput{TokensPerSecond: 3.3, Classification: ClassWitnessed, Engine: "fak-native", Comparability: ComparabilityApproximate, ObservedOn: "2026-08-23", Provenance: "https://github.com/anthony-chaudhary/fak/issues/8697", Note: "64 decoded tokens after 31 prompt tokens; target graph is P32/T64, so this is not an exact matched-envelope point"},
			Gap:       "No joint P32/T64 receipt; repeated synchronous Q4_K projection submissions leave native at about 47% of the diagnostic comparison.", NextIssue: issue(8324),
		},
		{
			ID: "coarse-resident-hybrid-graph", Name: "Coarse resident hybrid token graph", DependencyIDs: []string{"resident-q4k-baseline"}, Enabled: false, Status: StatusPartial,
			Expected: ExpectedRange{FloorTokensPerSecond: 5, RoofTokensPerSecond: 6.966061, Classification: ClassHypothesis, Provenance: "issue #8324 supplies the >=5 tok/s planning floor; issue #8697 supplies the diagnostic comparison used as a planning roof, not a physical limit"},
			Gap:      "Weights are resident, but activations, recurrent/KV state, and synchronization are not yet owned by one coarse fak-native token submission.", NextIssue: issue(8324),
		},
		{
			ID: "matched-native-parity", Name: "Matched native parity gate", DependencyIDs: []string{"coarse-resident-hybrid-graph"}, Enabled: false, Status: StatusAbsent,
			Expected: ExpectedRange{FloorTokensPerSecond: 6.62, RoofTokensPerSecond: 6.966061, Classification: ClassHypothesis, Provenance: "issue #8697 requires at least 95% of its 6.966061 tok/s comparison; 6.62 is the rounded planning threshold"},
			Gap:      "No exact same-artifact P32/T64 three-repetition fak-native before/after receipt with quality and execution-identity gates.", NextIssue: issue(8697),
		},
	}
	features := []Feature{
		{ID: "resident-q4k-weights", Name: "Resident Q4_K weights", RungID: "resident-q4k-baseline", Enabled: true, Status: StatusPresent, Observable: "receipt identifies fak-native metal/qwen35-hybrid-session-v1 and the exact artifact hash"},
		{ID: "coarse-token-submission", Name: "Coarse token-level Metal submission", RungID: "coarse-resident-hybrid-graph", Enabled: false, Status: StatusPartial, Observable: "receipt reports one owned token submission spanning activations, recurrent/KV state, and synchronization"},
		{ID: "matched-envelope-receipt", Name: "Matched P32/T64 native receipt", RungID: "matched-native-parity", Enabled: false, Status: StatusAbsent, Observable: "before/after receipt matches artifact, machine, P32/T64, repetitions, quality, and fak-native execution identity"},
	}
	metalApply := Applicability{EnvelopeID: metal.ID, Platform: "darwin/arm64", Backend: "metal"}
	cudaApply := Applicability{EnvelopeID: cuda.ID, Platform: "linux/amd64+nvidia-a100", Backend: "cuda"}
	levers := []Lever{
		lever("metal.resident-q4k-weights", "Resident Q4_K weights", metalApply, true, StatusPresent, nil, nil, "Retain the resident-weight baseline while isolating later submissions.", "docs/benchmarks/QWEN38-27B-LATEST.md and issue #8697", &WitnessedEffect{Summary: "3.3 tok/s fak-native after 31 prompt tokens and 64 decode tokens; approximate to this P32 envelope", Classification: ClassWitnessed, Provenance: "https://github.com/anthony-chaudhary/fak/issues/8697", Receipt: "issue #8697 evidence captured 2026-08-23"}, 8324, "Capture an exact P32/T64 three-repetition fak-native baseline receipt with artifact hash and execution identity."),
		lever("metal.command-buffer-amortization", "Command-buffer and synchronization amortization", metalApply, false, StatusPartial, []string{"metal.resident-q4k-weights"}, nil, "Reduce repeated synchronous Q4_K projection submission overhead; no throughput gain is assumed.", "issues #8324 and #8697 profile diagnosis", nil, 8324, "Run one-lever OFF/ON P32/T64 A/B on the pinned M3 Pro envelope; receipt must include commands, revisions, artifact hash, per-repetition latency/tok/s, quality, fak-native identity, and before/after profile."),
		lever("metal.fused-hybrid-graph-coverage", "Fused hybrid token graph coverage", metalApply, false, StatusPartial, []string{"metal.command-buffer-amortization"}, nil, "Keep activations and recurrent/KV state device-visible across a coarser token graph; no throughput gain is assumed.", "issue #8324", nil, 8324, "Run command-buffer-amortization ON with only fused graph coverage toggled OFF/ON; attach the exact matched P32/T64 end-to-end receipt and profile showing graph coverage."),
		lever("metal.paged-kv", "Paged KV allocation", metalApply, false, StatusAbsent, []string{"metal.resident-q4k-weights"}, nil, "Bound KV allocation and expose occupancy for serving; this is a serving hypothesis, not raw decode throughput evidence.", "issue #8395", nil, 8395, "Run the #8395 isolated paged-KV arm on identical prompts and arrival trace; receipt must report quality, TTFT p50/p95, ITL p50/p95, aggregate tok/s, peak memory, prefix-hit rate, and fallback count."),
		lever("metal.prefix-reuse", "Exact-prefix block reuse", metalApply, false, StatusAbsent, []string{"metal.paged-kv"}, nil, "Reuse exact prefix blocks without attributing gains to batching or prefill.", "issue #8395", nil, 8395, "Run the #8395 exact-prefix-reuse isolated arm with paged KV fixed ON and capture the complete serving receipt."),
		lever("metal.chunked-prefill", "Bounded chunked prefill", metalApply, false, StatusAbsent, []string{"metal.resident-q4k-weights"}, nil, "Bound prefill scheduling without attributing a decode-throughput effect.", "issue #8395", nil, 8395, "Run the #8395 chunked-prefill isolated arm on the identical prompts and arrival trace and capture the complete serving receipt."),
		lever("metal.continuous-batching", "Continuous batching", metalApply, false, StatusAbsent, []string{"metal.paged-kv"}, nil, "Improve serving occupancy under concurrent arrivals; no single-request decode gain is assumed.", "issue #8395", nil, 8395, "Run the #8395 continuous-batching isolated arm with paged KV fixed ON and capture the complete serving receipt."),
		lever("metal.matched-parity-receipt", "Matched native parity gate", metalApply, false, StatusAbsent, []string{"metal.fused-hybrid-graph-coverage"}, nil, "Plan for at least 95% of the separately classified llama.cpp comparison; this is not a fak-native measurement.", "issue #8697", nil, 8697, "Run matched fak-native and llama.cpp b9828 P32/T64 temperature-zero three-repetition commands on the exact M3 Pro/artifact; retain commands, revisions, hash, each latency/tok/s, deterministic quality, identities, and profiles."),
		lever("cuda.scalar-f32-activation-baseline", "Scalar f32 activation-product baseline", cudaApply, true, StatusPresent, nil, []string{"cuda.q8_1-activation-quant"}, "Retain the current P=1 path as the OFF arm; issue #8635 reports approximately 11 tok/s, not an accepted receipt in this graph.", "issue #8635", nil, 8635, "Capture the repeated same-artifact A100 baseline receipt with strict quality and fak-native execution identity before enabling Q8_1."),
		lever("cuda.q8_1-activation-quant", "Q8_1 activation quantization", cudaApply, false, StatusAbsent, nil, []string{"cuda.scalar-f32-activation-baseline"}, "Quantize each current resident activation vector under cosine >=0.995, exact-argmax, and maxAbs <=0.02 gates; no throughput gain is assumed.", "issue #8635", nil, 8635, "Run the strict Q4_K numerical gate for the Q8_1 OFF/ON arm and retain cosine, exact argmax, maxAbs, artifact identity, and raw output."),
		lever("cuda.dp4a-q4k-mmvq", "Signed DP4A Q4_K MMVQ", cudaApply, false, StatusAbsent, []string{"cuda.q8_1-activation-quant"}, nil, "Target the issue #8635 >=47.36 tok/s operating envelope without treating that target as witnessed fak-native throughput.", "issue #8635", nil, 8635, "With Q8_1 fixed ON, run DP4A MMVQ OFF/ON repeated same-artifact A100 decode A/B; retain per-run end-to-end tok/s, quality gates, raw logs, and fak-native identity."),
		lever("cuda.default-decode-routing", "Default P=1 decode routing", cudaApply, false, StatusAbsent, []string{"cuda.dp4a-q4k-mmvq"}, nil, "Route the validated MMVQ path by default only after full-model correctness and end-to-end gain.", "issue #8635", nil, 8635, "Capture full-model text/JSON/tool correctness without fallback, repeated A100 decode gain, and default-path inspection proving fak-native Q8_1/DP4A routing."),
	}
	return Graph{Schema: Schema, Envelope: metal, Comparison: comparison, Rungs: rungs, Features: features, Envelopes: []Envelope{metal, cuda}, Levers: levers}
}

func lever(id, name string, applicability Applicability, enabled bool, status Status, deps, conflicts []string, expected, provenance string, witnessed *WitnessedEffect, owner int, next string) Lever {
	return Lever{ID: id, Name: name, Applicability: applicability, Enabled: enabled, Status: status, DependencyIDs: deps, ConflictIDs: conflicts, Expected: PlanningEffect{Summary: expected, Classification: ClassHypothesis, Provenance: provenance}, Witnessed: witnessed, OwningIssue: issue(owner), NextWitness: next}
}

func issue(number int) IssueRef {
	return IssueRef{Number: number, URL: fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", number)}
}

// NextLever returns the first graph-order, dependency-ready lever without a
// witness. A disabled conflicting baseline does not block selection: the arm's
// experiment is expected to toggle it off.
func NextLever(graph Graph) (*Lever, error) {
	if err := Validate(graph); err != nil {
		return nil, err
	}
	byID := make(map[string]Lever, len(graph.Levers))
	for _, candidate := range graph.Levers {
		byID[candidate.ID] = candidate
	}
	for i := range graph.Levers {
		candidate := &graph.Levers[i]
		if candidate.Witnessed != nil || candidate.Enabled {
			continue
		}
		ready := true
		for _, dep := range candidate.DependencyIDs {
			if !byID[dep].Enabled {
				ready = false
				break
			}
		}
		if ready {
			copy := *candidate
			return &copy, nil
		}
	}
	return nil, nil
}

// DOT renders deterministic Graphviz DOT. Envelope clusters prevent Metal and
// CUDA throughput or edges from appearing as one curve.
func DOT(graph Graph) (string, error) {
	if err := Validate(graph); err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString("digraph native_performance {\n  rankdir=LR;\n")
	for _, envelope := range graph.Envelopes {
		fmt.Fprintf(&b, "  subgraph cluster_%s {\n    label=%q;\n", dotID(envelope.ID), envelope.ID+" | "+envelope.Backend+" | fak-native")
		for _, lever := range graph.Levers {
			if lever.Applicability.EnvelopeID != envelope.ID {
				continue
			}
			label := lever.ID + "\\n" + string(lever.Status)
			if lever.Enabled {
				label += " / enabled"
			}
			fmt.Fprintf(&b, "    %q [label=%q];\n", lever.ID, label)
		}
		b.WriteString("  }\n")
	}
	for _, lever := range graph.Levers {
		for _, dep := range lever.DependencyIDs {
			fmt.Fprintf(&b, "  %q -> %q [label=\"depends\"];\n", dep, lever.ID)
		}
		for _, conflict := range lever.ConflictIDs {
			if lever.ID < conflict {
				fmt.Fprintf(&b, "  %q -> %q [dir=both,style=dashed,color=red,label=\"conflicts\"];\n", lever.ID, conflict)
			}
		}
	}
	b.WriteString("}\n")
	return b.String(), nil
}

func dotID(value string) string {
	return strings.NewReplacer("-", "_", ".", "_", "/", "_").Replace(value)
}

// Validate rejects graph shapes that would make ordering or performance claims
// ambiguous. Expected and witnessed values deliberately use separate fields.
func Validate(graph Graph) error {
	var findings []string
	if graph.Schema != Schema {
		findings = append(findings, fmt.Sprintf("schema %q must be %q", graph.Schema, Schema))
	}
	if graph.Comparison.Classification != ClassComparison {
		findings = append(findings, "comparison classification must be comparison")
	}
	if err := validateThroughput("comparison", graph.Comparison); err != nil {
		findings = append(findings, err.Error())
	}

	byRung := make(map[string]Rung, len(graph.Rungs))
	for i, rung := range graph.Rungs {
		label := fmt.Sprintf("rung[%d]", i)
		if strings.TrimSpace(rung.ID) == "" {
			findings = append(findings, label+" has empty id")
			continue
		}
		if _, exists := byRung[rung.ID]; exists {
			findings = append(findings, fmt.Sprintf("duplicate rung id %q", rung.ID))
			continue
		}
		byRung[rung.ID] = rung
		if !validStatus(rung.Status) {
			findings = append(findings, fmt.Sprintf("rung %q has invalid status %q", rung.ID, rung.Status))
		}
		if rung.Expected.Classification != ClassHypothesis {
			findings = append(findings, fmt.Sprintf("rung %q expected range classification must be hypothesis; measurements belong in witnessed", rung.ID))
		}
		if err := validateRange(rung.ID, rung.Expected); err != nil {
			findings = append(findings, err.Error())
		}
		if rung.Witnessed != nil {
			if rung.Witnessed.Classification != ClassWitnessed {
				findings = append(findings, fmt.Sprintf("rung %q witnessed classification must be witnessed; projections belong in expected", rung.ID))
			}
			if err := validateThroughput("rung "+rung.ID+" witnessed", *rung.Witnessed); err != nil {
				findings = append(findings, err.Error())
			}
		}
	}
	for _, rung := range graph.Rungs {
		for _, dep := range rung.DependencyIDs {
			if _, ok := byRung[dep]; !ok {
				findings = append(findings, fmt.Sprintf("rung %q depends on unknown rung %q", rung.ID, dep))
			}
		}
	}
	featureIDs := make(map[string]struct{}, len(graph.Features))
	for i, feature := range graph.Features {
		label := fmt.Sprintf("feature[%d]", i)
		if strings.TrimSpace(feature.ID) == "" {
			findings = append(findings, label+" has empty id")
		} else if _, exists := featureIDs[feature.ID]; exists {
			findings = append(findings, fmt.Sprintf("duplicate feature id %q", feature.ID))
		} else {
			featureIDs[feature.ID] = struct{}{}
		}
		if _, exists := byRung[feature.RungID]; !exists {
			findings = append(findings, fmt.Sprintf("feature %q owns unknown rung %q", feature.ID, feature.RungID))
		}
		if !validStatus(feature.Status) {
			findings = append(findings, fmt.Sprintf("feature %q has invalid status %q", feature.ID, feature.Status))
		}
		if feature.Enabled && feature.Status == StatusAbsent {
			findings = append(findings, fmt.Sprintf("feature %q is enabled but absent", feature.ID))
		}
		if !feature.Enabled && feature.Status == StatusPresent {
			findings = append(findings, fmt.Sprintf("feature %q is present but disabled", feature.ID))
		}
		if strings.TrimSpace(feature.Observable) == "" {
			findings = append(findings, fmt.Sprintf("feature %q has empty observable", feature.ID))
		}
	}
	if cycle := rungCycle(graph.Rungs, byRung); len(cycle) > 0 {
		findings = append(findings, "dependency cycle: "+strings.Join(cycle, " -> "))
	}

	envelopes := make(map[string]Envelope, len(graph.Envelopes))
	for i, envelope := range graph.Envelopes {
		if strings.TrimSpace(envelope.ID) == "" {
			findings = append(findings, fmt.Sprintf("envelope[%d] has empty id", i))
		} else if _, exists := envelopes[envelope.ID]; exists {
			findings = append(findings, fmt.Sprintf("duplicate envelope id %q", envelope.ID))
		} else {
			envelopes[envelope.ID] = envelope
		}
		if envelope.Engine != "fak-native" || strings.TrimSpace(envelope.Backend) == "" || strings.TrimSpace(envelope.Hardware) == "" || strings.TrimSpace(envelope.Model) == "" || strings.TrimSpace(envelope.Quantization) == "" || strings.TrimSpace(envelope.ArtifactSHA256) == "" || envelope.MemoryGiB < 0 || envelope.DecodeTokens <= 0 || envelope.Repetitions <= 0 {
			findings = append(findings, fmt.Sprintf("envelope %q has invalid pinned applicability or fak-native identity", envelope.ID))
		}
	}
	if len(graph.Envelopes) == 0 || graph.Envelope.ID != graph.Envelopes[0].ID {
		findings = append(findings, "legacy envelope must equal the first pinned envelope")
	}
	byLever := make(map[string]Lever, len(graph.Levers))
	for i, lever := range graph.Levers {
		if strings.TrimSpace(lever.ID) == "" {
			findings = append(findings, fmt.Sprintf("lever[%d] has empty id", i))
			continue
		}
		if _, exists := byLever[lever.ID]; exists {
			findings = append(findings, fmt.Sprintf("duplicate lever id %q", lever.ID))
			continue
		}
		byLever[lever.ID] = lever
		if !validStatus(lever.Status) || (lever.Enabled && lever.Status == StatusAbsent) || (!lever.Enabled && lever.Status == StatusPresent) {
			findings = append(findings, fmt.Sprintf("lever %q has invalid enablement/status %t/%q", lever.ID, lever.Enabled, lever.Status))
		}
		envelope, exists := envelopes[lever.Applicability.EnvelopeID]
		if !exists {
			findings = append(findings, fmt.Sprintf("lever %q has invalid applicability envelope %q", lever.ID, lever.Applicability.EnvelopeID))
		} else if lever.Applicability.Backend != envelope.Backend || !validPlatform(lever.Applicability.Platform, envelope.Backend) {
			findings = append(findings, fmt.Sprintf("lever %q has invalid applicability %q/%q for envelope %q", lever.ID, lever.Applicability.Platform, lever.Applicability.Backend, envelope.ID))
		}
		if lever.Expected.Classification != ClassHypothesis || strings.TrimSpace(lever.Expected.Summary) == "" || strings.TrimSpace(lever.Expected.Provenance) == "" {
			findings = append(findings, fmt.Sprintf("lever %q expected effect must be a provenance-labeled hypothesis; witnessed evidence belongs in witnessed", lever.ID))
		}
		if lever.Witnessed != nil && (lever.Witnessed.Classification != ClassWitnessed || strings.TrimSpace(lever.Witnessed.Summary) == "" || strings.TrimSpace(lever.Witnessed.Provenance) == "" || strings.TrimSpace(lever.Witnessed.Receipt) == "") {
			findings = append(findings, fmt.Sprintf("lever %q witnessed effect must be receipt-backed witnessed evidence; projections belong in expected", lever.ID))
		}
		if lever.OwningIssue.Number <= 0 || strings.TrimSpace(lever.OwningIssue.URL) == "" || strings.TrimSpace(lever.NextWitness) == "" {
			findings = append(findings, fmt.Sprintf("lever %q must name owning issue and exact next witness requirement", lever.ID))
		}
	}
	for _, lever := range graph.Levers {
		refs := append(append([]string{}, lever.DependencyIDs...), lever.ConflictIDs...)
		seen := map[string]bool{}
		for _, ref := range refs {
			target, exists := byLever[ref]
			if !exists {
				findings = append(findings, fmt.Sprintf("lever %q references unknown lever %q", lever.ID, ref))
				continue
			}
			if target.Applicability.EnvelopeID != lever.Applicability.EnvelopeID {
				findings = append(findings, fmt.Sprintf("cross-envelope edge %q -> %q", lever.ID, ref))
			}
			if seen[ref] {
				findings = append(findings, fmt.Sprintf("lever %q repeats dependency/conflict reference %q", lever.ID, ref))
			}
			seen[ref] = true
		}
		for _, conflict := range lever.ConflictIDs {
			target, exists := byLever[conflict]
			if exists && !contains(target.ConflictIDs, lever.ID) {
				findings = append(findings, fmt.Sprintf("conflict edge %q <-> %q is not symmetric", lever.ID, conflict))
			}
			if exists && lever.Enabled && target.Enabled {
				findings = append(findings, fmt.Sprintf("conflicting levers %q and %q are both enabled", lever.ID, conflict))
			}
		}
	}
	if cycle := leverCycle(graph.Levers, byLever); len(cycle) > 0 {
		findings = append(findings, "lever dependency cycle: "+strings.Join(cycle, " -> "))
	}
	if len(findings) != 0 {
		sort.Strings(findings)
		return fmt.Errorf("invalid native performance graph: %s", strings.Join(findings, "; "))
	}
	return nil
}

func validPlatform(platform, backend string) bool {
	return (backend == "metal" && platform == "darwin/arm64") || (backend == "cuda" && platform == "linux/amd64+nvidia-a100")
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func validStatus(status Status) bool {
	return status == StatusPresent || status == StatusPartial || status == StatusAbsent
}

func validateRange(id string, r ExpectedRange) error {
	if !finitePositive(r.FloorTokensPerSecond) || !finitePositive(r.RoofTokensPerSecond) || r.FloorTokensPerSecond > r.RoofTokensPerSecond {
		return fmt.Errorf("rung %q has invalid expected range %v..%v", id, r.FloorTokensPerSecond, r.RoofTokensPerSecond)
	}
	if strings.TrimSpace(r.Provenance) == "" {
		return fmt.Errorf("rung %q expected range has no provenance", id)
	}
	return nil
}

func validateThroughput(label string, throughput Throughput) error {
	if !finitePositive(throughput.TokensPerSecond) {
		return fmt.Errorf("%s has invalid throughput %v", label, throughput.TokensPerSecond)
	}
	if strings.TrimSpace(throughput.Engine) == "" || strings.TrimSpace(throughput.Provenance) == "" || strings.TrimSpace(throughput.ObservedOn) == "" {
		return fmt.Errorf("%s must name engine, observation date, and provenance", label)
	}
	if throughput.Comparability != ComparabilityExact && throughput.Comparability != ComparabilityApproximate {
		return fmt.Errorf("%s has invalid comparability %q", label, throughput.Comparability)
	}
	return nil
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func rungCycle(rungs []Rung, byID map[string]Rung) []string {
	deps := func(id string) []string { return byID[id].DependencyIDs }
	ids := make([]string, 0, len(rungs))
	for _, rung := range rungs {
		ids = append(ids, rung.ID)
	}
	return dependencyCycle(ids, deps)
}

func leverCycle(levers []Lever, byID map[string]Lever) []string {
	deps := func(id string) []string { return byID[id].DependencyIDs }
	ids := make([]string, 0, len(levers))
	for _, lever := range levers {
		ids = append(ids, lever.ID)
	}
	return dependencyCycle(ids, deps)
}

func dependencyCycle(ids []string, deps func(string) []string) []string {
	state := make(map[string]uint8, len(ids))
	stack := make([]string, 0, len(ids))
	stackAt := make(map[string]int, len(ids))
	known := make(map[string]bool, len(ids))
	for _, id := range ids {
		known[id] = true
	}
	var cycle []string
	var visit func(string) bool
	visit = func(id string) bool {
		if state[id] == 1 {
			cycle = append(cycle, stack[stackAt[id]:]...)
			cycle = append(cycle, id)
			return true
		}
		if state[id] == 2 {
			return false
		}
		state[id] = 1
		stackAt[id] = len(stack)
		stack = append(stack, id)
		for _, dep := range deps(id) {
			if known[dep] && visit(dep) {
				return true
			}
		}
		stack = stack[:len(stack)-1]
		delete(stackAt, id)
		state[id] = 2
		return false
	}
	for _, id := range ids {
		if known[id] && visit(id) {
			return cycle
		}
	}
	return nil
}
