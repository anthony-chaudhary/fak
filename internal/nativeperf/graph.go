// Package nativeperf owns the committed hill-climb graph for fak-native raw-model
// performance. It describes planning and evidence; it does not run a model or
// promote projected throughput into a benchmark witness.
package nativeperf

import (
	"fmt"
	"math"
	"strings"
)

const Schema = "fak-native-performance-graph/v1"

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

type Graph struct {
	Schema     string     `json:"schema"`
	Envelope   Envelope   `json:"envelope"`
	Comparison Throughput `json:"comparison"`
	Rungs      []Rung     `json:"rungs"`
}

// ActiveGraph returns the committed Qwen3.8-27B Apple M3 Pro P32/T64 graph.
// The 3.3 tok/s native point and 6.966061 tok/s llama.cpp point are evidence;
// every value under Expected remains a planning hypothesis.
func ActiveGraph() Graph {
	return Graph{
		Schema: Schema,
		Envelope: Envelope{
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
		},
		Comparison: Throughput{
			TokensPerSecond: 6.966061,
			Classification:  ClassComparison,
			Engine:          "llama.cpp b9828",
			Comparability:   ComparabilityApproximate,
			ObservedOn:      "2026-08-23",
			Provenance:      "https://github.com/anthony-chaudhary/fak/issues/8697",
			Note:            "same machine, artifact, and P32/T64 target; comparison remains approximate until a joint matched receipt captures both engines",
		},
		Rungs: []Rung{
			{
				ID:            "resident-q4k-baseline",
				Name:          "Resident Q4_K native baseline",
				DependencyIDs: []string{},
				Enabled:       true,
				Status:        StatusPresent,
				Expected: ExpectedRange{
					FloorTokensPerSecond: 2.3,
					RoofTokensPerSecond:  3.3,
					Classification:       ClassHypothesis,
					Provenance:           "retest planning band from docs/benchmarks/QWEN38-27B-LATEST.md and issue #8697; endpoints remain evidence only in their original envelopes",
				},
				Witnessed: &Throughput{
					TokensPerSecond: 3.3,
					Classification:  ClassWitnessed,
					Engine:          "fak-native",
					Comparability:   ComparabilityApproximate,
					ObservedOn:      "2026-08-23",
					Provenance:      "https://github.com/anthony-chaudhary/fak/issues/8697",
					Note:            "64 decoded tokens after 31 prompt tokens; target graph is P32/T64, so this is not an exact matched-envelope point",
				},
				Gap:       "No joint P32/T64 receipt; repeated synchronous Q4_K projection submissions leave native at about 47% of the diagnostic comparison.",
				NextIssue: issue(8324),
			},
			{
				ID:            "coarse-resident-hybrid-graph",
				Name:          "Coarse resident hybrid token graph",
				DependencyIDs: []string{"resident-q4k-baseline"},
				Enabled:       false,
				Status:        StatusPartial,
				Expected: ExpectedRange{
					FloorTokensPerSecond: 5,
					RoofTokensPerSecond:  6.966061,
					Classification:       ClassHypothesis,
					Provenance:           "issue #8324 supplies the >=5 tok/s planning floor; issue #8697 supplies the diagnostic comparison used as a planning roof, not a physical limit",
				},
				Witnessed: nil,
				Gap:       "Weights are resident, but activations, recurrent/KV state, and synchronization are not yet owned by one coarse fak-native token submission.",
				NextIssue: issue(8324),
			},
			{
				ID:            "matched-native-parity",
				Name:          "Matched native parity gate",
				DependencyIDs: []string{"coarse-resident-hybrid-graph"},
				Enabled:       false,
				Status:        StatusAbsent,
				Expected: ExpectedRange{
					FloorTokensPerSecond: 6.62,
					RoofTokensPerSecond:  6.966061,
					Classification:       ClassHypothesis,
					Provenance:           "issue #8697 requires at least 95% of its 6.966061 tok/s comparison; 6.62 is the rounded planning threshold",
				},
				Witnessed: nil,
				Gap:       "No exact same-artifact P32/T64 three-repetition fak-native before/after receipt with quality and execution-identity gates.",
				NextIssue: issue(8697),
			},
		},
	}
}

func issue(number int) IssueRef {
	return IssueRef{
		Number: number,
		URL:    fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", number),
	}
}

// Validate rejects graph shapes that would make ordering or performance claims
// ambiguous. Expected and witnessed values deliberately use separate fields and
// closed classifications so a projection cannot masquerade as a measurement.
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

	byID := make(map[string]Rung, len(graph.Rungs))
	for i, rung := range graph.Rungs {
		label := fmt.Sprintf("rung[%d]", i)
		if strings.TrimSpace(rung.ID) == "" {
			findings = append(findings, label+" has empty id")
			continue
		}
		if _, exists := byID[rung.ID]; exists {
			findings = append(findings, fmt.Sprintf("duplicate rung id %q", rung.ID))
			continue
		}
		byID[rung.ID] = rung
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
			if _, ok := byID[dep]; !ok {
				findings = append(findings, fmt.Sprintf("rung %q depends on unknown rung %q", rung.ID, dep))
			}
		}
	}
	if cycle := dependencyCycle(graph.Rungs, byID); len(cycle) > 0 {
		findings = append(findings, "dependency cycle: "+strings.Join(cycle, " -> "))
	}
	if len(findings) != 0 {
		return fmt.Errorf("invalid native performance graph: %s", strings.Join(findings, "; "))
	}
	return nil
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

func dependencyCycle(rungs []Rung, byID map[string]Rung) []string {
	state := make(map[string]uint8, len(byID))
	stack := make([]string, 0, len(byID))
	stackAt := make(map[string]int, len(byID))
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
		for _, dep := range byID[id].DependencyIDs {
			if _, known := byID[dep]; known && visit(dep) {
				return true
			}
		}
		stack = stack[:len(stack)-1]
		delete(stackAt, id)
		state[id] = 2
		return false
	}
	for _, rung := range rungs {
		if _, known := byID[rung.ID]; known && visit(rung.ID) {
			return cycle
		}
	}
	return nil
}
