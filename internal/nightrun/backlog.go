package nightrun

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

// Backlog assembles the full candidate set of collection Tasks from every
// source, de-duplicated by id, in deterministic id order:
//
//  1. the in-binary benchmark catalog (internal/benchcatalog) — every benchmark
//     becomes a Task, its Requirements derived from the benchmark's cold-start
//     Need and its Value seeded from the system level it measures;
//  2. the curated open-witness registry — the named, still-open measured data the
//     project is blocked on (PENDING_MEASUREMENT / on-box re-measures);
//  3. an optional operator/agent overlay file (additive), so a new datum can be
//     enqueued without recompiling.
//
// A duplicate id across sources fails LOUD (the overlay must not silently shadow
// a built-in), mirroring computetarget's built-ins-plus-user-file discipline.
func Backlog(overlayPath string) ([]Task, error) {
	seen := map[string]Source{}
	var out []Task
	add := func(t Task, src Source) error {
		if prev, ok := seen[t.ID]; ok {
			return fmt.Errorf("nightrun: duplicate task id %q (already defined by %s source)", t.ID, prev)
		}
		seen[t.ID] = src
		out = append(out, t)
		return nil
	}

	for _, t := range benchmarkTasks() {
		if err := add(t, SourceBenchmark); err != nil {
			return nil, err
		}
	}
	for _, t := range witnessTasks() {
		if err := add(t, SourceWitness); err != nil {
			return nil, err
		}
	}
	overlay, err := loadOverlay(overlayPath)
	if err != nil {
		return nil, err
	}
	for _, t := range overlay {
		t.Source = SourceOverlay
		if err := add(t, SourceOverlay); err != nil {
			return nil, err
		}
	}
	return sortTasks(out), nil
}

// benchmarkTasks projects the benchmark catalog into collection Tasks. The
// benchmark's Need maps to a capability requirement and its Level seeds a Value:
// a serving/e2e benchmark that produces a hardware number is more valuable to
// collect than a zero-asset micro/component smoke that just re-runs.
func benchmarkTasks() []Task {
	var out []Task
	for _, b := range benchcatalog.All() {
		out = append(out, Task{
			ID:         "bench-" + b.Name,
			Title:      b.Summary,
			Source:     SourceBenchmark,
			Value:      benchValue(b),
			Requires:   benchRequires(b),
			Run:        b.Run,
			Acceptance: "a recorded result/number from `" + b.Run + "`",
			Doc:        b.Doc,
		})
	}
	return out
}

// benchRequires maps a benchmark's cold-start Need to nightrun requirements. A
// weights benchmark needs local weights; a dataset benchmark needs the dataset;
// an offline benchmark needs nothing.
func benchRequires(b benchcatalog.Bench) []Requirement {
	switch b.Need {
	case benchcatalog.NeedWeights:
		return []Requirement{ReqWeights}
	case benchcatalog.NeedDataset:
		return []Requirement{ReqDataset}
	default:
		return nil // offline
	}
}

// benchValue seeds a benchmark Task's importance: a serving/e2e benchmark that
// emits a real hardware number is coverage-class (worth filling per box); a
// micro/component zero-asset bench is a smoke check (the floor). The ledger's
// novelty/staleness then dominates the actual ordering per box.
func benchValue(b benchcatalog.Bench) Value {
	switch b.Level {
	case benchcatalog.LevelServing, benchcatalog.LevelE2E:
		return ValueCoverage
	default:
		return ValueSmoke
	}
}

// witnessTasks is the curated registry of the project's named, still-open
// measured data — the data nightrun most wants collected when a capable box is
// available. Each row is a TASK (work to do), never a result, so it cannot
// overclaim; the canonical authority is the linked issue/doc. These are seeded
// from the open witnesses the project tracks (the PENDING_MEASUREMENT /
// on-box-re-measure items); add a row here when a new measured datum is blocked
// on hardware/credentials, the same additive-leaf discipline the kernel uses.
func witnessTasks() []Task {
	return []Task{
		{
			ID:          "witness-q8-decode-matvec-bw",
			Title:       "re-measure the 7B Q8 GPU-resident decode mat-vec kernel bandwidth on the Mac Metal node after a mul_mv_q8_0 tune (residual was 96 vs ~143 GB/s)",
			Source:      SourceWitness,
			Value:       ValueWitness,
			Requires:    []Requirement{ReqMetal, ReqWeights},
			Run:         "FAK_METAL_DECODE=1 go test -tags=metal,cgo -run=^$ -bench=. ./internal/compute",
			Acceptance:  "a recorded GB/s for mul_mv_q8_0 in an experiments/benchmark/runs artifact, compared against the 143 GB/s llama-Metal reference",
			RecheckDays: 7,
			Doc:         "docs/notes/MAC-METAL-VERIFY-NODE.md",
		},
		{
			ID:          "witness-glm52-native-load-10min",
			Title:       "re-measure GLM-5.2 native cold load time on-box (target <10 min after the parallel quant-load + resident-expert levers)",
			Source:      SourceWitness,
			Value:       ValueWitness,
			Requires:    []Requirement{ReqCUDA, ReqWeights},
			Run:         "FAK_GGUF_LOAD_WORKERS=8 fak serve --gguf <glm-5.2.gguf> --backend cuda --load-only",
			Acceptance:  "a recorded wall-clock load time under 10 minutes captured from the load-path visibility log",
			RecheckDays: 14,
			Doc:         "docs/serving",
		},
		{
			ID:          "witness-glm52-vllm-throughput-parity",
			Title:       "collect raw GLM-5.2/vLLM throughput on a live H200 endpoint to close the #870 agentic battery (6/11 artifacts; the 5 remaining all need a live endpoint)",
			Source:      SourceWitness,
			Value:       ValueFrontier,
			Requires:    []Requirement{ReqCUDA, ReqNet},
			Run:         "experiments/agent-live/run.sh   # the committed H200 GLM-5.2/vLLM recipe",
			Acceptance:  "the 5 PENDING_MEASUREMENT artifacts populated from a real :8000/:8080 endpoint (no AUTHORITY row without a measured number)",
			RecheckDays: 30,
			Doc:         "BENCHMARK-AUTHORITY.md",
		},
		{
			ID:          "witness-terminalbench-live-credentialed",
			Title:       "run the credentialed Terminal-Bench 2.1 live submission so an authority row can exist (#902 packet is precredential/BLOCKED until a real run)",
			Source:      SourceWitness,
			Value:       ValueFrontier,
			Requires:    []Requirement{ReqNet},
			CredEnv:     []string{"ANTHROPIC_API_KEY"},
			Run:         "go run ./cmd/terminalbench -suite <official-suite> -out experiments/agent-live/terminalbench-live.json",
			Acceptance:  "a graded result_claim_allowed=true artifact from a credentialed run (#900/#925)",
			RecheckDays: 30,
			Doc:         "docs/notes/TERMINALBENCH",
		},
		{
			ID:          "witness-a100-qwen-serve-first-run",
			Title:       "collect the first-ever Qwen3.6-27B-on-one-A100 pure-fak-kernel serve numbers (tok/s + correctness) via the gcp-qwen-serve path",
			Source:      SourceWitness,
			Value:       ValueFrontier,
			Requires:    []Requirement{ReqCUDA},
			Run:         "experiments/benchmark gcp-qwen-serve.sh  →  fak serve + fak agent (qwen3.6-27b)",
			Acceptance:  "a recorded tok/s and a correctness cosine in an experiments/benchmark/runs/by-machine/a100 artifact",
			RecheckDays: 30,
			Doc:         "docs/HARDWARE-MATRIX.md",
		},
		{
			ID:          "witness-resume-cache-calibration",
			Title:       "collect the #940 resume-cache calibration back-test (projection vs real billing) — an offline measurement that needs no hardware",
			Source:      SourceWitness,
			Value:       ValueRegression,
			Requires:    nil, // offline back-test over recorded billing
			Run:         "fak resume validate --calibrate",
			Acceptance:  "a recorded calibration accuracy over the billing boundaries (current 97.7%) refreshed against new sessions",
			RecheckDays: 14,
			Doc:         "docs/proofs/async-addressing.md",
		},
	}
}

// loadOverlay reads the optional operator/agent overlay file: a JSON array of
// Tasks additive over the built-ins. A missing path is fine (built-ins only); a
// malformed file or a Task with no id/run fails loud so a typo can't silently
// drop a queued datum.
func loadOverlay(path string) ([]Task, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("nightrun: read overlay %s: %w", path, err)
	}
	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("nightrun: parse overlay %s: %w", path, err)
	}
	for i, t := range tasks {
		if strings.TrimSpace(t.ID) == "" {
			return nil, fmt.Errorf("nightrun: overlay %s task #%d has no id", path, i+1)
		}
		if strings.TrimSpace(t.Run) == "" {
			return nil, fmt.Errorf("nightrun: overlay %s task %q has no run command", path, t.ID)
		}
		if t.Value == "" {
			tasks[i].Value = ValueCoverage
		}
	}
	return tasks, nil
}

// DefaultOverlayRel is where nightrun looks for the operator overlay by default —
// committed under experiments/ so a queued datum is shareable across the fleet.
const DefaultOverlayRel = "experiments/nightrun/backlog.json"
