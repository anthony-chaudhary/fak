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
			Manual:     b.Manual,
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
			ID:          "witness-qwen38-27b-metal-auto-gateway-sweep",
			Title:       "auto-collect the Mac Qwen3.8-27B Metal serving curve sweep: decode, longgen, prefill sweep, and 2-stream concurrency against the fak serve gateway",
			Source:      SourceWitness,
			Value:       ValueFrontier,
			Requires:    []Requirement{ReqMetal, ReqWeights},
			Run:         "fak macbench all --gateway http://127.0.0.1:8080 --model qwen38:27b --json",
			Acceptance:  "a fak.macbench.result.v1 artifact with observed tok/s rows for decode-longgen, prefill-sweep, and 2stream; gateway reported only as loopback/remote-sanitized and bearer never printed",
			RecheckDays: 14,
			TimeoutSec:  7200,
			Doc:         "docs/benchmarks/QWEN38-27B-LATEST.md",
		},
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
			// A cold GLM-5.2 native load targets <10 min but needs headroom for first-run
			// compile/warmup over a 466GB checkpoint; 30 min bounds a genuinely-hung load
			// without truncating a healthy one (#992).
			TimeoutSec: 1800,
			// The Run carries a <glm-5.2.gguf> fill-me-in placeholder: an operator points it at
			// the local checkpoint. Authoritative recipe marker so it is never auto-run (#989).
			Manual: true,
			Doc:    "docs/serving",
		},
		{
			ID:     "witness-glm52-vllm-throughput-parity",
			Title:  "collect raw GLM-5.2/vLLM throughput on a live H200 endpoint to close the #870 agentic battery (6/11 artifacts; the 5 remaining all need a live endpoint)",
			Source: SourceWitness,
			Value:  ValueFrontier,
			// Loading the model to serve it needs local weights — ANDed with the GPU + live
			// endpoint requirements. On a weightless GPU box this is now reported infeasible
			// with the "needs local model weights" reason instead of feasible-but-failing (#990).
			Requires:    []Requirement{ReqCUDA, ReqNet, ReqWeights},
			Run:         "experiments/agent-live/run.sh   # the committed H200 GLM-5.2/vLLM recipe",
			Acceptance:  "the 5 PENDING_MEASUREMENT artifacts populated from a real :8000/:8080 endpoint (no AUTHORITY row without a measured number)",
			RecheckDays: 30,
			// A live serve + throughput sweep (cold load, warmup, then a measured run) needs a
			// large budget; 1h bounds a wedged endpoint without truncating a real sweep (#992).
			TimeoutSec: 3600,
			// The Run is a bare `script.sh   # comment` recipe — no placeholder, no arrow, so the
			// autoRunnable heuristic alone would exec it every sweep and record a spurious
			// failure. The explicit marker is the authoritative no-auto-run signal (#989).
			Manual: true,
			Doc:    "BENCHMARK-AUTHORITY.md",
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
			// The Run carries an <official-suite> placeholder an operator fills in; authoritative
			// no-auto-run marker so it is surfaced for a human, not exec'd as prose (#989).
			Manual: true,
			Doc:    "docs/notes/TERMINALBENCH",
		},
		{
			ID:     "witness-a100-qwen-serve-first-run",
			Title:  "collect the first-ever Qwen3.8-27B-on-one-A100 pure-fak-kernel serve numbers (tok/s + correctness) via the gcp-qwen-serve path",
			Source: SourceWitness,
			Value:  ValueFrontier,
			// A pure-fak-kernel serve must load the model weights to collect tok/s +
			// correctness — ReqWeights ANDed with the GPU requirement, so a weightless A100 box
			// reports it infeasible ("needs local model weights") rather than feasible-failing (#990).
			Requires:    []Requirement{ReqCUDA, ReqWeights},
			Run:         "experiments/benchmark gcp-qwen-serve.sh  →  fak serve + fak agent (qwen38:27b)",
			Acceptance:  "a recorded tok/s and a correctness cosine in an experiments/benchmark/runs/by-machine/a100 artifact",
			RecheckDays: 30,
			// A cold 27B load + serve + a correctness/throughput pass needs a large budget; 1h
			// bounds a wedged run without truncating a healthy cold collection (#992).
			TimeoutSec: 3600,
			// The Run is a prose `script.sh  →  fak serve + fak agent` recipe (the arrow heuristic
			// already skips it); the explicit marker makes the registry intent authoritative (#989).
			Manual: true,
			Doc:    "docs/HARDWARE-MATRIX.md",
		},
		{
			ID:          "witness-resume-cache-calibration",
			Title:       "collect the #940 resume-cache calibration back-test (projection vs real billing) — an offline measurement that needs no hardware",
			Source:      SourceWitness,
			Value:       ValueRegression,
			Requires:    nil, // offline back-test over recorded billing
			Run:         "fak resume validate -corpus ~/.claude/projects -json",
			Acceptance:  "a recorded calibration accuracy over the billing boundaries (current 97.7%) refreshed against new sessions",
			RecheckDays: 14,
			Doc:         "docs/proofs/async-addressing.md",
		},
		{
			// The first datum past the current saturation frontier: a CUDA llama.cpp
			// baseline on the SAME box and GGUF as the fak-kernel GLM-5.2 decode, so the
			// open fak-GPU-vs-llama.cpp-GPU comparison closes with both numbers measured on
			// one host. Surfaced the moment a CUDA box WITH weights and net is reachable —
			// it uses only the existing capability enum (cuda+weights+net), no new prober
			// (a `cuda-llama-cpp-build` token would be an "unknown requirement"→infeasible,
			// not a surfaced blocked state; that prober is separate, larger work, #1138).
			ID:     "witness-glm52-cuda-llamacpp-comparison",
			Title:  "collect a CUDA llama.cpp GLM-5.2 decode baseline on the same box+GGUF as the fak-kernel run, to close the open fak-GPU vs llama.cpp-GPU decode comparison (the next datum past the current saturation frontier)",
			Source: SourceWitness,
			Value:  ValueFrontier,
			// Loading GLM-5.2 to decode it through llama.cpp needs an NVIDIA GPU, the local
			// weights, and net (to fetch the llama.cpp build / shards) — ANDed, so a box
			// missing any one reports it infeasible with the precise reason rather than
			// feasible-but-failing.
			Requires:    []Requirement{ReqCUDA, ReqWeights, ReqNet},
			Run:         "llama.cpp/build/bin/llama-bench -m <glm-5.2.gguf> -ngl 99 -p 512 -n 128   # CUDA build; record prompt+decode tok/s next to the fak-kernel run",
			Acceptance:  "a recorded llama.cpp CUDA prompt+decode tok/s for GLM-5.2 on the same box/GGUF as the fak-kernel decode, so the comparison row carries two measured numbers",
			RecheckDays: 30,
			// A cold GLM-5.2 load + a llama.cpp decode sweep over a 466GB checkpoint needs a
			// large budget; 1h bounds a wedged run without truncating a healthy sweep.
			TimeoutSec: 3600,
			// The Run carries a <glm-5.2.gguf> fill-me-in placeholder and needs an operator
			// to point at the local llama.cpp CUDA build — authoritative no-auto-run marker
			// so it is surfaced for a human, never exec'd as prose.
			Manual: true,
			Doc:    "docs/nightrun/GPU-SERVER-OVERNIGHT-PLAN-2026-06-28.md",
		},
		{
			// The micro-context fabric's scale witness, collected BY THE LOOP instead of by
			// hand (#5842). The fabric shipped a working spine (`cmd/microcontextdemo`) plus
			// a quality ledger and a health scorecard, but nothing in the pipeline ever ran
			// it: every captured artifact under experiments/microcontext/ came from an agent
			// remembering to invoke it. This row is the ONE seam that makes the default live
			// in the pipeline — nightrun is a Go tick loop AND a registered member of the
			// `manage-benchmarks` super loop, so an unattended `run --apply` turn selects and
			// executes the verb on its own and records the outcome as a durable ledger row.
			//
			// Deliberately the SYNTHETIC selfcheck: it is offline (no weights, GPU, dataset,
			// or credential), so it is feasible on every box in the fleet and cannot go dark
			// waiting for hardware. Its scope line says so in the artifact — this collects
			// bounded harness fan-out and shared-base semantics, never model tokens/sec. The
			// live-endpoint scale points stay operator/credential work, filed separately.
			ID:     "witness-microcontext-fabric-spine",
			Title:  "collect the micro-context fabric's 10,000-logical-context spine witness on this box (one immutable shared base, 64 bounded physical workers) — the S0 harness datum the fabric epic #5785 headlines",
			Source: SourceWitness,
			Value:  ValueCoverage,
			// Offline by construction: the selfcheck drives a synthetic endpoint, so an empty
			// Requires is the honest declaration (every box is capable) rather than a gate
			// that would strand the datum on GPU/weights availability.
			Requires:   nil,
			Run:        "go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64",
			Acceptance: "a fak-microcontext-spine/1 report with verdict PASS over 10000 logical shards, shared_base_installs=1, and peak_in_flight never above the 64 declared worker slots (the selfcheck exits non-zero if any invariant breaks, so a ledger `collected` row IS the witness)",
			// A synthetic fan-out re-runs cheaply, so re-check weekly: this is the fabric's
			// regression tripwire (a spine that stops holding its invariants should surface
			// within a week, not after the default fortnight).
			RecheckDays: 7,
			Doc:         "docs/research/micro-context-fabrics.md",
		},
		{
			ID:          "witness-strix-halo-subkernels-ablations",
			Title:       "physical validation and differential ablation of Vulkan compute sub-kernels (argmax, matmul_f32, q8_matmul, q4k_matmul, rmsnorm, swiglu) on AMD Strix Halo APU (gfx1151, 40 CUs, 64GB UMA)",
			Source:      SourceWitness,
			Value:       ValueFrontier,
			Requires:    nil,
			Run:         "go run ./cmd/fak-dev amd-strix-validate --subkernels=all --ablate=all --json",
			Acceptance:  "a fak.strix.validation/v1 artifact with verdict PASS across all subkernels and ablations, logit cosine >= 0.999900, and verified digest",
			RecheckDays: 7,
			TimeoutSec:  120,
			Doc:         "docs/fleet-compute-nodes.md",
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
