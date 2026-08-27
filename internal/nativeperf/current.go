package nativeperf

import (
	"fmt"
	"strings"
	"time"
)

const CurrentSchema = "fak-native-performance-current/1"

type ConstraintType string

const (
	ConstraintCorrectness  ConstraintType = "correctness"
	ConstraintEvidence     ConstraintType = "evidence"
	ConstraintDependency   ConstraintType = "dependency"
	ConstraintCapacity     ConstraintType = "capacity"
	ConstraintCoordination ConstraintType = "coordination"
)

type ConstraintHorizon string

const (
	HorizonTransient   ConstraintHorizon = "transient"
	HorizonSemiDurable ConstraintHorizon = "semi-durable"
	HorizonStructural  ConstraintHorizon = "structural"
)

type ConstraintState string

const (
	ConstraintReady           ConstraintState = "ready"
	ConstraintWaitingEvidence ConstraintState = "waiting-evidence"
	ConstraintHeldCorrectness ConstraintState = "held-correctness"
	ConstraintCapacityBound   ConstraintState = "capacity-bound"
)

type EvidenceRef struct {
	Class   string `json:"class"`
	Summary string `json:"summary"`
	Ref     string `json:"ref"`
}

// CurrentConstraint records what limits a named performance outcome now and
// the checkable condition that removes the constraint. "Active constraint" is
// deliberate: a horizon says whether time may clear it; the word "temporary"
// alone is not an actionable state.
type CurrentConstraint struct {
	ID             string            `json:"id"`
	Name           string            `json:"name"`
	Type           ConstraintType    `json:"type"`
	Horizon        ConstraintHorizon `json:"horizon"`
	State          ConstraintState   `json:"state"`
	ObservedAt     string            `json:"observed_at"`
	ReviewBy       string            `json:"review_by"`
	Evidence       []EvidenceRef     `json:"evidence"`
	EnvelopeID     string            `json:"envelope"`
	Driver         string            `json:"driver"`
	AuthorityOwner string            `json:"authority_owner"`
	NextAction     string            `json:"next_action"`
	ExitWhen       string            `json:"exit_when"`
	ReadyLeverIDs  []string          `json:"ready_lever_ids,omitempty"`
}

type ReadyWave struct {
	ID               string   `json:"id"`
	EnvelopeID       string   `json:"envelope_id"`
	ReadyLeverIDs    []string `json:"ready_lever_ids"`
	ParallelWith     []string `json:"parallel_with,omitempty"`
	SerialWithinWave bool     `json:"serial_within_wave"`
	SerialReason     string   `json:"serial_reason,omitempty"`
}

type ReadyCollision struct {
	ID      string   `json:"id"`
	Kind    string   `json:"kind"`
	Members []string `json:"members"`
	Reason  string   `json:"reason"`
}

type OSSStateDefinition struct {
	State            string `json:"state"`
	Meaning          string `json:"meaning"`
	RequiredEvidence string `json:"required_evidence"`
}

type OSSWalkStep struct {
	Order       int    `json:"order"`
	Name        string `json:"name"`
	Requirement string `json:"requirement"`
	Exit        string `json:"exit"`
}

// OSSRoute is a current queue projection, not a claim that a candidate source
// has completed the walk. ProposedConstraintIDs remain proposals until State is
// mapped and the source has the required exhaustive study evidence.
type OSSRoute struct {
	Source                string   `json:"source"`
	Revision              string   `json:"revision"`
	State                 string   `json:"state"`
	Seam                  string   `json:"seam"`
	ProposedConstraintIDs []string `json:"proposed_constraint_ids,omitempty"`
	DedupedIssues         []int    `json:"deduped_issues,omitempty"`
	NextAction            string   `json:"next_action"`
}

type CurrentSnapshot struct {
	Schema      string               `json:"schema"`
	AsOf        string               `json:"as_of"`
	Definition  string               `json:"definition"`
	Constraints []CurrentConstraint  `json:"constraints"`
	ReadyWaves  []ReadyWave          `json:"ready_waves"`
	Collisions  []ReadyCollision     `json:"collisions"`
	OSSStates   []OSSStateDefinition `json:"oss_states"`
	OSSWalk     []OSSWalkStep        `json:"oss_walk"`
	OSSRoutes   []OSSRoute           `json:"oss_routes"`
}

// ReadyLevers returns every disabled, unwitnessed lever whose dependencies are
// enabled. NextLever remains the backward-compatible single-item projection.
func ReadyLevers(graph Graph) ([]Lever, error) {
	if err := Validate(graph); err != nil {
		return nil, err
	}
	byID := make(map[string]Lever, len(graph.Levers))
	for _, lever := range graph.Levers {
		byID[lever.ID] = lever
	}
	ready := make([]Lever, 0, len(graph.Levers))
	for _, lever := range graph.Levers {
		if lever.Enabled || lever.Witnessed != nil {
			continue
		}
		dependenciesReady := true
		for _, dependencyID := range lever.DependencyIDs {
			if !byID[dependencyID].Enabled {
				dependenciesReady = false
				break
			}
		}
		if dependenciesReady {
			ready = append(ready, lever)
		}
	}
	return ready, nil
}

func BuildCurrentSnapshot(graph Graph) (CurrentSnapshot, error) {
	ready, err := ReadyLevers(graph)
	if err != nil {
		return CurrentSnapshot{}, err
	}
	readyByEnvelope := map[string][]string{}
	for _, lever := range ready {
		readyByEnvelope[lever.Applicability.EnvelopeID] = append(readyByEnvelope[lever.Applicability.EnvelopeID], lever.ID)
	}

	const metalEnvelope = "qwen38-27b-q4km-m3pro-p32-t64"
	const cudaEnvelope = "qwen38-27b-q4k-a100-p1-decode"
	snapshot := CurrentSnapshot{
		Schema:     CurrentSchema,
		AsOf:       "2026-08-27",
		Definition: "An active constraint is the presently evidenced condition limiting a named performance outcome. Its type, horizon, owner, next action, and exit condition make it actionable; review_by prevents a stale observation from silently remaining current.",
		Constraints: []CurrentConstraint{
			{
				ID: "measurement-control-loop", Name: "Real profiling and regression control loop", Type: ConstraintEvidence, Horizon: HorizonSemiDurable, State: ConstraintWaitingEvidence,
				ObservedAt: "2026-08-25", ReviewBy: "2026-09-01", EnvelopeID: "portfolio",
				Driver:         "The classifier and receipt gates are built, but only synthetic Metal/CUDA profile bundles are committed and the scheduled workflow does not consume a returned campaign receipt.",
				AuthorityOwner: "internal/nativeperf profile and gate contracts; issue #8848",
				NextAction:     "Capture one real scrubbed Metal bundle and one real scrubbed CUDA bundle, then make the scheduled/manual workflow validate a returned request instead of only printing the handoff.",
				ExitWhen:       "Both native envelopes have accepted real profile bundles and one scheduled/manual run consumes a scrubbed request and records the gate verdict.",
				Evidence: []EvidenceRef{
					{Class: "contract", Summary: "Profile schema, classification, and one-lever selection are implemented.", Ref: "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#phaseprofile-bundles-and-bottleneck-selected-work"},
					{Class: "open", Summary: "The current acceptance table names both real profiler bundles OPEN.", Ref: "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#acceptance-status"},
					{Class: "control-loop", Summary: "The public workflow currently prints the private handoff and validator command; it does not fetch or gate a returned artifact.", Ref: ".github/workflows/native-performance-regression.yml"},
				},
			},
			{
				ID: "metal-resident-decode", Name: "Metal resident decode", Type: ConstraintDependency, Horizon: HorizonStructural, State: ConstraintReady,
				ObservedAt: "2026-08-23", ReviewBy: "2026-08-27", EnvelopeID: metalEnvelope,
				Driver:         "The near-matched native point is about 47% of the diagnostic llama.cpp comparison; repeated synchronous Q4_K submissions and an incomplete coarse hybrid token graph remain the issue-backed driver, pending a real profiler bundle.",
				AuthorityOwner: "issue #8324",
				NextAction:     "Run the exact P32/T64 one-lever command-buffer-amortization OFF/ON profile and receipt, then use it to confirm or revise the coarse resident graph work.",
				ExitWhen:       "A same-envelope quality-passing receipt proves the default fak-native path owns the coarse token submission and meets the issue's >=5 tok/s promotion floor, or a real profile selects a different driver.",
				ReadyLeverIDs:  []string{"metal.command-buffer-amortization"},
				Evidence: []EvidenceRef{
					{Class: "accepted", Summary: "The frozen full-run Metal envelope remains 2.3-2.9 decode tok/s with functional PASS.", Ref: "docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json"},
					{Class: "approximate", Summary: "The later observation is 3.3 vs 6.966061 tok/s, P31/T64 vs P32/T64, without a joint quality-complete receipt.", Ref: "https://github.com/anthony-chaudhary/fak/issues/8697"},
				},
			},
			{
				ID: "cuda-cold-decode", Name: "CUDA cold decode", Type: ConstraintDependency, Horizon: HorizonStructural, State: ConstraintReady,
				ObservedAt: "2026-08-25", ReviewBy: "2026-09-01", EnvelopeID: "q38-q4km-native-cuda-a100-cold-decode",
				Driver:         "The exact Q4_K_M A100 cold arm is correct at 11.8-12.1 tok/s. A distinct P=1 optimization envelope still uses scalar f32 activation products before the proposed Q8_1/DP4A path; its A/B must retain that separate identity.",
				AuthorityOwner: "issue #8635",
				NextAction:     "Run the strict Q8_1 OFF/ON numerical gate; only then run the DP4A MMVQ end-to-end A/B with Q8_1 fixed ON.",
				ExitWhen:       "The default fak-native CUDA decode path passes full-model quality with zero fallback and a repeated same-artifact end-to-end gain, or the measured profile selects another driver.",
				ReadyLeverIDs:  append([]string(nil), readyByEnvelope[cudaEnvelope]...),
				Evidence: []EvidenceRef{
					{Class: "accepted", Summary: "Five cold unique runs were 5/5 exact at 11.8-12.1 decode tok/s on A100-SXM4-40GB.", Ref: "docs/_witnesses/issue-8819-qwen38-cache-attribution/summary.json"},
					{Class: "hypothesis", Summary: "Q8_1 activation quantization followed by signed DP4A Q4_K MMVQ is the issue-owned P=1 sequence; no gain is assumed.", Ref: "https://github.com/anthony-chaudhary/fak/issues/8635"},
				},
			},
			{
				ID: "cuda-cache-correctness", Name: "CUDA cache restore correctness", Type: ConstraintCorrectness, Horizon: HorizonStructural, State: ConstraintHeldCorrectness,
				ObservedAt: "2026-08-25", ReviewBy: "2026-09-01", EnvelopeID: "q38-q4km-cuda-a100-cache-restore",
				Driver:         "The identical-prompt cache arm restored the wrong output in 5/5 attempts; its approximately 0.2 tok/s is diagnostic and cannot be optimized or promoted as parity.",
				AuthorityOwner: "issue #8848 follow-on from closed attribution issue #8819",
				NextAction:     "Instrument prefix snapshot, recurrent-state clone, host staging, restore, and backend-copy bytes/time; rerun five cold and five identical cache-hit requests with an unambiguous exact-output prompt.",
				ExitWhen:       "The identical-prompt arm is exact in every gated repetition with cache identity, restored state, zero fallback, and end-to-end timing in the same receipt.",
				Evidence:       []EvidenceRef{{Class: "diagnostic", Summary: "The cache arm was 0/5 exact; four confirmed hits were about 0.2 tok/s.", Ref: "docs/_witnesses/issue-8819-qwen38-cache-attribution/summary.json"}},
			},
			{
				ID: "laptop-placement", Name: "36 GiB laptop placement", Type: ConstraintCapacity, Horizon: HorizonStructural, State: ConstraintCapacityBound,
				ObservedAt: "2026-08-26", ReviewBy: "2026-09-09", EnvelopeID: "q38-q4km-native-metal-m3pro-capacity",
				Driver:         "The canonical no-FAK_Q4K_FREE_CPU control reached readiness and one native Metal token, but peak swap grew by 7,681,930,690 bytes; the fail-closed derived minimum is 44 GiB.",
				AuthorityOwner: "capacity receipt for closed issue #8971",
				NextAction:     "Place this exact no-free-CPU serving envelope on hardware meeting the 44 GiB derived bound; keep the 36 GiB laptop as control/orchestration or use only a separately named supported envelope.",
				ExitWhen:       "A same-artifact, same-environment native receipt proves zero positive swap on admitted hardware, or a newer measured bound supersedes 44 GiB.",
				Evidence:       []EvidenceRef{{Class: "witnessed-refusal", Summary: "The 36 GiB control derived a 44 GiB minimum after positive swap growth and restored the prior service.", Ref: "docs/_witnesses/issue-8971-streamed-q4k-capacity/canonical-no-free-cpu.json"}},
			},
			{
				ID: "native-serving-stack", Name: "Native serving stack", Type: ConstraintDependency, Horizon: HorizonStructural, State: ConstraintReady,
				ObservedAt: "2026-08-25", ReviewBy: "2026-09-03", EnvelopeID: metalEnvelope,
				Driver:         "Paged KV, exact-prefix reuse, chunked prefill, and continuous batching lack isolated real Qwen3.8 receipts; combining them before isolated arms would hide attribution.",
				AuthorityOwner: "issue #8395",
				NextAction:     "Work dependency-ready paged-KV and chunked-prefill as separate arms; advance prefix reuse and continuous batching only after paged KV is enabled and witnessed.",
				ExitWhen:       "Each mechanism has a quality-passing isolated receipt and the composed serving campaign reports TTFT/ITL p50/p95, aggregate throughput, peak memory, prefix-hit rate, and fallback count.",
				ReadyLeverIDs:  []string{"metal.paged-kv", "metal.chunked-prefill"},
				Evidence:       []EvidenceRef{{Class: "contract", Summary: "The graph records each serving mechanism as a separate absent lever with an exact witness requirement.", Ref: "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#metal-raw-decode-and-serving-levers"}},
			},
		},
		ReadyWaves: []ReadyWave{
			{ID: "metal", EnvelopeID: metalEnvelope, ReadyLeverIDs: append([]string(nil), readyByEnvelope[metalEnvelope]...), ParallelWith: []string{"cuda"}, SerialWithinWave: true, SerialReason: "Every arm shares one matched Metal envelope and must retain one-lever attribution without device contention."},
			{ID: "cuda", EnvelopeID: cudaEnvelope, ReadyLeverIDs: append([]string(nil), readyByEnvelope[cudaEnvelope]...), ParallelWith: []string{"metal"}, SerialWithinWave: true, SerialReason: "The Q8_1 candidate explicitly toggles the conflicting scalar-f32 baseline OFF inside one matched A/B."},
		},
		Collisions: []ReadyCollision{
			{ID: "metal-envelope", Kind: "shared-envelope", Members: append([]string(nil), readyByEnvelope[metalEnvelope]...), Reason: "These arms share the exact M3 Pro envelope; benchmark them serially and never combine them before each isolated receipt exists."},
			{ID: "cuda-activation-arm", Kind: "experiment-toggle", Members: []string{"cuda.scalar-f32-activation-baseline", "cuda.q8_1-activation-quant"}, Reason: "The two activation-product arms conflict; Q8_1 evidence must name the scalar baseline as OFF, not enable both."},
		},
		OSSStates: []OSSStateDefinition{
			{State: "watch", Meaning: "Pinned source identity retained without active implementation work.", RequiredEvidence: "repository, revision, license, and why it may matter"},
			{State: "candidate", Meaning: "A plausible source seam is named, but the exhaustive study or measured constraint binding is incomplete.", RequiredEvidence: "pinned source plus proposed seam; no performance issue inferred"},
			{State: "studied", Meaning: "A bounded study and candidate matrix are recorded; incomplete source classes remain explicit and block mapping.", RequiredEvidence: "study note, inventory map, completeness critic, explicit coverage limits, and dedupe readback"},
			{State: "mapped", Meaning: "One exact source seam is bound to a measured current constraint and a deduped issue.", RequiredEvidence: "path/line@revision, FAK seam, constraint ID, and issue readback"},
			{State: "mapped-needs-limiter", Meaning: "The exhaustive join found mapped backlog, but no measured limiter has selected which mapped seam should consume the next performance slot.", RequiredEvidence: "complete/qualified study, disposition counts, mapped issue evidence, and an explicit limiter-selection gap"},
			{State: "experimenting", Meaning: "A one-lever matched A/B is running or captured without a keep decision yet.", RequiredEvidence: "baseline/candidate receipts with quality, identity, memory, and end-to-end outcome"},
			{State: "kept", Meaning: "The adapted native path passed the A/B and is selected for the default or next composition.", RequiredEvidence: "accepted receipt, attribution/license, default-path witness, and rollback"},
			{State: "rejected", Meaning: "The source seam failed the A/B or no longer addresses the measured constraint.", RequiredEvidence: "negative receipt or superseding profile plus retained reason"},
		},
		OSSWalk: []OSSWalkStep{
			{Order: 1, Name: "source", Requirement: "Pin repository revision and license; inventory is discovery evidence, not a FAK gap.", Exit: "candidate or watch"},
			{Order: 2, Name: "seam", Requirement: "Complete the exhaustive study and name one exact source path/algorithm and one exact fak-native seam.", Exit: "studied"},
			{Order: 3, Name: "measured constraint", Requirement: "Bind the seam to a current constraint whose profiler/receipt evidence names that driver; otherwise return to watch.", Exit: "mapped or watch"},
			{Order: 4, Name: "deduped issue", Requirement: "Read back open and closed issues, then attach the route to one owner with a one-lever done condition.", Exit: "mapped with issue"},
			{Order: 5, Name: "A/B", Requirement: "Implement inside fak-native and capture a matched baseline/candidate with quality, identity, memory, fallback, and end-to-end accounting.", Exit: "experimenting"},
			{Order: 6, Name: "keep/reject", Requirement: "Keep only a quality-passing end-to-end gain; otherwise retain the negative result and reject or re-profile.", Exit: "kept or rejected"},
		},
		OSSRoutes: []OSSRoute{
			{Source: "vllm-project/vllm", Revision: "f18d0ba90d972a852a351c98be3f42b31372cfe4", State: "mapped-needs-limiter", Seam: "193 joined mechanism clusters: 183 actionable, including 168 partial and 13 conflict rows", ProposedConstraintIDs: []string{"measurement-control-loop", "native-serving-stack"}, NextAction: "Select from the 172 actionable partial/conflict rows using the measured limiter; the current prioritizer walking only five uncovered rows is backlog visibility, not performance closure."},
			{Source: "sgl-project/sglang", Revision: "536f570e6692eec0656ef9689db7591ca1d0e0a7", State: "studied", Seam: "12 serving and compatibility candidates; forge-history coverage remains explicitly partial", ProposedConstraintIDs: []string{"native-serving-stack"}, DedupedIssues: []int{8395}, NextAction: "Resolve the partial forge fence, then promote only a candidate selected by the serving limiter and existing issue dedupe."},
			{Source: "flashinfer-ai/flashinfer", Revision: "39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e", State: "mapped-needs-limiter", Seam: "22 decision-changing CUDA/kernel candidates, all deduped to existing FAK work or dependency-boundary rejection", ProposedConstraintIDs: []string{"measurement-control-loop", "cuda-cold-decode"}, NextAction: "Use the real CUDA profile to select one already-deduped seam; complete source accounting does not itself choose or close performance work."},
			{Source: "llm-d/llm-d", Revision: "bc20f73bd344b5a0faad5afca93831088aeee957", State: "mapped-needs-limiter", Seam: "20 serving-control candidates; two unduplicated gaps filed after complete dedupe", ProposedConstraintIDs: []string{"native-serving-stack"}, DedupedIssues: []int{9385, 9386}, NextAction: "Keep #9385/#9386 visible, but schedule them only when measured serving control or recovery is the active limiter."},
		},
	}
	if err := ValidateCurrentSnapshot(graph, snapshot); err != nil {
		return CurrentSnapshot{}, err
	}
	return snapshot, nil
}

func ValidateCurrentSnapshot(graph Graph, snapshot CurrentSnapshot) error {
	if snapshot.Schema != CurrentSchema || strings.TrimSpace(snapshot.AsOf) == "" || strings.TrimSpace(snapshot.Definition) == "" {
		return fmt.Errorf("current snapshot identity is incomplete")
	}
	asOf, err := time.Parse(time.DateOnly, snapshot.AsOf)
	if err != nil {
		return fmt.Errorf("current snapshot as_of: %w", err)
	}
	validTypes := map[ConstraintType]bool{ConstraintCorrectness: true, ConstraintEvidence: true, ConstraintDependency: true, ConstraintCapacity: true, ConstraintCoordination: true}
	validHorizons := map[ConstraintHorizon]bool{HorizonTransient: true, HorizonSemiDurable: true, HorizonStructural: true}
	validStates := map[ConstraintState]bool{ConstraintReady: true, ConstraintWaitingEvidence: true, ConstraintHeldCorrectness: true, ConstraintCapacityBound: true}
	constraintIDs := map[string]bool{}
	for _, constraint := range snapshot.Constraints {
		if constraintIDs[constraint.ID] {
			return fmt.Errorf("duplicate current constraint %q", constraint.ID)
		}
		constraintIDs[constraint.ID] = true
		if strings.TrimSpace(constraint.ID) == "" || strings.TrimSpace(constraint.Name) == "" || !validTypes[constraint.Type] || !validHorizons[constraint.Horizon] || !validStates[constraint.State] || strings.TrimSpace(constraint.EnvelopeID) == "" || strings.TrimSpace(constraint.Driver) == "" || strings.TrimSpace(constraint.AuthorityOwner) == "" || strings.TrimSpace(constraint.NextAction) == "" || strings.TrimSpace(constraint.ExitWhen) == "" || len(constraint.Evidence) == 0 {
			return fmt.Errorf("current constraint %q is incomplete", constraint.ID)
		}
		observed, observedErr := time.Parse(time.DateOnly, constraint.ObservedAt)
		review, reviewErr := time.Parse(time.DateOnly, constraint.ReviewBy)
		if observedErr != nil || reviewErr != nil || review.Before(observed) || review.Before(asOf) {
			return fmt.Errorf("current constraint %q has invalid observed/review dates", constraint.ID)
		}
		for _, evidence := range constraint.Evidence {
			if strings.TrimSpace(evidence.Class) == "" || strings.TrimSpace(evidence.Summary) == "" || strings.TrimSpace(evidence.Ref) == "" {
				return fmt.Errorf("current constraint %q has incomplete evidence", constraint.ID)
			}
		}
	}

	ready, err := ReadyLevers(graph)
	if err != nil {
		return err
	}
	wantReady := map[string]bool{}
	for _, lever := range ready {
		wantReady[lever.ID] = true
	}
	seenReady := map[string]bool{}
	waveIDs := map[string]bool{}
	for _, wave := range snapshot.ReadyWaves {
		if waveIDs[wave.ID] || strings.TrimSpace(wave.ID) == "" || strings.TrimSpace(wave.EnvelopeID) == "" || len(wave.ReadyLeverIDs) == 0 {
			return fmt.Errorf("current ready wave %q is invalid", wave.ID)
		}
		waveIDs[wave.ID] = true
		for _, leverID := range wave.ReadyLeverIDs {
			if !wantReady[leverID] || seenReady[leverID] {
				return fmt.Errorf("current ready wave %q has unknown or duplicate lever %q", wave.ID, leverID)
			}
			seenReady[leverID] = true
		}
		if wave.SerialWithinWave && strings.TrimSpace(wave.SerialReason) == "" {
			return fmt.Errorf("current ready wave %q lacks serial reason", wave.ID)
		}
	}
	for leverID := range wantReady {
		if !seenReady[leverID] {
			return fmt.Errorf("dependency-ready lever %q is missing from current waves", leverID)
		}
	}
	for _, wave := range snapshot.ReadyWaves {
		for _, peer := range wave.ParallelWith {
			if !waveIDs[peer] || peer == wave.ID {
				return fmt.Errorf("current ready wave %q has invalid parallel peer %q", wave.ID, peer)
			}
		}
	}

	validOSSStates := map[string]bool{}
	for _, definition := range snapshot.OSSStates {
		if validOSSStates[definition.State] || strings.TrimSpace(definition.State) == "" || strings.TrimSpace(definition.Meaning) == "" || strings.TrimSpace(definition.RequiredEvidence) == "" {
			return fmt.Errorf("invalid OSS state definition %q", definition.State)
		}
		validOSSStates[definition.State] = true
	}
	for _, required := range []string{"watch", "candidate", "studied", "mapped", "mapped-needs-limiter", "experimenting", "kept", "rejected"} {
		if !validOSSStates[required] {
			return fmt.Errorf("missing OSS state %q", required)
		}
	}
	for i, step := range snapshot.OSSWalk {
		if step.Order != i+1 || strings.TrimSpace(step.Name) == "" || strings.TrimSpace(step.Requirement) == "" || strings.TrimSpace(step.Exit) == "" {
			return fmt.Errorf("invalid OSS walk step at index %d", i)
		}
	}
	seenSources := map[string]bool{}
	for _, route := range snapshot.OSSRoutes {
		if seenSources[route.Source] || strings.TrimSpace(route.Source) == "" || strings.TrimSpace(route.Revision) == "" || !validOSSStates[route.State] || strings.TrimSpace(route.Seam) == "" || strings.TrimSpace(route.NextAction) == "" {
			return fmt.Errorf("invalid OSS route %q", route.Source)
		}
		seenSources[route.Source] = true
		for _, constraintID := range route.ProposedConstraintIDs {
			if !constraintIDs[constraintID] {
				return fmt.Errorf("OSS route %q references unknown constraint %q", route.Source, constraintID)
			}
		}
	}
	return nil
}

func RenderCurrentMarkdown(snapshot CurrentSnapshot) string {
	var b strings.Builder
	b.WriteString("---\ntitle: \"Current native-performance constraints\"\ndescription: \"Generated operational snapshot of current fak-native bottlenecks, ready work, collisions, and the OSS-to-performance decision walk.\"\n---\n\n")
	b.WriteString("# Current native-performance constraints\n\n")
	fmt.Fprintf(&b, "**As of:** %s  \n**Authority:** generated from `internal/nativeperf.BuildCurrentSnapshot`; immutable receipts remain the measurement evidence.  \n**Refresh:** `fak native-performance --current-md`\n\n", snapshot.AsOf)
	b.WriteString(snapshot.Definition + "\n\n")
	b.WriteString("## Current constraints\n\n")
	b.WriteString("| Constraint | Type / horizon / state | Envelope and driver | Evidence and authority | Next action / exit |\n|---|---|---|---|---|\n")
	for _, constraint := range snapshot.Constraints {
		evidence := make([]string, 0, len(constraint.Evidence))
		for _, item := range constraint.Evidence {
			evidence = append(evidence, fmt.Sprintf("[%s] %s (`%s`)", item.Class, item.Summary, item.Ref))
		}
		ready := ""
		if len(constraint.ReadyLeverIDs) > 0 {
			ready = "<br>Ready: `" + strings.Join(constraint.ReadyLeverIDs, "`, `") + "`"
		}
		fmt.Fprintf(&b, "| `%s` — %s | `%s` / `%s` / `%s`<br>Observed %s; review by %s | `%s`<br>%s%s | %s<br>Owner: %s | **Next:** %s<br>**Exit when:** %s |\n", constraint.ID, constraint.Name, constraint.Type, constraint.Horizon, constraint.State, constraint.ObservedAt, constraint.ReviewBy, constraint.EnvelopeID, constraint.Driver, ready, strings.Join(evidence, "<br>"), constraint.AuthorityOwner, constraint.NextAction, constraint.ExitWhen)
	}

	b.WriteString("\n## Dependency-ready work and collisions\n\n")
	b.WriteString("Every dependency-ready arm is shown. Metal and CUDA are independent waves and may progress in parallel; arms inside a matched envelope remain serial one-lever experiments.\n\n")
	b.WriteString("| Wave | Envelope | Ready arms | Parallel with | Within-wave rule |\n|---|---|---|---|---|\n")
	for _, wave := range snapshot.ReadyWaves {
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | `%s` | %s |\n", wave.ID, wave.EnvelopeID, strings.Join(wave.ReadyLeverIDs, "`, `"), strings.Join(wave.ParallelWith, "`, `"), wave.SerialReason)
	}
	b.WriteString("\n| Collision | Kind | Members | Why |\n|---|---|---|---|\n")
	for _, collision := range snapshot.Collisions {
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | %s |\n", collision.ID, collision.Kind, strings.Join(collision.Members, "`, `"), collision.Reason)
	}

	b.WriteString("\n## OSS-to-performance walk\n\n")
	b.WriteString("The closed walk is **source -> seam -> measured constraint -> deduped issue -> matched A/B -> keep/reject**. An exhaustive source list is discovery input, not permission to create or implement every idea.\n\n")
	b.WriteString("| State | Meaning | Required evidence |\n|---|---|---|\n")
	for _, definition := range snapshot.OSSStates {
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", definition.State, definition.Meaning, definition.RequiredEvidence)
	}
	b.WriteString("\n| # | Gate | Requirement | Exit |\n|---:|---|---|---|\n")
	for _, step := range snapshot.OSSWalk {
		fmt.Fprintf(&b, "| %d | %s | %s | `%s` |\n", step.Order, step.Name, step.Requirement, step.Exit)
	}
	b.WriteString("\n### Current source queue projection\n\n")
	b.WriteString("The complete discovery registry remains `docs/research/monitored-repositories.json`. These rows show only sources currently adjacent to a named performance constraint. `candidate`, `studied`, and `mapped-needs-limiter` are not implementation authorization; mapped backlog is not performance-closed.\n\n")
	b.WriteString("| Source @ revision | State | Seam | Proposed constraint / deduped issue | Next |\n|---|---|---|---|---|\n")
	for _, route := range snapshot.OSSRoutes {
		constraints := "—"
		if len(route.ProposedConstraintIDs) > 0 {
			constraints = "`" + strings.Join(route.ProposedConstraintIDs, "`, `") + "`"
		}
		issues := make([]string, 0, len(route.DedupedIssues))
		for _, issue := range route.DedupedIssues {
			issues = append(issues, fmt.Sprintf("#%d", issue))
		}
		if len(issues) == 0 {
			issues = append(issues, "—")
		}
		fmt.Fprintf(&b, "| `%s@%s` | `%s` | %s | %s / %s | %s |\n", route.Source, route.Revision, route.State, route.Seam, constraints, strings.Join(issues, ", "), route.NextAction)
	}

	b.WriteString("\n## Update contract\n\n")
	b.WriteString("Update the typed snapshot in the same change that accepts, rejects, or reclassifies evidence. Preserve immutable receipts; change a driver only from a compatible real profile or end-to-end receipt. Run `go test ./internal/nativeperf` and the focused `cmd/fak` native-performance tests, then regenerate this page with `fak native-performance --current-md`.\n")
	return b.String()
}
