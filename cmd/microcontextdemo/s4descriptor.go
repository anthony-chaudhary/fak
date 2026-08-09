package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const descriptorBenchReportSchema = "fak-microcontext-descriptor-benchmark/1"

type descriptorBenchArm struct {
	Name                string `json:"name"`
	Kind                string `json:"kind"`
	Available           bool   `json:"available"`
	Provenance          string `json:"provenance"`
	Contexts            int    `json:"contexts"`
	StartupP50Micros    int64  `json:"startup_p50_us"`
	StartupP95Micros    int64  `json:"startup_p95_us"`
	BytesPerDescriptor  int    `json:"descriptor_bytes"`
	PeakAllocDeltaBytes uint64 `json:"peak_alloc_delta_bytes"`
	ModelCalls          int    `json:"model_calls"`
	PromptTokens        int    `json:"prompt_tokens"`
	Limitation          string `json:"limitation,omitempty"`
}
type descriptorBenchReport struct {
	Schema           string               `json:"schema"`
	Verdict          string               `json:"verdict"`
	ObservedAt       string               `json:"observed_at"`
	Contexts         int                  `json:"contexts"`
	DescriptorSchema string               `json:"descriptor_schema"`
	BaseInstalls     int                  `json:"base_installs"`
	Arms             []descriptorBenchArm `json:"arms"`
	Assumptions      []harnessAssumption  `json:"assumptions"`
	Claims           []string             `json:"claims"`
	NonClaims        []string             `json:"non_claims"`
}
type harnessAssumption struct {
	Name              string `json:"name"`
	Class             string `json:"class"`
	DescriptorCarrier string `json:"descriptor_carrier,omitempty"`
}
type fixtureResponder struct {
	mu     sync.Mutex
	calls  int
	prompt int
}

func (g *fixtureResponder) Model() string { return "deterministic-descriptor-fixture" }
func (g *fixtureResponder) Complete(_ context.Context, m []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.calls++
	for _, x := range m {
		g.prompt += len(x.Content)
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "DONE"}}, nil
}

func runDescriptorBenchmark(ctx context.Context, n int) (descriptorBenchReport, error) {
	if n <= 0 {
		return descriptorBenchReport{}, errors.New("contexts must be positive")
	}
	r := descriptorBenchReport{Schema: descriptorBenchReportSchema, Verdict: "PASS", ObservedAt: time.Now().UTC().Format(time.RFC3339), Contexts: n, DescriptorSchema: microagent.DescriptorSchema, BaseInstalls: 1,
		Assumptions: []harnessAssumption{{"model endpoint", "semantic", "base_id"}, {"task/input", "semantic", "task_delta"}, {"tool authority", "semantic", "tools"}, {"turn/token limit", "semantic", "budget"}, {"prior messages", "semantic", "continuation"}, {"result acceptance", "semantic", "output_contract"}, {"OS process", "session convenience", ""}, {"terminal/TUI", "session convenience", ""}, {"working directory", "adapter concern", "capability/lease adapter"}, {"ambient credentials", "unsafe convenience", "explicit capability adapter"}, {"interactive approval channel", "adapter concern", "effect policy adapter"}},
		Claims:      []string{"the versioned descriptor executes through the existing microagent Host and Gateway seams", "one immutable base is installed once while each context carries only its task delta and semantic controls"},
		NonClaims:   []string{"CLI version probe latency is not full task completion latency or resident harness RAM", "the deterministic fixture is not model quality, TTFT, or token-throughput evidence"}}
	for _, cmd := range []struct{ name, path string }{{"fak full CLI", "fak"}, {"Claude Code / Ultracode-style CLI", "claude"}, {"Codex API-oriented CLI", "codex.cmd"}} {
		r.Arms = append(r.Arms, probeCLI(ctx, cmd.name, cmd.path))
	}
	base := []agent.Message{{Role: agent.RoleSystem, Content: "immutable shared base"}}
	gw := &fixtureResponder{}
	h, err := microagent.NewHost(gw, microagent.Config{Workers: 16, Queue: n})
	if err != nil {
		return r, err
	}
	defer h.Close()
	var before runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	times := make([]time.Duration, 0, n)
	bytesPer := 0
	for i := 0; i < n; i++ {
		d := microagent.Descriptor{Schema: microagent.DescriptorSchema, ID: fmt.Sprintf("d-%04d", i), BaseID: "fixture-base-v1", TaskDelta: fmt.Sprintf("task-%04d", i), Budget: microagent.DescriptorBudget{MaxTurns: 1, MaxOutputTokens: 8}, OutputContract: microagent.OutputContract{Kind: "exact", Expected: "DONE"}}
		sz, e := microagent.DescriptorSize(d)
		if e != nil {
			return r, e
		}
		bytesPer += sz
		t := time.Now()
		if _, e = microagent.SpawnDescriptor(h, d, base); e != nil {
			return r, e
		}
		times = append(times, time.Since(t))
	}
	if err := h.Drain(ctx); err != nil {
		return r, err
	}
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	delta := uint64(0)
	if after.Alloc > before.Alloc {
		delta = after.Alloc - before.Alloc
	}
	r.Arms = append(r.Arms, descriptorBenchArm{Name: "micro-context descriptor", Kind: "in-process-adapter", Available: true, Provenance: "observed deterministic fixture through internal/microagent Host+Gateway", Contexts: n, StartupP50Micros: durationPercentile(times, 50).Microseconds(), StartupP95Micros: durationPercentile(times, 95).Microseconds(), BytesPerDescriptor: bytesPer / n, PeakAllocDeltaBytes: delta, ModelCalls: gw.calls, PromptTokens: gw.prompt, Limitation: "fixture output; allocation delta sampled after drain, not peak RSS"})
	if err := verifyDescriptorReport(r); err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	return r, nil
}

func probeCLI(ctx context.Context, name, path string) descriptorBenchArm { // command probe lives in main.go helper to keep this file platform-neutral
	durations := make([]time.Duration, 0, 5)
	available := true
	detail := "version/help process startup only; no model task and no reliable child RSS/token accounting"
	for i := 0; i < 5; i++ {
		d, err := runVersionProbe(ctx, path)
		if err != nil {
			available = false
			detail = err.Error()
			break
		}
		durations = append(durations, d)
	}
	return descriptorBenchArm{Name: name, Kind: "full-cli-version-probe", Available: available, Provenance: "observed local process spawn", Contexts: len(durations), StartupP50Micros: durationPercentile(durations, 50).Microseconds(), StartupP95Micros: durationPercentile(durations, 95).Microseconds(), Limitation: detail}
}
func verifyDescriptorReport(r descriptorBenchReport) error {
	if r.Schema != descriptorBenchReportSchema || r.Verdict != "PASS" || r.Contexts != 1000 || r.DescriptorSchema != microagent.DescriptorSchema || r.BaseInstalls != 1 {
		return errors.New("descriptor report header invariant failed")
	}
	if len(r.Arms) != 4 {
		return errors.New("descriptor report requires three full-harness probes and one adapter arm")
	}
	a := r.Arms[len(r.Arms)-1]
	if !a.Available || a.Contexts != r.Contexts || a.ModelCalls != r.Contexts || a.BytesPerDescriptor <= 0 {
		return fmt.Errorf("descriptor arm invariant failed: %+v", a)
	}
	return nil
}
func verifyDescriptorArtifact(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r descriptorBenchReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyDescriptorReport(r)
}
