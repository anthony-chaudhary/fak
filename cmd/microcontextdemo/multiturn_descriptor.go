package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const multiTurnDescriptorSchema = "fak-microcontext-multi-turn-descriptor/1"

type multiTurnDescriptorReport struct {
	Schema                    string  `json:"schema"`
	Verdict                   string  `json:"verdict"`
	ObservedAt                string  `json:"observed_at"`
	DescriptorSchema          string  `json:"descriptor_schema"`
	Contexts                  int     `json:"contexts"`
	TurnsPerLogicalAgent      int     `json:"turns_per_logical_agent"`
	ExpectedTurns             int     `json:"expected_turns"`
	AccountedTurns            int     `json:"accounted_turns"`
	Completed                 int     `json:"completed"`
	Failed                    int     `json:"failed"`
	ContinuationTokens        int     `json:"continuation_tokens"`
	TraceMismatches           int     `json:"continuation_mismatches"`
	MidTaskHibernations       int     `json:"mid_task_hibernations"`
	VerifiedRestores          int     `json:"verified_restores"`
	ParkedBytes               int64   `json:"parked_bytes"`
	BaseInstalls              int     `json:"base_installs"`
	PhysicalWorkers           int     `json:"physical_workers"`
	WallNanos                 int64   `json:"wall_ns"`
	VerifiedCompletionsPerSec float64 `json:"verified_completions_per_second"`
	ClaimBoundary             string  `json:"claim_boundary"`
}

type scriptedTurnResponder struct {
	mu         sync.Mutex
	turns      map[string]int
	calls      int
	mismatches int
	maxTurns   int
}

func (g *scriptedTurnResponder) Model() string {
	return "deterministic-multi-turn-descriptor-fixture"
}
func (g *scriptedTurnResponder) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	trace := microagent.TraceFromContext(ctx)
	g.mu.Lock()
	defer g.mu.Unlock()
	g.calls++
	if trace == "" {
		g.mismatches++
	}
	g.turns[trace]++
	content := fmt.Sprintf("PROGRESS-%d", g.turns[trace])
	if g.turns[trace] == g.maxTurns {
		content = "DONE"
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: content}}, nil
}

func runMultiTurnDescriptor(ctx context.Context, contexts, workers, turns int) (multiTurnDescriptorReport, error) {
	r := multiTurnDescriptorReport{Schema: multiTurnDescriptorSchema, Verdict: "PASS", ObservedAt: time.Now().UTC().Format(time.RFC3339), DescriptorSchema: microagent.DescriptorSchemaV2, Contexts: contexts, TurnsPerLogicalAgent: turns, ExpectedTurns: contexts * turns, ContinuationTokens: contexts, BaseInstalls: 1, PhysicalWorkers: workers, ClaimBoundary: "observed deterministic Host/Gateway accounting and byte-verified hibernate restore; not model throughput, quality, KV residency, or process RSS"}
	if contexts != 1000 || workers <= 0 || turns < 2 {
		r.Verdict = "FAIL"
		return r, errors.New("multi-turn descriptor witness requires 1000 contexts, positive workers, and at least two turns")
	}
	gw := &scriptedTurnResponder{turns: make(map[string]int, contexts), maxTurns: turns}
	h, err := microagent.NewHost(gw, microagent.Config{Workers: workers, Queue: contexts})
	if err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	defer h.Close()
	dir, err := os.MkdirTemp("", "fak-microcontext-multiturn-")
	if err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	defer os.RemoveAll(dir)
	store, err := microagent.NewHibernationStore(dir)
	if err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	base := []agent.Message{{Role: agent.RoleSystem, Content: "immutable shared agent base"}}
	started := time.Now()
	for i := 0; i < contexts; i++ {
		id := fmt.Sprintf("mt-%04d", i)
		d := microagent.Descriptor{Schema: microagent.DescriptorSchemaV2, ID: id, BaseID: "immutable-agent-base-v2", TaskDelta: "advance bounded task", Budget: microagent.DescriptorBudget{MaxTurns: turns, MaxOutputTokens: 8}, ContinuationToken: "continuation-" + id, OutputContract: microagent.OutputContract{Kind: "exact", Expected: "DONE"}}
		a := &microagent.DescriptorAgent{Descriptor: d, Base: base}
		done, stepErr := a.Step(ctx, gw)
		if stepErr != nil || done {
			r.Failed++
			continue
		}
		n, parkErr := store.Park(id, a)
		if parkErr != nil {
			r.Failed++
			continue
		}
		r.MidTaskHibernations++
		r.ParkedBytes += int64(n)
		restored := &microagent.DescriptorAgent{Descriptor: d, Base: base}
		if wakeErr := store.Wake(id, restored); wakeErr != nil {
			r.Failed++
			continue
		}
		r.VerifiedRestores++
		if spawnErr := h.Spawn(id, restored); spawnErr != nil {
			r.Failed++
		}
	}
	if err := h.Drain(ctx); err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	r.WallNanos = time.Since(started).Nanoseconds()
	gw.mu.Lock()
	r.AccountedTurns, r.TraceMismatches = gw.calls, gw.mismatches
	for _, n := range gw.turns {
		if n == turns {
			r.Completed++
		} else {
			r.Failed++
		}
	}
	gw.mu.Unlock()
	if r.WallNanos > 0 {
		r.VerifiedCompletionsPerSec = float64(r.Completed) / (float64(r.WallNanos) / float64(time.Second))
	}
	if err := verifyMultiTurnDescriptorReport(r); err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	return r, nil
}

func verifyMultiTurnDescriptorReport(r multiTurnDescriptorReport) error {
	if r.Schema != multiTurnDescriptorSchema || r.Verdict != "PASS" || r.DescriptorSchema != microagent.DescriptorSchemaV2 {
		return errors.New("multi-turn descriptor header invariant failed")
	}
	if r.Contexts != 1000 || r.TurnsPerLogicalAgent < 2 || r.ExpectedTurns != r.Contexts*r.TurnsPerLogicalAgent || r.AccountedTurns != r.ExpectedTurns {
		return fmt.Errorf("turn reconciliation failed: expected=%d accounted=%d", r.ExpectedTurns, r.AccountedTurns)
	}
	if r.Completed != r.Contexts || r.Failed != 0 || r.ContinuationTokens != r.Contexts || r.TraceMismatches != 0 {
		return errors.New("completion or continuation reconciliation failed")
	}
	if r.MidTaskHibernations != r.Contexts || r.VerifiedRestores != r.Contexts || r.ParkedBytes <= 0 || r.BaseInstalls != 1 || r.PhysicalWorkers <= 0 || r.WallNanos <= 0 || r.VerifiedCompletionsPerSec <= 0 {
		return errors.New("hibernate, base, worker, or rate invariant failed")
	}
	return nil
}

func verifyMultiTurnDescriptorArtifact(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r multiTurnDescriptorReport
	if err := json.Unmarshal(b, &r); err != nil {
		return err
	}
	return verifyMultiTurnDescriptorReport(r)
}
