package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/coordination"
)

const coordinateSchema = "fak.coordinate/1"

type coordinateInput struct {
	Schema           string   `json:"schema"`
	Harness          string   `json:"harness"`
	TaskID           string   `json:"task_id"`
	Workers          int      `json:"workers"`
	Coordination     bool     `json:"coordination"`
	RawModelOnly     bool     `json:"raw_model_only,omitempty"`
	ContextState     string   `json:"context_state"`
	ContextAction    string   `json:"context_action"`
	CacheAction      string   `json:"cache_action"`
	QueueState       string   `json:"queue_state"`
	Placement        string   `json:"placement"`
	RequiredEffects  []string `json:"required_effects"`
	RequiredOutcomes []string `json:"required_outcomes"`
}
type coordinateReceipt struct {
	Schema              string   `json:"schema"`
	PlanID              string   `json:"plan_id"`
	TaskID              string   `json:"task_id"`
	HarnessNeutral      bool     `json:"harness_neutral"`
	Action              string   `json:"action"`
	ContextAction       string   `json:"context_action"`
	CacheAction         string   `json:"cache_action"`
	ComputeEngine       string   `json:"compute_engine"`
	Placement           string   `json:"placement"`
	Admission           string   `json:"admission"`
	Backpressure        string   `json:"backpressure"`
	RequiredEffects     []string `json:"required_effects"`
	RequiredOutcomes    []string `json:"required_outcomes"`
	Accepted            bool     `json:"accepted"`
	EvidenceSufficiency string   `json:"evidence_sufficiency"`
	Delegated           bool     `json:"delegated,omitempty"`
	DelegatedBehavior   string   `json:"delegated_behavior,omitempty"`
}

func cmdCoordinate(args []string) { os.Exit(runCoordinate(os.Stdout, os.Stderr, args)) }

func runCoordinate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("coordinate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	demo := fs.Bool("demo", false, "emit deterministic two-worker demo receipt")
	jsonMode := fs.Bool("json", false, "read one JSON request from stdin and write JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || (*demo == *jsonMode) {
		fmt.Fprintln(stderr, "usage: fak coordinate (--demo | --json)")
		return 2
	}
	var in coordinateInput
	if *demo {
		in = coordinateDemoInput("fak-native")
	} else {
		dec := json.NewDecoder(io.LimitReader(os.Stdin, 1<<20))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&in); err != nil {
			fmt.Fprintf(stderr, "coordinate: invalid input: %v\n", err)
			return 2
		}
		var extra any
		if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
			fmt.Fprintln(stderr, "coordinate: exactly one JSON object is required")
			return 2
		}
	}
	r, err := coordinate(in)
	if err != nil {
		fmt.Fprintf(stderr, "coordinate: %v\n", err)
		return 2
	}
	enc := json.NewEncoder(stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		fmt.Fprintf(stderr, "coordinate: encode: %v\n", err)
		return 1
	}
	return 0
}

func coordinateDemoInput(harness string) coordinateInput {
	return coordinateInput{Schema: coordinateSchema, Harness: harness, TaskID: "task/two-worker-demo", Workers: 2, Coordination: true, ContextState: "warm", ContextAction: "reuse_managed_context", CacheAction: "reuse_shared_prefix", QueueState: "available", Placement: "resident", RequiredEffects: []string{"bounded_tool_write"}, RequiredOutcomes: []string{"accepted_worker_result", "verified_fanin"}}
}

func coordinate(in coordinateInput) (coordinateReceipt, error) {
	if in.Schema != coordinateSchema || in.TaskID == "" || in.Workers < 1 {
		return coordinateReceipt{}, errors.New("schema, task_id, and positive workers are required")
	}
	if in.Harness != "claude" && in.Harness != "codex" && in.Harness != "opencode" && in.Harness != "fak-native" {
		return coordinateReceipt{}, fmt.Errorf("unknown harness %q", in.Harness)
	}
	if !in.Coordination {
		return coordinateReceipt{Schema: "fak.coordinate-receipt/1", PlanID: "delegate/" + in.TaskID, TaskID: in.TaskID, HarnessNeutral: true, Action: "delegate_existing_behavior", EvidenceSufficiency: "delegated", Delegated: true, DelegatedBehavior: "existing_harness_path"}, nil
	}
	if in.ContextAction == "" || in.CacheAction == "" || in.QueueState == "" || in.Placement == "" || len(in.RequiredEffects) == 0 || len(in.RequiredOutcomes) == 0 {
		return coordinateReceipt{}, errors.New("coordinated input is incomplete")
	}
	effects := append([]string(nil), in.RequiredEffects...)
	outcomes := append([]string(nil), in.RequiredOutcomes...)
	sort.Strings(effects)
	sort.Strings(outcomes)
	workers := make([]coordination.HarnessWorker, in.Workers)
	for i := range workers {
		workers[i] = coordination.HarnessWorker{ID: fmt.Sprintf("worker-%d", i+1), Role: "execute", Model: "selected"}
	}
	workflow := coordination.HarnessWorkflow{Harness: coordination.HarnessFakNative, Coordination: true, WorkID: in.TaskID, CorrelationID: "corr/" + in.TaskID, Workers: workers, Fanout: in.Workers, Concurrency: in.Workers, Lease: coordination.HarnessLease{Lane: "coordinate", Mode: "exclusive", TTL: time.Minute, Renewable: true}, Budgets: coordination.HarnessBudgets{Tokens: 10000, CostMicros: 100000, Duration: time.Minute}, Interaction: coordination.InteractionNone, Cancellation: coordination.CancellationIsolate, Exhaustion: coordination.ExhaustionDelay, Witness: coordination.WitnessRequired, Degradation: coordination.DegradationForbid}
	intent, err := coordination.NewHarnessAdapter().Normalize(workflow)
	if err != nil {
		return coordinateReceipt{}, err
	}
	action := "execute"
	admission := "admit"
	backpressure := "none"
	if in.QueueState != "available" {
		action = "queue"
		admission = "queue"
		backpressure = "capacity"
	}
	if in.Placement != "resident" {
		action = "load_then_execute"
		if admission == "queue" {
			action = "queue_then_load"
		}
	}
	if in.CacheAction == "build_shared_prefix" && action == "execute" {
		action = "build_cache_then_execute"
	}
	sum := sha256.Sum256([]byte(intent.WorkID))
	planID := "coord/" + fmt.Sprintf("%x", sum[:8]) + "/" + strings.ReplaceAll(action, "_", "-")
	r := coordinateReceipt{Schema: "fak.coordinate-receipt/1", PlanID: planID, TaskID: in.TaskID, HarnessNeutral: true, Action: action, ContextAction: in.ContextAction, CacheAction: in.CacheAction, ComputeEngine: "fak_native", Placement: in.Placement, Admission: admission, Backpressure: backpressure, RequiredEffects: effects, RequiredOutcomes: outcomes, Accepted: true, EvidenceSufficiency: "whole_path"}
	if in.RawModelOnly {
		r.Accepted = false
		r.EvidenceSufficiency = "raw_model_insufficient"
	}
	return r, nil
}
