package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/idempotency"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const effectBatchSchema = "fak-microcontext-effect-batch/1"

type selectedEffect struct {
	ContextID      string `json:"context_id"`
	Stage          string `json:"stage"`
	Capability     string `json:"capability"`
	Resource       string `json:"resource"`
	Operation      string `json:"operation"`
	IdempotencyKey string `json:"idempotency_key"`
	Value          string `json:"value"`
	Approved       bool   `json:"approved"`
	DryRun         bool   `json:"dry_run"`
}
type effectReceipt struct {
	ContextID        string `json:"context_id"`
	Resource         string `json:"resource"`
	IdempotencyKey   string `json:"idempotency_key"`
	Status           string `json:"status"`
	Dispatched       bool   `json:"dispatched"`
	Applied          bool   `json:"applied"`
	Replayed         bool   `json:"replayed"`
	ReadbackVerified bool   `json:"readback_verified"`
	ObservedValue    string `json:"observed_value,omitempty"`
	Reason           string `json:"reason,omitempty"`
}
type effectBatchReport struct {
	Schema                       string          `json:"schema"`
	Mode                         string          `json:"mode"`
	Selected                     int             `json:"selected"`
	Confirmed                    int             `json:"confirmed"`
	ReplayedConfirmed            int             `json:"replayed_confirmed"`
	Denied                       int             `json:"denied"`
	Conflicts                    int             `json:"conflicts"`
	Failed                       int             `json:"failed"`
	CancelledBeforeDispatch      int             `json:"cancelled_before_dispatch"`
	CancelledBeforeDispatchCalls int             `json:"cancelled_before_dispatch_calls"`
	UnknownPendingReadback       int             `json:"unknown_pending_readback"`
	UnknownLaterConfirmed        int             `json:"unknown_later_confirmed"`
	ApprovalNotRun               int             `json:"approval_not_run"`
	DryRuns                      int             `json:"dry_runs"`
	BreakerNotRun                int             `json:"breaker_not_run"`
	DispatchAttempts             int             `json:"dispatch_attempts"`
	PhysicalWrites               int             `json:"physical_writes"`
	DuplicatePhysicalApplies     int             `json:"duplicate_physical_applies"`
	RestartPhysicalApplies       int             `json:"restart_physical_applies"`
	JournalEntries               int             `json:"journal_entries"`
	FoldedConfirmed              int             `json:"folded_confirmed"`
	FoldedUnknown                int             `json:"folded_unknown"`
	FoldedResources              []string        `json:"folded_resources"`
	Receipts                     []effectReceipt `json:"receipts"`
	Notes                        []string        `json:"notes"`
}

type fixtureEffects struct {
	mu      sync.Mutex
	values  map[string]string
	applies map[string]int
	calls   atomic.Int64
}

func (f *fixtureEffects) apply(in selectedEffect) func() (string, error) {
	return func() (string, error) {
		f.calls.Add(1)
		if in.Value == "fixture-fail-before-write" {
			return "", errors.New("fixture apply failure")
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		f.values[in.Resource] = in.Value
		f.applies[in.IdempotencyKey]++
		return in.Value, nil
	}
}
func (f *fixtureEffects) read(ctx context.Context, in microagent.EffectIntent, result string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	f.mu.Lock()
	v, ok := f.values[in.Resource]
	f.mu.Unlock()
	if !ok || v != result {
		return fmt.Errorf("independent readback mismatch resource=%s", in.Resource)
	}
	return nil
}
func (f *fixtureEffects) duplicateApplies() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, v := range f.applies {
		if v > 1 {
			n += v - 1
		}
	}
	return n
}

func effectIntent(in selectedEffect) microagent.EffectIntent {
	return microagent.EffectIntent{ContextID: in.ContextID, Capability: in.Capability, Resource: in.Resource, Operation: in.Operation, IdempotencyToken: in.IdempotencyKey}
}
func executeSelectedEffect(ctx context.Context, c *microagent.EffectCoordinator, state *fixtureEffects, in selectedEffect, allowed []string) effectReceipt {
	r := effectReceipt{ContextID: in.ContextID, Resource: in.Resource, IdempotencyKey: in.IdempotencyKey}
	if in.Stage != "set-label" {
		r.Status = "denied"
		r.Reason = "stage_not_allowlisted"
		return r
	}
	if in.DryRun {
		r.Status = "dry_run"
		r.Reason = "no_dispatch"
		return r
	}
	if !in.Approved {
		r.Status = "not_run"
		r.Reason = "approval_required"
		return r
	}
	select {
	case <-ctx.Done():
		r.Status = "cancelled_before_dispatch"
		r.Reason = "context_cancelled"
		return r
	default:
	}
	r.Dispatched = true
	before := state.calls.Load()
	out, err := c.Run(ctx, effectIntent(in), allowed, state.apply(in), func(rc context.Context, ei microagent.EffectIntent, result string) error {
		return state.read(rc, ei, result)
	})
	r.Applied = state.calls.Load() > before
	r.Replayed = out.Replayed
	if err != nil {
		switch {
		case errors.Is(err, microagent.ErrAuthorityRefused):
			r.Status = "denied"
			r.Reason = "capability_denied"
		case errors.Is(err, microagent.ErrEffectConflict):
			r.Status = "conflict"
			r.Reason = "resource_lease_held"
		default:
			var ve *microagent.VerificationError
			if errors.As(err, &ve) {
				r.Status = "unknown_pending_readback"
				r.Reason = "dispatched_effect_not_yet_verified"
			} else {
				r.Status = "failed"
				r.Reason = "apply_failed"
			}
		}
		return r
	}
	r.ReadbackVerified = out.Verified
	r.ObservedValue = out.Result
	if out.Replayed {
		r.Status = "replayed_confirmed"
	} else {
		r.Status = "confirmed"
	}
	return r
}

func runEffectBatchSelfcheck(output string) error {
	dir, err := os.MkdirTemp("", "fak-effect-batch-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	ledger := filepath.Join(dir, "effects.jsonl")
	store, err := idempotency.Open(ledger, time.Hour)
	if err != nil {
		return err
	}
	coord := microagent.NewEffectCoordinator(store)
	state := &fixtureEffects{values: map[string]string{}, applies: map[string]int{}}
	allowed := []string{"issue.label.write"}
	report := effectBatchReport{Schema: effectBatchSchema, Mode: "controlled fixture effects through microagent EffectCoordinator"}
	base := []selectedEffect{{"ctx-1", "set-label", "issue.label.write", "issue:1", "set-label", "effect-1", "auth", true, false}, {"ctx-2", "set-label", "issue.label.write", "issue:2", "set-label", "effect-2", "security", true, false}, {"ctx-3", "set-label", "issue.label.write", "issue:3", "set-label", "effect-3", "triage", true, false}, {"ctx-4", "set-label", "issue.label.write", "issue:4", "set-label", "effect-4", "docs", true, false}, {"ctx-5", "set-label", "issue.label.write", "issue:5", "set-label", "effect-5", "cache", true, false}}
	for _, in := range base {
		report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord, state, in, allowed))
	}
	// Duplicate in the same process: durable idempotency must replay, never reapply.
	dup := base[1]
	dup.ContextID = "ctx-duplicate"
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord, state, dup, allowed))
	// Restart from the journal: a new coordinator/store must still replay.
	beforeRestart := state.calls.Load()
	store2, err := idempotency.Open(ledger, time.Hour)
	if err != nil {
		return err
	}
	coord2 := microagent.NewEffectCoordinator(store2)
	restart := base[2]
	restart.ContextID = "ctx-restart"
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, restart, allowed))
	report.RestartPhysicalApplies = int(state.calls.Load() - beforeRestart)
	// Selector output cannot grant authority or invent an effect stage.
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, selectedEffect{"ctx-stage-denied", "shell", "issue.label.write", "issue:6", "exec", "effect-x", "bad", true, false}, allowed))
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, selectedEffect{"ctx-cap-denied", "set-label", "issue.delete", "issue:6", "delete", "effect-y", "bad", true, false}, allowed))
	// Dry-run and approval posture do not dispatch.
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, selectedEffect{"ctx-dry", "set-label", "issue.label.write", "issue:7", "set-label", "effect-7", "dry", true, true}, allowed))
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, selectedEffect{"ctx-approval", "set-label", "issue.label.write", "issue:8", "set-label", "effect-8", "wait", false, false}, allowed))
	// Cancellation before admission never dispatches.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	report.Receipts = append(report.Receipts, executeSelectedEffect(cancelled, coord2, state, selectedEffect{"ctx-cancel-open", "set-label", "issue.label.write", "issue:9", "set-label", "effect-9", "never", true, false}, allowed))
	// Failure before write is visible and is not recorded as success.
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, selectedEffect{"ctx-fail", "set-label", "issue.label.write", "issue:10", "set-label", "effect-10", "fixture-fail-before-write", true, false}, allowed))
	// A dispatched write whose readback context is cancelled is UNKNOWN, not rollback. A later replay/readback confirms without reapply.
	postCtx, postCancel := context.WithCancel(context.Background())
	unknownIn := selectedEffect{"ctx-post-cancel", "set-label", "issue.label.write", "issue:11", "set-label", "effect-11", "landed", true, false}
	unknown := effectReceipt{ContextID: unknownIn.ContextID, Resource: unknownIn.Resource, IdempotencyKey: unknownIn.IdempotencyKey, Dispatched: true}
	beforeUnknown := state.calls.Load()
	out, uerr := coord2.Run(postCtx, effectIntent(unknownIn), allowed, func() (string, error) { v, e := state.apply(unknownIn)(); postCancel(); return v, e }, func(rc context.Context, ei microagent.EffectIntent, result string) error {
		return state.read(rc, ei, result)
	})
	unknown.Applied = state.calls.Load() > beforeUnknown
	unknown.Replayed = out.Replayed
	if uerr == nil {
		return errors.New("post-dispatch cancellation unexpectedly verified")
	}
	unknown.Status = "unknown_pending_readback"
	unknown.Reason = "dispatched_effect_not_yet_verified"
	report.Receipts = append(report.Receipts, unknown)
	confirm := unknownIn
	confirm.ContextID = "ctx-post-readback"
	later := executeSelectedEffect(context.Background(), coord2, state, confirm, allowed)
	if later.Status == "replayed_confirmed" {
		report.UnknownLaterConfirmed++
	}
	report.Receipts = append(report.Receipts, later)
	// Deterministic resource conflict while the first holder is inside apply.
	started := make(chan struct{})
	release := make(chan struct{})
	held := selectedEffect{"ctx-hold", "set-label", "issue.label.write", "issue:12", "set-label", "effect-12", "held", true, false}
	var first effectReceipt
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		first = effectReceipt{ContextID: held.ContextID, Resource: held.Resource, IdempotencyKey: held.IdempotencyKey, Dispatched: true}
		out, e := coord2.Run(context.Background(), effectIntent(held), allowed, func() (string, error) { close(started); <-release; return state.apply(held)() }, func(rc context.Context, ei microagent.EffectIntent, result string) error {
			return state.read(rc, ei, result)
		})
		first.Applied = e == nil
		first.Replayed = out.Replayed
		first.ReadbackVerified = out.Verified
		first.ObservedValue = out.Result
		if e == nil {
			first.Status = "confirmed"
		} else {
			first.Status = "failed"
		}
	}()
	<-started
	contender := held
	contender.ContextID = "ctx-contender"
	report.Receipts = append(report.Receipts, executeSelectedEffect(context.Background(), coord2, state, contender, allowed))
	close(release)
	wg.Wait()
	report.Receipts = append(report.Receipts, first)
	// Correlated unauthorized routing trips a breaker; remaining selected writes stay visibly not-run.
	denials := 0
	for i := 0; i < 5; i++ {
		in := selectedEffect{ContextID: fmt.Sprintf("ctx-bad-%d", i), Stage: "set-label", Capability: "issue.delete", Resource: fmt.Sprintf("issue:%d", 20+i), Operation: "delete", IdempotencyKey: fmt.Sprintf("bad-%d", i), Value: "bad", Approved: true}
		if denials >= 3 {
			report.Receipts = append(report.Receipts, effectReceipt{ContextID: in.ContextID, Resource: in.Resource, IdempotencyKey: in.IdempotencyKey, Status: "not_run", Reason: "circuit_breaker_open"})
			continue
		}
		r := executeSelectedEffect(context.Background(), coord2, state, in, allowed)
		if r.Status == "denied" {
			denials++
		}
		report.Receipts = append(report.Receipts, r)
	}
	report.Selected = len(report.Receipts)
	for _, r := range report.Receipts {
		switch r.Status {
		case "confirmed":
			report.Confirmed++
		case "replayed_confirmed":
			report.ReplayedConfirmed++
		case "denied":
			report.Denied++
		case "conflict":
			report.Conflicts++
		case "failed":
			report.Failed++
		case "cancelled_before_dispatch":
			report.CancelledBeforeDispatch++
			if r.Dispatched {
				report.CancelledBeforeDispatchCalls++
			}
		case "unknown_pending_readback":
			report.UnknownPendingReadback++
		case "dry_run":
			report.DryRuns++
		case "not_run":
			if r.Reason == "approval_required" {
				report.ApprovalNotRun++
			} else if r.Reason == "circuit_breaker_open" {
				report.BreakerNotRun++
			}
		}
		if (r.Status == "confirmed" || r.Status == "replayed_confirmed") && r.ReadbackVerified {
			report.FoldedConfirmed++
			report.FoldedResources = append(report.FoldedResources, r.Resource)
		}
		if r.Status == "unknown_pending_readback" {
			report.FoldedUnknown++
		}
	}
	report.DispatchAttempts = int(state.calls.Load())
	state.mu.Lock()
	for _, count := range state.applies {
		report.PhysicalWrites += count
	}
	state.mu.Unlock()
	report.DuplicatePhysicalApplies = state.duplicateApplies()
	report.JournalEntries = len(coord.Journal()) + len(coord2.Journal())
	sort.Strings(report.FoldedResources)
	report.Notes = []string{"selector emits a declared stage but never receives capability authority", "idempotency guarantees replay of recorded success; it does not prove a crashed remote write was absent", "post-dispatch cancellation is unknown until independent readback and is never called rollback", "only independently read-back confirmed receipts enter the fold", "dry-run, approval, resource conflict, partial failure, denial, and breaker states remain explicit"}
	if err := verifyEffectBatch(report); err != nil {
		return fmt.Errorf("%w: %+v", err, report)
	}
	b, _ := json.MarshalIndent(report, "", "  ")
	if output != "" {
		if err = os.WriteFile(output, append(b, '\n'), 0644); err != nil {
			return err
		}
	}
	fmt.Println(string(b))
	return nil
}
func verifyEffectBatch(r effectBatchReport) error {
	if r.Schema != effectBatchSchema {
		return errors.New("schema mismatch")
	}
	if r.Confirmed < 6 || r.ReplayedConfirmed < 3 {
		return errors.New("confirmed/replay witness missing")
	}
	if r.Denied < 5 || r.Conflicts != 1 || r.Failed != 1 {
		return errors.New("denial/conflict/failure witness missing")
	}
	if r.CancelledBeforeDispatch != 1 || r.CancelledBeforeDispatchCalls != 0 {
		return errors.New("pre-dispatch cancellation escaped")
	}
	if r.UnknownPendingReadback != 1 || r.UnknownLaterConfirmed != 1 {
		return errors.New("post-dispatch ambiguity not resolved")
	}
	if r.ApprovalNotRun != 1 || r.DryRuns != 1 || r.BreakerNotRun != 2 {
		return errors.New("posture/breaker witness missing")
	}
	if r.DispatchAttempts != 8 || r.PhysicalWrites != 7 {
		return errors.New("dispatch/write accounting mismatch")
	}
	if r.DuplicatePhysicalApplies != 0 || r.RestartPhysicalApplies != 0 {
		return errors.New("effect applied more than once")
	}
	if r.FoldedUnknown != 1 {
		return errors.New("unknown effect visibility missing")
	}
	for _, x := range r.Receipts {
		if x.Status == "unknown_pending_readback" && x.ReadbackVerified {
			return errors.New("unknown marked verified")
		}
		if (x.Status == "confirmed" || x.Status == "replayed_confirmed") && !x.ReadbackVerified {
			return errors.New("unverified effect folded")
		}
	}
	return nil
}
func verifyEffectBatchArtifact(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r effectBatchReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyEffectBatch(r)
}
