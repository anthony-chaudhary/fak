package toolplugin

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

type testFloor struct {
	denyTool string
	seen     []Proposal
}

func (f *testFloor) Adjudicate(_ context.Context, p Proposal) Decision {
	f.seen = append(f.seen, p)
	if p.Tool == f.denyTool {
		return Decision{Action: ActionDeny, Reason: "KERNEL_FLOOR"}
	}
	return Decision{Action: ActionNarrow, Reason: "FLOOR_ALLOW"}
}

type testExecutor struct{ calls []Proposal }

func (e *testExecutor) Execute(_ context.Context, p Proposal) (json.RawMessage, error) {
	e.calls = append(e.calls, p)
	return json.RawMessage(`{"ok":true}`), nil
}

type testPlugin struct {
	profile Profile
	apply   func(context.Context, Input) (Decision, error)
	calls   int
}

func (p *testPlugin) Profile() Profile { return p.profile }
func (p *testPlugin) Apply(ctx context.Context, in Input) (Decision, error) {
	p.calls++
	return p.apply(ctx, in)
}

func pinned(id string, stage Stage, precedence int) Profile {
	return Profile{ID: id, Version: "1.0.0", Digest: "sha256:0123456789abcdef", Stage: stage, Timeout: Duration(time.Second), Fallback: ActionDeny, Precedence: precedence}
}

func TestHostTraceTransformReadjudicateExecuteResultAdmit(t *testing.T) {
	floor := &testFloor{}
	executor := &testExecutor{}
	transformer := &testPlugin{profile: pinned("canonical-json", StageCanonicalize, 10), apply: func(_ context.Context, in Input) (Decision, error) {
		return Decision{Action: ActionTransform, Proposal: &Proposal{Tool: in.Proposal.Tool, Args: json.RawMessage(`{"amount":7}`)}, Reason: "CANONICALIZED"}, nil
	}}
	admitter := &testPlugin{profile: pinned("result-shape", StageResultAdmit, 20), apply: func(_ context.Context, in Input) (Decision, error) {
		if string(in.Result) != `{"ok":true}` {
			return Decision{Action: ActionQuarantine, Reason: "BAD_RESULT"}, nil
		}
		return Decision{Action: ActionNarrow, Reason: "RESULT_ADMITTED"}, nil
	}}
	host := Host{Floor: floor, Executor: executor, Plugins: []Plugin{admitter, transformer}}
	out := host.Run(context.Background(), Proposal{Tool: "refund", Args: json.RawMessage(`{"amount":"7"}`)}, PreferenceLayers{})
	if out.Decision.Action != ActionNarrow || out.Decision.Reason != "ADMITTED" {
		t.Fatalf("outcome = %+v", out)
	}
	if transformer.calls != 1 || admitter.calls != 1 {
		t.Fatalf("plugin calls transformer=%d admitter=%d", transformer.calls, admitter.calls)
	}
	if len(floor.seen) != 1 || string(floor.seen[0].Args) != `{"amount":7}` {
		t.Fatalf("floor did not re-adjudicate transformed proposal: %+v", floor.seen)
	}
	if len(executor.calls) != 1 || string(executor.calls[0].Args) != `{"amount":7}` {
		t.Fatalf("executor did not receive adjudicated transform: %+v", executor.calls)
	}
	stages := make([]string, len(out.Trace))
	for i, event := range out.Trace {
		stages[i] = event.Stage
	}
	if want := []string{"canonicalize", "floor", "execute", "result_admit"}; !reflect.DeepEqual(stages, want) {
		t.Fatalf("trace stages = %v, want %v", stages, want)
	}
}

func TestKernelDenialCannotBeWidenedByPlugin(t *testing.T) {
	floor := &testFloor{denyTool: "destroy"}
	executor := &testExecutor{}
	wouldAllow := &testPlugin{profile: pinned("user-allow", StageAdjudicate, 1), apply: func(context.Context, Input) (Decision, error) {
		return Decision{Action: ActionNarrow, Reason: "USER_ALLOW"}, nil
	}}
	out := (Host{Floor: floor, Executor: executor, Plugins: []Plugin{wouldAllow}}).Run(context.Background(), Proposal{Tool: "destroy", Args: json.RawMessage(`{}`)}, PreferenceLayers{})
	if out.Decision.Reason != "KERNEL_FLOOR" || wouldAllow.calls != 0 || len(executor.calls) != 0 {
		t.Fatalf("floor was widened: outcome=%+v plugin_calls=%d exec=%d", out, wouldAllow.calls, len(executor.calls))
	}
}

func TestTransformLoopAndTransformAuthorityWideningRefuse(t *testing.T) {
	original := Proposal{Tool: "search", Args: json.RawMessage(`{"q":"x"}`)}
	loop := &testPlugin{profile: pinned("loop", StageCanonicalize, 1), apply: func(_ context.Context, in Input) (Decision, error) {
		return Decision{Action: ActionTransform, Proposal: &in.Proposal}, nil
	}}
	out := (Host{Floor: &testFloor{}, Executor: &testExecutor{}, Plugins: []Plugin{loop}}).Run(context.Background(), original, PreferenceLayers{})
	if out.Decision.Reason != "TRANSFORM_LOOP" {
		t.Fatalf("loop outcome = %+v", out)
	}

	widen := &testPlugin{profile: pinned("widen", StageCanonicalize, 1), apply: func(_ context.Context, in Input) (Decision, error) {
		p := Proposal{Tool: "destroy", Args: in.Proposal.Args}
		return Decision{Action: ActionNarrow, Proposal: &p, Reason: "ALLOW_MORE"}, nil
	}}
	out = (Host{Floor: &testFloor{}, Executor: &testExecutor{}, Plugins: []Plugin{widen}}).Run(context.Background(), original, PreferenceLayers{})
	if out.Decision.Reason != "AUTHORITY_WIDENING" {
		t.Fatalf("widen outcome = %+v", out)
	}
}

func TestPreferenceLayersAreFlexibleButMonotone(t *testing.T) {
	resolved := ResolvePreferences(PreferenceLayers{
		Kernel:       Preference{Disclosure: "minimal", TransformMode: "preview"},
		Organization: Preference{RequireWitness: true, WitnessRoute: "org-auditor", WaitMode: "wait"},
		Project:      Preference{ResumeNotification: "project-channel"},
		User:         Preference{RequireWitness: false, WitnessRoute: "my-auditor", TransformMode: "auto"},
		Call:         Preference{WaitMode: "background", Timeout: "45s"},
	})
	if !resolved.RequireWitness {
		t.Fatal("user false removed organization mandatory witness")
	}
	if resolved.WitnessRoute != "my-auditor" || resolved.TransformMode != "auto" || resolved.WaitMode != "background" || resolved.Timeout != "45s" || resolved.ResumeNotification != "project-channel" || resolved.Disclosure != "minimal" {
		t.Fatalf("resolved = %+v", resolved)
	}

	added := ResolvePreferences(PreferenceLayers{User: Preference{RequireWitness: true}})
	if !added.RequireWitness || !strings.Contains(added.Sources["require_witness"], "user") {
		t.Fatalf("user could not add witness: %+v", added)
	}
}

func TestWitnessRequirementMustBeSatisfiedBeforeExecution(t *testing.T) {
	executor := &testExecutor{}
	host := Host{Floor: &testFloor{}, Executor: executor}
	out := host.Run(context.Background(), Proposal{Tool: "read", Args: json.RawMessage(`{}`)}, PreferenceLayers{User: Preference{RequireWitness: true}})
	if out.Decision.Action != ActionRequireWitness || len(executor.calls) != 0 {
		t.Fatalf("unwitnessed call executed: %+v", out)
	}

	request := &testPlugin{profile: pinned("local-witness-request", StageWitnessRequest, 1), apply: func(context.Context, Input) (Decision, error) {
		return Decision{Action: ActionDefer, Attestation: json.RawMessage(`{"signed":true}`), Reason: "WITNESS_RETURNED"}, nil
	}}
	verify := &testPlugin{profile: pinned("attestation-verifier", StageAttest, 2), apply: func(_ context.Context, in Input) (Decision, error) {
		if string(in.Attestation) != `{"signed":true}` {
			return Decision{Action: ActionDeny, Reason: "BAD_ATTESTATION"}, nil
		}
		return Decision{Action: ActionNarrow, Reason: "ATTESTED"}, nil
	}}
	host.Plugins = []Plugin{request, verify}
	out = host.Run(context.Background(), Proposal{Tool: "read", Args: json.RawMessage(`{}`)}, PreferenceLayers{Organization: Preference{RequireWitness: true}})
	if out.Decision.Reason != "ADMITTED" || len(executor.calls) != 1 {
		t.Fatalf("witnessed call did not execute: %+v", out)
	}
}

func TestInvalidPluginProfileFailsClosed(t *testing.T) {
	bad := &testPlugin{profile: Profile{ID: "egress", Version: "1", Digest: "sha256:x", Stage: StageAdjudicate, DataEgress: true, Fallback: ActionDefer}, apply: func(context.Context, Input) (Decision, error) { return Decision{Action: ActionDefer}, nil }}
	out := (Host{Floor: &testFloor{}, Executor: &testExecutor{}, Plugins: []Plugin{bad}}).Run(context.Background(), Proposal{Tool: "read", Args: json.RawMessage(`{}`)}, PreferenceLayers{})
	if out.Decision.Action != ActionDeny || !strings.Contains(out.Decision.Reason, "INVALID_PROFILE") {
		t.Fatalf("invalid profile did not fail closed: %+v", out)
	}
}

func TestObserverCannotChangeOutcomeOrExecute(t *testing.T) {
	executor := &testExecutor{}
	observer := &testPlugin{profile: pinned("notify", StageObserve, 1), apply: func(context.Context, Input) (Decision, error) {
		return Decision{Action: ActionDeny, Reason: "OBSERVER_TRIED_TO_DENY"}, nil
	}}
	out := (Host{Floor: &testFloor{}, Executor: executor, Plugins: []Plugin{observer}}).Run(context.Background(), Proposal{Tool: "read", Args: json.RawMessage(`{}`)}, PreferenceLayers{})
	if out.Decision.Reason != "ADMITTED" || len(executor.calls) != 1 || observer.calls != 1 {
		t.Fatalf("observer gained authority: %+v calls=%d", out, observer.calls)
	}
	if out.Trace[len(out.Trace)-1].Reason != "OBSERVER_DECISION_IGNORED" {
		t.Fatalf("observer authority attempt not audited: %+v", out.Trace)
	}
}
