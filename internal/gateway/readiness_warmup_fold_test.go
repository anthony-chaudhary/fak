package gateway

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/pkg/deploykit/coherence"
)

type bootCall struct {
	prompt    string
	maxTokens int
}

// recordingBootPlanner answers every boot turn and records its prompt and output
// ceiling. err fails the count probe only, so a separate warmup turn still passes.
type recordingBootPlanner struct {
	calls []bootCall
	err   error
}

func (p *recordingBootPlanner) Model() string { return "recording-boot-planner" }

func (p *recordingBootPlanner) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, o := range opts {
		o(&sp)
	}
	call := bootCall{}
	if len(msgs) > 0 {
		call.prompt = msgs[len(msgs)-1].Content
	}
	if sp.MaxTokens != nil {
		call.maxTokens = *sp.MaxTokens
	}
	p.calls = append(p.calls, call)
	if p.err != nil && call.prompt == coherence.CountPrompt(3) {
		return nil, p.err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "1 2 3"}}, nil
}

// With the fold opted in (FAK_WARMUP_SEPARATE_TURN=0) an armed count probe is the
// boot's first decode: READY costs one synthetic completion (#12640).
// fak-test:runtime fast est=50ms lane=default
func TestRunWarmupArmedProbeIsTheOnlyBootDecode(t *testing.T) {
	t.Setenv(warmupSeparateTurnEnv, "0")
	planner := &recordingBootPlanner{}
	srv := &Server{planner: planner}
	srv.ArmWarmupGate()
	srv.ArmCoherenceProbe(3)
	if _, err := srv.RunWarmup(context.Background()); err != nil {
		t.Fatalf("RunWarmup err = %v, want nil", err)
	}
	if len(planner.calls) != 1 || planner.calls[0].prompt != coherence.CountPrompt(3) {
		t.Fatalf("boot planner calls = %+v, want exactly the count probe", planner.calls)
	}
	if _, ok := srv.warmup.ready(); !ok {
		t.Fatal("warmup gate not complete after the probe answered")
	}
	if body := warmupHealthzBody(t, srv); body["ok"] != true {
		t.Fatalf("/healthz ok = %v, want true", body["ok"])
	}
}

// With no probe armed, by default, and with FAK_WARMUP_SEPARATE_TURN true or
// unparsable, the 1-token warmup turn runs ahead of the probe.
// fak-test:runtime fast est=50ms lane=default
func TestRunWarmupSeparateTurnArms(t *testing.T) {
	cases := []struct {
		name      string
		probe     int
		separate  string
		wantCalls int
	}{
		{"unarmed probe", 0, "", 1},
		{"armed probe default", 3, "", 2},
		{"separate turn env", 3, "1", 2},
		{"unparsable env", 3, "maybe", 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(warmupSeparateTurnEnv, tc.separate)
			planner := &recordingBootPlanner{}
			srv := &Server{planner: planner}
			srv.ArmWarmupGate()
			srv.ArmCoherenceProbe(tc.probe)
			if _, err := srv.RunWarmup(context.Background()); err != nil {
				t.Fatalf("RunWarmup err = %v, want nil", err)
			}
			if len(planner.calls) != tc.wantCalls || planner.calls[0].maxTokens != 1 {
				t.Fatalf("boot planner calls = %+v, want %d starting with a 1-token turn", planner.calls, tc.wantCalls)
			}
		})
	}
}

// When the probe is the first decode (fold opted in), its failure reads as the
// warmup forward's did: a typed V4.1 refusal stays backend_refused with its
// forward site. Behind a separate warmup turn, the default, the probe keeps
// coherence_probe_error.
// fak-test:runtime fast est=50ms lane=default
func TestRunWarmupFirstDecodeProbeFailureReason(t *testing.T) {
	refusal := fmt.Errorf("warmup: %w", &model.V41IndexWeightScaleOperationError{
		Layer: 20, Cause: &model.V41ForwardError{Stage: "indexer", Layer: 20, Err: model.ErrV41ForwardStage},
	})
	cases := []struct {
		name       string
		separate   string
		err        error
		wantReason string
		wantSite   bool
	}{
		{"typed refusal on first decode", "0", refusal, WarmupFailureBackendRefused, true},
		{"untyped error on first decode", "0", errors.New("boom"), WarmupFailureBackendError, false},
		{"untyped error behind separate turn", "1", errors.New("boom"), WarmupFailureCoherenceProbe, false},
		{"untyped error behind default turn", "", errors.New("boom"), WarmupFailureCoherenceProbe, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(warmupSeparateTurnEnv, tc.separate)
			srv := &Server{planner: &recordingBootPlanner{err: tc.err}}
			srv.ArmWarmupGate()
			srv.ArmCoherenceProbe(3)
			if _, err := srv.RunWarmup(context.Background()); !errors.Is(err, tc.err) {
				t.Fatalf("RunWarmup err = %v, want errors.Is(%v)", err, tc.err)
			}
			if reason, _, failed := srv.warmup.failure(); !failed || reason != tc.wantReason {
				t.Fatalf("warmup failure = %q (failed=%v), want %q", reason, failed, tc.wantReason)
			}
			if _, ok := srv.warmup.failureSite(); ok != tc.wantSite {
				t.Fatalf("forward site recorded = %v, want %v", ok, tc.wantSite)
			}
		})
	}
}
