package gateway

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deploykit/coherence"
)

// countProbePlanner answers the warmup turn with "ok" and the known-answer count
// probe with a scripted answer, recording the probe's sampling parameters.
type countProbePlanner struct {
	answer    string
	err       error
	probeTemp *float64
	probes    int
}

func (p *countProbePlanner) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if len(msgs) == 1 && strings.HasPrefix(msgs[0].Content, "Count from 1 to ") {
		p.probes++
		var sp agent.SampleParams
		for _, o := range opts {
			o(&sp)
		}
		p.probeTemp = sp.Temperature
		if p.err != nil {
			return nil, p.err
		}
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.answer}}, nil
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p *countProbePlanner) Model() string { return "count-probe-test" }

// TestRunWarmupCoherenceProbeGatesReadiness is the served-/healthz witness for the
// binary/SPIR-V skew defect: fluent-looking noise that passes every shape
// heuristic still flips readiness off once the known-answer probe is armed, a
// correct count stays ready, and the probe decodes greedily.
// fak-test:runtime fast est=1s lane=default
func TestRunWarmupCoherenceProbeGatesReadiness(t *testing.T) {
	cases := []struct {
		name     string
		answer   string
		wantOK   bool
		wantKind degenerateKind
	}{
		{"correct count", "1 2 3 4 5 6 7 8 9 10", true, decodeCoherent},
		{"skew noise passes shape heuristics", "ledger ocean window sparrow harbor", false, decodeSequenceMissing},
		{"degenerate shape keeps specific kind", "!!!!!!!!", false, decodePunctuation},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyDecode(tc.answer); tc.wantKind == decodeSequenceMissing && got != decodeCoherent {
				t.Fatalf("fixture must pass the shape heuristics, classifyDecode = %q", got)
			}
			planner := &countProbePlanner{answer: tc.answer}
			srv := &Server{planner: planner}
			srv.ArmWarmupGate()
			srv.ArmCoherenceProbe(coherence.DefaultCount)
			if _, err := srv.RunWarmup(context.Background()); err != nil {
				t.Fatalf("RunWarmup err = %v, want nil", err)
			}
			if planner.probes != 1 {
				t.Fatalf("count probe ran %d times, want 1", planner.probes)
			}
			if planner.probeTemp == nil || *planner.probeTemp != 0 {
				t.Fatalf("count probe temperature = %v, want explicit 0 (greedy)", planner.probeTemp)
			}
			if srv.warmup.pending() {
				t.Fatal("warmup gate still pending after a probe that answered")
			}
			code, body := healthzStatus(t, srv)
			if (body["ok"] == true) != tc.wantOK {
				t.Fatalf("/healthz ok = %v (status %d), want %v", body["ok"], code, tc.wantOK)
			}
			if tc.wantOK {
				return
			}
			if code != http.StatusServiceUnavailable {
				t.Fatalf("/healthz status = %d, want 503", code)
			}
			dd, _ := body["degenerate_decode"].(map[string]any)
			if dd == nil || dd["kind"] != string(tc.wantKind) {
				t.Fatalf("degenerate_decode = %v, want kind %q", body["degenerate_decode"], tc.wantKind)
			}
		})
	}
}

// TestRunWarmupCoherenceProbeErrorHoldsReadiness pins that a probe that cannot
// answer (planner error) leaves the warmup gate pending rather than advertising a
// ready serve whose coherence is unknown.
// fak-test:runtime fast est=1s lane=default
func TestRunWarmupCoherenceProbeErrorHoldsReadiness(t *testing.T) {
	sentinel := errors.New("backend refused")
	srv := &Server{planner: &countProbePlanner{err: sentinel}}
	srv.ArmWarmupGate()
	srv.ArmCoherenceProbe(coherence.DefaultCount)
	if _, err := srv.RunWarmup(context.Background()); !errors.Is(err, sentinel) {
		t.Fatalf("RunWarmup err = %v, want errors.Is(%v)", err, sentinel)
	}
	if !srv.warmup.pending() {
		t.Fatal("warmup gate released after a failed coherence probe; want pending")
	}
}

// TestRunWarmupUnarmedCoherenceProbeIsSilent pins default-silence: a serve that
// never arms the probe issues no count turn and readiness is unaffected.
// fak-test:runtime fast est=1s lane=default
func TestRunWarmupUnarmedCoherenceProbeIsSilent(t *testing.T) {
	planner := &countProbePlanner{answer: "noise"}
	srv := &Server{planner: planner}
	srv.ArmWarmupGate()
	if _, err := srv.RunWarmup(context.Background()); err != nil {
		t.Fatalf("RunWarmup err = %v, want nil", err)
	}
	if planner.probes != 0 {
		t.Fatalf("unarmed serve ran %d count probes, want 0", planner.probes)
	}
	if body := warmupHealthzBody(t, srv); body["ok"] != true {
		t.Fatalf("unarmed serve /healthz ok = %v, want true", body["ok"])
	}
}
