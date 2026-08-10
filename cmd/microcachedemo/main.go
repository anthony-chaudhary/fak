package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	fleetAgents            = 32
	callsPerAgent          = 8
	uniqueQueries          = 4
	generatedTokensPerMiss = 180
)

type result struct {
	Schema                  string  `json:"schema"`
	Agents                  int     `json:"agents"`
	Calls                   int     `json:"calls"`
	UniqueQueries           int     `json:"unique_queries"`
	BaselineUpstreamCalls   int     `json:"baseline_engine_calls"`
	FakUpstreamCalls        int     `json:"fak_engine_calls"`
	VDSOHits                int     `json:"vdso_hits"`
	UpstreamCallsAvoided    int     `json:"engine_calls_avoided"`
	UpstreamReductionPct    float64 `json:"engine_reduction_pct"`
	BaselineGeneratedTokens int     `json:"baseline_generated_tokens"`
	FakGeneratedTokens      int     `json:"fak_generated_tokens"`
	GeneratedTokensAvoided  int     `json:"generated_tokens_avoided"`
	SameAnswer              bool    `json:"same_answer"`
	PolicyDenied            int     `json:"policy_denied"`
	DeniedUpstreamCalls     int     `json:"denied_engine_calls"`
	PrivateAgentAMisses     int     `json:"private_agent_a_engine_calls"`
	PrivateAgentBMisses     int     `json:"private_agent_b_engine_calls"`
	ShareableCrossAgentHit  bool    `json:"shareable_cross_agent_hit"`
	PrincipalIsolation      bool    `json:"principal_isolation"`
}

type reuseObserver struct {
	v         *vdso.VDSO
	completes *atomic.Int64
}

func (e reuseObserver) Emit(ev abi.Event)            { e.completes.Add(1); e.v.Emit(ev) }
func (reuseObserver) Subscriptions() []abi.EventKind { return []abi.EventKind{abi.EvComplete} }

var runSequence atomic.Int64

type fixtureDriver struct{ calls atomic.Int64 }

func (e *fixtureDriver) Caps() []abi.Capability { return nil }
func (e *fixtureDriver) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	payload := []byte(fmt.Sprintf("answer:%s", string(c.Args.Inline)))
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))}}, nil
}

func main() {
	fs := flag.NewFlagSet("microcachedemo", flag.ExitOnError)
	selfcheck := fs.Bool("selfcheck", false, "assert cache reuse, policy, and principal-isolation invariants")
	jsonOut := fs.Bool("json", false, "emit the deterministic proof artifact as JSON")
	_ = fs.Parse(os.Args[1:])
	got, err := run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "microcachedemo:", err)
		os.Exit(1)
	}
	if *selfcheck {
		if err := check(got); err != nil {
			fmt.Fprintln(os.Stderr, "microcachedemo -selfcheck: FAIL:", err)
			os.Exit(1)
		}
		fmt.Printf("microcachedemo -selfcheck: PASS (%d agents - %d/%d calls served locally - %.1f%% engine work avoided - policy and tenant isolation hold)\n", got.Agents, got.VDSOHits, got.Calls, got.UpstreamReductionPct)
		return
	}
	if *jsonOut {
		b, _ := json.MarshalIndent(got, "", "  ")
		fmt.Println(string(b))
		return
	}
	render(os.Stdout, got)
}

func run() (result, error) {
	ctx := context.Background()
	runID := runSequence.Add(1)
	total := fleetAgents * callsPerAgent
	out := result{Schema: "fak-microcache-demo/1", Agents: fleetAgents, Calls: total, UniqueQueries: uniqueQueries, BaselineUpstreamCalls: total, BaselineGeneratedTokens: total * generatedTokensPerMiss}

	engineID := fmt.Sprintf("microcache-%d", os.Getpid())
	engine := &fixtureDriver{}
	abi.RegisterEngine(engineID, engine)
	cache := vdso.New(64)
	var completeEvents atomic.Int64
	cache.RegisterShareable("read_public_policy")
	abi.RegisterFastPath(1, cache)
	abi.RegisterEmitter(reuseObserver{v: cache, completes: &completeEvents})
	policy := adjudicator.Policy{Allow: map[string]bool{"read_public_policy": true, "read_private_record": true}, Deny: map[string]abi.ReasonCode{"refund_payment": abi.ReasonPolicyBlock}}
	k := kernel.New(engineID, kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(policy)}))

	var first string
	same := true
	for agent := 0; agent < fleetAgents; agent++ {
		for turn := 0; turn < callsPerAgent; turn++ {
			q := turn % uniqueQueries
			c := readCall("read_public_policy", fmt.Sprintf(`{"run":%d,"section":%d}`, runID, q), fmt.Sprintf("agent-%02d", agent))
			res, verdict := k.Syscall(ctx, c)
			if verdict.Kind != abi.VerdictAllow || res.Status != abi.StatusOK {
				return out, fmt.Errorf("public query agent=%d turn=%d verdict=%v status=%v", agent, turn, verdict.Kind, res.Status)
			}
			answer, err := resolve(ctx, res.Payload)
			if err != nil {
				return out, err
			}
			if agent == 0 && turn == 0 {
				first = string(answer)
			}
			if q == 0 && string(answer) != first {
				same = false
			}
		}
	}
	out.SameAnswer = same
	if completeEvents.Load() == 0 {
		return out, errors.New("cache observer received zero completion events")
	}
	ctr := k.Counters()
	out.FakUpstreamCalls = int(ctr.EngineCalls)
	out.VDSOHits = int(ctr.VDSOHits)
	out.UpstreamCallsAvoided = out.BaselineUpstreamCalls - out.FakUpstreamCalls
	out.UpstreamReductionPct = pct(out.UpstreamCallsAvoided, out.BaselineUpstreamCalls)
	out.FakGeneratedTokens = out.FakUpstreamCalls * generatedTokensPerMiss
	out.GeneratedTokensAvoided = out.BaselineGeneratedTokens - out.FakGeneratedTokens

	beforeDeny := engine.calls.Load()
	_, deny := k.Syscall(ctx, &abi.ToolCall{Tool: "refund_payment", Args: inline(`{"ticket":7}`), Meta: readHints("agent-00")})
	if deny.Kind == abi.VerdictDeny {
		out.PolicyDenied = 1
	}
	out.DeniedUpstreamCalls = int(engine.calls.Load() - beforeDeny)

	beforeA := engine.calls.Load()
	if _, v := k.Syscall(ctx, readCall("read_private_record", fmt.Sprintf(`{"run":%d,"ticket":9}`, runID), "agent-A")); v.Kind != abi.VerdictAllow {
		return out, errors.New("private agent A first call denied")
	}
	if _, v := k.Syscall(ctx, readCall("read_private_record", fmt.Sprintf(`{"run":%d,"ticket":9}`, runID), "agent-A")); v.Kind != abi.VerdictAllow {
		return out, errors.New("private agent A repeat denied")
	}
	out.PrivateAgentAMisses = int(engine.calls.Load() - beforeA)
	beforeB := engine.calls.Load()
	if _, v := k.Syscall(ctx, readCall("read_private_record", fmt.Sprintf(`{"run":%d,"ticket":9}`, runID), "agent-B")); v.Kind != abi.VerdictAllow {
		return out, errors.New("private agent B call denied")
	}
	out.PrivateAgentBMisses = int(engine.calls.Load() - beforeB)
	out.PrincipalIsolation = out.PrivateAgentAMisses == 1 && out.PrivateAgentBMisses == 1

	beforeShared := engine.calls.Load()
	if _, v := k.Syscall(ctx, readCall("read_public_policy", fmt.Sprintf(`{"run":%d,"section":99}`, runID), "agent-A")); v.Kind != abi.VerdictAllow {
		return out, errors.New("shareable warm denied")
	}
	if _, v := k.Syscall(ctx, readCall("read_public_policy", fmt.Sprintf(`{"run":%d,"section":99}`, runID), "agent-B")); v.Kind != abi.VerdictAllow {
		return out, errors.New("shareable cross-agent read denied")
	}
	out.ShareableCrossAgentHit = engine.calls.Load()-beforeShared == 1
	return out, nil
}

func readCall(tool, args, principal string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: inline(args), Meta: readHints(principal)}
}
func readHints(principal string) map[string]string {
	return map[string]string{"readOnlyHint": "true", "idempotentHint": "true", vdso.MetaPrincipal: principal}
}
func inline(s string) abi.Ref {
	return abi.Ref{Kind: abi.RefInline, Inline: []byte(s), Len: int64(len(s)), Scope: abi.ScopeFleet}
}
func resolve(ctx context.Context, ref abi.Ref) ([]byte, error) {
	if ref.Kind == abi.RefInline {
		return ref.Inline, nil
	}
	if r := abi.ActiveResolver(); r != nil {
		return r.Resolve(ctx, ref)
	}
	return nil, errors.New("no resolver for non-inline result")
}
func pct(part, whole int) float64 {
	if whole == 0 {
		return 0
	}
	return 100 * float64(part) / float64(whole)
}

func check(r result) error {
	var bad []string
	if r.Calls != fleetAgents*callsPerAgent || r.FakUpstreamCalls != uniqueQueries || r.VDSOHits != r.Calls-uniqueQueries {
		bad = append(bad, "fleet cache accounting")
	}
	if r.UpstreamReductionPct < 95 || r.GeneratedTokensAvoided <= 0 || !r.SameAnswer {
		bad = append(bad, "net engine reduction")
	}
	if r.PolicyDenied != 1 || r.DeniedUpstreamCalls != 0 {
		bad = append(bad, "pre-engine policy denial")
	}
	if !r.ShareableCrossAgentHit || !r.PrincipalIsolation {
		bad = append(bad, "shareability/principal isolation")
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return errors.New(fmt.Sprintf("%v", bad))
	}
	return nil
}

func render(w io.Writer, r result) {
	fmt.Fprintln(w, "FAK MICRO-CACHE - one shared kernel turns a swarm into four upstream calls")
	fmt.Fprintf(w, "fleet       %d agents x %d calls = %d identical-work opportunities\n", r.Agents, r.Calls/r.Agents, r.Calls)
	fmt.Fprintf(w, "engine      %d -> %d calls (%d local vDSO hits; %.1f%% upstream work avoided)\n", r.BaselineUpstreamCalls, r.FakUpstreamCalls, r.VDSOHits, r.UpstreamReductionPct)
	fmt.Fprintf(w, "generation  %d -> %d modeled output tokens (%d avoided)\n", r.BaselineGeneratedTokens, r.FakGeneratedTokens, r.GeneratedTokensAvoided)
	fmt.Fprintf(w, "safety      denied tool reached engine %d time(s)\n", r.DeniedUpstreamCalls)
	fmt.Fprintf(w, "tenancy     public cross-agent hit = %t; private A/B engine calls = %d/%d\n", r.ShareableCrossAgentHit, r.PrivateAgentAMisses, r.PrivateAgentBMisses)
	fmt.Fprintln(w, "VERDICT     native fak shares public repeated work fleet-wide, keeps private reads principal-scoped, and blocks unsafe work before compute")
	fmt.Fprintln(w, "PROOF       go run ./cmd/microcachedemo -selfcheck")
}
