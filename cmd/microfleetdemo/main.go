package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const (
	agents        = 24
	turnsPerAgent = 12
	contextCap    = 180
)

type report struct {
	Schema                   string         `json:"schema"`
	Agents                   int            `json:"agents"`
	Turns                    int            `json:"turns"`
	NaiveContextTokenTurns   int            `json:"naive_context_token_turns"`
	FakContextTokenTurns     int            `json:"fak_context_token_turns"`
	ContextTokenTurnsAvoided int            `json:"context_token_turns_avoided"`
	ContextReductionPct      float64        `json:"context_reduction_pct"`
	Compactions              int            `json:"compactions"`
	NaiveContextBytes        int            `json:"naive_context_bytes"`
	DescriptorBytes          int            `json:"descriptor_bytes"`
	DescriptorReductionPct   float64        `json:"descriptor_reduction_pct"`
	PeakResidentAgents       int            `json:"peak_resident_agents"`
	ParkedAgents             int            `json:"parked_agents"`
	ParkedBytes              int            `json:"parked_bytes"`
	ScheduledByTenant        map[string]int `json:"scheduled_by_tenant"`
	ToolAllowed              int            `json:"tool_allowed"`
	ToolDenied               int            `json:"tool_denied"`
	DeniedNeverDispatched    bool           `json:"denied_never_dispatched"`
	EgressAllowed            int            `json:"egress_allowed"`
	EgressDenied             int            `json:"egress_denied"`
	Checks                   []string       `json:"checks"`
}

type memoryAgent struct{ ctx *microagent.ManagedContext }

func (*memoryAgent) Step(context.Context, microagent.Gateway) (bool, error) { return true, nil }
func (a *memoryAgent) Freeze() ([]byte, error)                              { return a.ctx.Encode() }
func (a *memoryAgent) Thaw(b []byte) error                                  { return a.ctx.Decode(b) }

type countingBackend struct{ dispatched int }

func (b *countingBackend) Name() string { return "demo-memory" }
func (b *countingBackend) Dispatch(_ context.Context, _ microagent.ToolAction) (microagent.ToolResult, error) {
	b.dispatched++
	return microagent.ToolResult{Ran: true, ExitCode: 0, Stdout: []byte("ticket-found")}, nil
}

func main() {
	fs := flag.NewFlagSet("microfleetdemo", flag.ExitOnError)
	selfcheck := fs.Bool("selfcheck", false, "assert every native fak benefit shown by the demo")
	jsonOut := fs.Bool("json", false, "emit the proof artifact as JSON")
	_ = fs.Parse(os.Args[1:])
	r, err := run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "microfleetdemo:", err)
		os.Exit(1)
	}
	if *selfcheck {
		if err := check(r); err != nil {
			fmt.Fprintln(os.Stderr, "microfleetdemo -selfcheck: FAIL:", err)
			os.Exit(1)
		}
		fmt.Printf("microfleetdemo -selfcheck: PASS (%d agents - %.1f%% resident-context reduction - %d denied actions never dispatched - %d parked)\n", r.Agents, r.ContextReductionPct, r.ToolDenied+r.EgressDenied, r.ParkedAgents)
		return
	}
	if *jsonOut {
		b, _ := json.MarshalIndent(r, "", "  ")
		fmt.Println(string(b))
		return
	}
	render(os.Stdout, r)
}

func run() (report, error) {
	r := report{Schema: "fak-microfleet-demo/1", Agents: agents, Turns: agents * turnsPerAgent, ScheduledByTenant: map[string]int{}}
	base := strings.Repeat("shared customer-support handbook: refunds require approval; search is read-only. ", 90)
	baseTokens := estimate(base)
	q, err := microagent.NewTenantQueue([]microagent.TenantEnvelope{
		{Tenant: "interactive", Weight: 2, MaxQueued: agents, MaxConcurrent: 4, MaxSpendMicros: 1_000_000},
		{Tenant: "batch", Weight: 1, MaxQueued: agents, MaxConcurrent: 4, MaxSpendMicros: 1_000_000},
	})
	if err != nil {
		return r, err
	}
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < agents; i++ {
		tenant := "batch"
		interactive := false
		if i%2 == 0 {
			tenant, interactive = "interactive", true
		}
		if err := q.Submit(microagent.TenantTask{ID: fmt.Sprintf("agent-%02d", i), Tenant: tenant, CostMicros: 10, Interactive: interactive, EnqueuedAt: now}); err != nil {
			return r, err
		}
	}
	for {
		task, ok := q.Next(now)
		if !ok {
			break
		}
		r.ScheduledByTenant[task.Tenant]++
	}

	dir := filepath.Join(os.TempDir(), fmt.Sprintf("fak-microfleet-%d", os.Getpid()))
	store, err := microagent.NewHibernationStore(dir)
	if err != nil {
		return r, err
	}
	defer os.RemoveAll(dir)
	cap := microagent.NewResidentCap(4)
	for i := 0; i < agents; i++ {
		delta := fmt.Sprintf("resolve ticket %02d using search_kb; never refund directly", i)
		d := microagent.Descriptor{Schema: microagent.DescriptorSchema, ID: fmt.Sprintf("agent-%02d", i), BaseID: "support-handbook-v1", TaskDelta: delta, Tools: []string{"search_kb"}, Budget: microagent.DescriptorBudget{MaxTurns: turnsPerAgent, MaxOutputTokens: 96}, OutputContract: microagent.OutputContract{Kind: "nonempty"}}
		dsz, err := microagent.DescriptorSize(d)
		if err != nil {
			return r, err
		}
		r.DescriptorBytes += dsz
		r.NaiveContextBytes += len(base) + len(delta)
		ctx := microagent.NewManagedContext(contextCap)
		for turn := 0; turn < turnsPerAgent; turn++ {
			content := fmt.Sprintf("turn %02d ticket %02d evidence %s", turn, i, strings.Repeat("fact ", 28))
			ctx.Append("assistant", content, microagent.ArtifactPointer{Kind: "ticket", URI: fmt.Sprintf("mem://ticket/%02d/%02d", i, turn)})
			r.NaiveContextTokenTurns += baseTokens + estimate(delta) + estimate(content)
			r.FakContextTokenTurns += ctx.Tokens()
		}
		m := &memoryAgent{ctx: ctx}
		admitted := cap.Admit()
		if i < cap.Limit() && !admitted {
			return r, errors.New("resident cap refused an in-band agent")
		}
		if cap.Resident() > r.PeakResidentAgents {
			r.PeakResidentAgents = cap.Resident()
		}
		r.Compactions += ctx.Compactions()
		if !admitted {
			n, err := store.Park(d.ID, m)
			if err != nil {
				return r, err
			}
			r.ParkedAgents++
			r.ParkedBytes += n
		}
	}

	policy := adjudicator.Policy{Allow: map[string]bool{"search_kb": true}, Deny: map[string]abi.ReasonCode{"refund_payment": abi.ReasonPolicyBlock}}
	floor := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(policy)}))
	backend := &countingBackend{}
	exec, err := microagent.NewToolExecBackend(floor, backend)
	if err != nil {
		return r, err
	}
	if _, err := exec.Run(context.Background(), microagent.ToolAction{Tool: "search_kb"}); err != nil {
		return r, err
	}
	r.ToolAllowed++
	if _, err := exec.Run(context.Background(), microagent.ToolAction{Tool: "refund_payment"}); errors.Is(err, microagent.ErrActionDenied) {
		r.ToolDenied++
	} else {
		return r, fmt.Errorf("refund_payment: got %v, want denied", err)
	}
	r.DeniedNeverDispatched = backend.dispatched == 1
	egress, err := microagent.NewEgressPolicy(microagent.EgressTrustUntrusted, "api.example.com")
	if err != nil {
		return r, err
	}
	if egress.Decide(context.Background(), "https://api.example.com/tickets", &abi.ToolCall{Tool: "search_kb"}, floor).Kind == abi.VerdictAllow {
		r.EgressAllowed++
	}
	if egress.Decide(context.Background(), "https://outside.example.net/data", &abi.ToolCall{Tool: "search_kb"}, floor).Kind == abi.VerdictDeny {
		r.EgressDenied++
	}
	for cap.Resident() > 0 {
		cap.Release()
	}
	r.ContextTokenTurnsAvoided = r.NaiveContextTokenTurns - r.FakContextTokenTurns
	r.ContextReductionPct = percent(r.ContextTokenTurnsAvoided, r.NaiveContextTokenTurns)
	r.DescriptorReductionPct = percent(r.NaiveContextBytes-r.DescriptorBytes, r.NaiveContextBytes)
	r.Checks = []string{"shared base sent by reference", "bounded managed contexts compacted", "weighted tenants scheduled", "resident fleet hibernated", "tool calls kernel-adjudicated", "per-agent egress default-denied"}
	return r, nil
}

func check(r report) error {
	var bad []string
	if r.Agents != agents || r.Turns != agents*turnsPerAgent {
		bad = append(bad, "fleet accounting")
	}
	if r.ContextReductionPct < 80 || r.DescriptorReductionPct < 80 {
		bad = append(bad, "resident-context reduction below 80%")
	}
	if r.PeakResidentAgents > 4 || r.ParkedAgents != agents-4 {
		bad = append(bad, "residency bound")
	}
	if r.ScheduledByTenant["interactive"] != agents/2 || r.ScheduledByTenant["batch"] != agents/2 {
		bad = append(bad, "fair scheduling")
	}
	if r.ToolAllowed != 1 || r.ToolDenied != 1 || !r.DeniedNeverDispatched {
		bad = append(bad, "tool floor")
	}
	if r.EgressAllowed != 1 || r.EgressDenied != 1 {
		bad = append(bad, "egress floor")
	}
	if len(bad) > 0 {
		return errors.New(strings.Join(bad, ", "))
	}
	return nil
}

func estimate(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}
func percent(part, whole int) float64 {
	if whole == 0 {
		return 0
	}
	return 100 * float64(part) / float64(whole)
}
func render(w io.Writer, r report) {
	keys := make([]string, 0, len(r.ScheduledByTenant))
	for k := range r.ScheduledByTenant {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Fprintln(w, "FAK MICROFLEET — one process, many tiny agents, every native gate")
	fmt.Fprintf(w, "fleet       %d agents × %d turns; peak resident %d; parked %d (%d bytes)\n", r.Agents, r.Turns/r.Agents, r.PeakResidentAgents, r.ParkedAgents, r.ParkedBytes)
	fmt.Fprintf(w, "context     %d -> %d resident token-turns (%.1f%% avoided); %d compactions; descriptors %.1f%% smaller than copied bases\n", r.NaiveContextTokenTurns, r.FakContextTokenTurns, r.ContextReductionPct, r.Compactions, r.DescriptorReductionPct)
	for _, k := range keys {
		fmt.Fprintf(w, "scheduler   %-11s %d tasks\n", k, r.ScheduledByTenant[k])
	}
	fmt.Fprintf(w, "tool floor  %d allow · %d deny · denied dispatches = %t\n", r.ToolAllowed, r.ToolDenied, !r.DeniedNeverDispatched)
	fmt.Fprintf(w, "egress      %d allow · %d deny (off-list destination refused)\n", r.EgressAllowed, r.EgressDenied)
	fmt.Fprintln(w, "VERDICT     fak turns copied context + unbounded residents + best-effort safety into bounded references + fair scheduling + pre-exec denial")
	fmt.Fprintln(w, "PROOF       go run ./cmd/microfleetdemo -selfcheck")
}
