package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

const microSelfcheckAgents = 2

type microSelfcheckReceipt struct {
	Schema         string              `json:"schema"`
	ParentTaskID   string              `json:"parent_task_id"`
	Verdict        string              `json:"verdict"`
	Path           string              `json:"path"`
	Kernel         string              `json:"kernel"`
	Agents         int                 `json:"agents"`
	Done           int                 `json:"done"`
	HTTPCount      int64               `json:"http_count"`
	ProviderTokens int64               `json:"provider_tokens"`
	StoppedCount   int                 `json:"stopped"`
	ConcurrencyCap int                 `json:"concurrency_cap"`
	Offline        bool                `json:"offline"`
	Children       []microChildReceipt `json:"children"`
}

type receiptMeter struct {
	inner  agent.Planner
	tokens atomic.Int64
}

func (p *receiptMeter) Model() string { return p.inner.Model() }
func (p *receiptMeter) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	completion, err := p.inner.Complete(ctx, messages, tools, opts...)
	if completion != nil {
		p.tokens.Add(int64(completion.Usage.TotalTokens))
	}
	return completion, err
}

// runMicroSelfcheck is the milestone-zero value-chain witness. It deliberately
// uses the real gateway HTTP handler and the production SessionGateway/scheduler
// seams, but the gateway's deterministic mock engine: no key, network, or GPU.
func runMicroSelfcheck(ctx context.Context) (microSelfcheckReceipt, error) {
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "mock", VDSO: true, Logf: func(string, ...any) {}})
	if err != nil {
		return microSelfcheckReceipt{}, fmt.Errorf("start kernel: %w", err)
	}
	defer srv.Close()

	var requests atomic.Int64
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			requests.Add(1)
		}
		srv.Handler().ServeHTTP(w, r)
	}))
	defer httpServer.Close()

	children, err := foldMicroSelfcheckChildren()
	if err != nil {
		return microSelfcheckReceipt{}, fmt.Errorf("plan child leases: %w", err)
	}
	table := session.NewTable()
	base := &receiptMeter{inner: agent.NewHTTPPlanner(gatewayBaseURL(httpServer.URL), "selfcheck", "")}
	scheduler := microagent.NewScheduler(1)
	defer scheduler.Close()
	planner := microagent.NewSessionGateway(microagent.NewSchedulingGateway(base, scheduler), table)
	host, err := microagent.NewHost(planner, microagent.Config{Workers: microSelfcheckAgents, Queue: microSelfcheckAgents, Sessions: table})
	if err != nil {
		return microSelfcheckReceipt{}, fmt.Errorf("start microagent host: %w", err)
	}
	defer host.Close()

	runners := make(map[string]*microTurnAgent, microSelfcheckAgents)
	for i := 0; i < microSelfcheckAgents; i++ {
		id := fmt.Sprintf("value-%03d", i)
		runner := &microTurnAgent{id: id, turns: 1}
		runners[id] = runner
		if err := host.Spawn(id, runner); err != nil {
			return microSelfcheckReceipt{}, fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	if err := host.Drain(ctx); err != nil {
		return microSelfcheckReceipt{}, fmt.Errorf("drain: %w", err)
	}

	done := 0
	for _, result := range host.Reap() {
		if result.Done && result.Err == nil {
			done++
		}
	}
	retired := 0
	for _, state := range table.Snapshot() {
		if state.Run == session.Stopped && state.Reason == "done" {
			retired++
		}
	}
	for i := range children {
		answer, _, model := runners[children[i].SessionID].snapshot()
		if answer == "" {
			answer = fmt.Sprintf("session=%s model=%s stopped=done", children[i].SessionID, model)
		}
		children[i].EffectDigest, children[i].Witnessed = digestMicroReadback(answer)
		children[i].State = "stopped"
	}
	receipt := microSelfcheckReceipt{
		Schema: "fak-micro-selfcheck/2", ParentTaskID: microParentTaskID,
		Verdict: "PASS", Path: "kernel->session-gateway->scheduler->microagents",
		Kernel: "in-process-http/mock", Agents: microSelfcheckAgents, Done: done,
		HTTPCount: requests.Load(), ProviderTokens: base.tokens.Load(),
		StoppedCount: retired, ConcurrencyCap: 1, Offline: true, Children: children,
	}
	completeChildren := len(children) == microSelfcheckAgents
	for _, child := range children {
		completeChildren = completeChildren && child.LeaseID != "" && child.SessionID != "" && child.State == "stopped" && child.Witnessed && child.EffectDigest != ""
	}
	if done != microSelfcheckAgents || receipt.HTTPCount != microSelfcheckAgents || retired != microSelfcheckAgents || receipt.ProviderTokens <= 0 || !completeChildren {
		receipt.Verdict = "FAIL"
		return receipt, fmt.Errorf("incomplete value chain: done=%d requests=%d retired=%d tokens=%d", done, receipt.HTTPCount, retired, receipt.ProviderTokens)
	}
	return receipt, nil
}

func cmdMicroSelfcheck(jsonOut bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	receipt, err := runMicroSelfcheck(ctx)
	if jsonOut {
		encoded, marshalErr := json.Marshal(receipt)
		if marshalErr != nil {
			return marshalErr
		}
		fmt.Println(string(encoded))
	} else {
		fmt.Printf("%s micro value chain: %s; agents=%d/%d requests=%d sessions=%d tokens=%d offline=%v\n",
			receipt.Verdict, receipt.Path, receipt.Done, receipt.Agents, receipt.HTTPCount,
			receipt.StoppedCount, receipt.ProviderTokens, receipt.Offline)
	}
	return err
}
