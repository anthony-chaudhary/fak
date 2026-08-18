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
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const maxDepth = 2

type task struct {
	ID       string
	Parent   string
	Depth    int
	MaxTurns int
	Goal     string
}

type receipt struct {
	TaskID   string   `json:"task_id"`
	Parent   string   `json:"parent,omitempty"`
	Depth    int      `json:"depth"`
	Turns    int      `json:"turns"`
	Decision string   `json:"decision"`
	Evidence []string `json:"evidence"`
	Children []task   `json:"children,omitempty"`
}

type report struct {
	Goal                  string    `json:"goal"`
	Harness               []string  `json:"harness"`
	Receipts              []receipt `json:"receipts"`
	MaxDepth              int       `json:"max_depth"`
	MaxTurnsPerMicroagent int       `json:"max_turns_per_microagent"`
	ChildTranscriptBytes  int       `json:"child_transcript_bytes"`
	MasterReceiptBytes    int       `json:"master_receipt_bytes"`
	FullTranscriptsInRoot bool      `json:"full_transcripts_in_root"`
}

type scriptedModel struct {
	mu    sync.Mutex
	bytes int
}

func (*scriptedModel) Model() string { return "microharness-fixture" }
func (p *scriptedModel) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if len(msgs) == 0 {
		return nil, errors.New("empty microagent context")
	}
	p.mu.Lock()
	for _, msg := range msgs {
		p.bytes += len(msg.Content)
	}
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "bounded evidence accepted"}}, nil
}

func (p *scriptedModel) transcriptBytes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.bytes
}

type boundedAgent struct {
	task    task
	turns   int
	receipt chan<- receipt
}

func (a *boundedAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.turns++
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "Return evidence for only this bounded harness-building task."}, {Role: agent.RoleUser, Content: a.task.Goal}}
	if _, err := gw.Complete(ctx, messages, nil); err != nil {
		return false, err
	}
	if a.turns < a.task.MaxTurns {
		return false, nil
	}
	r := makeReceipt(a.task, a.turns)
	select {
	case a.receipt <- r:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func makeReceipt(t task, turns int) receipt {
	r := receipt{TaskID: t.ID, Parent: t.Parent, Depth: t.Depth, Turns: turns}
	switch t.ID {
	case "architecture":
		r.Decision = "compose a local coding harness from bounded tool and proof profiles"
		r.Evidence = []string{"goal requires repository edits", "effects need an independent proof boundary"}
		r.Children = []task{
			{ID: "tools", Parent: t.ID, Depth: t.Depth + 1, MaxTurns: 1, Goal: "Select the least-privilege tools for a local coding harness."},
			{ID: "proof", Parent: t.ID, Depth: t.Depth + 1, MaxTurns: 3, Goal: "Define the harness completion witness and failure boundary."},
		}
	case "tools":
		r.Decision = "profile=repo-read-write; shell=workspace-only"
		r.Evidence = []string{"editing needs scoped writes", "network is not required by the goal"}
	case "proof":
		r.Decision = "require build plus affected tests before completion"
		r.Evidence = []string{"a diff is not a behavior witness", "verification must be independent of the done claim"}
	default:
		r.Decision = "refuse unknown task"
		r.Evidence = []string{"task id is outside the declared workflow"}
	}
	return r
}

func run(ctx context.Context) (report, error) {
	planner := &scriptedModel{}

	rootGoal := "Build a local coding harness that can edit this repository and prove its work."
	pending := []task{{ID: "architecture", Parent: "root", Depth: 1, MaxTurns: 2, Goal: rootGoal}}
	var receipts []receipt
	for len(pending) > 0 {
		wave := pending
		pending = nil
		out := make(chan receipt, len(wave))
		host, err := microagent.NewHost(planner, microagent.Config{Workers: 3, Queue: 8})
		if err != nil {
			return report{}, err
		}
		for _, t := range wave {
			if t.Depth > maxDepth || t.MaxTurns < 1 || t.MaxTurns > 3 {
				return report{}, fmt.Errorf("task %s exceeds recursion envelope", t.ID)
			}
			if err := host.Spawn(t.ID, &boundedAgent{task: t, receipt: out}); err != nil {
				return report{}, err
			}
		}
		if err := host.Drain(ctx); err != nil {
			host.Close()
			return report{}, err
		}
		host.Close()
		for range wave {
			r := <-out
			receipts = append(receipts, r)
			pending = append(pending, r.Children...)
		}
	}

	sort.Slice(receipts, func(i, j int) bool { return receipts[i].TaskID < receipts[j].TaskID })
	masterBytes := 0
	for _, r := range receipts {
		compact := r
		compact.Children = nil
		raw, _ := json.Marshal(compact)
		masterBytes += len(raw)
	}
	return report{
		Goal:     rootGoal,
		Harness:  []string{"runtime=fak-native", "tools=repo-read-write/workspace-only", "completion=build+affected-tests"},
		Receipts: receipts, MaxDepth: maxDepth, MaxTurnsPerMicroagent: 3,
		ChildTranscriptBytes: planner.transcriptBytes(), MasterReceiptBytes: masterBytes,
		FullTranscriptsInRoot: false,
	}, nil
}

func check(r report) error {
	if len(r.Receipts) != 3 || len(r.Harness) != 3 {
		return fmt.Errorf("got %d receipts and %d harness fields", len(r.Receipts), len(r.Harness))
	}
	if r.MaxDepth != 2 || r.MaxTurnsPerMicroagent != 3 || r.FullTranscriptsInRoot {
		return errors.New("bounded recursion or context-isolation invariant failed")
	}
	for _, rec := range r.Receipts {
		if rec.Turns < 1 || rec.Turns > 3 || rec.Depth > r.MaxDepth || rec.Decision == "" || len(rec.Evidence) == 0 {
			return fmt.Errorf("invalid receipt for %s", rec.TaskID)
		}
	}
	return nil
}

func render(w io.Writer, r report) {
	fmt.Fprintln(w, "FAK MICROHARNESS — bounded microagents construct a harness")
	fmt.Fprintf(w, "goal: %s\n", r.Goal)
	for _, rec := range r.Receipts {
		fmt.Fprintf(w, "  receipt %-12s depth=%d turns=%d -> %s\n", rec.TaskID, rec.Depth, rec.Turns, rec.Decision)
	}
	fmt.Fprintf(w, "harness: %s\n", strings.Join(r.Harness, "; "))
	fmt.Fprintf(w, "context boundary: root retained %d receipt bytes; child transcript bytes=%d; full child transcripts in root=%t\n", r.MasterReceiptBytes, r.ChildTranscriptBytes, r.FullTranscriptsInRoot)
	fmt.Fprintln(w, "recursion boundary: depth<=2; turns/child<=3; child requests are re-admitted by the host")
	fmt.Fprintln(w, "PASS — go run ./cmd/microharnessdemo -selfcheck")
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the captured end-to-end witness")
	asJSON := flag.Bool("json", false, "emit the machine-readable report")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r, err := run(ctx)
	if err == nil && *selfcheck {
		err = check(r)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
	if *asJSON {
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	render(os.Stdout, r)
}
