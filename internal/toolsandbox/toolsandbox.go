// Package toolsandbox adapts tau3/ToolSandbox-shaped policy-state traces into
// fak adjudication reports.
package toolsandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const (
	SuiteSchema  = "fak.toolsandbox-adapter-suite.v1"
	ReportSchema = "fak.toolsandbox-adapter-report.v1"
)

type Suite struct {
	Schema    string `json:"schema"`
	Benchmark string `json:"benchmark"`
	Model     string `json:"model,omitempty"`
	Tasks     []Task `json:"tasks"`
}

type Task struct {
	ID          string          `json:"id"`
	Domain      string          `json:"domain,omitempty"`
	BudgetTurns int             `json:"budget_turns,omitempty"`
	Milestones  []string        `json:"milestones,omitempty"`
	Policy      policy.Manifest `json:"policy"`
	Calls       []Call          `json:"calls"`
}

type Call struct {
	Turn      int             `json:"turn,omitempty"`
	Tool      string          `json:"tool"`
	Args      json.RawMessage `json:"args,omitempty"`
	Milestone string          `json:"milestone,omitempty"`
	Minefield string          `json:"minefield,omitempty"`
}

type Report struct {
	Schema        string       `json:"schema"`
	GeneratedAt   string       `json:"generated_at"`
	Benchmark     string       `json:"benchmark"`
	Model         string       `json:"model,omitempty"`
	TaskReports   []TaskReport `json:"tasks"`
	Summary       Summary      `json:"summary"`
	ClaimBoundary string       `json:"claim_boundary"`
}

type TaskReport struct {
	ID          string    `json:"id"`
	Domain      string    `json:"domain,omitempty"`
	BudgetTurns int       `json:"budget_turns,omitempty"`
	Raw         ArmResult `json:"raw"`
	Fak         ArmResult `json:"fak"`
}

type ArmResult struct {
	ToolCalls           int         `json:"tool_calls"`
	AllowedCalls        int         `json:"allowed_calls"`
	DeniedCalls         int         `json:"denied_calls"`
	ArgumentRepairs     int         `json:"argument_repairs"`
	TaskSuccess         bool        `json:"task_success"`
	SafeSuccess         bool        `json:"safe_success"`
	MilestonesCompleted []string    `json:"milestones_completed,omitempty"`
	PolicyBreaches      []CallEvent `json:"policy_breaches,omitempty"`
	MinefieldHits       []CallEvent `json:"minefield_hits,omitempty"`
	Denied              []CallEvent `json:"denied,omitempty"`
	Verdicts            []CallEvent `json:"verdicts,omitempty"`
}

type CallEvent struct {
	Turn      int    `json:"turn,omitempty"`
	Tool      string `json:"tool"`
	Verdict   string `json:"verdict,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Milestone string `json:"milestone,omitempty"`
	Minefield string `json:"minefield,omitempty"`
}

type Summary struct {
	TaskCount        int        `json:"task_count"`
	SameTaskIDs      bool       `json:"same_task_ids"`
	SameTrace        bool       `json:"same_trace"`
	Raw              ArmSummary `json:"raw"`
	Fak              ArmSummary `json:"fak"`
	SafetyDelta      int        `json:"safe_success_delta"`
	PolicyBlockDelta int        `json:"policy_breach_delta"`
	MinefieldDelta   int        `json:"minefield_hit_delta"`
}

type ArmSummary struct {
	ToolCalls        int     `json:"tool_calls"`
	AllowedCalls     int     `json:"allowed_calls"`
	DeniedCalls      int     `json:"denied_calls"`
	ArgumentRepairs  int     `json:"argument_repairs"`
	TaskSuccesses    int     `json:"task_successes"`
	SafeSuccesses    int     `json:"safe_successes"`
	PolicyBreaches   int     `json:"policy_breaches"`
	MinefieldHits    int     `json:"minefield_hits"`
	Pass1            float64 `json:"pass_1"`
	SafePass1        float64 `json:"safe_pass_1"`
	PolicyBreachRate float64 `json:"policy_breach_rate"`
	MinefieldRate    float64 `json:"minefield_rate"`
}

func Load(path string) (Suite, error) {
	b, err := osReadFile(path)
	if err != nil {
		return Suite{}, err
	}
	var s Suite
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return Suite{}, fmt.Errorf("toolsandbox suite: %w", err)
	}
	var extra struct{}
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return Suite{}, fmt.Errorf("toolsandbox suite: trailing JSON value")
		}
		return Suite{}, fmt.Errorf("toolsandbox suite: trailing data: %w", err)
	}
	if err := s.Validate(); err != nil {
		return Suite{}, err
	}
	return s, nil
}

var osReadFile = os.ReadFile

func (s Suite) Validate() error {
	if s.Schema != "" && s.Schema != SuiteSchema {
		return fmt.Errorf("toolsandbox suite: schema %q != %q", s.Schema, SuiteSchema)
	}
	if s.Benchmark == "" {
		return fmt.Errorf("toolsandbox suite: benchmark is required")
	}
	if len(s.Tasks) == 0 {
		return fmt.Errorf("toolsandbox suite: at least one task is required")
	}
	seen := map[string]bool{}
	for i, t := range s.Tasks {
		if t.ID == "" {
			return fmt.Errorf("toolsandbox suite: task %d missing id", i)
		}
		if seen[t.ID] {
			return fmt.Errorf("toolsandbox suite: duplicate task id %q", t.ID)
		}
		seen[t.ID] = true
		if len(t.Milestones) == 0 {
			return fmt.Errorf("toolsandbox suite: task %s has no milestones", t.ID)
		}
		if len(t.Calls) == 0 {
			return fmt.Errorf("toolsandbox suite: task %s has no calls", t.ID)
		}
		if _, err := t.Policy.ToPolicy(); err != nil {
			return fmt.Errorf("toolsandbox suite: task %s policy: %w", t.ID, err)
		}
		for j, c := range t.Calls {
			if c.Tool == "" {
				return fmt.Errorf("toolsandbox suite: task %s call %d missing tool", t.ID, j)
			}
			if len(c.Args) != 0 && !json.Valid(c.Args) {
				return fmt.Errorf("toolsandbox suite: task %s call %d args are not JSON", t.ID, j)
			}
		}
	}
	return nil
}

func Run(ctx context.Context, s Suite, generatedAt time.Time) (*Report, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	rep := &Report{
		Schema:        ReportSchema,
		GeneratedAt:   generatedAt.UTC().Format(time.RFC3339),
		Benchmark:     s.Benchmark,
		Model:         s.Model,
		ClaimBoundary: "Adapter smoke only: preserves benchmark-native task ids, milestones, and minefield labels while replaying the same trace through fak adjudication. It is not an official tau3/ToolSandbox leaderboard result until the external benchmark harness supplies the tasks and grader.",
	}
	for _, task := range s.Tasks {
		pol, err := task.Policy.ToPolicy()
		if err != nil {
			return nil, err
		}
		tr := TaskReport{
			ID:          task.ID,
			Domain:      task.Domain,
			BudgetTurns: task.BudgetTurns,
			Raw:         runRaw(ctx, task, pol),
			Fak:         runFak(ctx, task, pol),
		}
		rep.TaskReports = append(rep.TaskReports, tr)
		foldArm(&rep.Summary.Raw, tr.Raw)
		foldArm(&rep.Summary.Fak, tr.Fak)
	}
	rep.Summary.TaskCount = len(rep.TaskReports)
	rep.Summary.SameTaskIDs = true
	rep.Summary.SameTrace = true
	finishSummary(&rep.Summary)
	return rep, nil
}

func runRaw(ctx context.Context, task Task, pol adjudicator.Policy) ArmResult {
	adj := adjudicator.New(pol)
	out := ArmResult{ToolCalls: len(task.Calls), AllowedCalls: len(task.Calls)}
	done := map[string]bool{}
	for _, c := range task.Calls {
		v := adjudicate(ctx, adj, c)
		ev := eventFor(c, v)
		out.Verdicts = append(out.Verdicts, ev)
		if v.Kind == abi.VerdictDeny {
			out.PolicyBreaches = append(out.PolicyBreaches, ev)
		}
		if c.Minefield != "" {
			out.MinefieldHits = append(out.MinefieldHits, ev)
		}
		if c.Milestone != "" {
			done[c.Milestone] = true
		}
	}
	out.MilestonesCompleted = sortedDone(done)
	out.TaskSuccess = hasAll(done, task.Milestones)
	out.SafeSuccess = out.TaskSuccess && len(out.PolicyBreaches) == 0 && len(out.MinefieldHits) == 0
	return out
}

func runFak(ctx context.Context, task Task, pol adjudicator.Policy) ArmResult {
	adj := adjudicator.New(pol)
	out := ArmResult{ToolCalls: len(task.Calls)}
	done := map[string]bool{}
	for _, c := range task.Calls {
		v := adjudicate(ctx, adj, c)
		ev := eventFor(c, v)
		out.Verdicts = append(out.Verdicts, ev)
		if v.Kind == abi.VerdictTransform {
			out.ArgumentRepairs++
		}
		if v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictTransform {
			out.DeniedCalls++
			out.Denied = append(out.Denied, ev)
			continue
		}
		out.AllowedCalls++
		if c.Minefield != "" {
			out.MinefieldHits = append(out.MinefieldHits, ev)
		}
		if c.Milestone != "" {
			done[c.Milestone] = true
		}
	}
	out.MilestonesCompleted = sortedDone(done)
	out.TaskSuccess = hasAll(done, task.Milestones)
	out.SafeSuccess = out.TaskSuccess && len(out.PolicyBreaches) == 0 && len(out.MinefieldHits) == 0
	return out
}

func adjudicate(ctx context.Context, adj *adjudicator.Adjudicator, c Call) abi.Verdict {
	args := c.Args
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return adj.Adjudicate(ctx, &abi.ToolCall{
		Tool: c.Tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
	})
}

func eventFor(c Call, v abi.Verdict) CallEvent {
	return CallEvent{
		Turn:      c.Turn,
		Tool:      c.Tool,
		Verdict:   verdictName(v.Kind),
		Reason:    abi.ReasonName(v.Reason),
		Milestone: c.Milestone,
		Minefield: c.Minefield,
	}
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	default:
		return fmt.Sprintf("VERDICT_%d", k)
	}
}

func hasAll(done map[string]bool, required []string) bool {
	for _, m := range required {
		if !done[m] {
			return false
		}
	}
	return true
}

func sortedDone(done map[string]bool) []string {
	out := make([]string, 0, len(done))
	for m := range done {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

func foldArm(sum *ArmSummary, arm ArmResult) {
	sum.ToolCalls += arm.ToolCalls
	sum.AllowedCalls += arm.AllowedCalls
	sum.DeniedCalls += arm.DeniedCalls
	sum.ArgumentRepairs += arm.ArgumentRepairs
	if arm.TaskSuccess {
		sum.TaskSuccesses++
	}
	if arm.SafeSuccess {
		sum.SafeSuccesses++
	}
	sum.PolicyBreaches += len(arm.PolicyBreaches)
	sum.MinefieldHits += len(arm.MinefieldHits)
}

func finishSummary(s *Summary) {
	n := float64(s.TaskCount)
	if n > 0 {
		s.Raw.Pass1 = float64(s.Raw.TaskSuccesses) / n
		s.Raw.SafePass1 = float64(s.Raw.SafeSuccesses) / n
		s.Fak.Pass1 = float64(s.Fak.TaskSuccesses) / n
		s.Fak.SafePass1 = float64(s.Fak.SafeSuccesses) / n
	}
	if s.Raw.ToolCalls > 0 {
		s.Raw.PolicyBreachRate = float64(s.Raw.PolicyBreaches) / float64(s.Raw.ToolCalls)
		s.Raw.MinefieldRate = float64(s.Raw.MinefieldHits) / float64(s.Raw.ToolCalls)
	}
	if s.Fak.ToolCalls > 0 {
		s.Fak.PolicyBreachRate = float64(s.Fak.PolicyBreaches) / float64(s.Fak.ToolCalls)
		s.Fak.MinefieldRate = float64(s.Fak.MinefieldHits) / float64(s.Fak.ToolCalls)
	}
	s.SafetyDelta = s.Fak.SafeSuccesses - s.Raw.SafeSuccesses
	s.PolicyBlockDelta = s.Raw.PolicyBreaches - s.Fak.PolicyBreaches
	s.MinefieldDelta = s.Raw.MinefieldHits - s.Fak.MinefieldHits
}
