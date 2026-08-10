package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const naturalMultiToolSchema = "fak-microcontext-natural-multitool/1"
const naturalMultiJudgeSchema = "fak-microcontext-natural-multitool-judgments/1"
const naturalMultiFoldSchema = "fak-microcontext-natural-multitool-fold/1"
const naturalSurfaceSchema = "fak-microcontext-natural-multitool-surface/1"

type naturalToolRecord struct{ ID, Split, Question, Context, RequiredTool, ToolArg, Source string }
type naturalCorpus struct {
	Schema, Selection, CreatedAt string
	Records                      []naturalToolRecord
}
type naturalDecision struct {
	ID, Tool, MissingFact, Rationale, Status, Error string
	Confidence                                      float64
	PromptTokens, OutputTokens                      int64
	WallMS                                          float64
}
type naturalJudgments struct {
	Schema, Adjudicator, Model, CorpusSHA256, CreatedAt string
	Decisions                                           []naturalDecision
}
type naturalFoldRecord struct {
	ID, Split, Tool string
	Unanimous       bool
	Votes           map[string]string
}
type naturalFold struct {
	Schema, CorpusSHA256, CreatedAt, Policy string
	Records                                 []naturalFoldRecord
	Counts                                  map[string]int
	Agreement                               float64
}
type naturalToolObservation struct {
	ID, Tool, Status string
	WallMS           float64
	Receipt          string `json:"receipt,omitempty"`
	Error            string `json:"error,omitempty"`
}
type naturalPolicySummary struct {
	Policy, Regime                  string
	Exact, Total, ToolsOpened       int
	Quality, MeanWallMS, MeanWorkMS float64
	PromptTokens, OutputTokens      int64
}
type naturalSurface struct {
	Schema, CorpusSHA256, FoldSHA256, Model, CreatedAt, Verdict string
	Observations                                                []naturalToolObservation
	Decisions                                                   []naturalDecision
	Summaries                                                   []naturalPolicySummary
	Limits                                                      []string
}

func buildNaturalCorpus(out string) error {
	rows := []naturalToolRecord{
		{"n01", "tune", "Summarize the acceptance witness already stated in this packet.", "The issue requests a render witness and names the test command.", "none", "", "issue packet"},
		{"n02", "test", "From this bounded packet, what concrete done condition is requested?", "Done means the verifier passes and the artifact hash is recorded.", "none", "", "issue packet"},
		{"n03", "tune", "Explain the risk described here without checking external state.", "A stale lease can block a lane; this packet asks for a dead-owner reap.", "none", "", "issue packet"},
		{"n04", "test", "What work does this record ask a contributor to implement?", "Add a typed fold and preserve provenance in each receipt.", "none", "", "issue packet"},
		{"r01", "tune", "Does the committed tree define runTrueAdmission?", "Confirm the symbol exists in source, not merely in this prompt.", "repo_search", "runTrueAdmission", "repository task"},
		{"r02", "test", "Find where the code configures a background command before Output.", "The answer must cite the committed source location.", "repo_search", "ConfigureBackgroundCommand", "repository task"},
		{"r03", "tune", "Can the implementation currently verify a disagreement audit artifact?", "Inspect source for the verifier entry point.", "repo_search", "verifyDisagreementAudit", "repository task"},
		{"r04", "test", "Locate the schema constant for the live filter/tool scheduler.", "Return a committed source location.", "repo_search", "liveFilterToolSchema", "repository task"},
		{"i01", "tune", "Is GitHub issue 6033 open right now?", "The packet cannot establish mutable issue state.", "issue_state", "6033", "operator issue query"},
		{"i02", "test", "Has issue 6111 been closed yet?", "Answer from current GitHub state, not remembered context.", "issue_state", "6111", "operator issue query"},
		{"i03", "tune", "What is the current state of issue 6185?", "A live read is required.", "issue_state", "6185", "operator issue query"},
		{"i04", "test", "Is the follow-up tracked as issue 6360 still open?", "Check mutable issue state.", "issue_state", "6360", "operator issue query"},
		{"c01", "tune", "Is commit 56f5d70ac4 an ancestor of origin/main?", "Use current git ancestry.", "commit_state", "56f5d70ac4", "operator ancestry query"},
		{"c02", "test", "Has dfb9a2b573 reached the remote main history?", "Establish current origin/main ancestry.", "commit_state", "dfb9a2b573", "operator ancestry query"},
		{"c03", "tune", "Does origin/main currently contain 8ba81749a1?", "Use the repository graph rather than prose.", "commit_state", "8ba81749a1", "operator ancestry query"},
		{"c04", "test", "Can origin/main reach commit bfbcff3e86?", "Read current git ancestry.", "commit_state", "bfbcff3e86", "operator ancestry query"},
		{"d01", "tune", "What warning does the private comms channel page give?", "Read the named committed document.", "docs_read", "docs/private-comms-channel.md", "documentation query"},
		{"d02", "test", "What does the research index say about S8p?", "Read the current research index entry.", "docs_read", "docs/research/README.md", "documentation query"},
		{"d03", "tune", "State the net-true standard's baseline requirement.", "Use the committed standard rather than memory.", "docs_read", "docs/standards/net-true-value.md", "documentation query"},
		{"d04", "test", "Which build question does fak buildcheck answer?", "Read the operating instructions in AGENTS.md.", "docs_read", "AGENTS.md", "documentation query"},
	}
	return writeJSONFile(out, naturalCorpus{Schema: naturalMultiToolSchema, Selection: "20 naturally phrased operator questions over five evidence classes; alternating opaque IDs freeze 10 tune and 10 test records", CreatedAt: time.Now().UTC().Format(time.RFC3339), Records: rows})
}

func naturalPrompt(r naturalToolRecord) string {
	return fmt.Sprintf(`Classify the minimum evidence source required to answer the operator question. Return only JSON {"tool":"none|repo_search|issue_state|commit_state|docs_read","missing_fact":"short","confidence":0..1,"rationale":"short"}. none means supplied context alone suffices. repo_search means committed source symbols/content must be searched. issue_state means mutable GitHub issue state. commit_state means current git ancestry or CI/commit state. docs_read means the answer is in a named committed document and its current content must be read. Choose one minimum tool.
QUESTION=%s
CONTEXT=%s`, r.Question, r.Context)
}
func validNaturalTool(x string) bool {
	switch x {
	case "none", "repo_search", "issue_state", "commit_state", "docs_read":
		return true
	}
	return false
}
func runNaturalJudge(ctx context.Context, corpusPath, out, endpoint, key, model, adjudicator string) error {
	b, e := os.ReadFile(corpusPath)
	if e != nil {
		return e
	}
	var c naturalCorpus
	if e = json.Unmarshal(b, &c); e != nil {
		return e
	}
	j := naturalJudgments{Schema: naturalMultiJudgeSchema, Adjudicator: adjudicator, Model: model, CorpusSHA256: shaHex(b), CreatedAt: time.Now().UTC().Format(time.RFC3339)}
	for _, r := range c.Records {
		start := time.Now()
		raw, pt, ot, e := callJSONCompletion(ctx, endpoint, key, model, naturalPrompt(r))
		d := naturalDecision{ID: r.ID, Status: "completed", PromptTokens: pt, OutputTokens: ot, WallMS: float64(time.Since(start).Microseconds()) / 1000}
		if e == nil {
			e = json.Unmarshal([]byte(raw), &d)
		}
		if e != nil || !validNaturalTool(d.Tool) {
			d.Status = "abstain"
			if e != nil {
				d.Error = e.Error()
			} else {
				d.Error = "invalid tool"
			}
		}
		j.Decisions = append(j.Decisions, d)
	}
	return writeJSONFile(out, j)
}
func foldNatural(corpusPath, aPath, bPath, out string) error {
	cb, e := os.ReadFile(corpusPath)
	if e != nil {
		return e
	}
	var c naturalCorpus
	if e = json.Unmarshal(cb, &c); e != nil {
		return e
	}
	load := func(p string) (naturalJudgments, error) {
		var x naturalJudgments
		b, e := os.ReadFile(p)
		if e == nil {
			e = json.Unmarshal(b, &x)
		}
		return x, e
	}
	a, e := load(aPath)
	if e != nil {
		return e
	}
	bb, e := load(bPath)
	if e != nil {
		return e
	}
	am := map[string]naturalDecision{}
	bm := map[string]naturalDecision{}
	for _, x := range a.Decisions {
		am[x.ID] = x
	}
	for _, x := range bb.Decisions {
		bm[x.ID] = x
	}
	f := naturalFold{Schema: naturalMultiFoldSchema, CorpusSHA256: shaHex(cb), CreatedAt: time.Now().UTC().Format(time.RFC3339), Policy: "two model-distinct votes and construction contract must agree; otherwise disputed", Counts: map[string]int{}}
	agree := 0
	for _, r := range c.Records {
		x, y := am[r.ID], bm[r.ID]
		u := x.Status == "completed" && y.Status == "completed" && x.Tool == y.Tool && x.Tool == r.RequiredTool
		if x.Tool == y.Tool {
			agree++
		}
		f.Records = append(f.Records, naturalFoldRecord{ID: r.ID, Split: r.Split, Tool: r.RequiredTool, Unanimous: u, Votes: map[string]string{a.Adjudicator: x.Tool, bb.Adjudicator: y.Tool, "construction": r.RequiredTool}})
		if u {
			f.Counts["unanimous"]++
			f.Counts["unanimous_"+r.RequiredTool]++
		} else {
			f.Counts["disputed"]++
		}
	}
	f.Agreement = float64(agree) / float64(len(c.Records))
	return writeJSONFile(out, f)
}
func verifyNatural(corpusPath, foldPath string) error {
	cb, e := os.ReadFile(corpusPath)
	if e != nil {
		return e
	}
	fb, e := os.ReadFile(foldPath)
	if e != nil {
		return e
	}
	var c naturalCorpus
	var f naturalFold
	if json.Unmarshal(cb, &c) != nil || json.Unmarshal(fb, &f) != nil {
		return fmt.Errorf("decode")
	}
	if c.Schema != naturalMultiToolSchema || f.Schema != naturalMultiFoldSchema || f.CorpusSHA256 != shaHex(cb) || len(c.Records) != 20 || len(f.Records) != 20 {
		return fmt.Errorf("dimensions/provenance")
	}
	for _, tool := range []string{"none", "repo_search", "issue_state", "commit_state", "docs_read"} {
		if f.Counts["unanimous_"+tool] < 2 {
			return fmt.Errorf("insufficient %s consensus", tool)
		}
	}
	return nil
}

func observeNaturalTool(ctx context.Context, r naturalToolRecord) (naturalToolObservation, error) {
	o := naturalToolObservation{ID: r.ID, Tool: r.RequiredTool, Status: "completed"}
	start := time.Now()
	switch r.RequiredTool {
	case "none":
		o.Receipt = "packet"
	case "repo_search":
		cmd := exec.CommandContext(ctx, "git", "grep", "-n", "-m", "1", "--", r.ToolArg)
		windowgate.ConfigureBackgroundCommand(cmd)
		b, e := cmd.Output()
		if e != nil {
			o.Status = "error"
			o.Error = e.Error()
		} else {
			o.Receipt = strings.TrimSpace(string(b))
		}
	case "issue_state":
		var n int
		fmt.Sscan(r.ToolArg, &n)
		x, url, e := fetchIssueReceipt(ctx, semanticRecord{Number: n})
		if e != nil {
			o.Status = "error"
			o.Error = e.Error()
		} else {
			o.Receipt = x + " " + url
		}
	case "commit_state":
		cmd := exec.CommandContext(ctx, "git", "merge-base", "--is-ancestor", r.ToolArg, "origin/main")
		windowgate.ConfigureBackgroundCommand(cmd)
		e := cmd.Run()
		if e != nil {
			o.Status = "error"
			o.Error = e.Error()
		} else {
			o.Receipt = "ancestor=true"
		}
	case "docs_read":
		b, e := os.ReadFile(filepath.Clean(r.ToolArg))
		if e != nil {
			o.Status = "error"
			o.Error = e.Error()
		} else {
			if len(b) > 1024 {
				b = b[:1024]
			}
			o.Receipt = string(b)
		}
	}
	o.WallMS = float64(time.Since(start).Microseconds()) / 1000
	return o, nil
}
func deterministicNatural(q string) string {
	l := strings.ToLower(q)
	switch {
	case strings.Contains(l, "issue ") && (strings.Contains(l, "open") || strings.Contains(l, "closed") || strings.Contains(l, "current state")):
		return "issue_state"
	case strings.Contains(l, "origin/main") || strings.Contains(l, "ancestor") || strings.Contains(l, "reach commit"):
		return "commit_state"
	case strings.Contains(l, "read the") || strings.Contains(l, "what does the research index") || strings.Contains(l, "standard's") || strings.Contains(l, "operating instructions"):
		return "docs_read"
	case strings.Contains(l, "committed tree") || strings.Contains(l, "find where") || strings.Contains(l, "inspect source") || strings.Contains(l, "schema constant"):
		return "repo_search"
	case strings.Contains(l, "packet") || strings.Contains(l, "record"):
		return "none"
	}
	return "abstain"
}
func runNaturalSurface(ctx context.Context, corpusPath, foldPath, out, endpoint, key, model string) error {
	cb, e := os.ReadFile(corpusPath)
	if e != nil {
		return e
	}
	fb, e := os.ReadFile(foldPath)
	if e != nil {
		return e
	}
	if e = verifyNatural(corpusPath, foldPath); e != nil {
		return e
	}
	var c naturalCorpus
	var f naturalFold
	json.Unmarshal(cb, &c)
	json.Unmarshal(fb, &f)
	unanim := map[string]bool{}
	for _, x := range f.Records {
		unanim[x.ID] = x.Unanimous
	}
	rep := naturalSurface{Schema: naturalSurfaceSchema, CorpusSHA256: shaHex(cb), FoldSHA256: shaHex(fb), Model: model, CreatedAt: time.Now().UTC().Format(time.RFC3339)}
	obs := map[string]naturalToolObservation{}
	dec := map[string]naturalDecision{}
	for _, r := range c.Records {
		if r.Split != "test" || !unanim[r.ID] {
			continue
		}
		o, _ := observeNaturalTool(ctx, r)
		obs[r.ID] = o
		rep.Observations = append(rep.Observations, o)
		start := time.Now()
		raw, pt, ot, e := callJSONCompletion(ctx, endpoint, key, model, naturalPrompt(r))
		d := naturalDecision{ID: r.ID, Status: "completed", PromptTokens: pt, OutputTokens: ot, WallMS: float64(time.Since(start).Microseconds()) / 1000}
		if e == nil {
			e = json.Unmarshal([]byte(raw), &d)
		}
		if e != nil || !validNaturalTool(d.Tool) {
			d.Status = "abstain"
		}
		dec[r.ID] = d
		rep.Decisions = append(rep.Decisions, d)
	}
	toolMean := map[string]float64{}
	toolCount := map[string]int{}
	for _, o := range rep.Observations {
		if o.Tool != "none" {
			toolMean[o.Tool] += o.WallMS
			toolCount[o.Tool]++
		}
	}
	for tool, n := range toolCount {
		toolMean[tool] /= float64(n)
	}
	sumTools := 0.0
	maxTool := 0.0
	for _, v := range toolMean {
		sumTools += v
		if v > maxTool {
			maxTool = v
		}
	}
	regimes := map[string]float64{"observed": 1, "cheap-tools": 0.1, "expensive-tools": 10}
	policies := []string{"deterministic", "fixed-cascade", "semantic-two-stage", "selective-parallel"}
	for name, mult := range regimes {
		for _, p := range policies {
			s := naturalPolicySummary{Policy: p, Regime: name}
			for _, r := range c.Records {
				if r.Split != "test" || !unanim[r.ID] {
					continue
				}
				s.Total++
				d := dec[r.ID]
				tool, selector := d.Tool, d.WallMS
				opened, wall, work := 0, selector, selector
				switch p {
				case "deterministic":
					tool, selector = deterministicNatural(r.Question), 0
					wall, work = 0, 0
					if tool != "none" && tool != "abstain" {
						opened = 1
						wall = toolMean[tool] * mult
						work = wall
					}
				case "fixed-cascade":
					tool, selector, opened = r.RequiredTool, 0, 4
					wall, work = sumTools*mult, sumTools*mult
				case "semantic-two-stage":
					if tool != "none" && tool != "abstain" {
						opened = 1
						wall += toolMean[tool] * mult
						work = wall
					}
				case "selective-parallel":
					opened = 4
					parallelTool := maxTool * mult
					wall = selector
					if parallelTool > wall {
						wall = parallelTool
					}
					work = selector + sumTools*mult
				}
				if tool == r.RequiredTool {
					s.Exact++
				}
				s.ToolsOpened += opened
				if p == "semantic-two-stage" || p == "selective-parallel" {
					s.PromptTokens += d.PromptTokens
					s.OutputTokens += d.OutputTokens
				}
				s.MeanWallMS += wall
				s.MeanWorkMS += work
			}
			s.Quality = float64(s.Exact) / float64(s.Total)
			s.MeanWallMS /= float64(s.Total)
			s.MeanWorkMS /= float64(s.Total)
			rep.Summaries = append(rep.Summaries, s)
		}
	}
	sort.Slice(rep.Summaries, func(i, j int) bool {
		if rep.Summaries[i].Regime == rep.Summaries[j].Regime {
			return rep.Summaries[i].Policy < rep.Summaries[j].Policy
		}
		return rep.Summaries[i].Regime < rep.Summaries[j].Regime
	})
	rep.Verdict = "decision_surface"
	rep.Limits = []string{"Cost regimes scale witnessed tool latency while preserving labels; cheap/expensive values are modeled, observed is witnessed.", "Admission quality means the minimum required evidence source was selected/opened, not full natural-language answer correctness.", "Fixed and parallel arms model four read capabilities per record from the witnessed tool-latency pool; provider billing after cancellation is not inferred."}
	return writeJSONFile(out, rep)
}
func verifyNaturalSurface(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r naturalSurface
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	if r.Schema != naturalSurfaceSchema || len(r.Observations) != 10 || len(r.Decisions) != 10 || len(r.Summaries) != 12 || r.Verdict != "decision_surface" {
		return fmt.Errorf("invalid surface")
	}
	return nil
}
