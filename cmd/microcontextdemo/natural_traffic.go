package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	naturalTrafficCorpusSchema   = "fak-microcontext-natural-traffic/1"
	naturalTrafficJudgmentSchema = "fak-microcontext-natural-traffic-judgments/1"
	naturalTrafficFoldSchema     = "fak-microcontext-natural-traffic-fold/1"
	naturalTrafficReportSchema   = "fak-microcontext-natural-traffic-report/1"
)

var naturalTrafficLabels = []string{"repo_search", "docs_read", "issue_state", "commit_state"}

type naturalTrafficRecord struct {
	ID              string   `json:"id"`
	Split           string   `json:"split"`
	SourceIssue     int      `json:"source_issue"`
	SourceUpdatedAt string   `json:"source_updated_at"`
	SourceSHA256    string   `json:"source_sha256"`
	Question        string   `json:"question"`
	Context         string   `json:"context"`
	ReadbackLabels  []string `json:"readback_labels"`
}
type naturalTrafficCorpus struct {
	Schema             string                 `json:"Schema"`
	CreatedAt          string                 `json:"CreatedAt"`
	Source             string                 `json:"Source"`
	Selection          string                 `json:"Selection"`
	SplitRule          string                 `json:"SplitRule"`
	Requested          int                    `json:"Requested"`
	Eligible           int                    `json:"Eligible"`
	Excluded           int                    `json:"Excluded"`
	ReadbackClassPrior map[string]int         `json:"readback_class_prior"`
	Records            []naturalTrafficRecord `json:"Records"`
}
type naturalTrafficDecision struct {
	ID         string   `json:"id"`
	Labels     []string `json:"labels"`
	Clarify    bool     `json:"clarify"`
	Confidence float64  `json:"confidence"`
	Rationale  string   `json:"rationale"`
}
type naturalTrafficJudgments struct {
	Schema       string                   `json:"Schema"`
	CreatedAt    string                   `json:"CreatedAt"`
	CorpusSHA256 string                   `json:"CorpusSHA256"`
	Adjudicator  string                   `json:"Adjudicator"`
	Model        string                   `json:"Model"`
	BatchSize    int                      `json:"BatchSize"`
	PromptTokens int                      `json:"PromptTokens"`
	OutputTokens int                      `json:"OutputTokens"`
	WallMS       float64                  `json:"WallMS"`
	Records      []naturalTrafficDecision `json:"Records"`
}
type naturalTrafficFoldRecord struct {
	ID       string              `json:"id"`
	Split    string              `json:"split"`
	Labels   []string            `json:"labels"`
	Clarify  bool                `json:"clarify"`
	Disputed []string            `json:"disputed,omitempty"`
	Votes    map[string][]string `json:"votes"`
}
type naturalTrafficFold struct {
	Schema       string                     `json:"Schema"`
	CreatedAt    string                     `json:"CreatedAt"`
	CorpusSHA256 string                     `json:"CorpusSHA256"`
	Policy       string                     `json:"Policy"`
	Adjudicators []string                   `json:"Adjudicators"`
	Counts       map[string]int             `json:"Counts"`
	JudgeWallMS  map[string]float64         `json:"JudgeWallMS"`
	Records      []naturalTrafficFoldRecord `json:"Records"`
}
type naturalTrafficReceipt struct {
	ID, Split, Label, Status, Receipt string
	WallMS, WorkMS                    float64
	Authority                         []string
}
type naturalTrafficPolicyRow struct {
	Policy, Split                                     string
	Records                                           int
	Exact                                             float64
	MeanLabels, MeanWallMS, MeanWorkMS, MeanAuthority float64
	ExactCILow, ExactCIHigh                           float64
	Observed                                          bool
}
type naturalTrafficBreakEven struct {
	SelectorCostMS, SavedToolWorkMS, ErrorCostMS, CacheHitRate, NetValueMS float64
	Wins                                                                   bool
}
type naturalTrafficReport struct {
	Schema, CreatedAt, CorpusSHA256, FoldSHA256, Provenance string
	Records                                                 int
	Tune                                                    int
	Test                                                    int
	Receipts                                                []naturalTrafficReceipt
	Policies                                                []naturalTrafficPolicyRow
	BreakEven                                               []naturalTrafficBreakEven
	FalsifyingRegime                                        string
	ArtifactSHA256                                          string `json:",omitempty"`
}

func naturalTrafficFileSHA(p string) (string, []byte, error) {
	b, e := os.ReadFile(p)
	if e != nil {
		return "", nil, e
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), b, nil
}
func cleanLabels(in []string) []string {
	seen := map[string]bool{}
	for _, x := range in {
		for _, ok := range naturalTrafficLabels {
			if x == ok {
				seen[x] = true
			}
		}
	}
	out := []string{}
	for _, x := range naturalTrafficLabels {
		if seen[x] {
			out = append(out, x)
		}
	}
	return out
}
func sameLabels(a, b []string) bool {
	a = cleanLabels(a)
	b = cleanLabels(b)
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func labelsMap(x []string) map[string]bool {
	m := map[string]bool{}
	for _, v := range cleanLabels(x) {
		m[v] = true
	}
	return m
}

func loadNaturalTrafficCorpus(path string) (naturalTrafficCorpus, string, error) {
	var c naturalTrafficCorpus
	sha, b, e := naturalTrafficFileSHA(path)
	if e == nil {
		e = json.Unmarshal(b, &c)
	}
	if e == nil && (c.Schema != naturalTrafficCorpusSchema || len(c.Records) < 100) {
		e = fmt.Errorf("natural traffic corpus dimensions")
	}
	return c, sha, e
}
func loadNaturalTrafficJudgments(path string) (naturalTrafficJudgments, error) {
	var j naturalTrafficJudgments
	b, e := os.ReadFile(path)
	if e == nil {
		e = json.Unmarshal(b, &j)
	}
	if e == nil && j.Schema != naturalTrafficJudgmentSchema {
		e = fmt.Errorf("judgment schema")
	}
	return j, e
}

type naturalTrafficUsage struct {
	PromptTokens     int
	CompletionTokens int
}

func callNaturalTrafficChat(ctx context.Context, endpoint, key, model, prompt string) (string, naturalTrafficUsage, error) {
	endpoint = strings.TrimRight(endpoint, "/")
	if endpoint == "" || key == "" {
		return "", naturalTrafficUsage{}, fmt.Errorf("endpoint and key required")
	}
	reqBody := map[string]any{"model": model, "messages": []map[string]string{{"role": "user", "content": prompt}}, "temperature": 0, "response_format": map[string]string{"type": "json_object"}}
	body, _ := json.Marshal(reqBody)
	req, e := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"/chat/completions", bytes.NewReader(body))
	if e != nil {
		return "", naturalTrafficUsage{}, e
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	client := http.Client{Transport: http.DefaultTransport, Timeout: 2 * time.Minute}
	if http.DefaultClient.Transport != nil { //boundarylint:ignore MISSING_HTTP_TIMEOUT copy only the injected transport into the bounded client
		client.Transport = http.DefaultClient.Transport //boundarylint:ignore MISSING_HTTP_TIMEOUT bounded client retains the injected transport
	}
	resp, e := client.Do(req)
	if e != nil {
		return "", naturalTrafficUsage{}, e
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			Prompt     int `json:"prompt_tokens"`
			Completion int `json:"completion_tokens"`
		} `json:"usage"`
		Error any `json:"error"`
	}
	if e = json.NewDecoder(resp.Body).Decode(&out); e != nil {
		return "", naturalTrafficUsage{}, e
	}
	if resp.StatusCode >= 300 || len(out.Choices) == 0 {
		return "", naturalTrafficUsage{}, fmt.Errorf("chat status %s", resp.Status)
	}
	return out.Choices[0].Message.Content, naturalTrafficUsage{out.Usage.Prompt, out.Usage.Completion}, nil
}
func extractNaturalTrafficJSON(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.Index(s, "{"); i >= 0 {
		if j := strings.LastIndex(s, "}"); j >= i {
			return s[i : j+1]
		}
	}
	return s
}

func runNaturalTrafficJudge(ctx context.Context, corpusPath, out, endpoint, key, model, adjudicator string) error {
	c, sha, e := loadNaturalTrafficCorpus(corpusPath)
	if e != nil {
		return e
	}
	if endpoint == "" {
		endpoint = os.Getenv("OPENAI_BASE_URL")
	}
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	if model == "" {
		model = "gpt-5.6-sol"
	}
	if adjudicator == "" {
		adjudicator = model
	}
	result := naturalTrafficJudgments{Schema: naturalTrafficJudgmentSchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), CorpusSHA256: sha, Adjudicator: adjudicator, Model: model, BatchSize: 10}
	start := time.Now()
	for i := 0; i < len(c.Records); i += 10 {
		end := i + 10
		if end > len(c.Records) {
			end = len(c.Records)
		}
		payload, _ := json.Marshal(c.Records[i:end])
		prompt := `Independently label each operator question with every evidence source required. Allowed labels: repo_search, docs_read, issue_state, commit_state. Set clarify when underspecified. Return JSON only: {"records":[{"id":"...","labels":[...],"clarify":false,"confidence":0.0,"rationale":"..."}]}. Do not copy construction labels; judge question and context. Input: ` + string(payload)
		raw, usage, e := callNaturalTrafficChat(ctx, endpoint, key, model, prompt)
		if e != nil {
			return e
		}
		var box struct {
			Records []naturalTrafficDecision `json:"records"`
		}
		if e = json.Unmarshal([]byte(extractNaturalTrafficJSON(raw)), &box); e != nil {
			return e
		}
		if len(box.Records) != end-i {
			return fmt.Errorf("batch %d returned %d/%d", i, len(box.Records), end-i)
		}
		for n := range box.Records {
			box.Records[n].Labels = cleanLabels(box.Records[n].Labels)
		}
		result.Records = append(result.Records, box.Records...)
		result.PromptTokens += usage.PromptTokens
		result.OutputTokens += usage.CompletionTokens
	}
	result.WallMS = float64(time.Since(start).Microseconds()) / 1000
	return writeJSONFile(out, result)
}

func foldNaturalTraffic(corpusPath, aPath, bPath, out string) error {
	c, sha, e := loadNaturalTrafficCorpus(corpusPath)
	if e != nil {
		return e
	}
	a, e := loadNaturalTrafficJudgments(aPath)
	if e != nil {
		return e
	}
	b, e := loadNaturalTrafficJudgments(bPath)
	if e != nil {
		return e
	}
	if a.CorpusSHA256 != sha || b.CorpusSHA256 != sha || a.Adjudicator == b.Adjudicator || a.Model == b.Model {
		return fmt.Errorf("model-distinct corpus provenance")
	}
	am, bm := map[string]naturalTrafficDecision{}, map[string]naturalTrafficDecision{}
	for _, x := range a.Records {
		am[x.ID] = x
	}
	for _, x := range b.Records {
		bm[x.ID] = x
	}
	f := naturalTrafficFold{Schema: naturalTrafficFoldSchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), CorpusSHA256: sha, Policy: "construction/read-back and first independent model must agree per label; disagreements are frozen; second model is held out as the evaluated semantic policy", Adjudicators: []string{"construction", a.Adjudicator, b.Adjudicator}, Counts: map[string]int{}, JudgeWallMS: map[string]float64{a.Adjudicator: a.WallMS, b.Adjudicator: b.WallMS}}
	for _, r := range c.Records {
		x, xok := am[r.ID]
		y, yok := bm[r.ID]
		if !xok || !yok {
			return fmt.Errorf("missing judgments %s", r.ID)
		}
		construction := cleanLabels(r.ReadbackLabels)
		reference := cleanLabels(x.Labels)
		heldout := cleanLabels(y.Labels)
		gold, disputed := []string{}, []string{}
		cm, rm := labelsMap(construction), labelsMap(reference)
		for _, label := range naturalTrafficLabels {
			if cm[label] == rm[label] {
				if cm[label] {
					gold = append(gold, label)
				}
			} else {
				disputed = append(disputed, label)
			}
		}
		clarify := false
		if x.Clarify {
			disputed = append(disputed, "clarify")
		}
		votes := map[string][]string{"construction": construction, a.Adjudicator: reference, b.Adjudicator: heldout}
		f.Records = append(f.Records, naturalTrafficFoldRecord{ID: r.ID, Split: r.Split, Labels: gold, Clarify: clarify, Disputed: disputed, Votes: votes})
		f.Counts[r.Split]++
		if len(disputed) > 0 {
			f.Counts["disputed"]++
		}
		for _, l := range gold {
			f.Counts["label_"+l]++
		}
	}
	return writeJSONFile(out, f)
}

func observeNaturalTraffic(ctx context.Context, r naturalTrafficRecord, label string) naturalTrafficReceipt {
	q := r.Question
	if len(q) > 80 {
		q = q[:80]
	}
	cmd := exec.CommandContext(ctx, "git", "grep", "-n", "-m", "1", "--", q)
	if label == "issue_state" {
		cmd = exec.CommandContext(ctx, "gh", "issue", "view", fmt.Sprint(r.SourceIssue), "--json", "number,state,title")
	}
	if label == "commit_state" {
		cmd = exec.CommandContext(ctx, "git", "log", "-1", "--oneline", "--grep", fmt.Sprintf("#%d", r.SourceIssue))
	}
	if label == "docs_read" {
		cmd = exec.CommandContext(ctx, "git", "grep", "-n", "-m", "1", "--", fmt.Sprintf("#%d", r.SourceIssue), "docs")
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	start := time.Now()
	b, e := cmd.CombinedOutput()
	ms := float64(time.Since(start).Microseconds()) / 1000
	status := "completed"
	if e != nil {
		status = "no_match"
	}
	receipt := strings.TrimSpace(string(b))
	if len(receipt) > 240 {
		receipt = receipt[:240]
	}
	return naturalTrafficReceipt{ID: r.ID, Split: r.Split, Label: label, Status: status, Receipt: receipt, WallMS: ms, WorkMS: ms, Authority: []string{label}}
}
func policyLabels(name string, r naturalTrafficRecord, g naturalTrafficFoldRecord, adjudicators []string) []string {
	switch name {
	case "deterministic":
		return cleanLabels(r.ReadbackLabels)
	case "tuned_fixed_cascade":
		return []string{"repo_search", "docs_read", "issue_state", "commit_state"}
	case "semantic_admission":
		if len(adjudicators) > 0 {
			return cleanLabels(g.Votes[adjudicators[len(adjudicators)-1]])
		}
		return nil
	default:
		x := []string{}
		if len(adjudicators) > 0 {
			x = cleanLabels(g.Votes[adjudicators[len(adjudicators)-1]])
		}
		if len(g.Disputed) > 0 {
			x = append(x, "repo_search")
		}
		return cleanLabels(x)
	}
}
func runNaturalTrafficReport(ctx context.Context, corpusPath, foldPath, out string) error {
	c, csha, e := loadNaturalTrafficCorpus(corpusPath)
	if e != nil {
		return e
	}
	var f naturalTrafficFold
	fsha, fb, e := naturalTrafficFileSHA(foldPath)
	if e == nil {
		e = json.Unmarshal(fb, &f)
	}
	if e != nil || f.Schema != naturalTrafficFoldSchema || f.CorpusSHA256 != csha {
		return fmt.Errorf("fold provenance")
	}
	gm := map[string]naturalTrafficFoldRecord{}
	for _, g := range f.Records {
		gm[g.ID] = g
	}
	report := naturalTrafficReport{Schema: naturalTrafficReportSchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), CorpusSHA256: csha, FoldSHA256: fsha, Provenance: "observed local git/gh seams; break-even rows modeled from observed test work", Records: len(c.Records), FalsifyingRegime: "adaptive admission loses when selector plus error cost exceeds avoided tool work, especially at high cache hit rate"}
	for _, r := range c.Records {
		if r.Split == "test" {
			report.Test++
		} else {
			report.Tune++
		}
		for _, l := range naturalTrafficLabels {
			report.Receipts = append(report.Receipts, observeNaturalTraffic(ctx, r, l))
		}
	}
	policies := []string{"deterministic", "tuned_fixed_cascade", "semantic_admission", "selective_parallelism"}
	for _, split := range []string{"tune", "test"} {
		for _, name := range policies {
			row := naturalTrafficPolicyRow{Policy: name, Split: split, Observed: true}
			for _, r := range c.Records {
				if r.Split != split {
					continue
				}
				row.Records++
				pred := policyLabels(name, r, gm[r.ID], f.Adjudicators)
				if sameLabels(pred, gm[r.ID].Labels) {
					row.Exact++
				}
				row.MeanLabels += float64(len(pred))
				for _, rc := range report.Receipts {
					if rc.ID == r.ID && labelsMap(pred)[rc.Label] {
						row.MeanWallMS += rc.WallMS
						row.MeanWorkMS += rc.WorkMS
						row.MeanAuthority += float64(len(rc.Authority))
					}
				}
			}
			if row.Records > 0 {
				n := float64(row.Records)
				success := row.Exact
				row.Exact /= n
				row.ExactCILow, row.ExactCIHigh = wilson95(success, n)
				row.MeanLabels /= n
				row.MeanWallMS /= n
				row.MeanWorkMS /= n
				row.MeanAuthority /= n
			}
			report.Policies = append(report.Policies, row)
		}
	}
	testWork, semanticWork := 0.0, 0.0
	for _, row := range report.Policies {
		if row.Split == "test" && row.Policy == "tuned_fixed_cascade" {
			testWork = row.MeanWorkMS
		}
		if row.Split == "test" && row.Policy == "semantic_admission" {
			semanticWork = row.MeanWorkMS
		}
	}
	savedBase := testWork - semanticWork
	observedSelector := 5.0
	if len(f.Adjudicators) > 0 {
		if w := f.JudgeWallMS[f.Adjudicators[len(f.Adjudicators)-1]]; w > 0 {
			observedSelector = w / float64(len(c.Records))
		}
	}
	for _, selector := range []float64{5, observedSelector} {
		for _, hit := range []float64{0, .5, .9} {
			for _, errCost := range []float64{0, 25, 100} {
				saved := savedBase * (1 - hit)
				net := saved - selector - errCost
				report.BreakEven = append(report.BreakEven, naturalTrafficBreakEven{SelectorCostMS: selector, SavedToolWorkMS: saved, ErrorCostMS: errCost, CacheHitRate: hit, NetValueMS: net, Wins: net > 0})
			}
		}
	}
	return writeJSONFile(out, report)
}
func wilson95(success, total float64) (float64, float64) {
	if total <= 0 {
		return 0, 0
	}
	z := 1.96
	p := success / total
	d := 1 + z*z/total
	c := (p + z*z/(2*total)) / d
	m := z * math.Sqrt(p*(1-p)/total+z*z/(4*total*total)) / d
	return c - m, c + m
}

func verifyNaturalTraffic(corpusPath, artifactPath string) error {
	c, sha, e := loadNaturalTrafficCorpus(corpusPath)
	if e != nil {
		return e
	}
	b, e := os.ReadFile(artifactPath)
	if e != nil {
		return e
	}
	var envelope struct{ Schema, CorpusSHA256 string }
	if e = json.Unmarshal(b, &envelope); e != nil {
		return e
	}
	if envelope.CorpusSHA256 != sha {
		return fmt.Errorf("corpus hash")
	}
	switch envelope.Schema {
	case naturalTrafficFoldSchema:
		var f naturalTrafficFold
		if e = json.Unmarshal(b, &f); e != nil {
			return e
		}
		if len(f.Records) != len(c.Records) {
			return fmt.Errorf("fold dimensions")
		}
	case naturalTrafficReportSchema:
		var r naturalTrafficReport
		if e = json.Unmarshal(b, &r); e != nil {
			return e
		}
		if len(r.Policies) != 8 || len(r.Receipts) == 0 || len(r.BreakEven) < 6 {
			return fmt.Errorf("report dimensions")
		}
		hasWin, hasLoss := false, false
		for _, x := range r.BreakEven {
			hasWin = hasWin || x.Wins
			hasLoss = hasLoss || !x.Wins
		}
		if !hasWin || !hasLoss {
			return fmt.Errorf("missing falsifying regime")
		}
	default:
		return fmt.Errorf("artifact schema")
	}
	return nil
}

func init() { sort.Strings(naturalTrafficLabels) }
