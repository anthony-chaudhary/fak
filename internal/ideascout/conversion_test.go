package ideascout

// Witnesses for the CONVERSION GATE (#6506).
//
// Two layers, the same shape the dedup and gather contracts already use:
//
//	the SHARED corpus (testdata/conversion_corpus.json), read by these tests AND by
//	tools/idea_scout_test.py, so the gate cannot exist on the interactive Go path
//	while the SCHEDULED Python path keeps filing into a 117-item untriaged pile;
//
//	end-to-end runs, which prove the ledger is actually WIRED — that a live run over
//	an untriaged stock above the cap creates no issue, that draining below the cap
//	files again, and that a source lane down on every topic stops the run from
//	reporting itself as a plain success.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const conversionCorpusPath = "testdata/conversion_corpus.json"

type conversionBacklogCase struct {
	Name   string          `json:"name"`
	Why    string          `json:"why"`
	Issues []ExistingIssue `json:"issues"`
	Expect BacklogStats    `json:"expect"`
}

type conversionGateCase struct {
	Name   string       `json:"name"`
	Why    string       `json:"why"`
	Stats  BacklogStats `json:"stats"`
	Cap    int          `json:"cap"`
	Paused bool         `json:"paused"`
	Reason string       `json:"reason"`
}

type conversionLaneCase struct {
	Name          string       `json:"name"`
	Why           string       `json:"why"`
	Topics        []Topic      `json:"topics"`
	FreshPerTopic int          `json:"fresh_per_topic"`
	Errors        []string     `json:"errors"`
	Expect        []LaneHealth `json:"expect"`
	Degraded      bool         `json:"degraded"`
}

type conversionCorpus struct {
	Schema               string                  `json:"schema"`
	DefaultUntriagedCap  int                     `json:"default_untriaged_cap"`
	StatusVocabulary     []string                `json:"status_vocabulary"`
	LaneStatusVocabulary []string                `json:"lane_status_vocabulary"`
	GateReasons          []string                `json:"gate_reasons"`
	Now                  string                  `json:"now"`
	BacklogCases         []conversionBacklogCase `json:"backlog_cases"`
	GateCases            []conversionGateCase    `json:"gate_cases"`
	LaneCases            []conversionLaneCase    `json:"lane_cases"`
}

func loadConversionCorpus(t *testing.T) conversionCorpus {
	t.Helper()
	raw, err := os.ReadFile(conversionCorpusPath)
	if err != nil {
		t.Fatalf("read shared conversion corpus %s (tools/idea_scout_test.py reads the same file): %v", conversionCorpusPath, err)
	}
	var c conversionCorpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("parse shared conversion corpus: %v", err)
	}
	if c.Schema != "fak/idea-scout-conversion-corpus@1" {
		t.Fatalf("corpus schema = %q, want fak/idea-scout-conversion-corpus@1", c.Schema)
	}
	if len(c.BacklogCases) == 0 || len(c.GateCases) == 0 || len(c.LaneCases) == 0 {
		t.Fatalf("conversion corpus is empty in at least one section: %d backlog, %d gate, %d lane",
			len(c.BacklogCases), len(c.GateCases), len(c.LaneCases))
	}
	return c
}

func corpusNow(t *testing.T, c conversionCorpus) time.Time {
	t.Helper()
	now, err := time.Parse(time.RFC3339, c.Now)
	if err != nil {
		t.Fatalf("corpus `now` is not RFC3339: %v", err)
	}
	return now
}

// TestSharedConversionCorpusVocabulary pins the words themselves. A status or gate
// reason that grows on one implementation and not the other would make the two
// scouts' run records mutually unreadable while both stayed green.
func TestSharedConversionCorpusVocabulary(t *testing.T) {
	c := loadConversionCorpus(t)

	if got := []string{StatusOK, StatusDegraded, StatusPaused}; !reflect.DeepEqual(got, c.StatusVocabulary) {
		t.Fatalf("status vocabulary = %v, corpus status_vocabulary = %v", got, c.StatusVocabulary)
	}
	if got := []string{LaneOK, LanePartial, LaneDown}; !reflect.DeepEqual(got, c.LaneStatusVocabulary) {
		t.Fatalf("lane status vocabulary = %v, corpus lane_status_vocabulary = %v", got, c.LaneStatusVocabulary)
	}
	if got := []string{GateUntriagedCap, GateIndexUnclassified}; !reflect.DeepEqual(got, c.GateReasons) {
		t.Fatalf("gate reasons = %v, corpus gate_reasons = %v", got, c.GateReasons)
	}
	if DefaultUntriagedCap != c.DefaultUntriagedCap {
		t.Fatalf("DefaultUntriagedCap = %d, corpus default_untriaged_cap = %d\n  (tools/idea_scout.py's DEFAULTS['untriaged_cap'] must carry the same number)", DefaultUntriagedCap, c.DefaultUntriagedCap)
	}
	if got := DefaultConfig().UntriagedCap; got != c.DefaultUntriagedCap {
		t.Fatalf("DefaultConfig().UntriagedCap = %d, corpus default_untriaged_cap = %d\n  (a default the config does not carry is a gate nobody runs)", got, c.DefaultUntriagedCap)
	}
}

// TestSharedConversionCorpusBacklog folds each corpus history into the ledger.
// The cases carry the two distinctions the whole measurement rests on: a CLOSED
// issue that still carries needs-triage was never triaged, and a corpus with no
// state at all is unclassified rather than empty.
func TestSharedConversionCorpusBacklog(t *testing.T) {
	c := loadConversionCorpus(t)
	now := corpusNow(t, c)
	for _, tc := range c.BacklogCases {
		t.Run(tc.Name, func(t *testing.T) {
			got := Backlog(tc.Issues, now)
			if !reflect.DeepEqual(got, tc.Expect) {
				t.Fatalf("Backlog() = %+v\n  want %+v\n  why: %s", got, tc.Expect, tc.Why)
			}
		})
	}
}

// TestSharedConversionCorpusGate pins the decision, including its two edges: the
// cap is a ceiling the stock may sit AT and still file, and a blind index big
// enough to matter fails CLOSED rather than reading as an empty backlog.
func TestSharedConversionCorpusGate(t *testing.T) {
	c := loadConversionCorpus(t)
	for _, tc := range c.GateCases {
		t.Run(tc.Name, func(t *testing.T) {
			gate := GateFiling(tc.Stats, tc.Cap)
			if gate.Paused != tc.Paused {
				t.Fatalf("GateFiling(%+v, cap=%d).Paused = %v, want %v\n  why: %s\n  detail: %s", tc.Stats, tc.Cap, gate.Paused, tc.Paused, tc.Why, gate.Detail)
			}
			if gate.Reason != tc.Reason {
				t.Fatalf("GateFiling(%+v, cap=%d).Reason = %q, want %q\n  why: %s", tc.Stats, tc.Cap, gate.Reason, tc.Reason, tc.Why)
			}
			if gate.Paused && gate.Detail == "" {
				t.Fatalf("a paused gate must say why in a form an operator can act on: %+v", gate)
			}
			if gate.Cap != tc.Cap || gate.Untriaged != tc.Stats.Untriaged {
				t.Fatalf("the gate must carry the cap and the stock it was measured against: %+v", gate)
			}
		})
	}
}

// TestSharedConversionCorpusSourceHealth pins the attribution: which lanes were
// attempted, which failed, and which failures are not source failures at all.
func TestSharedConversionCorpusSourceHealth(t *testing.T) {
	c := loadConversionCorpus(t)
	for _, tc := range c.LaneCases {
		t.Run(tc.Name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.FreshPerTopic = tc.FreshPerTopic
			got := SourceHealth(tc.Topics, cfg, tc.Errors)
			if !reflect.DeepEqual(got, tc.Expect) {
				t.Fatalf("SourceHealth() = %+v\n  want %+v\n  why: %s", got, tc.Expect, tc.Why)
			}
			if Degraded(got) != tc.Degraded {
				t.Fatalf("Degraded(%+v) = %v, want %v\n  why: %s", got, Degraded(got), tc.Degraded, tc.Why)
			}
			wantStatus := StatusOK
			if tc.Degraded {
				wantStatus = StatusDegraded
			}
			if status := RunStatus(FilingGate{}, got); status != wantStatus {
				t.Fatalf("RunStatus with health %+v = %q, want %q", got, status, wantStatus)
			}
		})
	}
}

// TestDegradedOutranksPaused pins the precedence the corpus declares in prose: a
// run whose source pool is incomplete cannot be summarised by the gate's verdict,
// because "nothing new worth filing" is not a claim an incomplete pool supports.
func TestDegradedOutranksPaused(t *testing.T) {
	paused := GateFiling(BacklogStats{Filed: 100, Untriaged: 99}, 40)
	if !paused.Paused {
		t.Fatalf("setup: expected a paused gate, got %+v", paused)
	}
	down := []LaneHealth{{Lane: "reddit", Attempted: 6, Failed: 6, Status: LaneDown}}
	if got := RunStatus(paused, down); got != StatusDegraded {
		t.Fatalf("RunStatus(paused, one lane down) = %q, want %q", got, StatusDegraded)
	}
	if got := RunStatus(paused, nil); got != StatusPaused {
		t.Fatalf("RunStatus(paused, healthy sources) = %q, want %q", got, StatusPaused)
	}
	if got := RunStatus(FilingGate{}, nil); got != StatusOK {
		t.Fatalf("RunStatus(open gate, healthy sources) = %q, want %q", got, StatusOK)
	}
}

// ============================================================================
// End to end: the gate has to be WIRED, not merely computable.
// ============================================================================

// gateFetcher serves one brand-new GitHub candidate (nothing dedups it) against a
// caller-supplied filing history, and records every issue creation attempt. It
// exists to answer one question: with a fresh, in-scope, un-filed candidate in
// hand, does a LIVE run still call `gh issue create` when the untriaged stock is
// over the cap?
type gateFetcher struct {
	scout      []ExistingIssue
	redditErr  error
	created    []IssuePlan
	labelsMade int
}

func (f *gateFetcher) FetchArxiv(string, int) (string, error) { return "", nil }
func (f *gateFetcher) FetchGitHub(string, int) ([]GitHubRepo, error) {
	return withDefaultRepoSize([]GitHubRepo{agedCandidateRepo()}), nil
}
func (f *gateFetcher) FetchGitHubFresh(string, int) ([]GitHubRepo, error) { return nil, nil }
func (f *gateFetcher) FetchHackerNews(string, int) (string, error)        { return "", nil }
func (f *gateFetcher) FetchReddit(string, int) (string, error)            { return "", f.redditErr }
func (f *gateFetcher) FetchExistingIssues(int) ([]ExistingIssue, error)   { return nil, nil }
func (f *gateFetcher) FetchScoutIssues(int) ([]ExistingIssue, error)      { return f.scout, nil }
func (f *gateFetcher) EnsureLabels() error                                { f.labelsMade++; return nil }
func (f *gateFetcher) AddToProject(string, string, string) error          { return nil }

func (f *gateFetcher) CreateIssue(issue IssuePlan, _ string) (string, error) {
	f.created = append(f.created, issue)
	return "https://github.com/o/r/issues/1", nil
}

// untriagedStock is `n` OPEN filings that still carry needs-triage, stamped with
// source IDs that match nothing the fetcher serves — so the ONLY thing that can
// stop today's filing is the conversion gate, never the dedup rungs.
func untriagedStock(n int, created string) []ExistingIssue {
	out := make([]ExistingIssue, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, ExistingIssue{
			Number:    100 + i,
			Title:     "idea-scout: an unrelated older filing",
			Body:      "<!-- idea-scout-source: arxiv:old-" + string(rune('a'+i)) + " -->",
			State:     "OPEN",
			CreatedAt: created,
			Labels:    []IssueLabel{{Name: ScoutLabel}, {Name: TriageLabel}},
		})
	}
	return out
}

// TestFilingPausesWhileTheUntriagedStockExceedsTheCap is the #6506 witness. Before
// the gate, this run filed: the candidate is new, in scope, above min_score and
// under max_issues, and every dedup rung passes it — which is exactly how 117
// untriaged issues accumulated while every run reported success.
func TestFilingPausesWhileTheUntriagedStockExceedsTheCap(t *testing.T) {
	dir := t.TempDir()
	fetcher := &gateFetcher{scout: untriagedStock(3, "2026-07-01T00:00:00Z")}
	now := time.Date(2026, 8, 11, 0, 0, 0, 0, time.UTC)

	result, err := Run(RunOptions{
		Workspace:  dir,
		ConfigPath: writeScoutConfig(t, dir, `"min_score": 1, "max_issues": 3, "scout_scan_limit": 50, "untriaged_cap": 2`),
		Fetcher:    fetcher,
		Live:       true,
		Today:      "2026-08-11",
		Now:        now,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if len(fetcher.created) != 0 {
		t.Fatalf("the scout filed %d issue(s) into a backlog already over the cap: %+v\n  the gate is the only thing between a novel candidate and an ever-growing untriaged stock", len(fetcher.created), fetcher.created)
	}
	if len(result.Filed) != 0 {
		t.Fatalf("filed = %+v, want nothing under a held gate", result.Filed)
	}
	if !result.FilingGate.Paused || result.FilingGate.Reason != GateUntriagedCap {
		t.Fatalf("filing_gate = %+v, want paused with reason %q", result.FilingGate, GateUntriagedCap)
	}
	if result.FilingGate.Cap != 2 || result.FilingGate.Untriaged != 3 {
		t.Fatalf("the gate must report the cap in force and the stock it measured: %+v", result.FilingGate)
	}
	if result.Status != StatusPaused {
		t.Fatalf("status = %q, want %q — a run that filed nothing because the gate held must not read as a plain success", result.Status, StatusPaused)
	}
	// The plan survives the pause: an operator draining the backlog still gets to
	// see what the scout would have filed.
	if len(result.Planned) != 1 {
		t.Fatalf("planned = %+v, want the one candidate still reported under a held gate", result.Planned)
	}
	if got := result.Backlog; got.Filed != 3 || got.Untriaged != 3 || got.OldestOpenDays != 41 {
		t.Fatalf("backlog ledger = %+v, want 3 filed / 3 untriaged / 41d oldest", got)
	}
	// Nothing was filed, so nothing may be recorded as filed.
	if _, err := os.Stat(CachePath(dir)); !os.IsNotExist(err) {
		t.Fatalf("a paused run wrote the seen-cache (stat err=%v); a source it never filed would then be treated as already filed", err)
	}

	var buf bytes.Buffer
	RenderHuman(&buf, result, DefaultConfig())
	out := buf.String()
	if !strings.Contains(out, "FILING PAUSED") || !strings.Contains(out, GateUntriagedCap) {
		t.Fatalf("the human report does not say the gate held:\n%s", out)
	}
	if strings.Contains(out, "FILED 1 issue") {
		t.Fatalf("the human report claims a filing that never happened:\n%s", out)
	}
}

// TestFilingResumesOnceTheStockIsDrainedUnderTheCap is the other half: the gate is
// a brake, not a kill switch, and it releases itself. "Re-enable only after
// draining" is therefore mechanical rather than a note somebody has to remember.
func TestFilingResumesOnceTheStockIsDrainedUnderTheCap(t *testing.T) {
	dir := t.TempDir()
	// Same three-issue history, but one has been triaged (the label came off) and
	// the cap now sits exactly at the remaining stock.
	stock := untriagedStock(3, "2026-07-01T00:00:00Z")
	stock[0].Labels = []IssueLabel{{Name: ScoutLabel}}
	fetcher := &gateFetcher{scout: stock}

	result, err := Run(RunOptions{
		Workspace:  dir,
		ConfigPath: writeScoutConfig(t, dir, `"min_score": 1, "max_issues": 3, "scout_scan_limit": 50, "untriaged_cap": 2`),
		Fetcher:    fetcher,
		Live:       true,
		Today:      "2026-08-11",
		Now:        time.Date(2026, 8, 11, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.FilingGate.Paused {
		t.Fatalf("the gate held at 2 untriaged against untriaged_cap=2: %+v\n  the cap is the ceiling the stock may EXCEED, not the last value it may reach", result.FilingGate)
	}
	if len(fetcher.created) != 1 || len(result.Filed) != 1 {
		t.Fatalf("a drained backlog must file again: created=%d filed=%+v", len(fetcher.created), result.Filed)
	}
	if result.Status != StatusOK {
		t.Fatalf("status = %q, want %q", result.Status, StatusOK)
	}
	if got := result.Backlog; got.Untriaged != 2 || got.Triaged != 1 {
		t.Fatalf("backlog ledger = %+v, want 2 untriaged / 1 triaged", got)
	}
}

// TestARunWithADeadSourceLaneIsDegraded pins the third Done condition: six Reddit
// 403s on six topics used to be six strings in an `errors` array under an exit-0
// run. The status now says the pool was incomplete.
func TestARunWithADeadSourceLaneIsDegraded(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	body := `{
		"topics": [
			{"key":"t1","github":"agent tool","reddit":"agent tool","terms":["agent","tool"]},
			{"key":"t2","github":"agent gateway","reddit":"agent gateway","terms":["agent","gateway"]}
		],
		"thresholds": {"min_score": 1, "max_issues": 3, "scout_scan_limit": 50, "fresh_per_topic": 0}
	}`
	if err := os.WriteFile(cfgPath, []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	fetcher := &gateFetcher{redditErr: errReddit403}

	result, err := Run(RunOptions{
		Workspace:  dir,
		ConfigPath: cfgPath,
		Fetcher:    fetcher,
		Today:      "2026-08-11",
		Now:        time.Date(2026, 8, 11, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.Status != StatusDegraded {
		t.Fatalf("status = %q, want %q: reddit failed on BOTH topics that armed it, so the candidate pool is incomplete and the run is not a plain success\n  health: %+v\n  errors: %v", result.Status, StatusDegraded, result.SourceHealth, result.Errors)
	}
	var reddit, github LaneHealth
	for _, lane := range result.SourceHealth {
		switch lane.Lane {
		case "reddit":
			reddit = lane
		case "github":
			github = lane
		}
	}
	if reddit.Status != LaneDown || reddit.Attempted != 2 || reddit.Failed != 2 {
		t.Fatalf("reddit health = %+v, want down 2/2", reddit)
	}
	if github.Status != LaneOK || github.Attempted != 2 {
		t.Fatalf("github health = %+v, want ok 2/2 — a healthy lane must not be dragged down with a dead one", github)
	}
	for _, lane := range result.SourceHealth {
		if lane.Lane == "github-fresh" {
			t.Fatalf("the fresh lane is disabled by fresh_per_topic=0 and must not be reported as attempted: %+v", lane)
		}
	}

	var buf bytes.Buffer
	RenderHuman(&buf, result, DefaultConfig())
	if out := buf.String(); !strings.Contains(out, "DEGRADED") || !strings.Contains(out, "source reddit") {
		t.Fatalf("the human report does not name the dead lane:\n%s", out)
	}
}

var errReddit403 = &laneError{"reddit status 403 Forbidden"}

type laneError struct{ msg string }

func (e *laneError) Error() string { return e.msg }
