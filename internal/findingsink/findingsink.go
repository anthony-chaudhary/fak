// Package findingsink is a general-purpose sink seam for scorecard findings: a producer folds
// its debt into neutral Findings and hands them to a Sink, without knowing whether the sink is
// a terminal dry-run, a durable local ledger, or GitHub issues.
//
// It generalizes the GitHub-only fan-out that each fak dispatcher (unwired-debt-dispatch, ...)
// wired inline. A Finding carries the sink-agnostic fields every sink needs (Key, Title, Grade,
// debt, Paths, Labels, a Summary and a Body); it may also carry the rich dogfoodissues
// ActionItem so the GitHub sink emits a fully-detailed issue rather than a thin one.
//
// Three sinks ship today:
//   - StdoutSink  -- the dry-run default: prints the planned findings, mutates nothing.
//   - LocalDBSink -- a durable append/upsert JSONL ledger under .fak/, content-addressed by
//     Key, deduped and updated in place on re-run. The local half of "sinks can be a local db
//     and also GitHub issues".
//   - GitHubSink  -- adapts Findings onto internal/dogfoodissues and reuses its dedup + gh sync.
//
// Invariant: Finding sink operations are fail-closed and deterministic across dry-run and live modes.
// Contract: Dry-run sinks guarantee that no persistent storage, local ledger, or remote issues are modified.
// Precondition: Each finding emitted to a sink must define a unique key identifying the debt unit.
// Guard: Sinks enforce bounds on finding batch size and gracefully quarantine corrupt record entries.
package findingsink

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// Finding is one sink-agnostic unit of scorecard debt.
type Finding struct {
	Key          string   `json:"key"`
	Title        string   `json:"title"`
	Grade        string   `json:"grade,omitempty"`
	DebtName     string   `json:"debt_name,omitempty"`
	DebtCount    int      `json:"debt_count,omitempty"`
	Paths        []string `json:"paths,omitempty"`
	Labels       []string `json:"labels,omitempty"`
	EvidencePath string   `json:"evidence_path,omitempty"`
	Summary      string   `json:"summary,omitempty"`
	Body         string   `json:"body,omitempty"`

	// issue is the optional rich form used by the GitHub sink; unexported so it never leaks
	// into the neutral local-db/stdout serialization.
	issue *dogfoodissues.ActionItem
}

// FromActionItem projects a rich dogfoodissues.ActionItem onto a neutral Finding, keeping the
// ActionItem attached for the GitHub sink.
func FromActionItem(ai dogfoodissues.ActionItem) Finding {
	cp := ai
	return Finding{
		Key:          ai.Key,
		Title:        ai.Title,
		Grade:        ai.Grade,
		DebtName:     ai.DebtName,
		DebtCount:    ai.DebtCount,
		Paths:        ai.Paths,
		Labels:       ai.Labels,
		EvidencePath: ai.EvidencePath,
		Summary:      ai.NextAction,
		Body:         ai.CurrentState,
		issue:        &cp,
	}
}

// FromActionItems maps a batch.
func FromActionItems(items []dogfoodissues.ActionItem) []Finding {
	out := make([]Finding, 0, len(items))
	for _, ai := range items {
		out = append(out, FromActionItem(ai))
	}
	return out
}

// actionItem returns the attached rich item, or synthesizes a minimal one from neutral fields
// so a Finding produced without an ActionItem still yields a usable GitHub issue.
func (f Finding) actionItem(evidence string) dogfoodissues.ActionItem {
	if f.issue != nil {
		return *f.issue
	}
	ev := evidence
	if f.EvidencePath != "" {
		ev = f.EvidencePath
	}
	return dogfoodissues.ActionItem{
		Key:          f.Key,
		Title:        f.Title,
		Grade:        f.Grade,
		DebtName:     f.DebtName,
		DebtCount:    f.DebtCount,
		EvidencePath: ev,
		NextAction:   f.Summary,
		CurrentState: f.Body,
		Paths:        f.Paths,
		Labels:       f.Labels,
	}
}

// EmitOptions carries the cross-sink knobs. Live flips a sink from dry-run to committing.
type EmitOptions struct {
	Live               bool
	Repo               string   // GitHub owner/repo ("" = current repo)
	Cap                int      // max findings to act on in one run (<=0 = no cap)
	Dir                string   // LocalDBSink workspace root (the ledger lands under <Dir>/.fak)
	Labels             []string // extra labels for created GitHub issues
	Evidence           string   // default evidence path for synthesized issues
	Limit              int      // existing-issue scan limit for the GitHub sink (0 = 300)
	ParentIssue        int
	ParentBaseline     float64
	CompletionStandard string
	TargetEnvelope     string
	WitnessedEnvelope  string
}

// Row is one sink outcome for one finding.
type Row struct {
	Key    string `json:"key"`
	Action string `json:"action"` // plan | create | update | noop | skip | error
	OK     bool   `json:"ok"`
	Ref    string `json:"ref,omitempty"` // URL (github) or ledger path (localdb)
	Detail string `json:"detail,omitempty"`
}

// Report is the machine-readable fold of a single Emit.
type Report struct {
	Sink    string `json:"sink"`
	Mode    string `json:"mode"` // dry-run | live
	Rows    []Row  `json:"rows"`
	Planned int    `json:"planned"`
	Created int    `json:"created"`
	Updated int    `json:"updated"`
	Skipped int    `json:"skipped"`
}

// Sink is the general-purpose destination for a batch of findings.
type Sink interface {
	Name() string
	Emit(findings []Finding, opt EmitOptions) (Report, error)
}

func mode(live bool) string {
	if live {
		return "live"
	}
	return "dry-run"
}

func capped(findings []Finding, cap int) []Finding {
	if cap > 0 && len(findings) > cap {
		return findings[:cap]
	}
	return findings
}

// ---- StdoutSink ----------------------------------------------------------------------------

// StdoutSink prints the planned findings and mutates nothing. It is the safe default.
type StdoutSink struct{ W io.Writer }

// Name returns the canonical identifier token for the stdout destination.
func (s StdoutSink) Name() string { return "stdout" }

// Emit prints the planned findings to the configured writer without modifying persistent state.
func (s StdoutSink) Emit(findings []Finding, opt EmitOptions) (Report, error) {
	findings = capped(findings, opt.Cap)
	w := s.W
	if w == nil {
		w = os.Stdout
	}
	rep := Report{Sink: s.Name(), Mode: "dry-run", Planned: len(findings)}
	for _, f := range findings {
		fmt.Fprintf(w, "PLAN  %-40s %s\n", f.Key, f.Title)
		if f.Summary != "" {
			fmt.Fprintf(w, "      -> %s\n", f.Summary)
		}
		rep.Rows = append(rep.Rows, Row{Key: f.Key, Action: "plan", OK: true, Detail: f.Title})
	}
	return rep, nil
}

// ---- LocalDBSink ---------------------------------------------------------------------------

// LocalDBSink upserts findings into a durable JSONL ledger under <Dir>/.fak, content-addressed
// by Key: a re-run updates an existing record in place and appends new ones, so the ledger
// converges instead of growing a duplicate per run. Under a dry-run it reports the plan
// (create vs update) without touching disk.
type LocalDBSink struct {
	// File overrides the ledger path (default <opt.Dir or .>/.fak/checkpoint-findings.jsonl).
	File string
}

// Name returns the canonical identifier token for the local JSONL database sink.
func (s LocalDBSink) Name() string { return "localdb" }

// ledgerRecord is one durable row: the neutral finding plus a monotone emit Count (deterministic
// given the inputs -- no wall-clock, so runs are reproducible).
type ledgerRecord struct {
	Finding
	State string `json:"state"` // open
	Count int    `json:"count"` // times this key has been emitted
}

func (s LocalDBSink) path(opt EmitOptions) string {
	if s.File != "" {
		return s.File
	}
	root := opt.Dir
	if root == "" {
		root = "."
	}
	return filepath.Join(root, ".fak", "checkpoint-findings.jsonl")
}

func (s LocalDBSink) load(path string) (map[string]ledgerRecord, []string, error) {
	byKey := map[string]ledgerRecord{}
	var order []string
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return byKey, order, nil
		}
		return nil, nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec ledgerRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue // tolerate a corrupt line rather than lose the ledger
		}
		if _, seen := byKey[rec.Key]; !seen {
			order = append(order, rec.Key)
		}
		byKey[rec.Key] = rec
	}
	return byKey, order, sc.Err()
}

// Emit upserts findings into the local JSONL ledger or reports planned operations in dry-run mode.
func (s LocalDBSink) Emit(findings []Finding, opt EmitOptions) (Report, error) {
	findings = capped(findings, opt.Cap)
	path := s.path(opt)
	rep := Report{Sink: s.Name(), Mode: mode(opt.Live), Planned: len(findings)}

	byKey, order, err := s.load(path)
	if err != nil {
		return rep, fmt.Errorf("load ledger %s: %w", path, err)
	}

	for _, f := range findings {
		prev, exists := byKey[f.Key]
		action := "create"
		if exists {
			action = "update"
		}
		if action == "create" {
			rep.Created++
			order = append(order, f.Key)
		} else {
			rep.Updated++
		}
		byKey[f.Key] = ledgerRecord{Finding: f, State: "open", Count: prev.Count + 1}
		rep.Rows = append(rep.Rows, Row{Key: f.Key, Action: action, OK: true, Ref: path})
	}

	if !opt.Live {
		rep.Mode = "dry-run"
		return rep, nil
	}
	if err := s.write(path, byKey, order); err != nil {
		return rep, err
	}
	return rep, nil
}

func (s LocalDBSink) write(path string, byKey map[string]ledgerRecord, order []string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir ledger dir: %w", err)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create ledger: %w", err)
	}
	w := bufio.NewWriter(f)
	// Stable output: emit in first-seen order (dedup keys defensively).
	seen := map[string]bool{}
	stable := make([]string, 0, len(order))
	for _, k := range order {
		if !seen[k] {
			seen[k] = true
			stable = append(stable, k)
		}
	}
	for _, k := range stable {
		rec := byKey[k]
		b, err := json.Marshal(rec)
		if err != nil {
			f.Close()
			return err
		}
		w.Write(b)
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ---- GitHubSink ----------------------------------------------------------------------------

// GitHubSink adapts findings onto internal/dogfoodissues and reuses its marker-keyed dedup and
// gh create/edit sync. Dry-run builds and reports the plan; live fetches existing issues,
// dedups, and syncs.
type GitHubSink struct{}

// Name returns the canonical identifier token for the GitHub issue sink.
func (s GitHubSink) Name() string { return "github" }

// Emit synchronizes findings against GitHub issues, building plans or applying remote edits.
func (s GitHubSink) Emit(findings []Finding, opt EmitOptions) (Report, error) {
	findings = capped(findings, opt.Cap)
	rep := Report{Sink: s.Name(), Mode: mode(opt.Live), Planned: len(findings)}

	items := make([]dogfoodissues.ActionItem, 0, len(findings))
	for _, f := range findings {
		items = append(items, f.actionItem(opt.Evidence))
	}

	var existing []dogfoodissues.Issue
	if opt.Live {
		limit := opt.Limit
		if limit <= 0 {
			limit = 300
		}
		var err error
		existing, err = dogfoodissues.FetchExistingIssues(opt.Repo, limit)
		if err != nil {
			return rep, err
		}
	}

	// DedupeCap == 0 means cap-to-zero in dogfoodissues (it would skip every item), so default
	// to a generous cap when the caller set none. The finding count is already bounded upstream
	// by capped(); this cap only governs the dedup pass.
	dedupeCap := 300
	if opt.Cap > 0 {
		dedupeCap = opt.Cap
	}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, dogfoodissues.BuildOptions{
		Live:          opt.Live,
		DedupeChecked: opt.Live,
		DedupeCap:     dedupeCap,
		ParentIssue:   opt.ParentIssue, ParentBaseline: opt.ParentBaseline, CompletionStandard: opt.CompletionStandard, TargetEnvelope: opt.TargetEnvelope, WitnessedEnvelope: opt.WitnessedEnvelope,
	})
	rep.Planned = len(plan)
	rep.Skipped = len(skipped)
	for _, sk := range skipped {
		rep.Rows = append(rep.Rows, Row{Key: sk.Key, Action: "skip", OK: false, Detail: sk.Reason})
	}
	for _, p := range plan {
		rep.Rows = append(rep.Rows, Row{Key: p.Key, Action: p.Action, OK: true})
	}

	if !opt.Live {
		return rep, nil
	}
	synced := dogfoodissues.Sync(plan, opt.Repo, opt.Labels, nil)
	rep.Rows = rep.Rows[:0]
	for _, row := range synced {
		r := Row{Key: row.Key, Action: row.Action, OK: row.OK, Ref: row.URL}
		if !row.OK {
			r.Detail = strings.TrimSpace(row.Stderr)
		}
		switch row.Action {
		case "create":
			rep.Created++
		case "update":
			rep.Updated++
		}
		rep.Rows = append(rep.Rows, r)
	}
	return rep, nil
}
