package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// dev_workspace.go — `fak dev workspace`, the first honest increment of #3426
// (epic #3256, workstream E): the local agentic-dev workspace as one verb.
//
// #3426 asks for a single verb that stands up the whole governed local spine —
// guard floor + gateway + memory + opt-in dispatch loop + a LIVE DECISION STREAM
// + hot-reload + offline. Two facts bound what can land as one leaf:
//
//   - the bare `fak dev` spelling is already the C2 namespace router (#2231,
//     epic #2228), pinned by TestDevListingIsDevTierOnly; repurposing it would
//     break a shipped contract, so the workspace launch lives UNDER the namespace
//     as the sub-verb `fak dev workspace` — the maintainer's own recommended
//     collision-free spelling in the #3426 triage;
//   - the full orchestration is epic-scale and cannot land whole.
//
// So this increment ships the piece that is UNIQUE to fak and ALREADY wired: the
// live decision-stream readout over the guard floor's durable audit journals
// (.dispatch-runs/guard-audit/*.jsonl), folded into the allowed / blocked /
// quarantined / witnessed view the issue names — plus a map of the spine the
// developer assembles by hand today. It is a read-only OPERATOR READOUT (the
// gen/next expected artifact): it does NOT yet orchestrate the seat, the opt-in
// loop, hot-reload, or offline --gguf; those follow as separate witnessed leaves,
// named under `not_yet_wired`. `--json` emits the agent-runnable schema.

// devSpineComponent is one hand-assembled piece of the governed local loop the
// verb maps: the command that stands it up today and the role it plays.
type devSpineComponent struct {
	Name    string `json:"name"`
	Command string `json:"command"`
	Role    string `json:"role"`
	OptIn   bool   `json:"opt_in,omitempty"`
}

// devStreamRow is one adjudication row surfaced in the stream tail: the durable
// journal Row folded down to the operator-facing fields (Class is the
// allowed/blocked/quarantined fold of Kind+Verdict).
type devStreamRow struct {
	Seq     uint64 `json:"seq"`
	Kind    string `json:"kind"`
	Class   string `json:"class"`
	Tool    string `json:"tool,omitempty"`
	Verdict string `json:"verdict,omitempty"`
	Reason  string `json:"reason,omitempty"`
	Witness string `json:"witness,omitempty"`
}

// devStreamView is the folded live view of the guard floor's audit journals — the
// "live decision stream" the issue names. Decisions counts only adjudicated calls
// (allowed+blocked+quarantined); Rows is every journal row (lifecycle/capability
// rows included). Witnessed is orthogonal: a row of any class that surfaced a
// bounded-disclosure claim.
type devStreamView struct {
	Dir         string         `json:"dir"`
	Journals    int            `json:"journals"`
	Rows        int            `json:"rows"`
	Decisions   int            `json:"decisions"`
	Allowed     int            `json:"allowed"`
	Blocked     int            `json:"blocked"`
	Quarantined int            `json:"quarantined"`
	Witnessed   int            `json:"witnessed"`
	Last        []devStreamRow `json:"last,omitempty"`
}

// devWorkspaceReport is the machine-readable `fak dev workspace --json` schema.
// NotYet names the capabilities #3426 still owes; Promotion names the EVIDENCE
// that would move this gen/next preview to gen/now; InvalidatedIf names the
// assumption whose failure retires the verb instead of promoting it.
type devWorkspaceReport struct {
	Issue         string              `json:"issue"`
	Epic          string              `json:"epic"`
	Repo          string              `json:"repo"`
	Preview       bool                `json:"preview"`
	Spine         []devSpineComponent `json:"spine"`
	Stream        devStreamView       `json:"stream"`
	NotYet        []string            `json:"not_yet_wired"`
	Promotion     []string            `json:"promotion_evidence"`
	InvalidatedIf string              `json:"invalidated_if"`
}

// devWorkspaceSpine is the governed local loop the issue names as "assembled by
// hand" — the verbs `fak dev workspace` will one day orchestrate as one command.
func devWorkspaceSpine() []devSpineComponent {
	return []devSpineComponent{
		{Name: "seat", Command: "fak accounts launch", Role: "credentialed Claude/Codex seat launch"},
		{Name: "guard floor", Command: "fak guard -- claude", Role: "in-process gateway floor + SessionStart/PreCompact/Stop/toolproc hooks"},
		{Name: "memory / recall", Command: "fak memory / fak recall", Role: "durable cross-session continuity"},
		{Name: "dispatch loop", Command: "fak loop", Role: "governed issue-dispatch over the repo backlog", OptIn: true},
		{Name: "goal sync", Command: "fak goal sync push", Role: "sync durable goal specs and registry to fak-private"},
		{Name: "decision stream", Command: "fak dev workspace", Role: "live allowed/blocked/quarantined/witnessed view of the guard floor"},
	}
}

// classifyRow folds a durable journal Row into the operator-facing stream class
// the issue names, or "" for a row that is NOT a per-call adjudication (lifecycle
// spawn/retire, capability fault/evict, a vDSO cache hit). It mirrors the closed
// Kind/Verdict vocabulary in internal/journal (DECIDE|DENY|RESULT_DENY|QUARANTINE
// with verdict names ALLOW|DENY|QUARANTINE|...).
func classifyRow(r journal.Row) string {
	switch r.Kind {
	case "DENY", "RESULT_DENY":
		return "blocked"
	case "QUARANTINE":
		return "quarantined"
	case "DECIDE":
		switch r.Verdict {
		case "DENY":
			return "blocked"
		case "QUARANTINE":
			return "quarantined"
		default:
			return "allowed"
		}
	default:
		return ""
	}
}

// scanStreamView reads every guard-floor audit journal under the repo and folds
// the rows into the live view. A missing directory is an honest zero (no guard
// floor has run here yet), and a damaged journal tail is skipped rather than
// bricking the readout — `fak audit verify` is the tamper auditor, not this.
func scanStreamView(root string, limit int) (devStreamView, error) {
	dir := guardAuditDir(root)
	st := devStreamView{Dir: dir}
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return st, nil
		}
		return st, err
	}
	var rows []journal.Row
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		st.Journals++
		// Segment-aware (#6488): the view's counts (Rows/Allowed/Blocked/Decisions) are
		// totals, so they must span a rotation cut. Only the rendered Last-N is a tail.
		// The dir walk sees live *.jsonl files only; the sealed .cut-<seq> archives come
		// in through the segment read, so nothing is counted twice.
		rr, rerr := journal.ReadAllSegments(filepath.Join(dir, e.Name()))
		if rerr != nil {
			continue
		}
		rows = append(rows, journal.WithoutCutAnchors(rr)...)
	}
	// Merge the per-process chains into one wall-clock stream (ties broken by seq)
	// so the tail reads as one developer's session, not interleaved by filename.
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].TSUnixNano != rows[j].TSUnixNano {
			return rows[i].TSUnixNano < rows[j].TSUnixNano
		}
		return rows[i].Seq < rows[j].Seq
	})
	st.Rows = len(rows)
	for _, r := range rows {
		if r.Witness != "" {
			st.Witnessed++
		}
		switch classifyRow(r) {
		case "allowed":
			st.Allowed++
			st.Decisions++
		case "blocked":
			st.Blocked++
			st.Decisions++
		case "quarantined":
			st.Quarantined++
			st.Decisions++
		}
	}
	// The tail is the last `limit` ADJUDICATIONS (lifecycle/capability rows are not
	// decisions), rendered oldest -> newest.
	for i := len(rows) - 1; i >= 0 && len(st.Last) < limit; i-- {
		cls := classifyRow(rows[i])
		if cls == "" {
			continue
		}
		st.Last = append(st.Last, devStreamRow{
			Seq:     rows[i].Seq,
			Kind:    rows[i].Kind,
			Class:   cls,
			Tool:    rows[i].Tool,
			Verdict: rows[i].Verdict,
			Reason:  rows[i].Reason,
			Witness: rows[i].Witness,
		})
	}
	for i, j := 0, len(st.Last)-1; i < j; i, j = i+1, j-1 {
		st.Last[i], st.Last[j] = st.Last[j], st.Last[i]
	}
	return st, nil
}

func runDevWorkspace(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dev workspace", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "repository root (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the machine-readable workspace schema")
	limit := fs.Int("limit", 5, "decision-stream rows to show in the tail (0 hides the tail)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak dev workspace [--repo DIR] [--limit N] [--json]")
		fmt.Fprintln(stderr, "  the local agentic-dev workspace (#3426, epic #3256): the guard-floor")
		fmt.Fprintln(stderr, "  decision stream (allowed/blocked/quarantined/witnessed) + the spine map.")
		fmt.Fprintln(stderr, "  preview: maps and reports; full orchestration lands as follow-on leaves.")
	}
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *limit < 0 {
		fmt.Fprintln(stderr, "dev workspace: --limit must be >= 0")
		return 2
	}
	root := *repo
	if root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "dev workspace: %v\n", err)
			return 1
		}
		root = findRepoRoot(cwd)
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "dev workspace: %v\n", err)
		return 1
	}
	stream, err := scanStreamView(absRoot, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "dev workspace: reading decision stream: %v\n", err)
		return 1
	}
	rep := devWorkspaceReport{
		Issue:   "#3426",
		Epic:    "#3256",
		Repo:    absRoot,
		Preview: true,
		Spine:   devWorkspaceSpine(),
		Stream:  stream,
		NotYet: []string{
			"one-command orchestration of the seat + guard floor",
			"opt-in governed dispatch loop with witness-gated closure",
			"hot-reload of fak.toml / policy without dropping the session",
			"offline-capable --gguf (no key)",
		},
		Promotion: []string{
			"a dogfooded session where THIS verb launched the guard floor that wrote these rows (not a hand-assembled script)",
			"an operator triaging a blocked call from this readout instead of raw .dispatch-runs/guard-audit/*.jsonl",
			"the four not_yet_wired legs landed and witnessed, flipping preview to false",
		},
		InvalidatedIf: "a developer never wants a fak-owned frontdoor — if `fak guard -- claude` plus the existing TUI already covers the need, this verb is redundant and retires into `fak info`",
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "dev workspace: %v\n", err)
			return 1
		}
		return 0
	}
	printDevWorkspace(stdout, rep)
	return 0
}

func printDevWorkspace(w io.Writer, rep devWorkspaceReport) {
	fmt.Fprintf(w, "fak dev workspace — local agentic-dev workspace (preview; %s, epic %s)\n", rep.Issue, rep.Epic)
	fmt.Fprintf(w, "repo: %s\n\n", rep.Repo)

	fmt.Fprintln(w, "spine (the governed local loop, assembled by hand today):")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, c := range rep.Spine {
		name := c.Name
		if c.OptIn {
			name += " (opt-in)"
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\n", name, c.Command, c.Role)
	}
	tw.Flush()

	s := rep.Stream
	fmt.Fprintln(w)
	fmt.Fprintf(w, "decision stream (guard floor: %s):\n", s.Dir)
	if s.Journals == 0 {
		fmt.Fprintln(w, "  no guard-floor decision journals yet — run 'fak guard -- claude' in this repo.")
	} else {
		fmt.Fprintf(w, "  journals=%d rows=%d decisions=%d allowed=%d blocked=%d quarantined=%d witnessed=%d\n",
			s.Journals, s.Rows, s.Decisions, s.Allowed, s.Blocked, s.Quarantined, s.Witnessed)
		if len(s.Last) > 0 {
			fmt.Fprintln(w, "  last:")
			for _, r := range s.Last {
				detail := r.Reason
				if r.Witness != "" {
					detail = "witness=" + r.Witness
				}
				fmt.Fprintf(w, "    #%d %-11s %-12s %s %s\n", r.Seq, r.Class, r.Kind, r.Tool, detail)
			}
		}
	}

	fmt.Fprintln(w, "\nnot yet wired (follow-on leaves of #3426):")
	for _, n := range rep.NotYet {
		fmt.Fprintf(w, "  - %s\n", n)
	}

	fmt.Fprintln(w, "\npromotion evidence (what moves this gen/next preview to gen/now):")
	for _, p := range rep.Promotion {
		fmt.Fprintf(w, "  - %s\n", p)
	}
	fmt.Fprintf(w, "\ninvalidated if: %s\n", rep.InvalidatedIf)
}
