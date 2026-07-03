package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/memview"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// notesMemoryBackend resolves the markdown memory store dir (explicit --store,
// else the committed mirror .claude/memory under the repo root) and wraps it as
// the read-only, read-time-re-verified memq notes backend (#2347).
func notesMemoryBackend(store string) (*memq.NotesBackend, string) {
	dir := pathutil.ExpandTilde(store)
	if dir == "" {
		root := resolveRoot("")
		if root == "" {
			root = "."
		}
		dir = memoryread.DefaultStore(root)
	} else if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	b, _ := memq.NewNotesBackend(dir) // a missing store is an empty corpus, never an error
	return b, "memory store " + filepath.ToSlash(dir)
}

// recallNote is one note in the `fak memory recall` envelope: the rendered body
// with its freshness verdict, or a withheld entry with the refusal evidence.
type recallNote struct {
	ID       string                   `json:"id"`
	Title    string                   `json:"title,omitempty"`
	Verdict  string                   `json:"verdict"` // fresh | unverified | withheld:<reason>
	Detail   string                   `json:"detail,omitempty"`
	Findings []recall.ArtifactFinding `json:"findings,omitempty"`
	Body     string                   `json:"body,omitempty"`
}

type recallEnvelope struct {
	Store    string       `json:"store"`
	Intent   string       `json:"intent"`
	Rendered []recallNote `json:"rendered"`
	Withheld []recallNote `json:"withheld,omitempty"`
	Stats    memq.Stats   `json:"stats"`
}

// runMemoryRecall is `fak memory recall` (#2346 R1): the loop-turn orientation
// block. It runs a private recall query over the notes backend, then re-pages
// each rendered note to emit its body tagged with the read-time verdict —
// fresh (every concrete claim verified), unverified (no checkable claims, or a
// claim the verifier could not decide; render hedged), or withheld (stale claim
// / sealed by the trust gate; evidence named, body never emitted).
func runMemoryRecall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memory recall", flag.ContinueOnError)
	fs.SetOutput(stderr)
	intent := fs.String("intent", "the task at hand", "the turn's intent (relevance ranking)")
	store := fs.String("store", "", "memory store dir (default: the committed mirror .claude/memory)")
	k := fs.Int("k", 0, "max notes (0 = driver default 5)")
	budget := fs.Int64("budget", 0, "byte budget for the block (0 = driver default 8192)")
	asJSON := fs.Bool("json", false, "emit the envelope as JSON (the full struct, unaffected by --format)")
	format := fs.String("format", "markdown", "surface encoding: markdown (default, rich prose) | json | toon | any memview.Register'd format")
	listFormats := fs.Bool("list-formats", false, "print the registered surface formats and exit")
	ablateFormats := fs.String("ablate-formats", "", "measure this note set's byte/token cost under every named format (comma-separated, or \"all\") instead of rendering")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *listFormats {
		fmt.Fprintln(stdout, strings.Join(memview.KnownFormats(), "\n"))
		return 0
	}

	backend, blabel := notesMemoryBackend(*store)
	c := ctx()
	res, err := memq.Run(c, backend, memoryRecallQuery(*intent, *k, *budget), memq.Caps{})
	if err != nil {
		fmt.Fprintf(stderr, "fak memory recall: %v\n", err)
		return 1
	}

	env := recallEnvelope{Store: backend.Dir(), Intent: *intent, Stats: res.Stats}
	cells, _ := backend.Cells(c)
	titles := make(map[string]string, len(cells))
	for _, cell := range cells {
		titles[cell.ID] = cell.Attrs["title"]
	}
	for _, it := range res.Rendered {
		env.Rendered = append(env.Rendered, recallRendered(c, backend, it.ID, titles[it.ID]))
	}
	for _, rf := range res.Refused {
		n := recallNote{ID: rf.ID, Title: titles[rf.ID], Verdict: "withheld:" + rf.Reason}
		// Name the evidence: the stale claim(s) that refused the page-in.
		if findings, verr := backend.Verify(c, rf.ID); verr == nil {
			for _, f := range findings {
				if f.Status == recall.ArtifactStale {
					n.Detail = fmt.Sprintf("%s %q: %s", f.Claim.Kind, f.Claim.Value, f.Detail)
					n.Findings = findings
					break
				}
			}
		}
		env.Withheld = append(env.Withheld, n)
	}

	if *asJSON {
		fmt.Fprintln(stdout, string(jsonIndent(env)))
		return 0
	}

	surface, serr := recallSurface(env)
	if serr != nil {
		fmt.Fprintf(stderr, "fak memory recall: %v\n", serr)
		return 1
	}
	if *ablateFormats != "" {
		formats, ferr := parseFormatList(*ablateFormats)
		if ferr != nil {
			fmt.Fprintf(stderr, "fak memory recall: %v\n", ferr)
			return 2
		}
		metrics, merr := memview.SweepFormats(surface, formats)
		if merr != nil {
			fmt.Fprintf(stderr, "fak memory recall: %v\n", merr)
			return 2
		}
		fmt.Fprintln(stdout, formatMetricsTable(metrics))
		return 0
	}
	if *format != "markdown" {
		body, eerr := memview.Encode(memview.Format(*format), surface)
		if eerr != nil {
			fmt.Fprintf(stderr, "fak memory recall: %v\n", eerr)
			return 2
		}
		stdout.Write(body)
		return 0
	}
	fmt.Fprintf(stdout, "# Verified memory recall (%s; intent: %q)\n", blabel, *intent)
	for _, n := range env.Rendered {
		fmt.Fprintf(stdout, "\n## %s (%s) [%s]\n\n%s\n", n.Title, n.ID, n.Verdict, strings.TrimRight(n.Body, "\n"))
	}
	if len(env.Rendered) == 0 {
		fmt.Fprintln(stdout, "\n(no notes rendered — empty store, or nothing relevant within budget)")
	}
	if len(env.Withheld) > 0 {
		fmt.Fprintln(stdout, "\nwithheld (never injected as fact):")
		for _, n := range env.Withheld {
			fmt.Fprintf(stdout, "  - %s [%s] %s\n", n.ID, n.Verdict, n.Detail)
		}
	}
	fmt.Fprintf(stdout, "\nstats: scanned=%d rendered=%d withheld=%d ~tokens=%d\n",
		env.Stats.CellsScanned, env.Stats.Rendered, len(env.Withheld), env.Stats.EstimatedTokens)
	return 0
}

// recallSurface translates a recallEnvelope into a memview.Surface — the
// caller-translates-in step every memview format consumer performs (the same
// discipline cmd/fak already applies for memview's ProvenanceEvent). Rendered
// notes carry their full body; withheld notes carry the refusal Detail instead,
// so the surfaced table stays faithful to what the text-mode renderer already
// prints (a withheld note's body is never surfaced under ANY format).
func recallSurface(env recallEnvelope) (memview.Surface, error) {
	fields := []string{"id", "title", "verdict", "body_or_detail"}
	rows := make([]memview.Row, 0, len(env.Rendered)+len(env.Withheld))
	for _, n := range env.Rendered {
		rows = append(rows, memview.Row{n.ID, n.Title, n.Verdict, strings.TrimRight(n.Body, "\n")})
	}
	for _, n := range env.Withheld {
		rows = append(rows, memview.Row{n.ID, n.Title, n.Verdict, n.Detail})
	}
	return memview.NewSurface("Verified memory recall", fields, rows)
}

// parseFormatList turns an --ablate-formats value into a format list: "all"
// (or "") sweeps every registered format, else a comma-separated name list.
func parseFormatList(raw string) ([]memview.Format, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" || strings.EqualFold(raw, "all") {
		return nil, nil
	}
	var out []memview.Format
	for _, tok := range strings.Split(raw, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		out = append(out, memview.Format(tok))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--ablate-formats: no format names in %q", raw)
	}
	return out, nil
}

// formatMetricsTable renders a SweepFormats result as a small fixed-width table —
// the ablation readout: same content, one row per format, byte/token cost.
func formatMetricsTable(metrics []memview.FormatMetrics) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-10s %10s %10s\n", "format", "bytes", "~tokens")
	for _, m := range metrics {
		fmt.Fprintf(&b, "%-10s %10d %10d\n", m.Format, m.Bytes, m.EstimatedTokens)
	}
	return strings.TrimRight(b.String(), "\n")
}

// memoryRecallQuery is intentionally not registered as a `fak memory drivers`
// strategy. #2347 adds a backend and CLI sugar over the markdown store; authored
// queries still use the existing driver list through `fak memory run --backend notes`.
func memoryRecallQuery(intent string, k int, budget int64) memq.Query {
	if k <= 0 {
		k = 5
	}
	if budget <= 0 {
		budget = 8192
	}
	return memq.Query{
		Intent: intent,
		Ops: []memq.Op{
			{Kind: memq.OpScan},
			{Kind: memq.OpFilter, Pred: &memq.Pred{Op: memq.PredAnd, Args: []memq.Pred{
				{Op: memq.PredEq, Field: "sealed", Value: "false"},
				{Op: memq.PredEq, Field: "tombstoned", Value: "false"},
			}}},
			{Kind: memq.OpRank, By: memq.RankRelevance, Desc: true},
			{Kind: memq.OpLimit, K: k},
			{Kind: memq.OpBudget, Bytes: budget},
			{Kind: memq.OpRender},
		},
	}
}

// recallRendered re-pages one rendered note (the same gated Materialize the
// executor ran) and tags it from the Verify seam: worst finding wins, and a
// note with nothing checkable is honestly "unverified", never "fresh".
func recallRendered(c context.Context, b *memq.NotesBackend, id, title string) recallNote {
	n := recallNote{ID: id, Title: title}
	body, err := b.Materialize(c, id)
	if err != nil {
		// The render already succeeded once; a refusal here means the world moved
		// between page-ins. Report it as withheld-shaped rather than fabricate.
		n.Verdict, n.Detail = "withheld:reverify", err.Error()
		return n
	}
	n.Body = string(body)
	findings, err := b.Verify(c, id)
	if err != nil {
		n.Verdict, n.Detail = "unverified", err.Error()
		return n
	}
	n.Findings = findings
	n.Verdict = "fresh"
	if len(findings) == 0 {
		n.Verdict, n.Detail = "unverified", "no concrete artifact claims to check"
		return n
	}
	for _, f := range findings {
		if f.Status != recall.ArtifactFresh {
			n.Verdict = "unverified"
			n.Detail = fmt.Sprintf("%s %q: %s", f.Claim.Kind, f.Claim.Value, f.Detail)
			break
		}
	}
	return n
}
