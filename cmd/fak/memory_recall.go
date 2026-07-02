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
// block. It runs the loop-recall driver over the notes backend, then re-pages
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
	asJSON := fs.Bool("json", false, "emit the envelope as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	backend, blabel := notesMemoryBackend(*store)
	d, ok := memq.Get("loop-recall")
	if !ok {
		fmt.Fprintln(stderr, "fak memory recall: loop-recall driver not registered")
		return 2
	}
	c := ctx()
	res, err := memq.Run(c, backend, d.Build(memq.Params{Intent: *intent, K: *k, Budget: *budget}), memq.Caps{})
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
