// cron_chain.go turns `fak cron` from a single-shot scheduler into a witnessed
// PIPELINE engine (#2888, the Hermes-inspiration epic #2871). Hermes cron jobs
// carry a `script` (a pre-run command whose stdout is injected into the prompt)
// and a `context_from` (chain job A's last output into job B's prompt). fak adopts
// both — but does it better by making every injected byte AUDITABLE:
//
//   - script output is recorded as a `fak-cron-output/1` ledger row tagged
//     provenance=OBSERVED, so a downstream reader can tell tool-collected data
//     apart from agent-authored text (the whole point — an injected script's
//     stdout must never masquerade as something the agent reasoned out).
//   - the A→B handoff is a `fak-cron-edge/1` ledger row, not a silent string
//     pass-through, so a chained pipeline is a traversable graph after the fact.
//
// `fak cron prompt --job B --context-from A` therefore refuses to chain from a
// job that has no WITNESSED output on the ledger — B can only consume what A
// provably produced. This is the fak-does-it-better angle: the injection and the
// handoff are queryable evidence, not trust.
//
// Scope (this slice): the `script` and `context_from` schedule-spec fields plus
// the OBSERVED provenance label and the ledger edge. Hermes' `workdir`
// (AGENTS.md/CLAUDE.md loading) and the skills/model/provider overrides are NOT
// adopted here (issue #2888 "Out of scope").
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	cronOutputSchema = "fak-cron-output/1" // a witnessed pre-run script output row
	cronEdgeSchema   = "fak-cron-edge/1"   // a witnessed A→B context handoff row

	// cronProvObserved labels an injected block as tool-collected (OBSERVED), the
	// provenance that keeps a script's stdout distinguishable from agent-authored
	// prompt text. It is the load-bearing word of this whole feature.
	cronProvObserved = "observed"
)

// cronOutputRecord is one pre-run script's witnessed stdout in the ledger. Slot
// keys it to the schedule slot (shared with the fire witness); Provenance is
// always cronProvObserved — a script output is collected, never authored.
type cronOutputRecord struct {
	Schema     string `json:"schema"`
	Job        string `json:"job"`
	Slot       string `json:"slot"`       // RFC3339 UTC, tick truncated to the interval
	Provenance string `json:"provenance"` // always "observed" — tool output, not authored
	Source     string `json:"source"`     // the script command that produced Text
	Text       string `json:"text"`       // the captured stdout (the injected bytes)
	ProducedAt string `json:"produced_at"`
}

// cronEdgeRecord is one witnessed A→B handoff: job ToJob consumed job FromJob's
// last OBSERVED output. It makes `context_from` a traversable ledger edge instead
// of an untraceable string pass-through.
type cronEdgeRecord struct {
	Schema     string `json:"schema"`
	FromJob    string `json:"from_job"`
	FromSlot   string `json:"from_slot"`
	ToJob      string `json:"to_job"`
	ToSlot     string `json:"to_slot"`
	ConsumedAt string `json:"consumed_at"`
}

// runCronPrompt assembles a job's witnessed prompt. With --script it runs the
// pre-run command and records its stdout as an OBSERVED ledger row; with
// --context-from A[,B...] it injects each named job's last witnessed output and
// records an A→B edge per source. The assembled prompt (OBSERVED context blocks
// followed by the authored --base text) is printed to stdout for a dispatcher to
// hand the agent. Exit 0 = assembled; exit 2 = usage/IO error or an unwitnessed
// context source (fail closed rather than chain from nothing).
func runCronPrompt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron prompt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	job := fs.String("job", "", "job/loop id this prompt belongs to (required)")
	ledger := fs.String("ledger", "", "witness ledger path, JSONL (required)")
	script := fs.String("script", "", "pre-run command; its stdout is recorded OBSERVED and injected")
	contextFrom := fs.String("context-from", "", "comma-separated upstream job ids whose last witnessed output to chain in")
	base := fs.String("base", "", "the authored base prompt appended after the OBSERVED context")
	interval := fs.Duration("interval", 0, "firing cadence; the tick is quantized to this slot (0 = one-shot)")
	at := fs.String("at", "", "wall-clock tick time (RFC3339); default now — injectable for tests")
	slot := fs.String("slot", "", "override the computed slot key directly (default: --at truncated to --interval)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*job) == "" {
		fmt.Fprintln(stderr, "fak cron prompt: --job is required")
		return 2
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak cron prompt: --ledger is required")
		return 2
	}

	now, slotKey, ok := resolveCronTimeAndSlot(stderr, "fak cron prompt", *at, *slot, *interval)
	if !ok {
		return 2
	}

	var blocks []string

	// --context-from: chain each upstream job's LAST witnessed output. A source
	// with no OBSERVED output on the ledger is refused (exit 2) — B consumes only
	// what A provably produced, and the handoff is recorded as an edge.
	for _, src := range splitCommaList(*contextFrom) {
		up, ok, err := cronLatestOutput(*ledger, src)
		if err != nil {
			fmt.Fprintf(stderr, "fak cron prompt: read ledger: %v\n", err)
			return 2
		}
		if !ok {
			fmt.Fprintf(stderr, "fak cron prompt: --context-from %q has no witnessed output in %s (refusing to chain from nothing)\n", src, *ledger)
			return 2
		}
		blocks = append(blocks, cronObservedBlock(up.Job, up.Slot, up.Source, up.Text))
		edge := cronEdgeRecord{
			Schema:     cronEdgeSchema,
			FromJob:    up.Job,
			FromSlot:   up.Slot,
			ToJob:      *job,
			ToSlot:     slotKey,
			ConsumedAt: now.Format(time.RFC3339),
		}
		if err := cronAppendJSONL(*ledger, edge); err != nil {
			fmt.Fprintf(stderr, "fak cron prompt: append edge: %v\n", err)
			return 2
		}
	}

	// --script: run the pre-run command, witness its stdout as OBSERVED, and
	// inject it into this job's own prompt (Hermes' `script` field).
	if strings.TrimSpace(*script) != "" {
		text, err := cronRunScript(*script)
		if err != nil {
			fmt.Fprintf(stderr, "fak cron prompt: %v\n", err)
			return 2
		}
		out := cronOutputRecord{
			Schema:     cronOutputSchema,
			Job:        *job,
			Slot:       slotKey,
			Provenance: cronProvObserved,
			Source:     *script,
			Text:       text,
			ProducedAt: now.Format(time.RFC3339),
		}
		if err := cronAppendJSONL(*ledger, out); err != nil {
			fmt.Fprintf(stderr, "fak cron prompt: append output: %v\n", err)
			return 2
		}
		blocks = append(blocks, cronObservedBlock(*job, slotKey, *script, text))
	}

	var b strings.Builder
	for _, blk := range blocks {
		b.WriteString(blk)
		b.WriteString("\n")
	}
	if base := strings.TrimSpace(*base); base != "" {
		b.WriteString(base)
		b.WriteString("\n")
	}
	fmt.Fprint(stdout, b.String())
	return 0
}

// runCronChain reads the A→B handoff edges back OUT of the ledger, so an operator
// can traverse a chained pipeline after the fact — the readback half of the
// `context_from` audit trail (#2888). runCronPrompt WRITES `fak-cron-edge/1` rows;
// without a reader they are write-only and the "auditable end-to-end" claim is only
// half-kept. This prints each recorded handoff as `FROM@slot -> TO@slot (consumed …)`
// (or --json for a machine read), optionally filtered to the edges touching --job
// (either side). Exit 0 = printed (empty ledger is a valid empty chain); exit 2 =
// usage/IO error.
func runCronChain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron chain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "witness ledger path, JSONL (required)")
	jobFilter := fs.String("job", "", "restrict to edges touching this job id (as source OR consumer)")
	asJSON := fs.Bool("json", false, "emit the edges as JSON instead of a table")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak cron chain: --ledger is required")
		return 2
	}
	edges, err := cronReadEdges(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron chain: read ledger: %v\n", err)
		return 2
	}
	if jf := strings.TrimSpace(*jobFilter); jf != "" {
		kept := edges[:0:0]
		for _, e := range edges {
			if e.FromJob == jf || e.ToJob == jf {
				kept = append(kept, e)
			}
		}
		edges = kept
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if edges == nil {
			edges = []cronEdgeRecord{} // encode an empty chain as [], never null
		}
		if err := enc.Encode(edges); err != nil {
			fmt.Fprintf(stderr, "fak cron chain: encode: %v\n", err)
			return 2
		}
		return 0
	}
	if len(edges) == 0 {
		fmt.Fprintln(stdout, "no cron chain edges recorded")
		return 0
	}
	for _, e := range edges {
		fmt.Fprintf(stdout, "%s@%s -> %s@%s  (consumed %s)\n",
			e.FromJob, e.FromSlot, e.ToJob, e.ToSlot, e.ConsumedAt)
	}
	return 0
}

// cronObservedBlock wraps an injected input in an OBSERVED provenance frame so a
// downstream reader (human or agent) can tell tool-collected data apart from
// agent-authored text. The frame names the origin job/slot and the producing
// script — the audit trail that makes the injection evidence, not trust.
func cronObservedBlock(job, slot, source, text string) string {
	src := fmt.Sprintf("cron job %q slot %s", job, slot)
	if strings.TrimSpace(source) != "" {
		src += " (script: " + strings.TrimSpace(source) + ")"
	}
	var b strings.Builder
	b.WriteString("=== BEGIN OBSERVED context — provenance: OBSERVED (tool output, NOT agent-authored) ===\n")
	b.WriteString("source: " + src + "\n")
	b.WriteString("---\n")
	b.WriteString(strings.TrimRight(text, "\r\n"))
	b.WriteString("\n=== END OBSERVED context ===\n")
	return b.String()
}

// cronRunScript runs a pre-run script through the platform shell and returns its
// captured stdout. A non-zero exit (or a shell that cannot start) is an error, so
// a broken collector fails the prompt closed rather than injecting empty context.
func cronRunScript(script string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	var c *exec.Cmd
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(ctx, "cmd", "/c", script)
	} else {
		c = exec.CommandContext(ctx, "sh", "-c", script)
	}
	c.WaitDelay = 5 * time.Second
	c.Cancel = func() error {
		if c.Process != nil && c.Process.Pid > 0 {
			procguard.KillPID(c.Process.Pid)
		}
		return nil
	}
	var out, errb bytes.Buffer
	c.Stdout = &out
	c.Stderr = &errb
	if err := c.Run(); err != nil {
		msg := strings.TrimSpace(errb.String())
		if msg != "" {
			return out.String(), fmt.Errorf("script %q failed: %v: %s", script, err, msg)
		}
		return out.String(), fmt.Errorf("script %q failed: %v", script, err)
	}
	return out.String(), nil
}

// cronLatestOutput returns the most-recent (last-appended) witnessed output row
// for job, ok=false when the job has none. Append order is chronological, so the
// last matching row is the job's "last output" in the Hermes sense.
func cronLatestOutput(ledger, job string) (cronOutputRecord, bool, error) {
	outs, err := cronReadOutputs(ledger)
	if err != nil {
		return cronOutputRecord{}, false, err
	}
	var latest cronOutputRecord
	found := false
	for _, o := range outs {
		if o.Job == job {
			latest = o
			found = true
		}
	}
	return latest, found, nil
}

// cronReadOutputs loads well-formed OBSERVED output rows from the ledger. A
// missing ledger is an empty history, not an error.
func cronReadOutputs(path string) ([]cronOutputRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return jsonlledger.Parse(string(b), func(r cronOutputRecord) bool {
		return r.Schema == cronOutputSchema && r.Job != ""
	}), nil
}

// cronReadEdges loads well-formed A→B handoff rows from the ledger. A missing
// ledger is an empty history, not an error.
func cronReadEdges(path string) ([]cronEdgeRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return jsonlledger.Parse(string(b), func(r cronEdgeRecord) bool {
		return r.Schema == cronEdgeSchema && r.FromJob != "" && r.ToJob != ""
	}), nil
}

// cronAppendJSONL appends one record as a JSONL line, creating the ledger (and
// its dir) on first write. Shared by the output and edge writers so the chain
// rows land in the same ledger the fire witness uses.
func cronAppendJSONL(path string, rec any) error {
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}
