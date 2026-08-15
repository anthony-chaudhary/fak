// Command fak-dos is the writable host adapter for DOS decisions.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dosdecision"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, args []string) int {
	if len(args) < 2 || args[0] != "decisions" {
		fmt.Fprintln(stderr, "usage: fak-dos decisions <add|remove|list> [flags]")
		return 2
	}
	switch args[1] {
	case "add":
		return runAdd(stdout, stderr, args[2:])
	case "remove", "cleanup":
		return runRemove(stdout, stderr, args[2:])
	case "list":
		return runList(stdout, stderr, args[2:])
	default:
		fmt.Fprintf(stderr, "fak-dos decisions: unknown command %q\n", args[1])
		return 2
	}
}

func commonFlags(name string, stderr io.Writer) (*flag.FlagSet, *string) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	ws := fs.String("workspace", "", "workspace root (default: cwd)")
	fs.Bool("json", false, "machine-readable JSON (the only output format; accepted for parity)")
	return fs, ws
}
func workspaceRoot(v string) (string, error) {
	if v == "" {
		return os.Getwd()
	}
	return filepath.Abs(v)
}

func runAdd(stdout, stderr io.Writer, args []string) int {
	fs, wsArg := commonFlags("fak-dos decisions add", stderr)
	key := fs.String("key", "", "idempotency key")
	action := fs.String("action", "", "structured action")
	severity := fs.String("severity", "", "structured severity")
	payloadText := fs.String("payload", "", "optional JSON object payload")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	ws, err := workspaceRoot(*wsArg)
	if err != nil {
		return fail(stderr, err)
	}
	var payload map[string]any
	if *payloadText != "" {
		if err := json.Unmarshal([]byte(*payloadText), &payload); err != nil {
			return fail(stderr, fmt.Errorf("--payload: %w", err))
		}
		if payload == nil {
			return fail(stderr, fmt.Errorf("--payload must be a JSON object"))
		}
	}
	row, created, err := Add(ws, *key, *action, *severity, payload, time.Now())
	if err != nil {
		return fail(stderr, err)
	}
	return writeJSON(stdout, stderr, map[string]any{"created": created, "row": row})
}

func runRemove(stdout, stderr io.Writer, args []string) int {
	fs, wsArg := commonFlags("fak-dos decisions remove", stderr)
	key := fs.String("key", "", "idempotency key")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	ws, err := workspaceRoot(*wsArg)
	if err != nil {
		return fail(stderr, err)
	}
	removed, err := Remove(ws, *key)
	if err != nil {
		return fail(stderr, err)
	}
	return writeJSON(stdout, stderr, map[string]any{"key": *key, "removed": removed})
}

func runList(stdout, stderr io.Writer, args []string) int {
	fs, wsArg := commonFlags("fak-dos decisions list", stderr)
	native := fs.Bool("native", true, "include `dos decisions --all --json` rows")
	all := fs.Bool("all", false, "also emit rows superseded by a released lease (resolved history)")
	summary := fs.Bool("summary", false, "emit {cleared, active, superseded} instead of the bare row array")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	ws, err := workspaceRoot(*wsArg)
	if err != nil {
		return fail(stderr, err)
	}
	rows := []dosdecision.Row{}
	if *native {
		nativeRows, err := nativeDecisions(ws, stderr)
		if err != nil {
			return fail(stderr, err)
		}
		rows = append(rows, nativeRows...)
	}
	host, err := Read(ws)
	if err != nil {
		return fail(stderr, err)
	}
	for _, r := range host {
		row, err := hostRow(r)
		if err != nil {
			return fail(stderr, err)
		}
		rows = append(rows, row)
	}
	// A refusal is over when its blocking lease is gone, so the active queue is the
	// revalidated one — an operator asked to resolve a collision must be looking at
	// a collision that still exists (#6494).
	live := dosdecision.LiveSet{}
	if dosdecision.NeedsLiveSet(rows) {
		live = liveLanes(ws)
	}
	res := dosdecision.Revalidate(rows, live)
	if *summary {
		return writeJSON(stdout, stderr, map[string]any{
			"cleared": res.Cleared, "active": res.Active, "superseded": res.Superseded,
		})
	}
	out := res.Active
	if *all {
		out = append(append([]dosdecision.Row{}, res.Active...), res.Superseded...)
	}
	return writeJSON(stdout, stderr, out)
}

// hostRow projects a host-authored row into the shape the revalidator reads. Host
// rows are never lease-dependent (`HOST_QUEUE_ITEM`), so they pass through, but
// projecting them keeps one row type on the output path.
func hostRow(r Row) (dosdecision.Row, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	var out dosdecision.Row
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// nativeDecisions and liveLanes are the two kernel reads this command performs.
// They are indirected so the fold between them is testable without a `dos` on PATH.
var (
	nativeDecisions = readNativeDecisions
	liveLanes       = readLiveLanes
)

// readNativeDecisions reads `dos decisions --all --json`. `--all` here is the
// KERNEL's meaning (include ORACLE/JUDGE-resolvable rows, not only HUMAN ones) —
// this adapter's own `--all` is about resolved history and is applied after the
// revalidation fold.
func readNativeDecisions(ws string, stderr io.Writer) ([]dosdecision.Row, error) {
	cmd := windowgate.Command("dos", "decisions", "--workspace", ws, "--all", "--json")
	b, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			fmt.Fprint(stderr, string(ee.Stderr))
		}
		return nil, fmt.Errorf("native dos decisions read: %w", err)
	}
	var rows []dosdecision.Row
	if err := json.Unmarshal(b, &rows); err != nil {
		return nil, fmt.Errorf("native dos decisions JSON: %w", err)
	}
	return rows, nil
}

// readLiveLanes reads `dos lease-lane live` — the authoritative WAL live set — from
// the workspace. It reports Known=false on any read or parse fault: an unreadable
// kernel must never be mistaken for an idle one, or the queue would empty itself
// whenever `dos` is missing. The workspace is passed as the working directory, not
// as a path argument, because an explicit path makes `dos lease-lane live`
// re-resolve to a different `.dos/` (the same rule `tools/dos_fleet_lease.py` obeys).
func readLiveLanes(ws string) dosdecision.LiveSet {
	cmd := windowgate.Command("dos", "lease-lane", "live")
	cmd.Dir = ws
	b, err := cmd.Output()
	if err != nil {
		return dosdecision.LiveSet{}
	}
	var raw []map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(b), &raw); err != nil {
		return dosdecision.LiveSet{}
	}
	lanes := make([]string, 0, len(raw))
	for _, r := range raw {
		if lane, _ := r["lane"].(string); lane != "" {
			lanes = append(lanes, lane)
		}
	}
	return dosdecision.LiveSet{Lanes: lanes, Known: true}
}

func writeJSON(w, stderr io.Writer, v any) int {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fail(stderr, err)
	}
	return 0
}
func fail(w io.Writer, err error) int { fmt.Fprintf(w, "fak-dos decisions: %v\n", err); return 1 }
