// Command fak-dos is the writable host adapter for DOS decisions.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"
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
	if err := fs.Parse(args); err != nil {
		return 2
	}
	ws, err := workspaceRoot(*wsArg)
	if err != nil {
		return fail(stderr, err)
	}
	rows := []any{}
	if *native {
		cmd := exec.Command("dos", "decisions", "--workspace", ws, "--all", "--json")
		b, err := cmd.Output()
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				fmt.Fprint(stderr, string(ee.Stderr))
			}
			return fail(stderr, fmt.Errorf("native dos decisions read: %w", err))
		}
		var nativeRows []any
		if err := json.Unmarshal(b, &nativeRows); err != nil {
			return fail(stderr, fmt.Errorf("native dos decisions JSON: %w", err))
		}
		rows = append(rows, nativeRows...)
	}
	host, err := Read(ws)
	if err != nil {
		return fail(stderr, err)
	}
	for _, r := range host {
		rows = append(rows, r)
	}
	return writeJSON(stdout, stderr, rows)
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
