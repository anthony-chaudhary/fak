package main

// `fak sidecar` -- the sidecar pane v0 (#2215, epic #2209): ONE read-only fold
// over the four per-agent runtime planes (sessions census, account usability,
// lane occupancy, context/cache posture) rendered IDENTICALLY on terminal and
// Slack from one shared render model (internal/sidecar).
//
// This file is the impure shell: it COLLECTS each plane's facts live, hands the
// typed inputs to the pure sidecar.Fold, and serializes the resulting Pane to the
// requested surface. It invents no UI framework — the render model and the two
// renderers live in internal/sidecar, exactly the execrollup / rollup.go split.
//
//	fak sidecar                     render the pane as terminal text (default)
//	fak sidecar --slack             emit the Slack Block Kit payload (JSON array)
//	fak sidecar --json              emit the machine-readable Pane envelope
//	fak sidecar --from FILE         read a captured `fleet_sessions.py json` payload
//	                                (file or '-') instead of running it
//	fak sidecar --lanes-from FILE   read a dos-top lane-occupancy JSON reading
//	fak sidecar --posture-from FILE read a gateway /debug/vars posture reading
//
// Every plane degrades HONESTLY: a source that cannot be read leaves its plane
// UNMEASURED (never a silent green), and the pane's OK bit reflects the gap. The
// sessions/accounts census is WITNESSED (read from on-disk transcripts); the lane
// and posture planes are OBSERVED (a live reading) — the pane RENDERS lane
// occupancy, it does not re-adjudicate the lease.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/sidecar"
)

func cmdSidecar(argv []string) { os.Exit(runSidecar(os.Stdout, os.Stderr, argv)) }

func runSidecar(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sidecar", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the machine-readable Pane envelope")
	asSlack := fs.Bool("slack", false, "emit the Slack Block Kit payload (JSON array of blocks)")
	from := fs.String("from", "", "read a captured `fleet_sessions.py json` payload (file or '-') instead of running it")
	lanesFrom := fs.String("lanes-from", "", "read a dos-top lane-occupancy JSON reading (file or '-')")
	postureFrom := fs.String("posture-from", "", "read a gateway /debug/vars posture reading (file or '-')")
	timeout := fs.Int("timeout", 60, "seconds for the fleet_sessions census collector")
	python := fs.String("python", "", "python interpreter for the census collector (default: auto)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak sidecar: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *asJSON && *asSlack {
		fmt.Fprintln(stderr, "fak sidecar: choose at most one of --json / --slack")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	in := collectSidecar(stderr, root, *python, *from, *lanesFrom, *postureFrom, *timeout)
	in.Workspace = root
	if h, err := os.Hostname(); err == nil {
		in.Host = h
	}
	in.GeneratedAt = time.Now().UTC().Format(time.RFC3339)

	pane := sidecar.Fold(in)

	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, pane); err != nil {
			fmt.Fprintf(stderr, "fak sidecar: encode: %v\n", err)
			return 1
		}
	case *asSlack:
		if err := writeIndentedJSON(stdout, sidecar.RenderSlack(pane)); err != nil {
			fmt.Fprintf(stderr, "fak sidecar: encode blocks: %v\n", err)
			return 1
		}
	default:
		fmt.Fprint(stdout, sidecar.RenderText(pane))
	}
	return 0
}

// collectSidecar measures each plane live, honoring the --*-from overrides. Any
// collector failure becomes an UNMEASURED plane in the fold, never a panic and
// never a silent pass.
func collectSidecar(stderr io.Writer, root, python, from, lanesFrom, postureFrom string, timeoutSec int) sidecar.Inputs {
	var in sidecar.Inputs

	// sessions + accounts — the B1/B2 census from the existing fleet_sessions.py
	// json contract. A captured --from payload keeps the render deterministic.
	census, censusErr := loadCensus(root, python, from, timeoutSec)
	if censusErr != "" {
		note := "fleet_sessions census unavailable: " + censusErr
		in.Sessions = sidecar.PlaneInput{Measured: false, Note: note}
		in.Accounts = sidecar.PlaneInput{Measured: false, Note: note}
	} else {
		in.SessionRows = censusSessions(census)
		in.Sessions = sidecar.PlaneInput{Measured: true, Note: "fleet_sessions.py json census"}
		in.AccountRows = censusAccounts(census)
		in.Accounts = sidecar.PlaneInput{Measured: true, Note: "fleet_sessions.py availability"}
	}

	// lanes — a dos-top occupancy reading. OBSERVED, rendered not re-adjudicated.
	// v0 has no in-process dos-top fold, so absent the --lanes-from reading the
	// plane reads UNMEASURED rather than fabricating an empty "no lanes held".
	if lanesFrom != "" {
		rows, err := loadLaneReading(lanesFrom)
		if err != "" {
			in.Lanes = sidecar.PlaneInput{Measured: false, Note: err}
		} else {
			in.LaneRows = rows
			in.Lanes = sidecar.PlaneInput{Measured: true, Note: "dos-top reading"}
		}
	} else {
		in.Lanes = sidecar.PlaneInput{Measured: false, Note: "no lane reading (pass --lanes-from)"}
	}

	// posture — a gateway /debug/vars reading. OBSERVED. Absent an address, the
	// plane reads UNMEASURED.
	if postureFrom != "" {
		post, err := loadPostureReading(postureFrom)
		if err != "" {
			in.PostureStatus = sidecar.PlaneInput{Measured: false, Note: err}
		} else {
			in.Posture = post
			in.PostureStatus = sidecar.PlaneInput{Measured: true, Note: "gateway /debug/vars"}
		}
	} else {
		in.PostureStatus = sidecar.PlaneInput{Measured: false, Note: "no gateway posture (pass --posture-from)"}
	}

	return in
}

// loadCensus reads the fleet_sessions.py json payload, from a captured file/stdin
// (--from) or by running the collector. It returns the parsed object and an empty
// error string on success, or a one-line reason on failure.
func loadCensus(root, python, from string, timeoutSec int) (map[string]any, string) {
	if from != "" {
		m, err := readJSONObject(from)
		if err != nil {
			return nil, err.Error()
		}
		return m, ""
	}
	python = resolvePython(python)
	payload, runErr := cadencereport.RunPyEnvelope(root,
		[]string{"tools/fleet_sessions.py", "json"}, python, time.Duration(timeoutSec)*time.Second)
	if runErr != "" {
		return nil, runErr
	}
	if payload == nil {
		return nil, "collector returned no payload"
	}
	return payload, ""
}

// resolvePython returns the explicit interpreter or the workspace default. The
// sidecar's --python flag defaults to "" (auto); without this resolution the
// empty string reaches cadencereport.RunPyEnvelope and exec.Command("") fails
// with "exec: no command", leaving the sessions/accounts census UNMEASURED on
// the default invocation — the very smoke the issue's witness exercises. This
// mirrors cadencereport.CollectWithScores, which resolves the default before
// the same RunPyEnvelope call.
func resolvePython(p string) string {
	if p == "" {
		return gardenbundle.DefaultPython()
	}
	return p
}

// censusSessions maps the fleet_sessions `rows` into sidecar SessionRows, skipping
// the synthetic `_probe` rows. The disposition word is normalized so the fold's
// live tally is stable.
func censusSessions(census map[string]any) []sidecar.SessionRow {
	rows, _ := census["rows"].([]any)
	out := make([]sidecar.SessionRow, 0, len(rows))
	for _, raw := range rows {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if asStr(m["project"]) == "_probe" {
			continue
		}
		out = append(out, sidecar.SessionRow{
			Session:     asStr(m["session"]),
			Account:     asStr(m["account"]),
			Harness:     "claude", // fleet_sessions is the ~/.claude census; codex/opencode/aider join is a follow-on (B1)
			Disposition: dispositionWord(asStr(m["disp"])),
		})
	}
	return out
}

// dispositionWord folds the fleet_sessions disp taxonomy into the short sidecar
// vocabulary (live / done / throttled / blocked / stopped).
func dispositionWord(disp string) string {
	switch {
	case disp == "":
		return ""
	case disp == "LIVE":
		return "live"
	case disp == "DONE":
		return "done"
	case disp == "STOPPED_LIMIT":
		return "throttled"
	case strings.HasPrefix(disp, "INFRA_"):
		return "blocked"
	default:
		return "stopped"
	}
}

// censusAccounts maps the fleet_sessions `accounts` availability list into sidecar
// AccountRows with the three-way usable/throttled/blocked state.
func censusAccounts(census map[string]any) []sidecar.AccountRow {
	accts, _ := census["accounts"].([]any)
	out := make([]sidecar.AccountRow, 0, len(accts))
	for _, raw := range accts {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		row := sidecar.AccountRow{Account: asStr(m["account"])}
		switch {
		case asBool(m["blocked"]):
			row.State = "blocked"
			row.Detail = asStr(m["block_reason"])
		case asBool(m["throttled"]):
			row.State = "throttled"
			if reset := asStr(m["reset"]); reset != "" {
				row.Detail = "resets " + reset
			}
		default:
			row.State = "usable"
		}
		out = append(out, row)
	}
	return out
}

// laneReading is the small JSON contract for a captured dos-top occupancy reading:
// {"lanes":[{"lane":"cmd","kind":"cluster","held":true,"owner":"worker-7"}, ...]}.
type laneReading struct {
	Lanes []struct {
		Lane  string `json:"lane"`
		Kind  string `json:"kind"`
		Held  bool   `json:"held"`
		Owner string `json:"owner"`
	} `json:"lanes"`
}

func loadLaneReading(path string) ([]sidecar.LaneRow, string) {
	data, err := readRaw(path)
	if err != nil {
		return nil, err.Error()
	}
	var lr laneReading
	if err := json.Unmarshal(data, &lr); err != nil {
		return nil, path + ": " + err.Error()
	}
	out := make([]sidecar.LaneRow, 0, len(lr.Lanes))
	for _, l := range lr.Lanes {
		out = append(out, sidecar.LaneRow{Lane: l.Lane, Kind: l.Kind, Held: l.Held, Owner: l.Owner})
	}
	return out, ""
}

// loadPostureReading reads a captured gateway posture JSON:
// {"cache_posture":"managed","compactions":3,"elisions":1,"sessions_joined":2}.
func loadPostureReading(path string) (sidecar.Posture, string) {
	data, err := readRaw(path)
	if err != nil {
		return sidecar.Posture{}, err.Error()
	}
	var p sidecar.Posture
	if err := json.Unmarshal(data, &p); err != nil {
		return sidecar.Posture{}, path + ": " + err.Error()
	}
	return p, ""
}

// readJSONObject reads a JSON object from a file or '-' (stdin).
func readJSONObject(path string) (map[string]any, error) {
	data, err := readRaw(path)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

func readRaw(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}

func asStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%v", v)
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
}
