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
//	fak sidecar --from FILE         read a captured legacy `fleet_sessions.py json`
//	                                payload (file or '-') instead of the Go census
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

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessiondesc"
	"github.com/anthony-chaudhary/fak/internal/sidecar"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdSidecar(argv []string) { os.Exit(runSidecar(os.Stdout, os.Stderr, argv)) }

func runSidecar(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sidecar", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the machine-readable Pane envelope")
	asSlack := fs.Bool("slack", false, "emit the Slack Block Kit payload (JSON array of blocks)")
	from := fs.String("from", "", "read a captured legacy `fleet_sessions.py json` payload (file or '-') instead of the Go census")
	lanesFrom := fs.String("lanes-from", "", "read a dos-top lane-occupancy JSON reading (file or '-')")
	postureFrom := fs.String("posture-from", "", "read a gateway /debug/vars posture reading (file or '-')")
	timeout := fs.Int("timeout", 60, "deprecated compatibility flag; the default census is in-process")
	python := fs.String("python", "", "deprecated compatibility flag; the default census never starts Python")
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

	// The default session plane is the in-process cross-harness census joined
	// through the exact-id session descriptor contract. --from deliberately keeps
	// the old captured payload path deterministic for offline fixtures.
	if from == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			in.Sessions = sidecar.PlaneInput{Measured: false, Note: "fleetmon census unavailable: " + err.Error()}
		} else {
			rows, note, joinErr := sidecarRowsFromFleetCensus(fleetmon.Census(home, time.Now()))
			if joinErr != nil {
				in.Sessions = sidecar.PlaneInput{Measured: false, Note: "session descriptor join refused: " + joinErr.Error()}
			} else {
				in.SessionRows = rows
				in.Sessions = sidecar.PlaneInput{Measured: true, Note: note}
			}
		}
		in.Accounts = sidecar.PlaneInput{
			Measured: false,
			Note:     "account availability is not part of the in-process session census (use --from for a legacy captured reading)",
		}
	} else {
		census, censusErr := loadCensus(from)
		if censusErr != "" {
			note := "legacy fleet_sessions capture unavailable: " + censusErr
			in.Sessions = sidecar.PlaneInput{Measured: false, Note: note}
			in.Accounts = sidecar.PlaneInput{Measured: false, Note: note}
		} else {
			in.SessionRows = censusSessions(census)
			in.Sessions = sidecar.PlaneInput{Measured: true, Note: "legacy fleet_sessions.py captured census"}
			in.AccountRows = censusAccounts(census)
			in.Accounts = sidecar.PlaneInput{Measured: true, Note: "legacy fleet_sessions.py captured availability"}
		}
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

// sidecarRowsFromFleetCensus composes the shipped cross-agent census with the
// shipped exact-id descriptor fold. NO_NAMESPACE sentinels stay typed in the
// plane note; observed-empty is a measured empty result. Duplicate or empty
// session ids are delegated to sessiondesc.Fold and therefore fail closed.
func sidecarRowsFromFleetCensus(census []fleetmon.CensusRow) ([]sidecar.SessionRow, string, error) {
	harnessRows := make([]sessiondesc.HarnessRow, 0, len(census))
	livenessByID := make(map[string]fleetmon.Liveness, len(census))
	unavailable := make([]string, 0)
	for _, row := range census {
		switch row.Kind {
		case fleetmon.KindNoNamespace:
			note := row.Agent + "=NO_NAMESPACE"
			if row.Note != "" {
				note += " (" + row.Note + ")"
			}
			unavailable = append(unavailable, note)
		case fleetmon.KindSession:
			harnessRows = append(harnessRows, sessiondesc.HarnessRow{
				SessionID: row.Session,
				Agent:     row.Agent,
			})
			livenessByID[row.Session] = row.Liveness
		default:
			return nil, "", fmt.Errorf("fleetmon census row for %q has unknown kind %q", row.Agent, row.Kind)
		}
	}

	descriptors, err := sessiondesc.Fold(sessiondesc.Sources{
		HarnessStatus: sessiondesc.SourceObserved,
		Harness:       harnessRows,
	})
	if err != nil {
		return nil, "", err
	}

	rows := make([]sidecar.SessionRow, 0, len(descriptors))
	for _, descriptor := range descriptors {
		rows = append(rows, sidecar.SessionRow{
			Session:     descriptor.ID,
			Account:     descriptor.Harness.Identity,
			Harness:     descriptor.Harness.Agent,
			Disposition: strings.ToLower(string(livenessByID[descriptor.ID])),
		})
	}

	note := "fleetmon census -> fak.session.descriptor.v1"
	if len(descriptors) == 0 {
		note += "; measured empty"
	}
	if len(unavailable) > 0 {
		note += "; source unavailable: " + strings.Join(unavailable, "; ")
	}
	return rows, note, nil
}

// loadCensus reads the legacy offline fleet_sessions.py JSON contract from a
// captured file/stdin. The default path never calls it and never starts Python.
func loadCensus(from string) (map[string]any, string) {
	m, err := readJSONObject(from)
	if err != nil {
		return nil, err.Error()
	}
	return m, ""
}

// censusSessions maps legacy --from rows into sidecar SessionRows, skipping the
// synthetic `_probe` rows. Default collection uses sidecarRowsFromFleetCensus.
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
			Harness:     "claude", // the legacy capture only enumerated ~/.claude
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
		case scorecard.True(m["blocked"]):
			row.State = "blocked"
			row.Detail = asStr(m["block_reason"])
		case scorecard.True(m["throttled"]):
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

// readRaw is pathutil.ReadFileOrStdin under the name this file's three loaders read best
// with. The file-or-'-' convention has ONE definition (internal/pathutil); this used to be a
// byte-identical private re-derivation of it.
func readRaw(path string) ([]byte, error) { return pathutil.ReadFileOrStdin(path) }

func asStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%v", v)
}
