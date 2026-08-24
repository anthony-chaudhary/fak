package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const dispatchTrajectoryAuditSchema = "fleet-dispatch-trajectory-audit/1"

type dispatchTrajectoryFriction struct {
	ID       string `json:"id"`
	Sessions int    `json:"sessions"`
	Events   int    `json:"events"`
}

type dispatchTrajectoryRun struct {
	Issue     int            `json:"issue"`
	Log       string         `json:"log"`
	SpawnedAt time.Time      `json:"spawned_at"`
	Bytes     int64          `json:"bytes"`
	Claim     string         `json:"claim"`
	Reason    string         `json:"reason,omitempty"`
	Friction  map[string]int `json:"friction,omitempty"`
}

type dispatchTrajectoryRecommendation struct {
	ID       string `json:"id"`
	Evidence string `json:"evidence"`
	Action   string `json:"action"`
}

type dispatchTrajectoryAuditReport struct {
	Schema          string                             `json:"schema"`
	RunsDir         string                             `json:"runs_dir"`
	Since           time.Time                          `json:"since"`
	Sessions        int                                `json:"sessions"`
	WitnessedShips  int                                `json:"witnessed_ships"`
	NoCommit        int                                `json:"no_commit"`
	MissingWitness  int                                `json:"missing_witness"`
	ShipYield       float64                            `json:"ship_yield"`
	LogBytes        int64                              `json:"log_bytes"`
	Friction        []dispatchTrajectoryFriction       `json:"friction"`
	Runs            []dispatchTrajectoryRun            `json:"runs"`
	Recommendations []dispatchTrajectoryRecommendation `json:"recommendations"`
	Provenance      []string                           `json:"provenance"`
}

var dispatchTrajectoryLogRE = regexp.MustCompile(`^resolve-(\d+)-(\d{8}-\d{6})\.log$`)

var dispatchTrajectorySignatures = []struct {
	id string
	re *regexp.Regexp
}{
	{"commit_lock_contention", regexp.MustCompile(`(?i)LOCK_BUSY|commit lane: (?:busy|stale)`)},
	{"prestaged_path_overlap", regexp.MustCompile(`PRESTAGED_PATH_OVERLAP`)},
	{"patch_application_failure", regexp.MustCompile(`Invalid patch:`)},
	{"dependency_timeout", regexp.MustCompile(`(?i)timed out|timeout after`)},
	{"peer_wip_interference", regexp.MustCompile(`(?i)unrelated untracked|peer WIP|shared tree`)},
	{"full_tree_gate_retry", regexp.MustCompile(`(?i)== go build ==|scripts\\ci\.ps1|make ci`)},
	{"crash_restart_exhausted", regexp.MustCompile(`CRASH_RESTART_EXHAUSTED`)},
	{"missing_prompt", regexp.MustCompile(`No prompt provided via stdin`)},
}

func runDispatchTrajectoryAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch trajectory-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory containing resolve worker logs and witnesses")
	windowH := fs.Float64("window-h", 24, "scan workers spawned within the last N hours (0 = all)")
	sinceRaw := fs.String("since", "", "exact RFC3339 lower bound (overrides --window-h)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	since := time.Time{}
	if strings.TrimSpace(*sinceRaw) != "" {
		parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(*sinceRaw))
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch trajectory-audit: --since: %v\n", err)
			return 2
		}
		since = parsed
	} else if *windowH > 0 {
		since = time.Now().Add(-time.Duration(*windowH * float64(time.Hour)))
	}
	rep, err := auditDispatchTrajectories(*runsDir, since)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch trajectory-audit: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch trajectory-audit: encode: %v\n", err)
			return 1
		}
		return 0
	}
	renderDispatchTrajectoryAudit(stdout, rep)
	return 0
}

func auditDispatchTrajectories(runsDir string, since time.Time) (dispatchTrajectoryAuditReport, error) {
	rep := dispatchTrajectoryAuditReport{
		Schema: dispatchTrajectoryAuditSchema, RunsDir: runsDir, Since: since,
		Provenance: []string{
			"Worker outcomes come only from resolve-*.witness structured artifacts.",
			"Friction counts come from resolve-*.log signatures and are observed, not causal attribution.",
			"Repository commits and GitHub closures outside these worker witnesses are not credited to the dispatch trajectories.",
		},
	}
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return rep, err
	}
	frictionSessions := map[string]int{}
	frictionEvents := map[string]int{}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		m := dispatchTrajectoryLogRE.FindStringSubmatch(entry.Name())
		if m == nil {
			continue
		}
		spawned, err := time.ParseInLocation("20060102-150405", m[2], time.UTC)
		if err != nil || (!since.IsZero() && spawned.Before(since)) {
			continue
		}
		issue, _ := strconv.Atoi(m[1])
		path := filepath.Join(runsDir, entry.Name())
		body, err := os.ReadFile(path)
		if err != nil {
			return rep, err
		}
		info, err := entry.Info()
		if err != nil {
			return rep, err
		}
		run := dispatchTrajectoryRun{Issue: issue, Log: entry.Name(), SpawnedAt: spawned, Bytes: info.Size(), Claim: "MISSING", Friction: map[string]int{}}
		witnessPath := strings.TrimSuffix(path, ".log") + ".witness"
		if raw, err := os.ReadFile(witnessPath); err == nil {
			var witness struct {
				Claim  string `json:"claim"`
				Reason string `json:"reason"`
			}
			if json.Unmarshal(raw, &witness) == nil {
				run.Claim, run.Reason = witness.Claim, witness.Reason
			}
		}
		switch run.Claim {
		case "CLAIM_WITNESSED":
			rep.WitnessedShips++
		case "CLAIM_NO_COMMIT":
			rep.NoCommit++
		default:
			rep.MissingWitness++
		}
		for _, sig := range dispatchTrajectorySignatures {
			count := len(sig.re.FindAll(body, -1))
			if count == 0 {
				continue
			}
			run.Friction[sig.id] = count
			frictionSessions[sig.id]++
			frictionEvents[sig.id] += count
		}
		if len(run.Friction) == 0 {
			run.Friction = nil
		}
		rep.LogBytes += info.Size()
		rep.Runs = append(rep.Runs, run)
	}
	sort.Slice(rep.Runs, func(i, j int) bool { return rep.Runs[i].SpawnedAt.Before(rep.Runs[j].SpawnedAt) })
	rep.Sessions = len(rep.Runs)
	if rep.Sessions > 0 {
		rep.ShipYield = float64(rep.WitnessedShips) / float64(rep.Sessions)
	}
	for _, sig := range dispatchTrajectorySignatures {
		if frictionEvents[sig.id] > 0 {
			rep.Friction = append(rep.Friction, dispatchTrajectoryFriction{ID: sig.id, Sessions: frictionSessions[sig.id], Events: frictionEvents[sig.id]})
		}
	}
	sort.Slice(rep.Friction, func(i, j int) bool {
		if rep.Friction[i].Sessions != rep.Friction[j].Sessions {
			return rep.Friction[i].Sessions > rep.Friction[j].Sessions
		}
		return rep.Friction[i].ID < rep.Friction[j].ID
	})
	rep.Recommendations = dispatchTrajectoryRecommendations(rep)
	return rep, nil
}

func dispatchTrajectoryRecommendations(rep dispatchTrajectoryAuditReport) []dispatchTrajectoryRecommendation {
	counts := map[string]dispatchTrajectoryFriction{}
	for _, row := range rep.Friction {
		counts[row.ID] = row
	}
	out := []dispatchTrajectoryRecommendation{}
	if rep.NoCommit > 0 {
		out = appendTrajectoryRecommendation(out, "admit_for_yield", fmt.Sprintf("%d/%d trajectories ended without a witnessed commit", rep.NoCommit, rep.Sessions), "Refill in small disjoint tranches and stop admitting a backend/lane when its recent witnessed-ship yield falls below the operator threshold.")
	}
	if row := counts["commit_lock_contention"]; row.Sessions > 0 {
		out = appendTrajectoryRecommendation(out, "serialize_epilogues", fmt.Sprintf("%d sessions hit commit-lock contention (%d events)", row.Sessions, row.Events), "Keep implementation parallel, but queue commit/push/close epilogues through one controller instead of making every worker poll the shared commit lock.")
	}
	if row := counts["peer_wip_interference"]; row.Sessions > 0 {
		out = append(out, dispatchTrajectoryRecommendation{ID: "validate_owned_paths", Evidence: fmt.Sprintf("%d sessions encountered peer-WIP interference", row.Sessions), Action: "Generate each worker's exact fak validate --mine command from its declared lease tree and treat full live-tree CI as observational, not the worker's completion gate."})
	}
	if row := counts["dependency_timeout"]; row.Sessions > 0 {
		out = append(out, dispatchTrajectoryRecommendation{ID: "bound_planning", Evidence: fmt.Sprintf("%d sessions recorded dependency timeouts", row.Sessions), Action: "Price at full target, then launch bounded 4-8 member waves; fall back to explicit disjoint ticks when contract audit exceeds its deadline."})
	}
	if row := counts["missing_prompt"]; row.Sessions > 0 {
		out = append(out, dispatchTrajectoryRecommendation{ID: "durable_worker_fuel", Evidence: fmt.Sprintf("%d sessions launched without stdin fuel", row.Sessions), Action: "Persist rendered issue fuel before spawn and restart from that artifact, never from an already-consumed stdin stream."})
	}
	return out
}

func appendTrajectoryRecommendation(out []dispatchTrajectoryRecommendation, id, evidence, action string) []dispatchTrajectoryRecommendation {
	return append(out, dispatchTrajectoryRecommendation{ID: id, Evidence: evidence, Action: action})
}

func renderDispatchTrajectoryAudit(w io.Writer, rep dispatchTrajectoryAuditReport) {
	fmt.Fprintf(w, "dispatch trajectory audit — %d sessions, %d witnessed ships, %d no-commit (yield %.1f%%)\n", rep.Sessions, rep.WitnessedShips, rep.NoCommit, rep.ShipYield*100)
	fmt.Fprintf(w, "logs: %.1f MiB  since: %s\n", float64(rep.LogBytes)/(1024*1024), rep.Since.Format(time.RFC3339))
	if len(rep.Friction) > 0 {
		fmt.Fprintln(w, "\nfriction (sessions / events):")
		for _, row := range rep.Friction {
			fmt.Fprintf(w, "  %-28s %3d / %d\n", row.ID, row.Sessions, row.Events)
		}
	}
	if len(rep.Recommendations) > 0 {
		fmt.Fprintln(w, "\nrepeatability actions:")
		for _, row := range rep.Recommendations {
			fmt.Fprintf(w, "  %s — %s\n    %s\n", row.ID, row.Evidence, row.Action)
		}
	}
	fmt.Fprintln(w, "\nprovenance:")
	for _, row := range rep.Provenance {
		fmt.Fprintf(w, "  - %s\n", row)
	}
}
