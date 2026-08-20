package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const orchestrationStatusSchema = "fak.codex_orchestration_status.v1"

const orchestrationEvidenceNotObserved = "not_observed"

type orchestrationOutcomeStatus struct {
	Verdict            string `json:"verdict"`
	EffectReadback     string `json:"effect_readback"`
	IndependentWitness string `json:"independent_witness"`
	Reconciliation     string `json:"reconciliation"`
	Reason             string `json:"reason"`
}

type orchestrationWorkerStatus struct {
	RoleID       string    `json:"role_id"`
	PID          int       `json:"pid,omitempty"`
	State        string    `json:"state"`
	ProcessAlive bool      `json:"process_alive"`
	LogPath      string    `json:"log_path,omitempty"`
	LogBytes     int64     `json:"log_bytes"`
	UpdatedAt    time.Time `json:"updated_at,omitempty"`
	TurnsStarted int       `json:"turns_started"`
	TurnsDone    int       `json:"turns_completed"`
	LastEvent    string    `json:"last_event,omitempty"`
}

type orchestrationRunStatus struct {
	Schema           string                      `json:"schema"`
	SessionID        string                      `json:"session_id"`
	RunID            string                      `json:"run_id"`
	LaunchedAt       time.Time                   `json:"launched_at"`
	RequestedProfile string                      `json:"requested_profile"`
	ResolvedProfile  string                      `json:"resolved_profile"`
	WorkClass        string                      `json:"work_class"`
	State            string                      `json:"state"`
	Running          int                         `json:"running"`
	Completed        int                         `json:"completed"`
	Exited           int                         `json:"exited"`
	Outcome          orchestrationOutcomeStatus  `json:"outcome"`
	Workers          []orchestrationWorkerStatus `json:"workers"`
}

func runOrchestrationStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("orchestration status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("home", "", "state root (default: current directory)")
	sessionID := fs.String("session", "", "specific launch session id (default: newest)")
	asJSON := fs.Bool("json", false, "emit versioned machine-readable status")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak orchestration status [--session ID] [--home DIR] [--json]")
		return 2
	}
	root := *home
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
			return 1
		}
	}
	receipt, err := newestOrchestrationReceipt(root, *sessionID)
	if err != nil {
		fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
		return 1
	}
	status := inspectOrchestrationRun(root, receipt)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(status); err != nil {
			fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "Orchestration %s - %s (worker execution)\n", status.RunID, status.State)
	fmt.Fprintf(stdout, "  Outcome %s | effects=%s | witness=%s | reconciliation=%s\n", status.Outcome.Verdict, status.Outcome.EffectReadback, status.Outcome.IndependentWitness, status.Outcome.Reconciliation)
	fmt.Fprintf(stdout, "  %s\n", status.Outcome.Reason)
	fmt.Fprintf(stdout, "  %d running | %d completed | %d exited | launched %s\n", status.Running, status.Completed, status.Exited, status.LaunchedAt.Local().Format("2006-01-02 15:04:05"))
	for _, worker := range status.Workers {
		activity := "no events"
		if worker.LastEvent != "" {
			activity = fmt.Sprintf("%s | turns %d/%d", worker.LastEvent, worker.TurnsDone, worker.TurnsStarted)
		}
		fmt.Fprintf(stdout, "  %-10s %-9s pid=%-7d %s | %s\n", worker.RoleID, worker.State, worker.PID, humanBytes(worker.LogBytes), activity)
	}
	fmt.Fprintf(stdout, "  machine: fak orchestration status --session %s --json\n", status.SessionID)
	return 0
}

func newestOrchestrationReceipt(home, sessionID string) (codexOrchestrationLaunchReceipt, error) {
	if sessionID != "" {
		if r, ok := readCodexOrchestrationLaunchReceipt(home, sessionID); ok {
			return r, nil
		}
		return codexOrchestrationLaunchReceipt{}, fmt.Errorf("launch receipt %q not found", sessionID)
	}
	paths, err := filepath.Glob(filepath.Join(home, "fak-orchestration-launches", "*.json"))
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, err
	}
	sort.Slice(paths, func(i, j int) bool {
		ai, _ := os.Stat(paths[i])
		aj, _ := os.Stat(paths[j])
		return ai.ModTime().After(aj.ModTime())
	})
	for _, p := range paths {
		id := filepath.Base(p)
		id = id[:len(id)-len(filepath.Ext(id))]
		if r, ok := readCodexOrchestrationLaunchReceipt(home, id); ok {
			return r, nil
		}
	}
	return codexOrchestrationLaunchReceipt{}, fmt.Errorf("no launch receipts under %s", filepath.Join(home, "fak-orchestration-launches"))
}

func inspectOrchestrationRun(home string, receipt codexOrchestrationLaunchReceipt) orchestrationRunStatus {
	out := orchestrationRunStatus{Schema: orchestrationStatusSchema, SessionID: receipt.SessionID, RunID: receipt.RunID, LaunchedAt: receipt.LaunchedAt, RequestedProfile: receipt.RequestedProfile, ResolvedProfile: receipt.ResolvedProfile, WorkClass: receipt.WorkClass, State: "complete", Outcome: unverifiedOrchestrationOutcome(), Workers: make([]orchestrationWorkerStatus, 0, len(receipt.Workers))}
	for _, worker := range receipt.Workers {
		ws := orchestrationWorkerStatus{RoleID: worker.RoleID, PID: worker.PID, LogPath: worker.LogPath, ProcessAlive: processalive.Check(worker.PID)}
		path := worker.LogPath
		if !filepath.IsAbs(path) {
			path = filepath.Join(home, path)
		}
		inspectOrchestrationLog(path, &ws)
		switch {
		case ws.TurnsStarted > 0 && ws.TurnsDone >= ws.TurnsStarted:
			ws.State = "completed"
			out.Completed++
		case ws.ProcessAlive:
			ws.State = "running"
			out.Running++
		default:
			ws.State = "exited"
			out.Exited++
		}
		out.Workers = append(out.Workers, ws)
	}
	if out.Running > 0 {
		out.State = "running"
	} else if out.Exited > 0 {
		out.State = "attention"
	}
	return out
}

func unverifiedOrchestrationOutcome() orchestrationOutcomeStatus {
	return orchestrationOutcomeStatus{
		Verdict:            "unverified",
		EffectReadback:     orchestrationEvidenceNotObserved,
		IndependentWitness: orchestrationEvidenceNotObserved,
		Reconciliation:     orchestrationEvidenceNotObserved,
		Reason:             "worker and turn events do not prove effects, independent witness acceptance, or reconciliation",
	}
}

func inspectOrchestrationLog(path string, ws *orchestrationWorkerStatus) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	ws.LogBytes = info.Size()
	ws.UpdatedAt = info.ModTime().UTC()
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var event struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil || event.Type == "" {
			continue
		}
		ws.LastEvent = event.Type
		if event.Type == "turn.started" {
			ws.TurnsStarted++
		}
		if event.Type == "turn.completed" {
			ws.TurnsDone++
		}
	}
}
