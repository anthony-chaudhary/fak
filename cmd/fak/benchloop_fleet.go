package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const benchFleetSchema = "fak.bench-loop.fleet.v1"

type benchFleetPlan struct {
	PerMachineNext map[string]struct {
		MachineID        string `json:"machine_id"`
		Benchmark        string `json:"workload_kind"`
		Model            string `json:"model"`
		Precision        string `json:"precision"`
		Reason           string `json:"reason"`
		SuggestedCommand string `json:"suggested_command"`
	} `json:"per_machine_next"`
}
type benchFleetRequest struct {
	Schema      string `json:"schema"`
	ID          string `json:"id"`
	Machine     string `json:"machine"`
	NodeClass   string `json:"node_class"`
	Benchmark   string `json:"benchmark"`
	Model       string `json:"model"`
	Precision   string `json:"precision"`
	Reason      string `json:"reason"`
	Command     string `json:"command"`
	State       string `json:"state"`
	RequestedAt string `json:"requested_at"`
	Path        string `json:"path,omitempty"`
	// Durable dispatch history (#6503). The recurring loop reports its result from
	// these counters rather than from whatever the current tick happened to claim,
	// and holds an unavailable node instead of re-dispatching it every fifteen minutes.
	Attempts      int     `json:"attempts,omitempty"`
	Failures      int     `json:"failures,omitempty"`
	LastAttemptAt string  `json:"last_attempt_at,omitempty"`
	HeldSince     string  `json:"held_since,omitempty"`
	HeldReason    string  `json:"held_reason,omitempty"`
	Seconds       float64 `json:"seconds,omitempty"`
	Measured      bool    `json:"measured,omitempty"`
}
type benchFleetReport struct {
	Schema      string              `json:"schema"`
	GeneratedAt string              `json:"generated_at"`
	Apply       bool                `json:"apply"`
	Queue       string              `json:"queue"`
	Machines    int                 `json:"machines"`
	Enqueued    int                 `json:"enqueued"`
	Existing    int                 `json:"existing"`
	Held        int                 `json:"held"`
	Released    int                 `json:"released"`
	Requests    []benchFleetRequest `json:"requests"`
	Next        string              `json:"next"`
}

func runBenchFleet(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "dispatch" {
		return runBenchFleetDispatch(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "status" {
		return runBenchFleetStatus(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("fak bench-loop fleet", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "persist one deduplicated work request per benchmark node")
	jsonOut := fs.Bool("json", false, "emit the machine-readable report")
	nowArg := fs.String("now", "", "planner time in yyyyMMddTHHmmssZ form")
	queueArg := fs.String("queue", "", "queue directory")
	workspace := fs.String("workspace", ".", "repository root")
	python := fs.String("python", "python", "Python executable used by the benchmark planner")
	planJSON := fs.String("plan-json", "", "read planner JSON from this file")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak bench-loop fleet [--apply] [--json] [--now STAMP] [--plan-json FILE]")
		return 2
	}
	root, err := filepath.Abs(*workspace)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop fleet: root: %v\n", err)
		return 1
	}
	queue := *queueArg
	if queue == "" {
		queue = filepath.Join(root, ".fak", "bench-fleet", "requests")
	}
	stamp := *nowArg
	if stamp == "" {
		stamp = time.Now().UTC().Format("20060102T150405Z")
	}
	payload, err := loadBenchFleetPlan(root, stamp, *python, *planJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop fleet: %v\n", err)
		return 1
	}
	var plan benchFleetPlan
	if err := json.Unmarshal(payload, &plan); err != nil {
		fmt.Fprintf(stderr, "fak bench-loop fleet: decode planner: %v\n", err)
		return 1
	}
	report := benchFleetReport{Schema: benchFleetSchema, GeneratedAt: time.Now().UTC().Format(time.RFC3339), Apply: *apply, Queue: filepath.ToSlash(queue), Machines: len(plan.PerMachineNext)}
	machines := make([]string, 0, len(plan.PerMachineNext))
	for machine := range plan.PerMachineNext {
		machines = append(machines, machine)
	}
	sort.Strings(machines)
	for _, machine := range machines {
		row := plan.PerMachineNext[machine]
		if row.MachineID == "" {
			row.MachineID = machine
		}
		command := cleanBenchSuggestedCommand(row.SuggestedCommand)
		sum := sha256.Sum256([]byte(strings.Join([]string{row.MachineID, row.Benchmark, row.Model, row.Precision, command}, "\x00")))
		id := hex.EncodeToString(sum[:8])
		req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: id, Machine: row.MachineID, NodeClass: benchNodeClass(row.MachineID), Benchmark: row.Benchmark, Model: row.Model, Precision: row.Precision, Reason: row.Reason, Command: command, State: "planned", RequestedAt: report.GeneratedAt}
		path := filepath.Join(queue, safeBenchMachine(row.MachineID)+"-"+id+".json")
		req.Path = filepath.ToSlash(path)
		if *apply {
			if _, err := os.Stat(path); err == nil {
				req.State = "already_queued"
				report.Existing++
				if refreshBenchFleetHold(root, path) {
					report.Released++
				}
			} else if !os.IsNotExist(err) {
				fmt.Fprintf(stderr, "fak bench-loop fleet: inspect %s: %v\n", path, err)
				return 1
			} else {
				req.State = "queued"
				// Preflight the node's own session/credential before the cell is ever
				// dispatched: a node with no configured route is enqueued already held on
				// the gap it names, so the loop reports a configuration gap instead of
				// spending a claim on it every fifteen minutes (#6503).
				if _, _, _, state, routeErr := benchFleetRoute(root, req); routeErr != nil && state != benchloop.FleetRunning {
					req.State = state
					_, req.HeldReason = benchloop.NormalizeFleetState(state)
					req.HeldSince = report.GeneratedAt
					report.Held++
				}
				if err := writeBenchFleetRequest(path, req); err != nil {
					fmt.Fprintf(stderr, "fak bench-loop fleet: queue %s: %v\n", row.MachineID, err)
					return 1
				}
				report.Enqueued++
			}
		}
		report.Requests = append(report.Requests, req)
	}
	if *apply {
		report.Next = "node runners claim queued requests; reruns are idempotent until the plan changes"
	} else {
		report.Next = "rerun with --apply to persist the per-node work queue"
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		renderBenchFleet(stdout, report)
	}
	return 0
}

func runBenchFleetStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-loop fleet status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit machine-readable status")
	workspace := fs.String("workspace", ".", "repository root")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root, err := filepath.Abs(*workspace)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop fleet status: %v\n", err)
		return 1
	}
	queue := filepath.Join(root, ".fak", "bench-fleet", "requests")
	entries, err := os.ReadDir(queue)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak bench-loop fleet status: %v\n", err)
		return 1
	}
	type status struct {
		Schema   string                 `json:"schema"`
		Queue    string                 `json:"queue"`
		Queued   int                    `json:"queued"`
		Utility  benchloop.FleetUtility `json:"utility"`
		Requests []benchFleetRequest    `json:"requests"`
		Nodes    []benchloop.FleetNode  `json:"nodes"`
	}
	got := status{Schema: "fak.bench-loop.fleet-status.v1", Queue: filepath.ToSlash(queue), Nodes: benchloop.RegisteredFleetNodes()}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		b, e := os.ReadFile(filepath.Join(queue, entry.Name()))
		if e != nil {
			continue
		}
		var req benchFleetRequest
		if json.Unmarshal(b, &req) == nil {
			got.Requests = append(got.Requests, req)
		}
	}
	got.Queued = len(got.Requests)
	sort.Slice(got.Requests, func(i, j int) bool { return got.Requests[i].Machine < got.Requests[j].Machine })
	cells := make([]benchloop.FleetCell, 0, len(got.Requests))
	for _, req := range got.Requests {
		cells = append(cells, benchFleetCell(req))
	}
	got.Utility = benchloop.SummarizeFleet(cells)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(got)
	} else {
		fmt.Fprintf(stdout, "bench fleet queue: %d request(s) in %s\n", got.Queued, got.Queue)
		for _, req := range got.Requests {
			fmt.Fprintf(stdout, "- %s: %s (%s)\n", req.Machine, req.Command, req.State)
		}
		fmt.Fprintf(stdout, "utility: attempted=%d measured=%d held=%d repeated-failures=%d compute=%.0fs result=%d (%s)\n",
			got.Utility.Attempted, got.Utility.Measured, got.Utility.Held, got.Utility.RepeatedFailures, got.Utility.ComputeSeconds, got.Utility.Result, got.Utility.Reason)
	}
	// status stays a read-only report and always exits 0; the loop's own result is
	// carried by the dispatch tick, which is what the scheduler runs.
	return 0
}

// refreshBenchFleetHold re-preflights an already-queued row: a node that was held on
// a missing session or credential returns to the queue the moment its route resolves,
// instead of sitting out the rest of its hold window. It reports whether it released
// a hold, which the plan report counts so an operator sees the recovery.
func refreshBenchFleetHold(root, path string) bool {
	req, err := readBenchFleetRequest(path)
	if err != nil {
		return false
	}
	if state, _ := benchloop.NormalizeFleetState(req.State); state != benchloop.FleetHeld {
		return false
	}
	if _, _, _, state, routeErr := benchFleetRoute(root, req); routeErr != nil || state != benchloop.FleetRunning {
		return false
	}
	req.State, req.HeldSince, req.HeldReason = "queued", "", ""
	return writeBenchFleetRequest(path, req) == nil
}

func loadBenchFleetPlan(root, stamp, python, path string) ([]byte, error) {
	if path != "" {
		return os.ReadFile(path)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, filepath.Join(root, "tools", "bench_plan.py"), "--workspace", root, "--now", stamp, "--json")
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("planner: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return out, nil
}
func cleanBenchSuggestedCommand(s string) string {
	if i := strings.Index(s, ": "); i >= 0 {
		s = s[i+2:]
	}
	if i := strings.Index(s, "  #"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

func benchNodeClass(machine string) string {
	m := strings.ToLower(machine)
	switch {
	case strings.Contains(m, "a100") || strings.Contains(m, "h100") || strings.Contains(m, "l4") || strings.Contains(m, "gpu"):
		return "gpu"
	case strings.Contains(m, "cpu"):
		return "cpu"
	case strings.Contains(m, "mac"):
		return "mac"
	default:
		return "control"
	}
}
func safeBenchMachine(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}
func writeBenchFleetRequest(path string, req benchFleetRequest) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp := path + fmt.Sprintf(".%d.tmp", os.Getpid())
	if err = os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	if err = os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
func renderBenchFleet(w io.Writer, r benchFleetReport) {
	mode := "DRY_RUN"
	if r.Apply {
		mode = "APPLIED"
	}
	fmt.Fprintf(w, "bench fleet %s: %d nodes, %d queued, %d existing, %d held on configuration\n", mode, r.Machines, r.Enqueued, r.Existing, r.Held)
	for _, x := range r.Requests {
		fmt.Fprintf(w, "- %-16s %-7s %-14s %s\n", x.Machine, x.NodeClass, x.State, x.Command)
	}
	fmt.Fprintf(w, "next: %s\n", r.Next)
}

func runBenchFleetInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-loop install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	interval := fs.Int("interval", 15, "cadence in minutes")
	task := fs.String("task", "FakBenchmarkFleetLoop", "Windows Scheduled Task name")
	remove := fs.Bool("remove", false, "remove the Scheduled Task")
	workspace := fs.String("workspace", ".", "repository root the task benchmarks")
	force := fs.Bool("force", false, "arm the schedule without a witnessed numeric benchmark")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if runtime.GOOS != "windows" {
		fmt.Fprintln(stderr, "fak bench-loop install: use `fak bench-loop fleet --apply` from cron on non-Windows hosts")
		return 1
	}
	if *interval < 1 {
		fmt.Fprintln(stderr, "fak bench-loop install: --interval must be positive")
		return 2
	}
	if *remove {
		if !runBenchFleetSchtasks(stderr, "", "/Delete", "/TN", *task, "/F") {
			return 1
		}
		fmt.Fprintf(stdout, "removed %s\n", *task)
		return 0
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop install: executable: %v\n", err)
		return 1
	}
	root, err := filepath.Abs(*workspace)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop install: root: %v\n", err)
		return 1
	}
	// A recurring loop is only worth arming once one real node has produced a witnessed
	// benchmark number; before #6503 this schedule re-spent compute on held and failing
	// cells every fifteen minutes and reported result 0 for it.
	if ok, why := benchloop.FleetReenableAllowed(benchloop.SummarizeFleet(benchFleetQueueCells(root))); !ok && !*force {
		fmt.Fprintf(stderr, "fak bench-loop install: refusing to arm %s: %s\n", *task, why)
		fmt.Fprintln(stderr, "fak bench-loop install: run one node explicitly (`fak bench-loop fleet dispatch`) or pass --force")
		return 1
	}
	// The tick payload is committed tooling, not an ephemeral %TEMP% script that any
	// cleanup could delete out from under the scheduler (#6503).
	runner := filepath.Join(root, "tools", "scheduled-tasks", "fak-bench-fleet-tick.cmd")
	if _, err := os.Stat(runner); err != nil {
		fmt.Fprintf(stderr, "fak bench-loop install: tick payload: %v\n", err)
		return 1
	}
	tr := fmt.Sprintf("cmd.exe /d /s /c \"\"%s\" \"%s\" \"%s\"\"", runner, exe, root)
	if !runBenchFleetSchtasks(stderr, root, "/Create", "/TN", *task, "/SC", "MINUTE", "/MO", fmt.Sprint(*interval), "/TR", tr, "/F", "/RL", "LIMITED") {
		return 1
	}
	fmt.Fprintf(stdout, "installed %s every %dm: %s\n", *task, *interval, tr)
	return 0
}

// runBenchFleetSchtasks runs one schtasks.exe invocation for `fak bench-loop install`,
// reporting failure with the scheduler's own combined output appended -- schtasks explains
// itself on stdout, so swallowing it would leave the operator with a bare exit status. dir
// is the working directory of the schtasks.exe process itself, not the created task's --
// the task carries the workspace explicitly in its /TR command line -- and "" leaves it
// inherited, which is all a deletion needs. Both the install and the --remove path refuse
// through this one rule.
func runBenchFleetSchtasks(stderr io.Writer, dir string, args ...string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "schtasks.exe", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	configureDispatchHelperCommand(cmd)
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop install: %v: %s\n", err, strings.TrimSpace(string(out)))
		return false
	}
	return true
}
