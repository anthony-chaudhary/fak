package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

type benchFleetWitness struct {
	Schema     string   `json:"schema"`
	RequestID  string   `json:"request_id"`
	Machine    string   `json:"machine"`
	Benchmark  string   `json:"benchmark,omitempty"`
	Model      string   `json:"model,omitempty"`
	Precision  string   `json:"precision,omitempty"`
	State      string   `json:"state"`
	Route      string   `json:"route"`
	Command    []string `json:"command,omitempty"`
	StartedAt  string   `json:"started_at"`
	FinishedAt string   `json:"finished_at"`
	ExitCode   int      `json:"exit_code"`
	Output     string   `json:"output"`
	Error      string   `json:"error,omitempty"`
}

type benchFleetDispatchReport struct {
	Schema     string                 `json:"schema"`
	Queue      string                 `json:"queue"`
	Considered int                    `json:"considered"`
	Claimed    int                    `json:"claimed"`
	Succeeded  int                    `json:"succeeded"`
	Failed     int                    `json:"failed"`
	Waiting    int                    `json:"waiting"`
	Reconciled int                    `json:"reconciled"`
	Skipped    []benchFleetSkip       `json:"skipped,omitempty"`
	Utility    benchloop.FleetUtility `json:"utility"`
	Witnesses  []benchFleetWitness    `json:"witnesses"`
}

// benchFleetSkip records a durable queue row this tick deliberately did NOT spend a
// dispatch on, with the reason an operator reads off the report: a node held on a
// missing session, a cell inside its failure backoff, or a claim another dispatcher
// still owns. Before #6503 those rows were re-dispatched every fifteen minutes.
type benchFleetSkip struct {
	RequestID string `json:"request_id"`
	Machine   string `json:"machine"`
	State     string `json:"state"`
	Reason    string `json:"reason"`
}

type benchFleetExec func(name string, args ...string) ([]byte, int, error)

func runBenchFleetDispatch(stdout, stderr io.Writer, argv []string) int {
	return runBenchFleetDispatchWithExec(stdout, stderr, argv, defaultBenchFleetExec)
}

func runBenchFleetDispatchWithExec(stdout, stderr io.Writer, argv []string, run benchFleetExec) int {
	fs := flag.NewFlagSet("fak bench-loop fleet dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("workspace", ".", "repository root")
	queueFlag := fs.String("queue", "", "request queue")
	max := fs.Int("max", 16, "maximum requests per tick")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	queue := *queueFlag
	if queue == "" {
		queue = filepath.Join(*root, ".fak", "bench-fleet", "requests")
	}
	report := benchFleetDispatchReport{Schema: "fak.bench-fleet.dispatch.v1", Queue: filepath.ToSlash(queue)}
	entries, err := os.ReadDir(queue)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak bench-loop fleet dispatch: %v\n", err)
		return 1
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	now := time.Now().UTC()
	cells := make([]benchloop.FleetCell, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		path := filepath.Join(queue, e.Name())
		req, err := readBenchFleetRequest(path)
		if err != nil {
			continue
		}
		report.Considered++
		if req.State == benchloop.FleetRunning {
			// A claim only means something while its owner is alive: reconcile it against
			// the lock's own pid/host read-back, not against the row's say-so, so a
			// dispatcher killed mid-run cannot strand the cell in "running" forever.
			if state, reason := benchloop.ReconcileFleetRunning(readBenchFleetClaim(path, req), now); state != benchloop.FleetRunning {
				req.State, req.HeldReason = state, reason
				_ = os.Remove(path + ".claim")
				_ = writeBenchFleetRequest(path, req)
				report.Reconciled++
			}
		}
		cell := benchFleetCell(req)
		dispatch, why := benchloop.ShouldDispatchFleetCell(cell, now)
		if dispatch && report.Claimed >= *max {
			dispatch, why = false, "tick dispatch budget reached"
		}
		if !dispatch {
			if why != "" {
				report.Skipped = append(report.Skipped, benchFleetSkip{RequestID: req.ID, Machine: req.Machine, State: req.State, Reason: why})
			}
			cells = append(cells, cell)
			continue
		}
		lock, err := os.OpenFile(path+".claim", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			cells = append(cells, cell)
			continue
		}
		host, _ := os.Hostname()
		_, _ = fmt.Fprintf(lock, "%d %s %s\n", os.Getpid(), host, now.Format(time.RFC3339))
		lock.Close()
		report.Claimed++
		req.State = benchloop.FleetRunning
		_ = writeBenchFleetRequest(path, req)
		witness := executeBenchFleetRequest(*root, req, run)
		witnessDir := filepath.Join(filepath.Dir(queue), "witnesses")
		_ = os.MkdirAll(witnessDir, 0o755)
		if b, e := json.MarshalIndent(witness, "", "  "); e == nil {
			_ = writeAtomic(filepath.Join(witnessDir, req.ID+".json"), append(b, '\n'))
		}
		if witness.State == "succeeded" {
			if err := ingestBenchFleetWitness(*root, req, witness); err != nil {
				markBenchFleetRunFailed(&witness, &req, path, "ingest benchmark witness", err)
			} else if err := updateBenchFleetCatalog(*root); err != nil {
				markBenchFleetRunFailed(&witness, &req, path, "update benchmark catalog", err)
			}
		}
		req = applyBenchFleetOutcome(req, witness)
		_ = writeBenchFleetRequest(path, req)
		_ = os.Remove(path + ".claim")
		cells = append(cells, benchFleetCell(req))
		report.Witnesses = append(report.Witnesses, witness)
		switch witness.State {
		case "succeeded":
			report.Succeeded++
		case "failed":
			report.Failed++
		default:
			report.Waiting++
		}
	}
	// The result reports the DURABLE queue, not this tick's luck: a queue whose cells
	// are all failed or held claims nothing this tick and used to exit 0 for it (#6503).
	report.Utility = benchloop.SummarizeFleet(cells)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		fmt.Fprintf(stdout, "bench fleet dispatch: claimed=%d succeeded=%d failed=%d waiting=%d reconciled=%d\n", report.Claimed, report.Succeeded, report.Failed, report.Waiting, report.Reconciled)
		fmt.Fprintf(stdout, "utility: attempted=%d measured=%d held=%d repeated-failures=%d compute=%.0fs result=%d (%s)\n",
			report.Utility.Attempted, report.Utility.Measured, report.Utility.Held, report.Utility.RepeatedFailures, report.Utility.ComputeSeconds, report.Utility.Result, report.Utility.Reason)
	}
	return report.Utility.Result
}

// benchFleetCell projects one durable queue row onto the health rules in
// internal/benchloop. cmd/fak owns the on-disk JSON; the package owns the decisions.
func benchFleetCell(req benchFleetRequest) benchloop.FleetCell {
	return benchloop.FleetCell{
		Machine: req.Machine, Benchmark: req.Benchmark, State: req.State,
		HeldReason: req.HeldReason, HeldSince: parseBenchFleetTime(req.HeldSince),
		LastAttempt: parseBenchFleetTime(req.LastAttemptAt), Attempts: req.Attempts,
		Failures: req.Failures, Seconds: req.Seconds, Measured: req.Measured,
	}
}

// benchFleetQueueCells reads the whole durable request queue under root. Callers
// that judge the loop as a whole -- the status verb and the install gate -- read the
// cells rather than a single tick, because one tick that claimed nothing is not
// evidence that the fleet measured anything.
func benchFleetQueueCells(root string) []benchloop.FleetCell {
	queue := filepath.Join(root, ".fak", "bench-fleet", "requests")
	entries, err := os.ReadDir(queue)
	if err != nil {
		return nil
	}
	cells := make([]benchloop.FleetCell, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		req, err := readBenchFleetRequest(filepath.Join(queue, e.Name()))
		if err != nil {
			continue
		}
		cells = append(cells, benchFleetCell(req))
	}
	return cells
}

// applyBenchFleetOutcome folds one execution back into the durable row: the state,
// the attempt/failure counters the utility report is computed from, the compute the
// cell has cost so far, and whether the node has ever produced a benchmark number.
// HeldSince tracks when the current hold was last confirmed, so each re-probe
// restarts the hold window instead of re-dispatching an unavailable node every tick.
func applyBenchFleetOutcome(req benchFleetRequest, w benchFleetWitness) benchFleetRequest {
	req.State = w.State
	req.Attempts++
	req.LastAttemptAt = w.FinishedAt
	req.Seconds += benchFleetWitnessSeconds(w)
	if _, _, ok := benchloop.FleetMeasurement(w.Output); ok {
		req.Measured = true
	}
	state, gap := benchloop.NormalizeFleetState(w.State)
	switch state {
	case benchloop.FleetFailed:
		req.Failures++
		req.HeldSince, req.HeldReason = "", ""
	case benchloop.FleetHeld:
		req.HeldSince, req.HeldReason = w.FinishedAt, gap
	default:
		req.Failures = 0
		req.HeldSince, req.HeldReason = "", ""
	}
	return req
}

// benchFleetWitnessSeconds is the wall clock one execution spent, used as the
// compute-cost axis of the utility report. An unparsable pair costs zero rather
// than poisoning the total.
func benchFleetWitnessSeconds(w benchFleetWitness) float64 {
	started, finished := parseBenchFleetTime(w.StartedAt), parseBenchFleetTime(w.FinishedAt)
	if started.IsZero() || finished.IsZero() || finished.Before(started) {
		return 0
	}
	return finished.Sub(started).Seconds()
}

// readBenchFleetClaim reads back the lock a running row claims to be held by. The
// lock carries "<pid> <host> <rfc3339>", so this host can independently disprove a
// claim its own dead dispatcher left behind; a claim taken elsewhere is judged only
// by its lease, because a remote pid is not readable from here.
func readBenchFleetClaim(path string, req benchFleetRequest) benchloop.FleetClaim {
	claim := benchloop.FleetClaim{Started: parseBenchFleetTime(req.LastAttemptAt)}
	b, err := os.ReadFile(path + ".claim")
	if err != nil {
		return claim
	}
	claim.Present = true
	fields := strings.Fields(string(b))
	if len(fields) > 0 {
		claim.PID, _ = strconv.Atoi(fields[0])
	}
	if len(fields) > 1 {
		claim.Host = fields[1]
	}
	if len(fields) > 2 {
		if started := parseBenchFleetTime(fields[2]); !started.IsZero() {
			claim.Started = started
		}
	}
	if claim.Started.IsZero() {
		// A lock written before the loop recorded its owner (#6503) carries no stamp,
		// and its row may carry no attempt time either. The lock's own mtime is the
		// independent reading that keeps such a claim from being immortal.
		if info, statErr := os.Stat(path + ".claim"); statErr == nil {
			claim.Started = info.ModTime().UTC()
		}
	}
	host, _ := os.Hostname()
	claim.Local = claim.Host != "" && host != "" && strings.EqualFold(claim.Host, host)
	claim.Alive = claim.Local && dispatchaudit.ProcessAlive(claim.PID)
	return claim
}

func parseBenchFleetTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(s))
	if err != nil {
		return time.Time{}
	}
	return t.UTC()
}

// markBenchFleetRunFailed demotes a run that EXECUTED cleanly but whose post-run
// bookkeeping (witness ingest, catalog update) failed. A witness the fleet cannot record
// is not a success, so the witness and its queue request both flip to "failed" carrying
// stage as the reason, and the request is written back so the queue reflects it. stage
// names the step that failed; err is appended in the "<stage>: <err>" shape an operator
// already reads off the witness file.
func markBenchFleetRunFailed(w *benchFleetWitness, req *benchFleetRequest, path, stage string, err error) {
	w.State = "failed"
	w.Error = stage + ": " + err.Error()
	req.State = w.State
	_ = writeBenchFleetRequest(path, *req)
}

func executeBenchFleetRequest(root string, req benchFleetRequest, run benchFleetExec) benchFleetWitness {
	w := benchFleetWitness{Schema: "fak.bench-fleet.witness.v1", RequestID: req.ID, Machine: req.Machine, Benchmark: req.Benchmark, Model: req.Model, Precision: req.Precision, StartedAt: time.Now().UTC().Format(time.RFC3339)}
	name, args, route, state, err := benchFleetRoute(root, req)
	w.Route = route
	w.State = state
	if err != nil {
		w.Error = err.Error()
		w.FinishedAt = time.Now().UTC().Format(time.RFC3339)
		return w
	}
	w.Command = append([]string{name}, args...)
	out, exit, runErr := run(name, args...)
	w.Output = string(out)
	w.ExitCode = exit
	if runErr != nil {
		lowerOutput := strings.ToLower(w.Output)
		if w.Route == "dgxbridge" && (strings.Contains(lowerOutput, "no slack channel") || strings.Contains(lowerOutput, "missing")) {
			w.State = "waiting_credentials"
		} else if (strings.HasPrefix(w.Route, "mac:") || strings.HasPrefix(w.Route, "workstation:")) && benchFleetSessionUnavailable(lowerOutput) {
			w.State = "waiting_session"
		} else if strings.HasPrefix(w.Route, "workstation:") && (strings.Contains(lowerOutput, "command not found") || strings.Contains(lowerOutput, "no such file or directory") || strings.Contains(lowerOutput, "cannot find the path")) {
			w.State = "waiting_provision"
		} else if strings.HasPrefix(w.Route, "gcp:") && (strings.Contains(lowerOutput, "no such file or directory") || strings.Contains(lowerOutput, "command not found") || strings.Contains(lowerOutput, "need -hf")) {
			w.State = "waiting_provision"
		} else {
			w.State = "failed"
		}
		w.Error = runErr.Error()
	} else if !hasBenchNodeWitness(w.Output) {
		w.State = "failed"
		w.Error = "remote witness missing FAK_BENCH_NODE marker"
	} else {
		w.State = "succeeded"
	}
	w.FinishedAt = time.Now().UTC().Format(time.RFC3339)
	return w
}

func benchFleetSessionUnavailable(output string) bool {
	for _, marker := range []string{"timed out", "timeout", "502 bad gateway", "failed to respond", "connection closed", "connection refused", "permission denied", "host key verification failed", "no route to host"} {
		if strings.Contains(output, marker) {
			return true
		}
	}
	return false
}

func hasBenchNodeWitness(output string) bool {
	for _, line := range strings.Split(strings.ReplaceAll(output, "\r\n", "\n"), "\n") {
		if value, ok := strings.CutPrefix(strings.TrimSpace(line), "FAK_BENCH_NODE="); ok {
			value = strings.TrimSpace(value)
			return value != "" && !strings.ContainsAny(value, "$()")
		}
	}
	return false
}

func benchFleetRemoteCommand(req benchFleetRequest) string {
	prefix := "printf 'FAK_BENCH_NODE='; hostname; cd ~/fak && "
	// Every gcp-* node run below obeys ONE rule: execute the bounded, committed workload
	// on the node under its own Go toolchain — except on Container-Optimized OS
	// (gcp-g2-l4-32), which mounts the persistent home filesystem noexec, where the SAME
	// workload runs against the SAME provisioned source inside the pinned Go container.
	// The table carries only what differs per benchmark (its command, and for a run that
	// needs a credential in the container, the extra `docker run` flags and the container
	// form of the command); the rule itself is written once, below the table.
	if strings.HasPrefix(req.Machine, "gcp-") {
		for _, row := range []struct{ benchmark, run, dockerFlags, dockerRun string }{
			// Replay the committed airline trace with output isolated under /tmp. This is
			// bounded, deterministic real turn-tax execution on every provisioned node.
			{benchmark: "turn-tax", run: "go run ./cmd/fak turntax -suite turntax-airline -out /tmp/fak-turntax-report.json"},
			// Use the same bounded synthetic session workload on every node so the fleet
			// compares the real session execution path without an unbounded model load.
			{benchmark: "session-benchmark", run: "go run ./cmd/sessionbench -synthetic tiny -agents 2 -turns 2 -reps 1 -out /tmp/fak-sessionbench.json"},
			// Assemble the committed cross-model cards on each node with outputs isolated
			// under /tmp; the canonical fleet witness captures the resulting parity summary.
			{benchmark: "parity", run: "go run ./cmd/paritybench -out-json /tmp/fak-parity.json -out-md /tmp/fak-parity.md"},
			// Bound the generated fan-out matrix so every scheduled node produces a real,
			// reproducible topology witness without leaving output files in the source tree.
			{benchmark: "fan-benchmark", run: "go run ./cmd/fanbench -agents 1,4 -sub-turns 1 -trials 1 -prefixes smoke -out /tmp/fak-fanbench.json -csv /tmp/fak-fanbench.csv"},
			// The concept replay is intentionally bounded and deterministic. Run it on each
			// provisioned node to fill the topology cell with node-authored lineage rather
			// than accepting the control point's local replay as a remote witness.
			{benchmark: "concept-benchmark", run: "go run ./cmd/conceptbench -replay cmd/conceptbench/testdata/replay"},
			// Replace the planner's <task> placeholder with the real bounded agent workload.
			// The node reads its mode-0600 gateway credential locally; no secret enters argv
			// or the canonical witness captured by the control point. The container arm reads
			// the same mode-0600 file, handing it to the container as an env var instead.
			{
				benchmark:   "agent-live",
				run:         "export FAK_GROQ_API_KEY=$(cat $HOME/.config/fak/groq.key); go run ./cmd/fak agent -provider openai -base-url https://api.groq.com/openai/v1 -api-key-env FAK_GROQ_API_KEY -model qwen/qwen3.6-27b -max-turns 10",
				dockerFlags: "-e FAK_GROQ_API_KEY=$(cat $HOME/.config/fak/groq.key) ",
				dockerRun:   "go run ./cmd/fak agent -provider openai -base-url https://api.groq.com/openai/v1 -api-key-env FAK_GROQ_API_KEY -model qwen/qwen3.6-27b -max-turns 10",
			},
		} {
			if req.Benchmark != row.benchmark {
				continue
			}
			if req.Machine == "gcp-g2-l4-32" {
				containerRun := row.dockerRun
				if containerRun == "" {
					containerRun = row.run
				}
				return "printf 'FAK_BENCH_NODE='; hostname; docker run --rm " + row.dockerFlags + "-v $HOME/fak:/src -w /src golang:1.26 /usr/local/go/bin/" + containerRun
			}
			return prefix + "export PATH=$HOME/.local/go/bin:$PATH; " + row.run
		}
	}
	if req.Benchmark == "qwen36" && (strings.HasPrefix(req.Machine, "gcp-") || req.Machine == "a100" || req.Machine == "cpu-server-a") {
		// The planner explicitly asks for Qwen3.6 through a gateway. Each compute node
		// launches and witnesses its own request; the provisioned credential stays in a
		// mode-0600 file and is never placed in argv, the request JSON, or the witness.
		credential := "key=$HOME/.config/fak/groq.key; test -s $key; "
		if req.Machine == "a100" || req.Machine == "cpu-server-a" {
			// Lab nodes may not hold gateway credentials. Still execute a real node-local
			// benchmark and report provisioning honestly instead of replaying placeholder prose.
			return benchFleetLabPrefix() + "go run ./cmd/sessionbench -synthetic tiny -agents 2 -turns 2 -reps 1"
		}
		return "printf 'FAK_BENCH_NODE='; hostname; " + credential +
			"curl -fsS -o /tmp/fak-qwen36-response.json -w 'FAK_BENCH_HTTP=%{http_code} FAK_BENCH_SECONDS=%{time_total}' " +
			"-H \"Authorization: Bearer $(cat $key)\" -H 'Content-Type: application/json' " +
			"-d '{\"model\":\"qwen/qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with benchmark ok\"}],\"max_tokens\":8}' " +
			"https://api.groq.com/openai/v1/chat/completions && echo && " +
			"python3 -c 'import json; d=json.load(open(\"/tmp/fak-qwen36-response.json\")); assert d.get(\"model\")==\"qwen/qwen3.6-27b\"; print(\"FAK_BENCH_MODEL=\"+d[\"model\"]); print(\"FAK_BENCH_COMPLETION_TOKENS=\"+str(d.get(\"usage\",{}).get(\"completion_tokens\",0)))'"
	}
	if req.Benchmark == "radix-benchmark" && (strings.HasPrefix(req.Machine, "gcp-") || req.Machine == "a100" || req.Machine == "cpu-server-a") {
		if req.Machine == "gcp-g2-l4-32" {
			// COS home is noexec, so run the provisioned source and real weights in Go's container.
			return "printf 'FAK_BENCH_NODE='; hostname; docker run --rm -v $HOME/fak:/src -v $HOME/models/smollm2-135m:/models/smollm2-135m -w /src golang:1.26 /usr/local/go/bin/go run ./cmd/radixbench -hf /models/smollm2-135m -lean -quant -reps 1 -only few-shot"
		}
		if req.Machine == "a100" || req.Machine == "cpu-server-a" {
			return benchFleetLabPrefix() + "go run ./cmd/radixbench -live=false -reps 1 -only few-shot"
		}
		return prefix + "export PATH=$HOME/.local/go/bin:$PATH; go run ./cmd/radixbench -hf ~/models/smollm2-135m -lean -quant -reps 1 -only few-shot"
	}
	if req.Benchmark == "model-benchmark" && (strings.HasPrefix(req.Machine, "gcp-") || req.Machine == "a100" || req.Machine == "cpu-server-a") {
		if req.Machine == "gcp-g2-l4-32" {
			// Container-Optimized OS mounts the persistent home filesystem noexec.
			// Run the provisioned source and weights in the pinned Go container instead.
			return "printf 'FAK_BENCH_NODE='; hostname; docker run --rm -v $HOME/fak:/src -v $HOME/models/smollm2-135m:/models/smollm2-135m -w /src golang:1.26 /usr/local/go/bin/go run ./cmd/modelbench -hf /models/smollm2-135m -quant -decode-steps 4 -decode-reps 1 -prefill-reps 1"
		}
		if req.Machine == "a100" || req.Machine == "cpu-server-a" {
			return benchFleetLabPrefix() + "go run ./cmd/modelbench -synthetic tiny -decode-steps 4 -decode-reps 1 -prefill-reps 1"
		}
		return prefix + "export PATH=$HOME/.local/go/bin:$PATH; go run ./cmd/modelbench -hf ~/models/smollm2-135m -quant -decode-steps 4 -decode-reps 1 -prefill-reps 1"
	}
	if req.Benchmark == "gpu-benchmark" {
		if req.Machine == "a100" {
			return benchFleetLabPrefix() + "CUDA_HOME=/usr/local/cuda FAK_CUDA_ARCH=sm_80 bash tools/run_485_acceptance_on_gpu.sh"
		}
		if req.Machine == "gcp-g2-l4-32" {
			return "printf 'FAK_BENCH_NODE='; hostname; curl -fsS -o /tmp/fak-bench-response -w 'FAK_BENCH_HTTP=%{http_code} FAK_BENCH_SECONDS=%{time_total}' -H 'Content-Type: application/json' -d '{\"model\":\"qwen2.5-0.5b-gpu\",\"messages\":[{\"role\":\"user\",\"content\":\"Say benchmark ok\"}],\"max_tokens\":4}' http://127.0.0.1:8082/v1/chat/completions && echo && cat /tmp/fak-bench-response"
		}
		arch := ""
		switch req.Machine {
		case "gcp-g2-l4":
			arch = "sm_89"
		case "gcp-a3-high-h100-1g":
			// The current catalog ID is fulfilled by the sanctioned A100 serve node.
			arch = "sm_80"
		}
		if arch != "" {
			return prefix + "export PATH=$HOME/.local/go/bin:$PATH; CUDA_HOME=/usr/local/cuda FAK_CUDA_ARCH=" + arch + " bash internal/compute/build_cuda.sh binary ./cmd/gpucheck /tmp/fak-gpucheck && /tmp/fak-gpucheck -hf ~/models/qwen05 -n 4"
		}
	}
	return prefix + req.Command
}

func benchFleetLabPrefix() string {
	return "printf 'FAK_BENCH_NODE='; hostname; export PATH=/usr/local/go/bin:/usr/local/cuda/bin:$PATH GOCACHE=/tmp/gocache GOPATH=/tmp/gopath GOTOOLCHAIN=auto; mkdir -p /tmp/fak-bench-results; repo=$(mktemp -d /tmp/fak-bench.XXXXXX) && git clone --depth 1 https://github.com/anthony-chaudhary/fak $repo && cd $repo && "
}

func benchFleetRoute(root string, req benchFleetRequest) (string, []string, string, string, error) {
	remote := benchFleetRemoteCommand(req)
	switch req.Machine {
	case "gcp-g2-l4":
		return "gcloud", []string{"compute", "ssh", "fak-cuda-build-l4", "--zone", "us-central1-b", "--quiet", "--command", remote}, "gcp:ssh/fak-cuda-build-l4", "running", nil
	case "gcp-g2-l4-32":
		return "gcloud", []string{"compute", "ssh", "fak-realmodel", "--zone", "us-central1-a", "--quiet", "--command", remote}, "gcp:ssh/fak-realmodel", "running", nil
	case "gcp-a3-high-h100-1g":
		return "gcloud", []string{"compute", "ssh", "fak-qwen-serve", "--zone", "us-central1-f", "--quiet", "--command", remote}, "gcp:ssh/fak-qwen-serve", "running", nil
	case "a100", "cpu-server-a":
		bridge := filepath.Clean(filepath.Join(root, "..", "fak-private", ".dgxbridge-verify", "dgxbridge-fresh.exe"))
		if runtime.GOOS != "windows" {
			bridge = filepath.Clean(filepath.Join(root, "..", "fak-private", ".dgxbridge-verify", "dgxbridge-fresh"))
		}
		if _, err := os.Stat(bridge); err != nil {
			return "", nil, "dgxbridge", "waiting_credentials", errors.New("private bridge unavailable")
		}
		cfg := benchFleetRoutes(root)
		channel := cfg.A100Channel
		if req.Machine == "cpu-server-a" {
			channel = cfg.CPUServerChannel
		}
		args := []string{}
		if channel != "" {
			args = append(args, "-channel", channel)
		}
		// The bridge's file-readback path wedges under busy Slack channels. Transcript
		// mode plus a generous timeout is the independently verified lab runbook.
		// Transport the script as base64 so the Slack hub cannot reinterpret URL
		// slashes, shell separators, or quoted JSON before bash receives it.
		encoded := base64.StdEncoding.EncodeToString([]byte(remote))
		remote = "printf %s " + encoded + " | base64 -d | bash"
		args = append(args, "-timeout", "4m", "-remote-out", "/tmp/fak-bench-results/"+req.ID+".bridge.out", "run", remote)
		return bridge, args, "dgxbridge", "running", nil
	case "workstation-a":
		cfg := benchFleetRoutes(root)
		if cfg.WorkstationHost == "" || cfg.WorkstationUser == "" || cfg.WorkstationIdentityFile == "" {
			return "", nil, "workstation:ssh", "waiting_session", errors.New("workstation SSH session is not configured")
		}
		distro := cfg.WorkstationDistro
		if distro == "" {
			distro = "Ubuntu"
		}
		script := benchFleetWorkstationScript(req)
		encoded := base64.StdEncoding.EncodeToString([]byte(script))
		remotePS := "$ErrorActionPreference='Stop'; $p=Join-Path $env:TEMP 'fak-bench-" + req.ID + ".sh'; " +
			"[IO.File]::WriteAllBytes($p,[Convert]::FromBase64String('" + encoded + "')); " +
			"$wp=('/mnt/'+$p.Substring(0,1).ToLower()+$p.Substring(2).Replace('\\','/')); try { " +
			"& wsl.exe -d '" + strings.ReplaceAll(distro, "'", "''") + "' -- bash $wp; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Remove-Item -LiteralPath $p -ErrorAction SilentlyContinue }"
		destination := cfg.WorkstationUser + "@" + cfg.WorkstationHost
		return "ssh", []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", cfg.WorkstationIdentityFile, destination, "powershell.exe -NoProfile -NonInteractive -Command \"" + strings.ReplaceAll(remotePS, "\"", "\\\"") + "\""}, "workstation:ssh/" + cfg.WorkstationHost, "running", nil
	case "node-macos-a":
		remote = "printf 'FAK_BENCH_NODE='; hostname; " + req.Command
		remote = "set -eu; export PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=auto; " +
			"repo=; for p in \"$HOME/fak-3xbench/fak\" \"$HOME/.fak-mac-bench/fak\" \"$HOME/fak-3xbench\"; do " +
			"if test -d \"$p/.git\" && test -d \"$p/cmd/livecodebench\"; then repo=$p; break; fi; done; " +
			"test -n \"$repo\"; cd \"$repo\"; " + remote
		cfg := benchFleetRoutes(root)
		host := strings.TrimSpace(cfg.MacHost)
		if host == "" {
			return "", nil, "mac:ssh", "waiting_session", errors.New("mac runner host is not configured")
		}
		args := []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "IdentitiesOnly=yes"}
		if cfg.MacIdentityFile != "" {
			args = append(args, "-i", cfg.MacIdentityFile)
		}
		args = append(args, host, remote)
		return "ssh", args, "mac:ssh/" + host, "running", nil
	default:
		return "", nil, "unknown", "waiting_route", fmt.Errorf("no route for machine %q", req.Machine)
	}
}

type benchFleetRouteConfig struct {
	A100Channel             string `json:"a100_channel"`
	CPUServerChannel        string `json:"cpu_server_channel"`
	MacHost                 string `json:"mac_host"`
	MacIdentityFile         string `json:"mac_identity_file"`
	WorkstationHost         string `json:"workstation_host"`
	WorkstationUser         string `json:"workstation_user"`
	WorkstationIdentityFile string `json:"workstation_identity_file"`
	WorkstationDistro       string `json:"workstation_distro"`
}

func benchFleetRoutes(root string) benchFleetRouteConfig {
	var cfg benchFleetRouteConfig
	b, err := os.ReadFile(filepath.Join(root, ".fak", "bench-fleet", "routes.json"))
	if err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_A100_CHANNEL")); value != "" {
		cfg.A100Channel = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_CPU_SERVER_CHANNEL")); value != "" {
		cfg.CPUServerChannel = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_MAC_HOST")); value != "" {
		cfg.MacHost = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_MAC_IDENTITY_FILE")); value != "" {
		cfg.MacIdentityFile = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_WORKSTATION_HOST")); value != "" {
		cfg.WorkstationHost = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_WORKSTATION_USER")); value != "" {
		cfg.WorkstationUser = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_WORKSTATION_IDENTITY_FILE")); value != "" {
		cfg.WorkstationIdentityFile = value
	}
	if value := strings.TrimSpace(os.Getenv("FAK_BENCH_WORKSTATION_DISTRO")); value != "" {
		cfg.WorkstationDistro = value
	}
	return cfg
}

func benchFleetWorkstationScript(req benchFleetRequest) string {
	command := req.Command
	switch req.Benchmark {
	case "gpu-benchmark":
		// This weight-independent acceptance run executes real sm_89 CUDA kernels and
		// emits correctness, VRAM, and timing witnesses on the registered laptop GPU.
		command = "CUDA_HOME=$HOME/cudaenv FAK_CUDA_ARCH=sm_89 bash tools/run_485_acceptance_on_gpu.sh"
	case "session-benchmark":
		command = "go run ./cmd/sessionbench -synthetic tiny -agents 2 -turns 2 -prefix 64 -result 16 -decode 4 -reps 1 -val-scale 32,2,1,4,8"
	case "model-benchmark":
		command = "go run ./cmd/modelbench -hf internal/model/.cache/smollm2-135m -quant -decode-steps 4 -decode-reps 1 -prefill-reps 1"
	}
	return "set -euo pipefail\n" +
		"printf 'FAK_BENCH_NODE='; hostname\n" +
		"printf 'FAK_BENCH_GPU='; nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p'\n" +
		"export PATH=/usr/local/go/bin:$HOME/cudaenv/bin:$PATH\n" +
		"if [ ! -d $HOME/fak/.git ]; then git clone --depth 1 https://github.com/anthony-chaudhary/fak $HOME/fak; fi\n" +
		"git -C $HOME/fak fetch --depth 1 origin main\n" +
		"git -C $HOME/fak reset --hard origin/main\n" +
		"cd $HOME/fak\n" + command + "\n"
}

type benchFleetRunManifest struct {
	Schema    string            `json:"$schema"`
	RunID     string            `json:"run_id"`
	MachineID string            `json:"machine_id"`
	Timestamp string            `json:"timestamp"`
	Git       map[string]any    `json:"git"`
	Harness   map[string]any    `json:"harness"`
	Model     map[string]any    `json:"model"`
	Config    map[string]any    `json:"config"`
	Tags      []string          `json:"tags"`
	Artifacts map[string]string `json:"artifacts"`
}

func ingestBenchFleetWitness(root string, req benchFleetRequest, witness benchFleetWitness) error {
	t, err := time.Parse(time.RFC3339, witness.FinishedAt)
	if err != nil {
		return err
	}
	timestamp := t.UTC().Format("20060102T150405Z")
	runID := req.Machine + "-bench-fleet-" + req.ID + "-" + timestamp
	dir := filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", req.Machine, timestamp+"-bench-fleet-"+req.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	witnessName := "witness.json"
	wb, err := json.MarshalIndent(witness, "", "  ")
	if err != nil {
		return err
	}
	if err := writeAtomic(filepath.Join(dir, witnessName), append(wb, '\n')); err != nil {
		return err
	}
	manifest := benchFleetRunManifest{
		Schema: "benchmark/run-manifest.v1", RunID: runID, MachineID: req.Machine, Timestamp: timestamp,
		Git:     map[string]any{"rev": "unknown", "branch": "main", "dirty": false},
		Harness: map[string]any{"name": "fak-bench-fleet", "version": "1"},
		Model:   map[string]any{"name": req.Model, "precision": req.Precision},
		Config:  map[string]any{"benchmark": req.Benchmark, "route": witness.Route, "request_id": req.ID},
		Tags:    []string{"bench-fleet", req.NodeClass, req.Benchmark}, Artifacts: map[string]string{"witness": witnessName},
	}
	mb, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(filepath.Join(dir, "manifest.json"), append(mb, '\n'))
}

func updateBenchFleetCatalog(root string) error {
	path := filepath.Join(root, "experiments", "benchmark", "catalog.json")
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var catalog map[string]json.RawMessage
	if err := json.Unmarshal(b, &catalog); err != nil {
		return err
	}
	var machines map[string]map[string]any
	if err := json.Unmarshal(catalog["machines"], &machines); err != nil {
		return err
	}
	var runs []map[string]any
	if err := json.Unmarshal(catalog["runs"], &runs); err != nil {
		return err
	}
	seen := make(map[string]bool, len(runs))
	for _, run := range runs {
		if id, _ := run["run_id"].(string); id != "" {
			seen[id] = true
		}
	}
	pattern := filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", "*", "*-bench-fleet-*", "manifest.json")
	manifests, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	for _, path := range manifests {
		mb, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var m benchFleetRunManifest
		if err := json.Unmarshal(mb, &m); err != nil {
			return err
		}
		if seen[m.RunID] {
			continue
		}
		model, _ := m.Model["name"].(string)
		precision, _ := m.Model["precision"].(string)
		rel, err := filepath.Rel(root, filepath.Dir(path))
		if err != nil {
			return err
		}
		runs = append(runs, map[string]any{"run_id": m.RunID, "machine_id": m.MachineID, "timestamp": m.Timestamp, "model": model, "precision": precision, "path": filepath.ToSlash(rel), "provenance": "measured", "tags": m.Tags})
		seen[m.RunID] = true
		if machine := machines[m.MachineID]; machine != nil {
			count := 0
			switch value := machine["runs"].(type) {
			case float64:
				count = int(value)
			case int:
				count = value
			}
			machine["runs"] = count + 1
			lastRun, _ := machine["last_run"].(string)
			if m.Timestamp > lastRun {
				machine["last_run"] = m.Timestamp
			}
		}
	}
	machinesJSON, err := json.Marshal(machines)
	if err != nil {
		return err
	}
	runsJSON, err := json.Marshal(runs)
	if err != nil {
		return err
	}
	catalog["machines"] = machinesJSON
	catalog["runs"] = runsJSON
	catalog["last_updated"] = json.RawMessage(strconv.Quote(time.Now().UTC().Format(time.RFC3339)))
	out, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(path, append(out, '\n'))
}

func readBenchFleetRequest(path string) (benchFleetRequest, error) {
	var r benchFleetRequest
	b, e := os.ReadFile(path)
	if e != nil {
		return r, e
	}
	e = json.Unmarshal(b, &r)
	return r, e
}
func writeAtomic(path string, b []byte) error {
	tmp := path + fmt.Sprintf(".tmp-%d", os.Getpid())
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

const (
	benchFleetExecTimeout = 10 * time.Minute
	benchFleetWaitDelay   = 5 * time.Second
)

func newBenchFleetExecCommand(name string, timeout time.Duration, args ...string) (*exec.Cmd, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	c := exec.CommandContext(ctx, name, args...)
	// Some launchers return while a descendant still owns the redirected pipe handles.
	// Bound that wait so one route cannot pin the recurring scheduled tick forever.
	c.WaitDelay = benchFleetWaitDelay
	configureDispatchHelperCommand(c)
	return c, cancel
}

func defaultBenchFleetExec(name string, args ...string) ([]byte, int, error) {
	c, cancel := newBenchFleetExecCommand(name, benchFleetExecTimeout, args...)
	defer cancel()
	var b bytes.Buffer
	c.Stdout = &b
	c.Stderr = &b
	err := c.Run()
	// The Windows Cloud SDK launcher may return before its child has flushed the
	// inherited redirected handles. A direct gcloud retry provides the independent
	// node marker instead of accepting an empty exit-0 witness.
	if err == nil && b.Len() == 0 && name == "gcloud" {
		cancel()
		c, cancel = newBenchFleetExecCommand("gcloud.cmd", benchFleetExecTimeout, args...)
		c.Stdout = &b
		c.Stderr = &b
		err = c.Run()
	}
	if err == nil {
		return b.Bytes(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return b.Bytes(), ee.ExitCode(), err
	}
	return b.Bytes(), -1, err
}
