package main

import (
	"bytes"
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
	Schema     string              `json:"schema"`
	Queue      string              `json:"queue"`
	Considered int                 `json:"considered"`
	Claimed    int                 `json:"claimed"`
	Succeeded  int                 `json:"succeeded"`
	Failed     int                 `json:"failed"`
	Waiting    int                 `json:"waiting"`
	Witnesses  []benchFleetWitness `json:"witnesses"`
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
	for _, e := range entries {
		if report.Claimed >= *max {
			break
		}
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		path := filepath.Join(queue, e.Name())
		req, err := readBenchFleetRequest(path)
		if err != nil {
			continue
		}
		report.Considered++
		if req.State != "queued" && !strings.HasPrefix(req.State, "waiting_") {
			continue
		}
		lock, err := os.OpenFile(path+".claim", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			continue
		}
		lock.Close()
		report.Claimed++
		req.State = "running"
		_ = writeBenchFleetRequest(path, req)
		witness := executeBenchFleetRequest(*root, req, run)
		req.State = witness.State
		_ = writeBenchFleetRequest(path, req)
		_ = os.Remove(path + ".claim")
		witnessDir := filepath.Join(filepath.Dir(queue), "witnesses")
		_ = os.MkdirAll(witnessDir, 0o755)
		if b, e := json.MarshalIndent(witness, "", "  "); e == nil {
			_ = writeAtomic(filepath.Join(witnessDir, req.ID+".json"), append(b, '\n'))
		}
		if witness.State == "succeeded" {
			if err := ingestBenchFleetWitness(*root, req, witness); err != nil {
				witness.State = "failed"
				witness.Error = "ingest benchmark witness: " + err.Error()
				req.State = witness.State
				_ = writeBenchFleetRequest(path, req)
			} else if err := updateBenchFleetCatalog(*root); err != nil {
				witness.State = "failed"
				witness.Error = "update benchmark catalog: " + err.Error()
				req.State = witness.State
				_ = writeBenchFleetRequest(path, req)
			}
		}
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
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(report)
	} else {
		fmt.Fprintf(stdout, "bench fleet dispatch: claimed=%d succeeded=%d failed=%d waiting=%d\n", report.Claimed, report.Succeeded, report.Failed, report.Waiting)
	}
	if report.Failed > 0 {
		return 1
	}
	return 0
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
		} else if strings.HasPrefix(w.Route, "mac:") && (strings.Contains(lowerOutput, "timed out") || strings.Contains(lowerOutput, "timeout") || strings.Contains(lowerOutput, "502 bad gateway") || strings.Contains(lowerOutput, "failed to respond") || strings.Contains(lowerOutput, "connection closed")) {
			w.State = "waiting_session"
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
	if req.Benchmark == "model-benchmark" && strings.HasPrefix(req.Machine, "gcp-") {
		if req.Machine == "gcp-g2-l4-32" {
			// Container-Optimized OS mounts the persistent home filesystem noexec.
			// Run the provisioned source and weights in the pinned Go container instead.
			return "printf 'FAK_BENCH_NODE='; hostname; docker run --rm -v $HOME/fak:/src -v $HOME/models/smollm2-135m:/models/smollm2-135m -w /src golang:1.26 /usr/local/go/bin/go run ./cmd/modelbench -hf /models/smollm2-135m -quant -decode-steps 4 -decode-reps 1 -prefill-reps 1"
		}
		return prefix + "export PATH=$HOME/.local/go/bin:$PATH; go run ./cmd/modelbench -hf ~/models/smollm2-135m -quant -decode-steps 4 -decode-reps 1 -prefill-reps 1"
	}
	if req.Benchmark == "gpu-benchmark" {
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
		return bridge, []string{"run", remote}, "dgxbridge", "running", nil
	case "workstation-a":
		if runtime.GOOS != "windows" {
			return "", nil, "local-control", "waiting_session", errors.New("workstation runner is only available on its Windows control node")
		}
		command := "Write-Output ('FAK_BENCH_NODE=' + $env:COMPUTERNAME); Set-Location -LiteralPath '" + strings.ReplaceAll(root, "'", "''") + "'; " + req.Command
		return "powershell.exe", []string{"-NoProfile", "-NonInteractive", "-Command", command}, "local-control", "running", nil
	case "node-macos-a":
		host := benchFleetMacHost(root)
		if host == "" {
			return "", nil, "mac:tailscale", "waiting_session", errors.New("mac runner host is not configured")
		}
		return "tailscale", []string{"ssh", host, remote}, "mac:tailscale/" + host, "running", nil
	default:
		return "", nil, "unknown", "waiting_route", fmt.Errorf("no route for machine %q", req.Machine)
	}
}

type benchFleetRouteConfig struct {
	MacHost string `json:"mac_host"`
}

func benchFleetMacHost(root string) string {
	if host := strings.TrimSpace(os.Getenv("FAK_BENCH_MAC_HOST")); host != "" {
		return host
	}
	b, err := os.ReadFile(filepath.Join(root, ".fak", "bench-fleet", "routes.json"))
	if err != nil {
		return ""
	}
	var cfg benchFleetRouteConfig
	if json.Unmarshal(b, &cfg) != nil {
		return ""
	}
	return strings.TrimSpace(cfg.MacHost)
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
func defaultBenchFleetExec(name string, args ...string) ([]byte, int, error) {
	c := exec.Command(name, args...)
	configureDispatchHelperCommand(c)
	var b bytes.Buffer
	c.Stdout = &b
	c.Stderr = &b
	err := c.Run()
	if err == nil {
		return b.Bytes(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return b.Bytes(), ee.ExitCode(), err
	}
	return b.Bytes(), -1, err
}
