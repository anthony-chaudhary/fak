package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const schema = "fak-nvidia-nemo-api-demo-audit/1"

type toolTotals struct {
	Calls  int            `json:"calls"`
	ByTool map[string]int `json:"by_tool"`
}

type artifact struct {
	Name        string         `json:"name"`
	Kind        string         `json:"kind"`
	Bytes       int64          `json:"bytes"`
	SHA256      string         `json:"sha256"`
	JSONLEvents int            `json:"jsonl_events,omitempty"`
	ToolCalls   int            `json:"tool_calls,omitempty"`
	ToolByName  map[string]int `json:"tool_by_name,omitempty"`
}

type report struct {
	Schema             string     `json:"schema"`
	GeneratedUTC       string     `json:"generated_utc"`
	Campaign           string     `json:"campaign"`
	Provider           string     `json:"provider"`
	ProviderBaseURL    string     `json:"provider_base_url"`
	ClientHarness      string     `json:"client_harness"`
	InvokedModel       string     `json:"invoked_model"`
	SourceDir          string     `json:"source_dir"`
	Pattern            string     `json:"pattern"`
	ArtifactCount      int        `json:"artifact_count"`
	ArtifactBytes      int64      `json:"artifact_bytes"`
	OutputTrajectories int        `json:"output_trajectories"`
	Tools              toolTotals `json:"tools"`
	Artifacts          []artifact `json:"artifacts"`
}

type event struct {
	Type string `json:"type"`
	Part struct {
		Type string `json:"type"`
		Tool string `json:"tool"`
	} `json:"part"`
}

func main() {
	dir := flag.String("dir", ".goal-runs", "directory containing the local campaign artifacts")
	pattern := flag.String("pattern", "*nemotron*", "artifact basename glob")
	out := flag.String("out", "", "write canonical JSON to this path (stdout when empty)")
	generated := flag.String("generated-utc", "", "stable RFC3339 timestamp for a checked-in manifest")
	flag.Parse()

	*dir = pathutil.ExpandTilde(*dir)
	names, err := filepath.Glob(filepath.Join(*dir, *pattern))
	must(err)
	sort.Strings(names)
	r := report{
		Schema:          schema,
		Campaign:        "NVIDIA-hosted Nemo API demo / OpenCode fallback fleet, 2026-08-11..12",
		Provider:        "NVIDIA-hosted API demo endpoint (NVIDIA NIM/API Catalog)",
		ProviderBaseURL: "https://integrate.api.nvidia.com/v1",
		ClientHarness:   "OpenCode (not Codex and not fak fleet-accounts launch)",
		InvokedModel:    "nvidia/nvidia/nemotron-3-super-120b-a12b",
		SourceDir:       filepath.ToSlash(*dir), Pattern: *pattern,
		Tools: toolTotals{ByTool: map[string]int{}},
	}
	if *generated != "" {
		if _, err := time.Parse(time.RFC3339, *generated); err != nil {
			must(fmt.Errorf("generated-utc: %w", err))
		}
		r.GeneratedUTC = *generated
	} else {
		r.GeneratedUTC = time.Now().UTC().Format(time.RFC3339)
	}

	for _, name := range names {
		a, err := inspect(name)
		must(err)
		r.Artifacts = append(r.Artifacts, a)
		r.ArtifactBytes += a.Bytes
		if a.Kind == "trajectory-output" {
			r.OutputTrajectories++
		}
		r.Tools.Calls += a.ToolCalls
		for k, v := range a.ToolByName {
			r.Tools.ByTool[k] += v
		}
	}
	r.ArtifactCount = len(r.Artifacts)
	payload, err := json.MarshalIndent(r, "", "  ")
	must(err)
	payload = append(payload, '\n')
	if *out == "" {
		_, err = os.Stdout.Write(payload)
	} else {
		err = os.WriteFile(*out, payload, 0o644)
	}
	must(err)
}

func inspect(name string) (artifact, error) {
	f, err := os.Open(name)
	if err != nil {
		return artifact{}, err
	}
	defer f.Close()
	h := sha256.New()
	data, err := io.ReadAll(io.TeeReader(f, h))
	if err != nil {
		return artifact{}, err
	}
	a := artifact{Name: filepath.Base(name), Kind: kind(name), Bytes: int64(len(data)), SHA256: hex.EncodeToString(h.Sum(nil))}
	if a.Kind != "trajectory-output" {
		return a, nil
	}
	a.ToolByName = map[string]int{}
	scan := bufio.NewScanner(strings.NewReader(string(data)))
	scan.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scan.Scan() {
		var e event
		if json.Unmarshal(scan.Bytes(), &e) != nil {
			continue
		}
		a.JSONLEvents++
		if e.Type == "tool_use" || e.Part.Type == "tool" {
			a.ToolCalls++
			tool := e.Part.Tool
			if tool == "" {
				tool = "unknown"
			}
			a.ToolByName[tool]++
		}
	}
	if err := scan.Err(); err != nil {
		return artifact{}, err
	}
	return a, nil
}

func kind(name string) string {
	base := filepath.Base(name)
	switch {
	case strings.HasSuffix(base, ".out.log"):
		return "trajectory-output"
	case strings.HasSuffix(base, ".err.log"):
		return "trajectory-error"
	case strings.HasSuffix(base, ".in.txt"):
		return "prompt-input"
	case strings.HasSuffix(base, ".worktree"):
		return "worktree-marker"
	case strings.HasSuffix(base, ".pid"):
		return "pid-marker"
	default:
		return "other"
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "nvidia-nemo-api-demo-audit:", err)
		os.Exit(1)
	}
}
