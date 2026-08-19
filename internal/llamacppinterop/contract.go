// Package llamacppinterop defines fak's versioned delegation seam for llama.cpp.
package llamacppinterop

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

const Schema = "fak.llamacppinterop/1"

type Outcome string

const (
	OutcomeDelegate Outcome = "delegate"
	OutcomeAbstain  Outcome = "abstain"
	OutcomeRefuse   Outcome = "refuse"
)

type Capability struct {
	Binary  string `json:"binary"`
	Version string `json:"version"`
	Server  bool   `json:"server"`
}
type Result struct {
	Schema     string     `json:"schema"`
	Outcome    Outcome    `json:"outcome"`
	Reason     string     `json:"reason"`
	Capability Capability `json:"capability"`
	Argv       []string   `json:"argv,omitempty"`
}
type Runner interface {
	Output(context.Context, string, ...string) ([]byte, error)
}
type ExecRunner struct{}

func (ExecRunner) Output(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

var versionRE = regexp.MustCompile(`(?i)(?:version|build)\s*[: ]\s*([0-9]+(?:\.[0-9]+){0,2}|[a-f0-9]{7,40})`)

func Discover(ctx context.Context, r Runner, binary string) Result {
	if strings.TrimSpace(binary) == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "llama.cpp binary is empty"}
	}
	b, err := r.Output(ctx, binary, "--version")
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: fmt.Sprintf("version probe failed: %v", err)}
	}
	m := versionRE.FindStringSubmatch(string(b))
	if len(m) < 2 {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "llama.cpp version is not parseable"}
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "llama.cpp capability discovered", Capability: Capability{Binary: binary, Version: m[1], Server: strings.Contains(strings.ToLower(binary), "server")}}
}
func Plan(cap Capability, model string, d quantmeta.Descriptor) Result {
	if cap.Binary == "" || cap.Version == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "unproven llama.cpp capability"}
	}
	if strings.TrimSpace(model) == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "model path is empty"}
	}
	if d.Artifact == nil || !strings.EqualFold(d.Artifact.ContainerID, "gguf") {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "llama.cpp delegation requires a GGUF artifact", Capability: cap}
	}
	argv := []string{cap.Binary, "-m", model}
	if cap.Server {
		argv = append(argv, "--host", "127.0.0.1", "--port", "0")
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "delegate to versioned llama.cpp runtime", Capability: cap, Argv: argv}
}

type Health struct {
	Status string `json:"status"`
}

func CheckHealth(ctx context.Context, c *http.Client, url string) Result {
	if c == nil {
		c = &http.Client{Timeout: 2 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(url, "/")+"/health", nil)
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: err.Error()}
	}
	resp, err := c.Do(req)
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "health request failed: " + err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: fmt.Sprintf("health returned HTTP %d", resp.StatusCode)}
	}
	var h Health
	if json.NewDecoder(resp.Body).Decode(&h) != nil || !(h.Status == "ok" || h.Status == "ready") {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "health response is not ready"}
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "llama.cpp server is healthy"}
}
