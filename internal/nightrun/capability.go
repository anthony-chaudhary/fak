package nightrun

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Capabilities is the probed fact-sheet of THIS box — everything next() needs to
// decide what data the box can actually collect tonight. It is a value type so a
// test can construct one literally and pin the selector's output; the live
// Probe wires it from the environment.
type Capabilities struct {
	// Box is a stable identifier for this machine (its hostname, or FAK_BOX_ID),
	// recorded on every ledger row so a collected datum is attributable.
	Box string `json:"box"`
	// GPU is the detected accelerator kind: "cuda" | "metal" | "vulkan" | "none".
	GPU string `json:"gpu"`
	// Weights is true when local model weights are present (an export cache dir, a
	// GGUF, or FAK_MODEL_DIR / FAK_GGUF pointing at one).
	Weights bool `json:"weights"`
	// Datasets is true when an external benchmark dataset is checked out under
	// testdata/ (e.g. a WebVoyager export).
	Datasets bool `json:"datasets"`
	// Net is true when outbound network is assumed available (false under
	// FAK_OFFLINE).
	Net bool `json:"net"`
	// Creds maps a credential env-var NAME to whether it is set (never the value).
	Creds map[string]bool `json:"creds"`
}

// knownCredEnv is the closed set of credential env-var NAMES nightrun probes for.
// A Task may require any name; one outside this set is simply re-checked live at
// Satisfies time, so the set is an optimisation, not a gate.
var knownCredEnv = []string{
	"ANTHROPIC_API_KEY", "FAK_GATEWAY_KEY", "HF_TOKEN", "OPENAI_API_KEY",
}

// Probe seams: every external read the probe makes goes through one of these so a
// test drives a fully deterministic box. ProbeLocal wires them to the real OS.
type probeEnv struct {
	getenv   func(string) string
	look     func(string) (string, error)
	exists   func(string) bool
	hostname func() (string, error)
	goos     string
}

// ProbeLocal builds the live Capabilities of the box this process runs on, rooted
// at root for the on-disk checks (weights cache, datasets). It reads the
// environment, looks for nvidia-smi, and falls back to the platform default
// (darwin → metal) — the same load-bearing rule tools/bench_plan.py uses ("no
// CUDA on the mac"), but resolved for the LOCAL box instead of a static roster.
func ProbeLocal(root string) Capabilities {
	return probe(root, probeEnv{
		getenv:   os.Getenv,
		look:     exec.LookPath,
		exists:   pathExists,
		hostname: os.Hostname,
		goos:     runtime.GOOS,
	})
}

func probe(root string, e probeEnv) Capabilities {
	c := Capabilities{Creds: map[string]bool{}}

	c.Box = strings.TrimSpace(e.getenv("FAK_BOX_ID"))
	if c.Box == "" {
		if h, err := e.hostname(); err == nil {
			c.Box = strings.TrimSpace(h)
		}
	}
	if c.Box == "" {
		c.Box = "unknown-box"
	}

	c.GPU = detectGPU(e)
	c.Weights = detectWeights(root, e)
	c.Datasets = detectDatasets(root, e)
	c.Net = !truthy(e.getenv("FAK_OFFLINE"))

	for _, name := range knownCredEnv {
		c.Creds[name] = strings.TrimSpace(e.getenv(name)) != ""
	}
	return c
}

// detectGPU resolves the accelerator kind. An explicit FAK_BACKEND wins (the same
// knob the compute HAL reads); otherwise nvidia-smi on PATH ⇒ cuda; otherwise a
// darwin host ⇒ metal (the Apple GPU the metal backend builds against); else none.
func detectGPU(e probeEnv) string {
	switch strings.ToLower(strings.TrimSpace(e.getenv("FAK_BACKEND"))) {
	case "cuda":
		return "cuda"
	case "metal":
		return "metal"
	case "vulkan":
		return "vulkan"
	}
	if _, err := e.look("nvidia-smi"); err == nil {
		return "cuda"
	}
	if e.goos == "darwin" {
		return "metal"
	}
	return "none"
}

// detectWeights is true when a real model checkpoint is reachable: an explicit
// FAK_MODEL_DIR / FAK_GGUF that exists, or the in-repo export cache.
func detectWeights(root string, e probeEnv) bool {
	for _, env := range []string{"FAK_MODEL_DIR", "FAK_GGUF", "FAK_EXPORT_DIR"} {
		if p := strings.TrimSpace(e.getenv(env)); p != "" && e.exists(p) {
			return true
		}
	}
	return e.exists(filepath.Join(root, "internal", "model", ".cache"))
}

// detectDatasets is true when an external benchmark dataset is checked out — the
// WebVoyager export or a tau trace dir under testdata/.
func detectDatasets(root string, e probeEnv) bool {
	for _, rel := range []string{
		filepath.Join("testdata", "webvoyager"),
		filepath.Join("testdata", "tau2"),
	} {
		if e.exists(filepath.Join(root, rel)) {
			return true
		}
	}
	return false
}

// Satisfies reports whether the box can run a Task, and a human reason when it
// cannot (the first unmet requirement). The check is ANDed over Requires plus
// CredEnv; an offline Task is always feasible.
func (c Capabilities) Satisfies(t Task) (bool, string) {
	for _, r := range t.Requires {
		if ok, why := c.meets(r); !ok {
			return false, why
		}
	}
	for _, name := range t.CredEnv {
		if !c.hasCred(name) {
			return false, fmt.Sprintf("needs credential %s (not set)", name)
		}
	}
	return true, ""
}

func (c Capabilities) meets(r Requirement) (bool, string) {
	switch r {
	case ReqOffline, "":
		return true, ""
	case ReqWeights:
		if c.Weights {
			return true, ""
		}
		return false, "needs local model weights (none found)"
	case ReqDataset:
		if c.Datasets {
			return true, ""
		}
		return false, "needs an external dataset under testdata/ (none found)"
	case ReqCUDA:
		if c.GPU == "cuda" {
			return true, ""
		}
		return false, fmt.Sprintf("needs an NVIDIA GPU (box gpu=%s)", c.gpuOrNone())
	case ReqMetal:
		if c.GPU == "metal" {
			return true, ""
		}
		return false, fmt.Sprintf("needs an Apple GPU (box gpu=%s)", c.gpuOrNone())
	case ReqNet:
		if c.Net {
			return true, ""
		}
		return false, "needs network (FAK_OFFLINE is set)"
	default:
		return false, "unknown requirement " + string(r)
	}
}

func (c Capabilities) hasCred(name string) bool {
	if c.Creds == nil {
		return false
	}
	return c.Creds[name]
}

func (c Capabilities) gpuOrNone() string {
	if c.GPU == "" {
		return "none"
	}
	return c.GPU
}

// CredNames returns the credential names this box has, sorted — for the human
// capability summary.
func (c Capabilities) CredNames() []string {
	var out []string
	for name, ok := range c.Creds {
		if ok {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

func pathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
