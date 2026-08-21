// Package modelloadplan selects a model artifact before any download or allocation.
package modelloadplan

import (
	"fmt"
	"sort"
	"strings"
)

const (
	Schema       = "fak.model-load-plan/v1alpha1"
	ModelID      = "Qwen/Qwen3.8-27B"
	GGUFRevision = "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
	OpenRouterID = "qwen/qwen3.8-27b"
)

type Request struct {
	Setup       string `json:"setup"`
	Goal        string `json:"goal"`
	LocalPolicy string `json:"local_policy"`
	Memory      string `json:"memory_topology"`
	DeviceBytes int64  `json:"device_bytes"`
	HostBytes   int64  `json:"host_bytes"`
	DiskBytes   int64  `json:"disk_bytes"`
}

type Candidate struct {
	ID            string   `json:"id"`
	Kind          string   `json:"kind"`
	Quantization  string   `json:"quantization,omitempty"`
	URI           string   `json:"uri"`
	ArtifactBytes int64    `json:"artifact_bytes,omitempty"`
	RuntimeBytes  int64    `json:"runtime_bytes,omitempty"`
	DeviceBytes   int64    `json:"device_bytes,omitempty"`
	HostBytes     int64    `json:"host_bytes,omitempty"`
	Fits          bool     `json:"fits"`
	Reasons       []string `json:"reasons"`
	NextCommand   string   `json:"next_command,omitempty"`
}

type Plan struct {
	Schema     string      `json:"schema"`
	Model      string      `json:"model"`
	Request    Request     `json:"request"`
	Selected   *Candidate  `json:"selected,omitempty"`
	Candidates []Candidate `json:"candidates"`
	Provenance []string    `json:"provenance"`
}

type variant struct {
	quant string
	file  string
	bytes int64
}

var variants = []variant{
	{"IQ2_S", "Qwen3.8-27B-UD-IQ2_S.gguf", 8371970048},
	{"Q3_K_XL", "Qwen3.8-27B-UD-Q3_K_XL.gguf", 13146393504},
	{"Q4_K_M", "Qwen3.8-27B-UD-Q4_K_M.gguf", 16464440224},
	{"Q5_K_M", "Qwen3.8-27B-UD-Q5_K_M.gguf", 19771509664},
	{"Q6_K", "Qwen3.8-27B-UD-Q6_K.gguf", 21983677344},
	{"Q8_0", "Qwen3.8-27B-Q8_0.gguf", 29047086048},
}

func Normalize(r Request) (Request, error) {
	r.Setup = strings.ToLower(strings.TrimSpace(r.Setup))
	r.Goal = strings.ToLower(strings.TrimSpace(r.Goal))
	r.LocalPolicy = strings.ToLower(strings.TrimSpace(r.LocalPolicy))
	r.Memory = strings.ToLower(strings.TrimSpace(r.Memory))
	if r.Setup == "" {
		r.Setup = "personal"
	}
	if r.Goal == "" {
		r.Goal = "balanced"
	}
	if r.LocalPolicy == "" {
		r.LocalPolicy = "auto"
	}
	if r.Memory == "" {
		r.Memory = "unified"
	}
	if !oneOf(r.Setup, "personal", "shared", "batch") {
		return r, fmt.Errorf("setup must be personal, shared, or batch")
	}
	if !oneOf(r.Goal, "balanced", "quality", "latency", "cost") {
		return r, fmt.Errorf("goal must be balanced, quality, latency, or cost")
	}
	if !oneOf(r.LocalPolicy, "auto", "require", "disable") {
		return r, fmt.Errorf("local policy must be auto, require, or disable")
	}
	if !oneOf(r.Memory, "unified", "split") {
		return r, fmt.Errorf("memory topology must be unified or split")
	}
	if r.DeviceBytes < 0 || r.HostBytes < 0 || r.DiskBytes < 0 {
		return r, fmt.Errorf("capacities cannot be negative")
	}
	return r, nil
}

func Build(req Request) (Plan, error) {
	r, err := Normalize(req)
	if err != nil {
		return Plan{}, err
	}
	p := Plan{Schema: Schema, Model: ModelID, Request: r, Provenance: []string{
		"Hugging Face unsloth/Qwen3.8-27B-GGUF file metadata retrieved 2026-08-19",
		"OpenRouter qwen/qwen3.8-27b catalog metadata retrieved 2026-08-19",
	}}
	reserve := map[string]int64{"personal": 2 << 30, "shared": 6 << 30, "batch": 10 << 30}[r.Setup]
	for _, v := range variants {
		c := Candidate{ID: "local/" + strings.ToLower(v.quant), Kind: "local", Quantization: v.quant,
			URI: "hf://unsloth/Qwen3.8-27B-GGUF@" + GGUFRevision + "/" + v.file, ArtifactBytes: v.bytes}
		c.RuntimeBytes = v.bytes + v.bytes/12 + reserve
		c.DeviceBytes, c.HostBytes = placement(r, c.RuntimeBytes)
		c.Fits, c.Reasons = localFit(r, c)
		c.NextCommand = "fak serve --gguf " + c.URI + " --model qwen38:27b"
		p.Candidates = append(p.Candidates, c)
	}
	hosted := Candidate{ID: "hosted/openrouter", Kind: "hosted", URI: OpenRouterID, Fits: r.LocalPolicy != "require",
		Reasons:     []string{"hosted route avoids local weight residency and disk use"},
		NextCommand: "fak serve --provider openai --base-url https://openrouter.ai/api/v1 --model " + OpenRouterID + " --api-key-env OPENROUTER_API_KEY"}
	if !hosted.Fits {
		hosted.Reasons = []string{"rejected because local policy is require"}
	}
	p.Candidates = append(p.Candidates, hosted)
	rank(p.Candidates, r)
	for i := range p.Candidates {
		if p.Candidates[i].Fits {
			chosen := p.Candidates[i]
			p.Selected = &chosen
			break
		}
	}
	return p, nil
}

func placement(r Request, runtime int64) (device, host int64) {
	if r.Memory == "unified" {
		return runtime, 0
	}
	device = min64(runtime, r.DeviceBytes)
	return device, runtime - device
}

func localFit(r Request, c Candidate) (bool, []string) {
	if r.LocalPolicy == "disable" {
		return false, []string{"rejected because local policy is disable"}
	}
	var reasons []string
	if r.DiskBytes == 0 {
		reasons = append(reasons, "disk capacity is unknown")
	} else if c.ArtifactBytes > r.DiskBytes {
		reasons = append(reasons, "artifact exceeds disk capacity")
	}
	if r.Memory == "unified" {
		cap := r.DeviceBytes
		if cap == 0 {
			cap = r.HostBytes
		}
		if cap == 0 {
			reasons = append(reasons, "unified memory capacity is unknown")
		} else if c.RuntimeBytes > cap {
			reasons = append(reasons, "runtime estimate exceeds unified memory capacity")
		}
	} else {
		if r.DeviceBytes == 0 && r.HostBytes == 0 {
			reasons = append(reasons, "device and host memory capacities are unknown")
		} else if c.RuntimeBytes > r.DeviceBytes+r.HostBytes {
			reasons = append(reasons, "runtime estimate exceeds combined device and host memory")
		} else if c.HostBytes > 0 {
			reasons = append(reasons, fmt.Sprintf("plans %d bytes of host offload", c.HostBytes))
		}
	}
	fit := true
	for _, reason := range reasons {
		if strings.Contains(reason, "unknown") || strings.Contains(reason, "exceeds") || strings.Contains(reason, "rejected") {
			fit = false
		}
	}
	if fit {
		reasons = append(reasons, "fits declared memory and disk capacities")
	}
	return fit, reasons
}

func rank(cs []Candidate, r Request) {
	preferred := map[string][]string{
		"balanced": {"Q4_K_M", "Q5_K_M", "Q3_K_XL", "Q6_K", "IQ2_S", "Q8_0"},
		"quality":  {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_XL", "IQ2_S"},
		"latency":  {"Q4_K_M", "Q3_K_XL", "IQ2_S", "Q5_K_M", "Q6_K", "Q8_0"},
		"cost":     {"Q4_K_M", "Q3_K_XL", "IQ2_S", "Q5_K_M", "Q6_K", "Q8_0"},
	}[r.Goal]
	pos := map[string]int{}
	for i, q := range preferred {
		pos[q] = i
	}
	sort.SliceStable(cs, func(i, j int) bool {
		a, b := cs[i], cs[j]
		if a.Fits != b.Fits {
			return a.Fits
		}
		if a.Kind != b.Kind {
			if r.LocalPolicy == "disable" || r.Goal == "cost" || (r.Setup == "shared" && r.LocalPolicy == "auto") {
				return a.Kind == "hosted"
			}
			return a.Kind == "local"
		}
		if a.Kind == "local" {
			return pos[a.Quantization] < pos[b.Quantization]
		}
		return a.ID < b.ID
	})
}

func oneOf(v string, values ...string) bool {
	for _, x := range values {
		if v == x {
			return true
		}
	}
	return false
}
func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
