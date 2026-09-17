package perfrsiscore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

const Schema = "fak.launch-context.v1"

const Unmeasured = "unmeasured"

// LaunchContext is the versioned, digest-bound record that pins the axes which
// make a performance measurement attributable to its cause. Every axis is
// fail-closed: an absent or explicitly unmeasured value is never inferred, and
// the digest covers the axis values rather than itself so tampering is evident.
type LaunchContext struct {
	Schema          string          `json:"schema"`
	Digest          string          `json:"digest"`
	Hardware        HardwareContext `json:"hardware"`
	CodeVersion     CodeVersion     `json:"code_version"`
	LaunchParams    LaunchParams    `json:"launch_params"`
	ChangeSet       ChangeSet       `json:"change_set"`
	Kernels         []string        `json:"kernels"`
	InputConditions InputConditions `json:"input_conditions"`
	Threads         ThreadContext   `json:"threads"`
	Intervals       []Interval      `json:"intervals"`
	CodePaths       []string        `json:"code_paths"`
}

type HardwareContext struct {
	Platform      string  `json:"platform"`
	Backend       string  `json:"backend"`
	DriverVersion string  `json:"driver_version"`
	KernelVersion string  `json:"kernel_version"`
	Governor      string  `json:"governor"`
	ClocksMHz     int     `json:"clocks_mhz"`
	ThermalC      float64 `json:"thermal_c"`
	PowerW        float64 `json:"power_w"`
}

type CodeVersion struct {
	SHA   string `json:"sha"`
	Dirty bool   `json:"dirty"`
}

type LaunchParams struct {
	Argv          []string          `json:"argv"`
	EnvKeys       []string          `json:"env_keys"`
	ResolvedFlags map[string]string `json:"resolved_flags"`
}

type ChangeSet struct {
	Base string `json:"base"`
	Head string `json:"head"`
}

type InputConditions struct {
	Shapes       []string `json:"shapes"`
	DTypes       []string `json:"dtypes"`
	BatchSize    int      `json:"batch_size"`
	Concurrency  int      `json:"concurrency"`
	PromptTokens int      `json:"prompt_tokens"`
	PrefixTokens int      `json:"prefix_tokens"`
	DecodeTokens int      `json:"decode_tokens"`
}

type ThreadContext struct {
	Workers   int `json:"workers"`
	Observers int `json:"observers"`
	Engine    int `json:"engine"`
}

type Interval struct {
	Name string `json:"name"`
	NS   int64  `json:"ns"`
}

func (c LaunchContext) ComputeDigest() string {
	canonical := c.canonical()
	b, err := json.Marshal(canonical)
	if err != nil {
		panic(fmt.Sprintf("launch context digest: canonical marshal failed: %v", err))
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// canonical returns a normalized copy whose digest field is zeroed and whose
// variable-length slices and maps are sorted, so the digest is deterministic
// across runs and map iteration order.
func (c LaunchContext) canonical() LaunchContext {
	c.Digest = ""
	c.Kernels = sortedStrings(c.Kernels)
	c.CodePaths = sortedStrings(c.CodePaths)
	c.LaunchParams.Argv = sortedStrings(c.LaunchParams.Argv)
	c.LaunchParams.EnvKeys = sortedStrings(c.LaunchParams.EnvKeys)
	if len(c.LaunchParams.ResolvedFlags) == 0 {
		c.LaunchParams.ResolvedFlags = map[string]string{}
	}
	c.InputConditions.Shapes = sortedStrings(c.InputConditions.Shapes)
	c.InputConditions.DTypes = sortedStrings(c.InputConditions.DTypes)
	c.Intervals = sortedIntervals(c.Intervals)
	return c
}

// sortedStrings returns a sorted copy of in. nil and empty inputs share one
// canonical shape: a non-nil empty slice (JSON []), never null.
func sortedStrings(in []string) []string {
	out := make([]string, len(in))
	copy(out, in)
	sort.Strings(out)
	return out
}

// sortedIntervals returns a sorted copy of in. nil and empty inputs share one
// canonical shape: a non-nil empty slice (JSON []), never null.
func sortedIntervals(in []Interval) []Interval {
	out := make([]Interval, len(in))
	copy(out, in)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].NS < out[j].NS
	})
	return out
}

func (c LaunchContext) Witnessed() map[string]bool {
	return map[string]bool{
		"hardware.platform":        witnessed(c.Hardware.Platform),
		"hardware.backend":         witnessed(c.Hardware.Backend),
		"hardware.driver_version":  witnessed(c.Hardware.DriverVersion),
		"hardware.kernel_version":  witnessed(c.Hardware.KernelVersion),
		"hardware.governor":        witnessed(c.Hardware.Governor),
		"hardware.clocks_mhz":      c.Hardware.ClocksMHz != 0,
		"hardware.thermal_c":       c.Hardware.ThermalC != 0,
		"hardware.power_w":         c.Hardware.PowerW != 0,
		"code_version.sha":         witnessed(c.CodeVersion.SHA),
		"code_version.dirty":       witnessed(c.CodeVersion.SHA),
		"launch_params.argv":       len(c.LaunchParams.Argv) > 0,
		"launch_params.env_keys":   anyWitnessed(c.LaunchParams.EnvKeys),
		"launch_params.flags":      len(c.LaunchParams.ResolvedFlags) > 0,
		"change_set.base":          witnessed(c.ChangeSet.Base),
		"change_set.head":          witnessed(c.ChangeSet.Head),
		"kernels":                  len(c.Kernels) > 0,
		"input_conditions.shapes":  len(c.InputConditions.Shapes) > 0,
		"input_conditions.dtypes":  len(c.InputConditions.DTypes) > 0,
		"input_conditions.batch":   c.InputConditions.BatchSize != 0,
		"input_conditions.concurr": c.InputConditions.Concurrency != 0,
		"input_conditions.prompt":  c.InputConditions.PromptTokens != 0,
		"input_conditions.prefix":  c.InputConditions.PrefixTokens != 0,
		"input_conditions.decode":  c.InputConditions.DecodeTokens != 0,
		"threads.workers":          c.Threads.Workers != 0,
		"threads.observers":        c.Threads.Observers != 0,
		"threads.engine":           c.Threads.Engine != 0,
		"intervals":                len(c.Intervals) > 0,
		"code_paths":               len(c.CodePaths) > 0,
	}
}

func witnessed(value string) bool {
	trimmed := strings.TrimSpace(value)
	return trimmed != "" && trimmed != Unmeasured
}

func anyWitnessed(values []string) bool {
	for _, value := range values {
		if witnessed(value) {
			return true
		}
	}
	return false
}

func (c LaunchContext) WitnessedCount() int {
	count := 0
	for _, ok := range c.Witnessed() {
		if ok {
			count++
		}
	}
	return count
}

func (c LaunchContext) AxisCount() int { return len(c.Witnessed()) }

func (c LaunchContext) Validate() error {
	if c.Schema != Schema {
		return fmt.Errorf("schema %q, want %q; fix: set schema to %s", c.Schema, Schema, Schema)
	}
	if c.Digest != "" && c.Digest != c.ComputeDigest() {
		return fmt.Errorf("digest %q does not match recomputed digest %q; fix: recompute the digest after changing axis values", c.Digest, c.ComputeDigest())
	}
	for _, interval := range c.Intervals {
		if strings.TrimSpace(interval.Name) == "" {
			return errors.New("interval name is required; fix: provide a non-empty interval name")
		}
		if interval.NS < 0 {
			return fmt.Errorf("interval %q has negative duration %d; fix: provide a non-negative nanosecond duration", interval.Name, interval.NS)
		}
	}
	return nil
}

func DecodeLaunchContext(b []byte) (LaunchContext, error) {
	var c LaunchContext
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, fmt.Errorf("decode launch context: %w; fix: provide valid JSON conforming to %s", err, Schema)
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return c, errors.New("decode launch context: trailing JSON value; fix: remove trailing JSON data after launch context object")
	}
	if c.Schema != Schema {
		return c, fmt.Errorf("decode launch context: schema %q, want %q; fix: set schema to %s", c.Schema, Schema, Schema)
	}
	if c.Digest != "" && c.Digest != c.ComputeDigest() {
		return c, fmt.Errorf("decode launch context: digest %q does not match recomputed digest %q; fix: recompute the digest after changing axis values", c.Digest, c.ComputeDigest())
	}
	return c, nil
}
