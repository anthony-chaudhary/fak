package session

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// ResourceBudget is a per-session hard ceiling for the wrapped process tree.
// Zero disables an axis, matching the existing optional session budget axes.
// CPU is cumulative user+system process-tree seconds; peak RSS is the maximum
// observed resident-set high-water mark; disk axes are bytes per second.
type ResourceBudget struct {
	PeakRSSBytes        uint64  `json:"peak_rss_bytes,omitempty"`
	CPUSeconds          float64 `json:"cpu_seconds,omitempty"`
	ProcessCount        int     `json:"process_count,omitempty"`
	ReadBytesPerSecond  uint64  `json:"read_bytes_per_second,omitempty"`
	WriteBytesPerSecond uint64  `json:"write_bytes_per_second,omitempty"`
}

func (b ResourceBudget) IsZero() bool {
	return b.PeakRSSBytes == 0 && b.CPUSeconds == 0 && b.ProcessCount == 0 && b.ReadBytesPerSecond == 0 && b.WriteBytesPerSecond == 0
}

// ResourceUsage is one observation of the wrapped process tree. Have* keeps an
// unavailable OS counter distinct from an observed zero; a missing axis never
// trips a budget.
type ResourceUsage struct {
	PeakRSSBytes        uint64  `json:"peak_rss_bytes,omitempty"`
	CPUSeconds          float64 `json:"cpu_seconds,omitempty"`
	ProcessCount        int     `json:"process_count,omitempty"`
	ReadBytesPerSecond  uint64  `json:"read_bytes_per_second,omitempty"`
	WriteBytesPerSecond uint64  `json:"write_bytes_per_second,omitempty"`
	HavePeakRSS         bool    `json:"have_peak_rss,omitempty"`
	HaveCPUSeconds      bool    `json:"have_cpu_seconds,omitempty"`
	HaveProcessCount    bool    `json:"have_process_count,omitempty"`
	HaveReadRate        bool    `json:"have_read_rate,omitempty"`
	HaveWriteRate       bool    `json:"have_write_rate,omitempty"`
}

func (u ResourceUsage) IsZero() bool {
	return !u.HavePeakRSS && !u.HaveCPUSeconds && !u.HaveProcessCount && !u.HaveReadRate && !u.HaveWriteRate
}

// ResourceBudgetAxis names the first breached axis in deterministic order.
type ResourceBudgetAxis string

const (
	ResourceAxisPeakRSS    ResourceBudgetAxis = "peak_rss"
	ResourceAxisCPUSeconds ResourceBudgetAxis = "cpu_seconds"
	ResourceAxisProcess    ResourceBudgetAxis = "process_count"
	ResourceAxisReadRate   ResourceBudgetAxis = "read_bytes_per_second"
	ResourceAxisWriteRate  ResourceBudgetAxis = "write_bytes_per_second"
)

// ResourceBudgetDecision is the pure boundary verdict for a resource sample.
type ResourceBudgetDecision struct {
	Stop  bool
	Axis  ResourceBudgetAxis
	Usage ResourceUsage
}

const ReasonResourceBudgetExhausted = "RESOURCE_BUDGET_EXHAUSTED"

// Decide checks only axes the caller could actually observe. The ceiling is
// inclusive: reaching it is already exhaustion, so a runaway cannot consume one
// extra sample interval past the declared cap.
func (b ResourceBudget) Decide(u ResourceUsage) ResourceBudgetDecision {
	if b.PeakRSSBytes > 0 && u.HavePeakRSS && u.PeakRSSBytes >= b.PeakRSSBytes {
		return ResourceBudgetDecision{Stop: true, Axis: ResourceAxisPeakRSS, Usage: u}
	}
	if b.CPUSeconds > 0 && u.HaveCPUSeconds && u.CPUSeconds >= b.CPUSeconds {
		return ResourceBudgetDecision{Stop: true, Axis: ResourceAxisCPUSeconds, Usage: u}
	}
	if b.ProcessCount > 0 && u.HaveProcessCount && u.ProcessCount >= b.ProcessCount {
		return ResourceBudgetDecision{Stop: true, Axis: ResourceAxisProcess, Usage: u}
	}
	if b.ReadBytesPerSecond > 0 && u.HaveReadRate && u.ReadBytesPerSecond >= b.ReadBytesPerSecond {
		return ResourceBudgetDecision{Stop: true, Axis: ResourceAxisReadRate, Usage: u}
	}
	if b.WriteBytesPerSecond > 0 && u.HaveWriteRate && u.WriteBytesPerSecond >= b.WriteBytesPerSecond {
		return ResourceBudgetDecision{Stop: true, Axis: ResourceAxisWriteRate, Usage: u}
	}
	return ResourceBudgetDecision{Usage: u}
}

// ParseResourceBudget parses the compact operator spelling:
// peak-rss=8GiB,cpu-seconds=3600,processes=8,io-read-bps=1GiB,io-write-bps=512MiB.
func ParseResourceBudget(spec string) (ResourceBudget, error) {
	var b ResourceBudget
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return b, fmt.Errorf("empty resource budget")
	}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, val, ok := strings.Cut(part, "=")
		if !ok {
			return b, fmt.Errorf("resource budget item %q must be key=value", part)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		val = strings.TrimSpace(val)
		if val == "" {
			return b, fmt.Errorf("resource budget key %q has an empty value", key)
		}
		var err error
		switch key {
		case "peak-rss", "peak_rss", "rss":
			b.PeakRSSBytes, err = parseResourceBytes(val)
		case "cpu-seconds", "cpu_seconds", "cpu":
			b.CPUSeconds, err = parseResourceSeconds(val)
		case "processes", "process-count", "process_count":
			b.ProcessCount, err = parseResourceCount(val)
		case "io-read-bps", "read-bps", "read_bytes_per_second", "disk-read-bps":
			b.ReadBytesPerSecond, err = parseResourceRate(val)
		case "io-write-bps", "write-bps", "write_bytes_per_second", "disk-write-bps":
			b.WriteBytesPerSecond, err = parseResourceRate(val)
		default:
			return b, fmt.Errorf("unknown resource budget key %q", key)
		}
		if err != nil {
			return b, fmt.Errorf("%s: %w", key, err)
		}
	}
	return b, nil
}

func parseResourceSeconds(v string) (float64, error) {
	if d, err := time.ParseDuration(v); err == nil {
		if d < 0 {
			return 0, fmt.Errorf("must be non-negative")
		}
		return d.Seconds(), nil
	}
	n, err := strconv.ParseFloat(strings.ReplaceAll(v, "_", ""), 64)
	if err != nil || math.IsNaN(n) || math.IsInf(n, 0) || n < 0 {
		return 0, fmt.Errorf("want non-negative seconds or duration")
	}
	return n, nil
}

func parseResourceCount(v string) (int, error) {
	n, err := strconv.ParseInt(strings.ReplaceAll(v, "_", ""), 10, 32)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("want a positive process count")
	}
	return int(n), nil
}

func parseResourceRate(v string) (uint64, error) {
	for _, suffix := range []string{"/s", "/sec", "ps", "bytes/s", "byte/s"} {
		if strings.HasSuffix(strings.ToLower(v), suffix) {
			v = strings.TrimSpace(v[:len(v)-len(suffix)])
			break
		}
	}
	return parseResourceBytes(v)
}

func parseResourceBytes(v string) (uint64, error) {
	raw := strings.ToLower(strings.TrimSpace(v))
	units := []struct {
		suffix string
		mult   uint64
	}{
		{"kib", 1 << 10}, {"mib", 1 << 20}, {"gib", 1 << 30}, {"tib", 1 << 40},
		{"kb", 1 << 10}, {"mb", 1 << 20}, {"gb", 1 << 30}, {"tb", 1 << 40},
		{"b", 1},
	}
	mult := uint64(1)
	for _, unit := range units {
		if strings.HasSuffix(raw, unit.suffix) {
			mult = unit.mult
			raw = strings.TrimSpace(strings.TrimSuffix(raw, unit.suffix))
			break
		}
	}
	n, err := strconv.ParseUint(strings.ReplaceAll(raw, "_", ""), 10, 64)
	if err != nil || n > ^uint64(0)/mult {
		return 0, fmt.Errorf("want non-negative byte amount with optional unit")
	}
	return n * mult, nil
}

// SetResourceBudget configures the per-tree resource ceilings. A terminal
// session is immutable, matching the other live control setters.
func (t *Table) SetResourceBudget(trace string, b ResourceBudget) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Resource = b })
}

// ObserveResource records a process-tree sample and drains a live session at
// the first breached resource ceiling. It deliberately does not run for a
// session with no configured budget, avoiding revision churn on the normal path.
func (t *Table) ObserveResource(trace string, usage ResourceUsage) (State, bool, ResourceBudgetDecision) {
	if t == nil {
		return DefaultState(trace), false, ResourceBudgetDecision{Usage: usage}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false, ResourceBudgetDecision{Usage: usage}
	}
	if cur.Run == Paused || cur.Run == Draining || cur.Run == Terminating {
		return cur, false, ResourceBudgetDecision{Usage: usage}
	}
	if cur.Resource.IsZero() {
		return cur, false, ResourceBudgetDecision{Usage: usage}
	}
	cur.ResourceUsage = usage
	decision := cur.Resource.Decide(usage)
	if !decision.Stop {
		return t.putLocked(cur), false, decision
	}
	cur.Run = Draining
	cur.Reason = ReasonResourceBudgetExhausted
	final := t.finalizeDrainLocked(cur)
	return final, true, decision
}
