package headroom

import (
	"context"
	"time"
)

type BenchArm struct {
	Name       string        `json:"name"`
	Report     BenchReport   `json:"report"`
	Duration   time.Duration `json:"duration"`
	NSPerInput int64         `json:"ns_per_input"`
}

type ComparisonReport struct {
	Schema   string     `json:"schema"`
	Corpus   int        `json:"corpus"`
	Arms     []BenchArm `json:"arms"`
	Complete bool       `json:"complete"`
	Missing  []string   `json:"missing,omitempty"`
}

// CompareBench runs every named compressor over one frozen corpus. It measures
// only local transformation cost and byte preservation/savings; task quality,
// provider tokens, TTFT, regrowth, and total cost remain external obligations.
func CompareBench(names []string, inputs []BenchInput) ComparisonReport {
	report := ComparisonReport{Schema: "fak-headroom-comparison/1", Corpus: len(inputs)}
	for _, name := range names {
		comp, ok := Lookup(name)
		if !ok {
			report.Missing = append(report.Missing, name)
			continue
		}
		start := time.Now()
		arm := BenchArm{Name: name, Report: RunBench(comp, inputs), Duration: time.Since(start)}
		if len(inputs) > 0 {
			arm.NSPerInput = arm.Duration.Nanoseconds() / int64(len(inputs))
		}
		report.Arms = append(report.Arms, arm)
	}
	report.Complete = len(report.Missing) == 0 && len(report.Arms) == len(names)
	return report
}

// noCompression is the tuned local pass-through arm. Provider prefix-cache
// behavior is measured only by the end-to-end witness, not this local runner.
type noCompression struct{}

func (noCompression) Name() string { return "none" }
func (noCompression) Compress(_ context.Context, in Input) (Output, error) {
	return passthrough(in), nil
}

func init() { Register(noCompression{}) }
