package sessionjournal

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"
)

const BenchmarkSchema = "fak-agentquery-benchmark/1"

type BenchmarkReport struct {
	Schema      string          `json:"schema"`
	OS          string          `json:"os"`
	Arch        string          `json:"arch"`
	GoVersion   string          `json:"go_version"`
	ObservedAt  string          `json:"observed_at"`
	Repetitions int             `json:"repetitions"`
	Cases       []BenchmarkCase `json:"cases"`
	Decision    IndexDecision   `json:"index_decision"`
}

type BenchmarkCase struct {
	Events int             `json:"events"`
	Bytes  int             `json:"journal_bytes"`
	Paths  []BenchmarkPath `json:"paths"`
}

type BenchmarkPath struct {
	Name               string  `json:"name"`
	MedianNS           int64   `json:"median_ns"`
	MedianAllocatedB   uint64  `json:"median_allocated_bytes"`
	ResultDigest       string  `json:"result_digest"`
	ProcessStarts      int     `json:"process_starts"`
	OperatorSteps      int     `json:"operator_steps"`
	ExternalDependency *string `json:"external_dependency"`
	MeasurementScope   string  `json:"measurement_scope"`
}

type IndexDecision struct {
	UseIndex              bool   `json:"use_index"`
	Reason                string `json:"reason"`
	ReopenAtEvents        int    `json:"reopen_at_events"`
	ReopenAtMedianMS      int64  `json:"reopen_at_median_ms"`
	LargestDirectMedianMS int64  `json:"largest_direct_median_ms"`
}

type benchmarkGroup struct {
	Lane, State   string
	Count         int
	Min, Max, Sum int64
}
type benchSample struct {
	ns        int64
	allocated uint64
	digest    string
}

func RunBenchmark(eventCounts []int, repetitions int, observed time.Time) (BenchmarkReport, error) {
	if repetitions < 1 || repetitions > 20 {
		return BenchmarkReport{}, fmt.Errorf("repetitions must be 1..20")
	}
	report := BenchmarkReport{Schema: BenchmarkSchema, OS: runtime.GOOS, Arch: runtime.GOARCH, GoVersion: runtime.Version(), ObservedAt: observed.UTC().Format(time.RFC3339), Repetitions: repetitions}
	for _, count := range eventCounts {
		if count < 2 || count > 1_000_000 || count%2 != 0 {
			return BenchmarkReport{}, fmt.Errorf("event counts must be even and within 2..1000000")
		}
		journal := deterministicJournal(count, observed)
		bc := BenchmarkCase{Events: count, Bytes: len(journal)}
		paths := []struct {
			name          string
			starts, steps int
			dependency    *string
			scope         string
			run           func([]byte, time.Time) ([]benchmarkGroup, error)
		}{
			{"direct_in_process_fold", 0, 1, nil, "includes JSONL decode, authoritative fold, and aggregate", directBenchmarkPath},
			{"ps_json_jq_pipeline", 2, 3, stringPtr("jq"), "models fak ps JSON serialization plus jq-equivalent decode/fold; excludes OS process startup", psJSONJQBenchmarkPath},
			{"direct_jsonl_scan", 0, 2, nil, "single-pass JSONL decode and aggregate without authoritative fold", scanBenchmarkPath},
		}
		var canonical string
		for _, path := range paths {
			samples := make([]benchSample, 0, repetitions)
			for i := 0; i < repetitions; i++ {
				var before, after runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&before)
				start := time.Now()
				groups, err := path.run(journal, observed)
				elapsed := time.Since(start)
				runtime.ReadMemStats(&after)
				if err != nil {
					return BenchmarkReport{}, fmt.Errorf("%s: %w", path.name, err)
				}
				digest := groupDigest(groups)
				if canonical == "" {
					canonical = digest
				} else if digest != canonical {
					return BenchmarkReport{}, fmt.Errorf("%s result mismatch", path.name)
				}
				samples = append(samples, benchSample{elapsed.Nanoseconds(), after.TotalAlloc - before.TotalAlloc, digest})
			}
			sort.Slice(samples, func(i, j int) bool { return samples[i].ns < samples[j].ns })
			allocs := append([]benchSample(nil), samples...)
			sort.Slice(allocs, func(i, j int) bool { return allocs[i].allocated < allocs[j].allocated })
			mid := repetitions / 2
			bc.Paths = append(bc.Paths, BenchmarkPath{Name: path.name, MedianNS: samples[mid].ns, MedianAllocatedB: allocs[mid].allocated, ResultDigest: samples[mid].digest, ProcessStarts: path.starts, OperatorSteps: path.steps, ExternalDependency: path.dependency, MeasurementScope: path.scope})
		}
		report.Cases = append(report.Cases, bc)
	}
	largest := report.Cases[len(report.Cases)-1].Paths[0].MedianNS / int64(time.Millisecond)
	report.Decision = IndexDecision{UseIndex: false, Reason: "bounded direct fold remains dependency-free; add an index only when the largest supported journal repeatedly exceeds the latency threshold", ReopenAtEvents: 200_000, ReopenAtMedianMS: 1000, LargestDirectMedianMS: largest}
	if largest >= report.Decision.ReopenAtMedianMS {
		report.Decision.UseIndex = true
		report.Decision.Reason = "direct fold crossed the declared latency threshold; evaluate a disposable reproducible index"
	}
	return report, nil
}

func deterministicJournal(events int, observed time.Time) []byte {
	var b strings.Builder
	for i := 0; i < events/2; i++ {
		id := fmt.Sprintf("agent-%06d", i)
		lane := []string{"cmd", "docs", "gateway"}[i%3]
		start := observed.Add(-time.Duration((i%1000)+1) * time.Second)
		end := start.Add(time.Duration((i%250)+1) * time.Millisecond)
		fmt.Fprintf(&b, `{"schema":"%s","kind":"open","id":"%s","ts":"%s","registration":{"registration_id":"%s","lane":"%s"}}`+"\n", Schema, id, start.Format(time.RFC3339Nano), id, lane)
		fmt.Fprintf(&b, `{"schema":"%s","kind":"close","id":"%s","ts":"%s"}`+"\n", Schema, id, end.Format(time.RFC3339Nano))
	}
	return []byte(b.String())
}

func directBenchmarkPath(journal []byte, observed time.Time) ([]benchmarkGroup, error) {
	events, h := ParseEventsReport(string(journal))
	if h.Degraded() {
		return nil, fmt.Errorf("generated journal degraded")
	}
	return groupsFromEvents(events, observed), nil
}
func scanBenchmarkPath(journal []byte, observed time.Time) ([]benchmarkGroup, error) {
	scanner := bufio.NewScanner(bytes.NewReader(journal))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	events := make([]Event, 0)
	for scanner.Scan() {
		var e Event
		if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
			return nil, err
		}
		events = append(events, e)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return groupsFromEvents(events, observed), nil
}
func psJSONJQBenchmarkPath(journal []byte, observed time.Time) ([]benchmarkGroup, error) {
	events := ParseEvents(string(journal))
	sessions := FoldEvents(events)
	payload, err := json.Marshal(sessions)
	if err != nil {
		return nil, err
	}
	var decoded []Session
	if err = json.Unmarshal(payload, &decoded); err != nil {
		return nil, err
	}
	return groupsFromSessions(decoded, observed), nil
}
func groupsFromEvents(events []Event, observed time.Time) []benchmarkGroup {
	return groupsFromSessions(FoldEvents(events), observed)
}
func groupsFromSessions(sessions []Session, observed time.Time) []benchmarkGroup {
	type key struct{ lane, state string }
	m := map[key]*benchmarkGroup{}
	for _, s := range sessions {
		lane := ""
		if s.Registration != nil {
			lane = s.Registration.Lane
		}
		state := "closed"
		if !s.Closed {
			state = "live"
		}
		elapsed := s.LastSeen.Sub(s.StartedAt).Milliseconds()
		if s.Closed {
			elapsed = s.LastSeen.Sub(s.StartedAt).Milliseconds()
		}
		k := key{lane, state}
		g := m[k]
		if g == nil {
			g = &benchmarkGroup{Lane: lane, State: state, Min: elapsed, Max: elapsed}
			m[k] = g
		}
		g.Count++
		g.Sum += elapsed
		if elapsed < g.Min {
			g.Min = elapsed
		}
		if elapsed > g.Max {
			g.Max = elapsed
		}
	}
	out := make([]benchmarkGroup, 0, len(m))
	for _, g := range m {
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Lane != out[j].Lane {
			return out[i].Lane < out[j].Lane
		}
		return out[i].State < out[j].State
	})
	return out
}
func groupDigest(groups []benchmarkGroup) string {
	b, _ := json.Marshal(groups)
	sum := sha256.Sum256(b)
	return fmt.Sprintf("sha256:%x", sum[:])
}
func stringPtr(v string) *string { return &v }
