package sessionmine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"
)

const RefreshBenchmarkSchema = "fak-sessionmine-refresh-benchmark/1"

type RefreshBenchmarkOptions struct {
	Sizes       []int
	Repetitions int
	WorkDir     string
}
type RefreshBenchmarkPhase struct {
	Name           string  `json:"name"`
	P50MS          float64 `json:"p50_ms"`
	P95MS          float64 `json:"p95_ms"`
	Reused         int     `json:"reused"`
	Rebuilt        int     `json:"rebuilt"`
	FreshnessLagMS float64 `json:"freshness_lag_ms,omitempty"`
}
type RefreshBenchmarkScale struct {
	Sessions        int                     `json:"sessions"`
	IndexBytes      int64                   `json:"index_bytes"`
	BytesPerSession float64                 `json:"bytes_per_session"`
	PeakHeapBytes   uint64                  `json:"peak_heap_bytes"`
	Phases          []RefreshBenchmarkPhase `json:"phases"`
	Pass            bool                    `json:"pass"`
	Failures        []string                `json:"failures"`
}
type RefreshBenchmarkReport struct {
	Schema      string                  `json:"schema"`
	OS          string                  `json:"os"`
	Arch        string                  `json:"arch"`
	GoVersion   string                  `json:"go_version"`
	Repetitions int                     `json:"repetitions"`
	SLO         RefreshBenchmarkSLO     `json:"slo"`
	Scales      []RefreshBenchmarkScale `json:"scales"`
	Pass        bool                    `json:"pass"`
}
type RefreshBenchmarkSLO struct {
	WarmP95MS       float64 `json:"warm_p95_ms"`
	ChangedP95MS    float64 `json:"changed_p95_ms"`
	WarmReuseRatio  float64 `json:"warm_reuse_ratio"`
	BytesPerSession float64 `json:"bytes_per_session"`
	FreshnessLagMS  float64 `json:"freshness_lag_ms"`
}

func DefaultRefreshBenchmarkSLO() RefreshBenchmarkSLO {
	return RefreshBenchmarkSLO{WarmP95MS: 5000, ChangedP95MS: 5000, WarmReuseRatio: .99, BytesPerSession: 4096, FreshnessLagMS: 60000}
}

func BenchmarkRefresh(ctx context.Context, opts RefreshBenchmarkOptions) (RefreshBenchmarkReport, error) {
	if len(opts.Sizes) == 0 {
		opts.Sizes = []int{1000, 10000, 100000}
	}
	if opts.Repetitions == 0 {
		opts.Repetitions = 3
	}
	if opts.Repetitions < 1 {
		return RefreshBenchmarkReport{}, fmt.Errorf("repetitions must be positive")
	}
	for _, n := range opts.Sizes {
		if n < 1 || n > 100000 {
			return RefreshBenchmarkReport{}, fmt.Errorf("sizes must be 1..100000")
		}
	}
	root := opts.WorkDir
	cleanup := func() {}
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "fak-sessionmine-bench-")
		if err != nil {
			return RefreshBenchmarkReport{}, err
		}
		cleanup = func() { _ = os.RemoveAll(root) }
	}
	defer cleanup()
	rep := RefreshBenchmarkReport{Schema: RefreshBenchmarkSchema, OS: runtime.GOOS, Arch: runtime.GOARCH, GoVersion: runtime.Version(), Repetitions: opts.Repetitions, SLO: DefaultRefreshBenchmarkSLO(), Pass: true, Scales: []RefreshBenchmarkScale{}}
	for _, n := range opts.Sizes {
		scale, err := benchmarkRefreshScale(ctx, root, n, opts.Repetitions, rep.SLO)
		if err != nil {
			return RefreshBenchmarkReport{}, err
		}
		rep.Scales = append(rep.Scales, scale)
		rep.Pass = rep.Pass && scale.Pass
	}
	return rep, nil
}
func benchmarkRefreshScale(ctx context.Context, base string, n, reps int, slo RefreshBenchmarkSLO) (RefreshBenchmarkScale, error) {
	dir := filepath.Join(base, fmt.Sprintf("n-%d", n))
	src := filepath.Join(dir, "codex")
	idx := filepath.Join(dir, "index.json")
	if err := os.MkdirAll(src, 0755); err != nil {
		return RefreshBenchmarkScale{}, err
	}
	body := []byte("{\"timestamp\":\"2026-08-19T00:00:00Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"shell_command\",\"arguments\":\"{}\"}}\n")
	for i := 0; i < n; i++ {
		if err := os.WriteFile(filepath.Join(src, fmt.Sprintf("s-%06d.jsonl", i)), body, 0644); err != nil {
			return RefreshBenchmarkScale{}, err
		}
	}
	opts := RefreshOptions{Mine: Options{CodexRoot: src, MinSupport: 1, Limit: 25}, IndexPath: idx}
	peak := uint64(0)
	sample := func() {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.HeapAlloc > peak {
			peak = m.HeapAlloc
		}
	}
	measure := func(name string, prepare func(int) time.Time) (RefreshBenchmarkPhase, error) {
		vals := make([]float64, 0, reps)
		phase := RefreshBenchmarkPhase{Name: name}
		for i := 0; i < reps; i++ {
			stamp := prepare(i)
			runtime.GC()
			sample()
			start := time.Now()
			select {
			case <-ctx.Done():
				return phase, ctx.Err()
			default:
			}
			result, err := MineIndexed(opts.Mine, opts.IndexPath)
			sample()
			if err != nil {
				return phase, err
			}
			vals = append(vals, float64(time.Since(start).Microseconds())/1000)
			phase.Reused = result.ReusedFiles
			phase.Rebuilt = result.ParsedFiles
			if !stamp.IsZero() {
				phase.FreshnessLagMS = float64(time.Since(stamp).Microseconds()) / 1000
			}
		}
		sort.Float64s(vals)
		phase.P50MS = nearestFloat(vals, .50)
		phase.P95MS = nearestFloat(vals, .95)
		return phase, nil
	}
	cold, err := measure("cold", func(int) time.Time { _ = os.Remove(idx); return time.Time{} })
	if err != nil {
		return RefreshBenchmarkScale{}, err
	}
	warm, err := measure("unchanged", func(int) time.Time { return time.Time{} })
	if err != nil {
		return RefreshBenchmarkScale{}, err
	}
	changed, err := measure("one_changed", func(i int) time.Time {
		stamp := time.Now().UTC()
		p := filepath.Join(src, "s-000000.jsonl")
		_ = os.WriteFile(p, append(body, byte('\n')), 0644)
		_ = os.Chtimes(p, stamp, stamp)
		return stamp
	})
	if err != nil {
		return RefreshBenchmarkScale{}, err
	}
	info, err := os.Stat(idx)
	if err != nil {
		return RefreshBenchmarkScale{}, err
	}
	scale := RefreshBenchmarkScale{Sessions: n, IndexBytes: info.Size(), BytesPerSession: float64(info.Size()) / float64(n), PeakHeapBytes: peak, Phases: []RefreshBenchmarkPhase{cold, warm, changed}, Pass: true, Failures: []string{}}
	ratio := float64(warm.Reused) / float64(n)
	if warm.P95MS > slo.WarmP95MS {
		scale.Failures = append(scale.Failures, "warm_p95")
	}
	if changed.P95MS > slo.ChangedP95MS {
		scale.Failures = append(scale.Failures, "changed_p95")
	}
	if ratio < slo.WarmReuseRatio {
		scale.Failures = append(scale.Failures, "warm_reuse")
	}
	if scale.BytesPerSession > slo.BytesPerSession {
		scale.Failures = append(scale.Failures, "bytes_per_session")
	}
	if changed.FreshnessLagMS > slo.FreshnessLagMS {
		scale.Failures = append(scale.Failures, "freshness_lag")
	}
	scale.Pass = len(scale.Failures) == 0
	return scale, nil
}
func nearestFloat(v []float64, q float64) float64 {
	if len(v) == 0 {
		return 0
	}
	i := int(float64(len(v))*q+.999999) - 1
	if i < 0 {
		i = 0
	}
	if i >= len(v) {
		i = len(v) - 1
	}
	return v[i]
}
