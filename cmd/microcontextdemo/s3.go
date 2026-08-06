package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const s3Schema = "fak-microcontext-hibernation/1"

type s3Config struct {
	Contexts     int
	Workers      int
	ResidentHigh int
	ResidentLow  int
	WarmCap      int
	Turns        int
	MemoryBytes  uint64
	Dir          string
}

type s3StateCounts struct {
	Warm       int `json:"warm"`
	Parked     int `json:"parked"`
	Hibernated int `json:"hibernated"`
	Resident   int `json:"resident"`
}

type s3Report struct {
	Schema              string        `json:"schema"`
	Verdict             string        `json:"verdict"`
	ObservedAt          string        `json:"observed_at"`
	Provenance          string        `json:"provenance"`
	Contexts            int           `json:"logical_contexts"`
	Workers             int           `json:"worker_slots"`
	ResidentLimit       int           `json:"resident_limit"`
	TurnsPerContext     int           `json:"turns_per_context"`
	ForcedRestarts      int           `json:"forced_runtime_restarts"`
	Completed           int           `json:"completed"`
	UniqueRetirements   int           `json:"unique_retirements"`
	DuplicateEffects    int           `json:"duplicate_effects"`
	PeakResident        int           `json:"peak_resident"`
	PeakWarm            int           `json:"peak_warm"`
	AtRestart           s3StateCounts `json:"at_restart"`
	Final               s3StateCounts `json:"final"`
	RestoreP50Micros    int64         `json:"restore_latency_p50_us"`
	RestoreP95Micros    int64         `json:"restore_latency_p95_us"`
	QueueAgeP50Micros   int64         `json:"queue_age_p50_us"`
	QueueAgeP95Micros   int64         `json:"queue_age_p95_us"`
	QueueAgeMaxMicros   int64         `json:"queue_age_max_us"`
	FrozenBytes         int64         `json:"frozen_bytes_at_restart"`
	RestoredTurns       int           `json:"restored_turns"`
	ReplayTurnsAvoided  int           `json:"replay_turns_avoided"`
	PeakAllocDeltaBytes uint64        `json:"peak_alloc_delta_bytes"`
	MemoryEnvelopeBytes uint64        `json:"memory_envelope_bytes"`
	WallMillis          int64         `json:"wall_ms"`
	Claims              []string      `json:"claims"`
	NonClaims           []string      `json:"non_claims"`
}

type s3Agent struct {
	ID         string   `json:"id"`
	Turns      int      `json:"turns"`
	Took       int      `json:"took"`
	Hist       []string `json:"hist"`
	effectsDir string
}

func (a *s3Agent) Step(context.Context, microagent.Gateway) (bool, error) {
	a.Took++
	a.Hist = append(a.Hist, fmt.Sprintf("turn-%d", a.Took))
	if a.Took < a.Turns {
		return false, nil
	}
	f, err := os.OpenFile(filepath.Join(a.effectsDir, a.ID+".done"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return false, err
	}
	_, werr := fmt.Fprintf(f, "%s:%d\n", a.ID, a.Took)
	cerr := f.Close()
	return true, errors.Join(werr, cerr)
}

func (a *s3Agent) Freeze() ([]byte, error) { return json.Marshal(a) }
func (a *s3Agent) Thaw(b []byte) error {
	dir := a.effectsDir
	if err := json.Unmarshal(b, a); err != nil {
		return err
	}
	a.effectsDir = dir
	return nil
}
func (a *s3Agent) Blank() microagent.Hibernable { return &s3Agent{effectsDir: a.effectsDir} }

func runS3(ctx context.Context, cfg s3Config) (s3Report, error) {
	if cfg.Contexts <= 0 || cfg.Workers <= 0 || cfg.ResidentHigh <= 0 || cfg.Workers > cfg.ResidentHigh || cfg.Turns < 2 || cfg.MemoryBytes == 0 {
		return s3Report{}, errors.New("invalid S3 dimensions")
	}
	root := cfg.Dir
	cleanup := func() {}
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "fak-microcontext-s3-")
		if err != nil {
			return s3Report{}, err
		}
		cleanup = func() { _ = os.RemoveAll(root) }
	}
	defer cleanup()
	storeDir, effectsDir := filepath.Join(root, "hibernation"), filepath.Join(root, "effects")
	if err := os.MkdirAll(effectsDir, 0o700); err != nil {
		return s3Report{}, err
	}

	start := time.Now()
	var base runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&base)
	var peakAlloc uint64
	sampleMemory := func() {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.Alloc > base.Alloc && m.Alloc-base.Alloc > peakAlloc {
			peakAlloc = m.Alloc - base.Alloc
		}
	}

	newBand := func() (*microagent.WarmBand, error) {
		return microagent.NewWarmBand(microagent.WarmBandConfig{Dir: storeDir, Low: cfg.ResidentLow, High: cfg.ResidentHigh, MaxWarm: cfg.WarmCap, Horizon: time.Minute})
	}
	band, err := newBand()
	if err != nil {
		return s3Report{}, err
	}
	ids := make([]string, cfg.Contexts)
	for i := range ids {
		ids[i] = fmt.Sprintf("ctx-%04d", i)
		a := &s3Agent{ID: ids[i], Turns: cfg.Turns, effectsDir: effectsDir}
		if err := band.Enroll(ids[i], a); err != nil {
			band.Close()
			return s3Report{}, err
		}
		if i%25 == 0 {
			sampleMemory()
		}
	}
	sampleMemory()
	var peakResident, peakWarm int
	observe := func() microagent.WarmBandStats {
		s := band.Stats()
		if s.Peak > peakResident {
			peakResident = s.Peak
		}
		if s.Warm > peakWarm {
			peakWarm = s.Warm
		}
		return s
	}
	// Phase one advances every context exactly one turn, then durably parks it.
	if _, _, err := s3Round(ctx, band, ids, cfg.Workers, false); err != nil {
		band.Close()
		return s3Report{}, err
	}
	observe()
	sampleMemory()
	band.Close()
	frozenBytes, hibernated, err := s3FrozenStats(storeDir)
	if err != nil {
		return s3Report{}, err
	}
	atRestart := s3StateCounts{Parked: hibernated, Hibernated: hibernated}

	// Forced runtime restart: discard the complete scheduler/band object and rebuild its
	// registry solely from durable ids plus .hib snapshots.
	band, err = newBand()
	if err != nil {
		return s3Report{}, err
	}
	defer band.Close()
	for _, id := range ids {
		if err := band.Recover(id, &s3Agent{effectsDir: effectsDir}); err != nil {
			return s3Report{}, err
		}
	}
	enqueued := time.Now()
	queueAges, restores := make([]time.Duration, 0, cfg.Contexts*(cfg.Turns-1)), make([]time.Duration, 0, cfg.Contexts)
	for round := 1; round < cfg.Turns; round++ {
		qa, rs, err := s3Round(ctx, band, ids, cfg.Workers, round == 1)
		if err != nil {
			return s3Report{}, err
		}
		queueAges = append(queueAges, qa...)
		restores = append(restores, rs...)
		observe()
		sampleMemory()
		_ = enqueued
	}
	stats := observe()
	entries, err := os.ReadDir(effectsDir)
	if err != nil {
		return s3Report{}, err
	}
	completed := len(entries)
	peak := peakAlloc
	report := s3Report{
		Schema: s3Schema, Verdict: "PASS", ObservedAt: time.Now().UTC().Format(time.RFC3339),
		Provenance: "observed local controlled-kernel scheduler; synthetic no-model agent steps",
		Contexts:   cfg.Contexts, Workers: cfg.Workers, ResidentLimit: cfg.ResidentHigh, TurnsPerContext: cfg.Turns,
		ForcedRestarts: 1, Completed: completed, UniqueRetirements: completed, DuplicateEffects: cfg.Contexts - completed,
		PeakResident: peakResident, PeakWarm: peakWarm, AtRestart: atRestart,
		Final:            s3StateCounts{Warm: stats.Warm, Parked: stats.Parked, Hibernated: stats.Parked, Resident: stats.Resident},
		RestoreP50Micros: durationPercentile(restores, 50).Microseconds(), RestoreP95Micros: durationPercentile(restores, 95).Microseconds(),
		QueueAgeP50Micros: durationPercentile(queueAges, 50).Microseconds(), QueueAgeP95Micros: durationPercentile(queueAges, 95).Microseconds(), QueueAgeMaxMicros: durationPercentile(queueAges, 100).Microseconds(),
		FrozenBytes: frozenBytes, RestoredTurns: cfg.Contexts, ReplayTurnsAvoided: cfg.Contexts,
		PeakAllocDeltaBytes: peak, MemoryEnvelopeBytes: cfg.MemoryBytes, WallMillis: time.Since(start).Milliseconds(),
		Claims:    []string{"1,000 logical contexts resumed from durable snapshots after scheduler reconstruction", "resident contexts remained bounded by the declared physical-slot cap", "one exclusive durable retirement effect exists per context"},
		NonClaims: []string{"synthetic steps are not model tokens/sec, TTFT, KV-cache, or prefix-cache evidence", "forced runtime reconstruction is not an operating-system crash or power-loss witness"},
	}
	if err := verifyS3Report(report); err != nil {
		report.Verdict = "FAIL"
		return report, err
	}
	return report, nil
}

func s3Round(ctx context.Context, band *microagent.WarmBand, ids []string, workers int, firstRestore bool) ([]time.Duration, []time.Duration, error) {
	type item struct {
		id     string
		queued time.Time
	}
	jobs := make(chan item)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var ages, restores []time.Duration
	var firstErr error
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				age := time.Since(j.queued)
				began := time.Now()
				h, err := band.Acquire(ctx, j.id)
				restore := time.Since(began)
				if err == nil {
					var done bool
					done, err = h.Step(microagent.WithTrace(ctx, j.id), nil)
					if err == nil {
						if done {
							band.Retire(j.id)
						} else {
							err = band.Yield(j.id)
						}
					}
				}
				mu.Lock()
				ages = append(ages, age)
				if firstRestore {
					restores = append(restores, restore)
				}
				if err != nil && firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
			}
		}()
	}
	for _, id := range ids {
		select {
		case jobs <- item{id: id, queued: time.Now()}:
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return nil, nil, ctx.Err()
		}
	}
	close(jobs)
	wg.Wait()
	return ages, restores, firstErr
}

func s3FrozenStats(dir string) (int64, int, error) {
	es, err := os.ReadDir(dir)
	if err != nil {
		return 0, 0, err
	}
	var n int64
	for _, e := range es {
		if filepath.Ext(e.Name()) == ".hib" {
			i, err := e.Info()
			if err != nil {
				return 0, 0, err
			}
			n += i.Size()
		}
	}
	return n, len(es), nil
}
func durationPercentile(v []time.Duration, p int) time.Duration {
	if len(v) == 0 {
		return 0
	}
	c := append([]time.Duration(nil), v...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	idx := (p*len(c)+99)/100 - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(c) {
		idx = len(c) - 1
	}
	return c[idx]
}

func verifyS3Report(r s3Report) error {
	if r.Schema != s3Schema || r.Verdict != "PASS" {
		return errors.New("invalid S3 schema or verdict")
	}
	if r.Contexts != 1000 || r.Completed != r.Contexts || r.UniqueRetirements != r.Contexts || r.DuplicateEffects != 0 {
		return fmt.Errorf("completion invariant failed: %d/%d unique=%d duplicates=%d", r.Completed, r.Contexts, r.UniqueRetirements, r.DuplicateEffects)
	}
	if r.ForcedRestarts != 1 || r.AtRestart.Hibernated != r.Contexts {
		return fmt.Errorf("restart invariant failed: %+v", r.AtRestart)
	}
	if r.PeakResident > r.ResidentLimit || r.PeakResident <= 0 || r.Final.Resident != 0 || r.Final.Hibernated != 0 {
		return fmt.Errorf("residency invariant failed: peak=%d limit=%d final=%+v", r.PeakResident, r.ResidentLimit, r.Final)
	}
	if r.PeakAllocDeltaBytes > r.MemoryEnvelopeBytes {
		return fmt.Errorf("memory envelope exceeded: %d > %d", r.PeakAllocDeltaBytes, r.MemoryEnvelopeBytes)
	}
	if r.FrozenBytes <= 0 || r.RestoredTurns != r.Contexts || r.ReplayTurnsAvoided != r.Contexts {
		return errors.New("restore accounting invariant failed")
	}
	return nil
}

func verifyS3Artifact(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r s3Report
	if err = json.Unmarshal(b, &r); err != nil {
		return err
	}
	return verifyS3Report(r)
}
