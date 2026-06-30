package main

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

const (
	fanbenchOverlapSchema = "fak.fanbench-overlap.v1"
	fanbenchOverlapTool   = "fanbench_overlap_infer"
	fanbenchOverlapEngine = "fanbench-overlap"
	fanbenchOverlapNote   = "Measures fak's actual host-driven Submit/Reap shape: Submit serially adjudicates, then W concurrent Reap calls run inline Engine.Complete after the kernel has released its pending-map mutex. This is NOT an MPI non-blocking progress thread, NOT MPI/HPC latency hiding, and carries no MPI/InfiniBand throughput or message-rate claim."
)

type fanbenchOverlapOptions struct {
	Width       int
	EngineDelay time.Duration
	Timeout     time.Duration
}

type fanbenchOverlapReport struct {
	Schema                string  `json:"schema"`
	Mode                  string  `json:"mode"`
	Width                 int     `json:"width"`
	EngineDelayNs         int64   `json:"engine_delay_ns"`
	SerialAdjudicateNs    int64   `json:"serial_adjudicate_ns"`
	SerialInferNs         int64   `json:"serial_infer_ns"`
	SerializedTotalNs     int64   `json:"serialized_total_ns"`
	MeasuredWallNs        int64   `json:"measured_wall_ns"`
	OverlapEfficiency     float64 `json:"overlap_efficiency"`
	ConcurrentEngineCalls int64   `json:"concurrent_engine_calls"`
	MaxConcurrentReap     int64   `json:"max_concurrent_reap"`
	HonestyCaveat         string  `json:"honesty_caveat"`
}

func writeOverlap(ctx context.Context, outPath string, opt fanbenchOverlapOptions) error {
	rep, err := runOverlap(ctx, opt)
	if err != nil {
		return err
	}
	return benchcli.WriteReport(outPath, rep)
}

func runOverlap(ctx context.Context, opt fanbenchOverlapOptions) (fanbenchOverlapReport, error) {
	opt = normalizeOverlapOptions(opt)

	adjNs, err := measureSerialAdjudication(ctx, opt)
	if err != nil {
		return fanbenchOverlapReport{}, err
	}
	inferNs, err := measureSerialInference(ctx, opt)
	if err != nil {
		return fanbenchOverlapReport{}, err
	}
	wallNs, calls, maxActive, err := measureConcurrentReap(ctx, opt)
	if err != nil {
		return fanbenchOverlapReport{}, err
	}

	total := adjNs + inferNs
	efficiency := 0.0
	if wallNs > 0 {
		efficiency = float64(total) / float64(wallNs)
	}
	return fanbenchOverlapReport{
		Schema:                fanbenchOverlapSchema,
		Mode:                  "concurrent-reap-overlap",
		Width:                 opt.Width,
		EngineDelayNs:         opt.EngineDelay.Nanoseconds(),
		SerialAdjudicateNs:    adjNs,
		SerialInferNs:         inferNs,
		SerializedTotalNs:     total,
		MeasuredWallNs:        wallNs,
		OverlapEfficiency:     efficiency,
		ConcurrentEngineCalls: calls,
		MaxConcurrentReap:     maxActive,
		HonestyCaveat:         fanbenchOverlapNote,
	}, nil
}

func normalizeOverlapOptions(opt fanbenchOverlapOptions) fanbenchOverlapOptions {
	if opt.Width < 1 {
		opt.Width = 8
	}
	if opt.EngineDelay <= 0 {
		opt.EngineDelay = 5 * time.Millisecond
	}
	if opt.Timeout <= 0 {
		opt.Timeout = 5*time.Second + 4*opt.EngineDelay*time.Duration(opt.Width)
	}
	return opt
}

func measureSerialAdjudication(ctx context.Context, opt fanbenchOverlapOptions) (int64, error) {
	ctx, cancel := context.WithTimeout(ctx, opt.Timeout)
	defer cancel()

	k := newOverlapKernel(fanbenchOverlapEngine + "-adjudicate")
	t0 := time.Now()
	for i := 0; i < opt.Width; i++ {
		if _, v := k.Submit(ctx, overlapCall(i)); v.Kind != abi.VerdictAllow {
			return 0, fmt.Errorf("overlap adjudicate submit %d: verdict=%v reason=%s", i, v.Kind, abi.ReasonName(v.Reason))
		}
	}
	return time.Since(t0).Nanoseconds(), nil
}

func measureSerialInference(ctx context.Context, opt fanbenchOverlapOptions) (int64, error) {
	ctx, cancel := context.WithTimeout(ctx, opt.Timeout)
	defer cancel()

	eng := newAtomicOverlapEngine(opt.EngineDelay, 0)
	abi.RegisterEngine(fanbenchOverlapEngine+"-serial", eng)
	k := newOverlapKernel(fanbenchOverlapEngine + "-serial")
	handles, err := submitOverlapCalls(ctx, k, opt.Width)
	if err != nil {
		return 0, err
	}

	t0 := time.Now()
	for i, h := range handles {
		if _, err := k.Reap(ctx, h); err != nil {
			return 0, fmt.Errorf("serial reap %d: %w", i, err)
		}
	}
	return time.Since(t0).Nanoseconds(), nil
}

func measureConcurrentReap(ctx context.Context, opt fanbenchOverlapOptions) (wallNs, calls, maxActive int64, err error) {
	ctx, cancel := context.WithTimeout(ctx, opt.Timeout)
	defer cancel()

	eng := newAtomicOverlapEngine(opt.EngineDelay, opt.Width)
	abi.RegisterEngine(fanbenchOverlapEngine+"-concurrent", eng)
	k := newOverlapKernel(fanbenchOverlapEngine + "-concurrent")

	t0 := time.Now()
	handles, err := submitOverlapCalls(ctx, k, opt.Width)
	if err != nil {
		return 0, 0, 0, err
	}

	var wg sync.WaitGroup
	errs := make(chan error, len(handles))
	for i, h := range handles {
		wg.Add(1)
		go func(i int, h abi.SubmissionHandle) {
			defer wg.Done()
			if _, err := k.Reap(ctx, h); err != nil {
				errs <- fmt.Errorf("concurrent reap %d: %w", i, err)
			}
		}(i, h)
	}
	wg.Wait()
	close(errs)
	if err, ok := <-errs; ok {
		return 0, 0, 0, err
	}
	return time.Since(t0).Nanoseconds(), eng.calls.Load(), eng.maxActive.Load(), nil
}

func submitOverlapCalls(ctx context.Context, k *kernel.Kernel, width int) ([]abi.SubmissionHandle, error) {
	handles := make([]abi.SubmissionHandle, 0, width)
	for i := 0; i < width; i++ {
		h, v := k.Submit(ctx, overlapCall(i))
		if v.Kind != abi.VerdictAllow {
			return nil, fmt.Errorf("overlap submit %d: verdict=%v reason=%s", i, v.Kind, abi.ReasonName(v.Reason))
		}
		handles = append(handles, h)
	}
	return handles, nil
}

func newOverlapKernel(engineID string) *kernel.Kernel {
	k := kernel.New(engineID, kernel.WithAdjudicators([]abi.Adjudicator{
		adjudicator.New(adjudicator.Policy{Allow: map[string]bool{fanbenchOverlapTool: true}}),
	}))
	k.SetVDSO(false)
	return k
}

func overlapCall(i int) *abi.ToolCall {
	body := []byte(`{"request":` + strconv.Itoa(i) + `}`)
	return &abi.ToolCall{
		Tool: fanbenchOverlapTool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
	}
}

type atomicOverlapEngine struct {
	delay   time.Duration
	barrier int64
	release chan struct{}
	once    sync.Once

	calls     atomic.Int64
	arrived   atomic.Int64
	active    atomic.Int64
	maxActive atomic.Int64
}

func newAtomicOverlapEngine(delay time.Duration, barrier int) *atomicOverlapEngine {
	return &atomicOverlapEngine{
		delay:   delay,
		barrier: int64(barrier),
		release: make(chan struct{}),
	}
}

func (e *atomicOverlapEngine) Caps() []abi.Capability { return nil }

func (e *atomicOverlapEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	active := e.active.Add(1)
	e.observeActive(active)
	defer e.active.Add(-1)

	if e.barrier > 0 {
		if e.arrived.Add(1) >= e.barrier {
			e.once.Do(func() { close(e.release) })
		}
		select {
		case <-e.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if e.delay > 0 {
		timer := time.NewTimer(e.delay)
		select {
		case <-timer.C:
		case <-ctx.Done():
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			return nil, ctx.Err()
		}
	}

	body := []byte(`{"ok":true,"engine":"fanbench-overlap","seq":` + strconv.FormatUint(c.SeqNo, 10) + `}`)
	return &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine": "fanbench-overlap",
		},
	}, nil
}

func (e *atomicOverlapEngine) observeActive(active int64) {
	for {
		old := e.maxActive.Load()
		if active <= old || e.maxActive.CompareAndSwap(old, active) {
			return
		}
	}
}
