package compute

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
//
// Closed in-memory oracle over the SM90 pull-style FP4 MegaMoE task-shape
// contract and K128 stage-ordering invariants adapted from
// flashinfer-ai/flashinfer PR #7961 @74c8d6bd0943165f8ec85c898908f3cebcef875a
// (Apache-2.0; skeleton in cuda/mega_moe_fp4.cuh). Software evidence only:
// these tests pin structure and ordering, never hardware or served-token
// speed.

// Contract constants mirrored from cuda/mega_moe_fp4.cuh.
const (
	megaMoeFP4M               = 256
	megaMoeFP4TokenTileNarrow = 72
	megaMoeFP4TokenTileWide   = 80
	megaMoeFP4MaxRowsPerTask  = 80
	megaMoeFP4KStage          = 128
	megaMoeFP4PipelineDepth   = 2
	megaMoeFP4MaxInFlightRows = 4
	megaMoeFP4MaxTabledTokens = 4096
)

// Sentinel errors: the closed contract the oracle asserts against.
var (
	errMegaMoeFP4RowsOutOfRange  = errors.New("megamoe-fp4: rows outside the single-task range")
	errMegaMoeFP4StageOrder      = errors.New("megamoe-fp4: stage order breaks the K128 pipeline dependency")
	errMegaMoeFP4PullBufferFull  = errors.New("megamoe-fp4: in-flight rows at the contracted cap")
	errMegaMoeFP4Undersized      = errors.New("megamoe-fp4: pull buffer below the in-flight cap")
	errMegaMoeFP4UnknownEnvelope = errors.New("megamoe-fp4: envelope absent from the tactic table")
	errMegaMoeFP4UnknownBucket   = errors.New("megamoe-fp4: token count outside the contracted buckets")
)

// selectTokenTileWidth picks the swap-AB token-tile width for an expert's
// row count: N72 up to 72 rows, N80 above, so one task holds the expert's
// 65-80 rows instead of a near-empty second N64 tile.
func selectTokenTileWidth(rows int) (int, error) {
	if rows < 1 || rows > megaMoeFP4MaxRowsPerTask {
		return 0, fmt.Errorf("%w: rows=%d", errMegaMoeFP4RowsOutOfRange, rows)
	}
	if rows <= megaMoeFP4TokenTileNarrow {
		return megaMoeFP4TokenTileNarrow, nil
	}
	return megaMoeFP4TokenTileWide, nil
}

// scalePoolRows sizes the activation-scale pool to the exact row count. The
// rejected N64 alternative padded the pool to a tile multiple; the contract
// is that it is never padded.
func scalePoolRows(rows int) int { return rows }

type megaMoeFP4Event struct{ start, end int }

// scheduleMegaMoeFP4Stages replays S K128 stages under the contracted
// dependencies: load(s) issues once the WGMMA that last consumed its double
// buffer finished (stage s-2 at depth 2), and wgmma(s) starts no earlier
// than load(s)'s end or the previous stage's wgmma end.
func scheduleMegaMoeFP4Stages(loadDur, wgmmaDur []int) ([]megaMoeFP4Event, []megaMoeFP4Event, error) {
	if len(loadDur) != len(wgmmaDur) {
		return nil, nil, fmt.Errorf("%w: mismatched stage counts", errMegaMoeFP4StageOrder)
	}
	for _, d := range append(append([]int{}, loadDur...), wgmmaDur...) {
		if d <= 0 {
			return nil, nil, fmt.Errorf("%w: non-positive stage duration", errMegaMoeFP4StageOrder)
		}
	}
	n := len(loadDur)
	loads := make([]megaMoeFP4Event, n)
	wgmmas := make([]megaMoeFP4Event, n)
	wEnds := [megaMoeFP4PipelineDepth]int{}
	for s := 0; s < n; s++ {
		lStart := 0
		if s >= megaMoeFP4PipelineDepth {
			lStart = wEnds[s%megaMoeFP4PipelineDepth]
		}
		loads[s] = megaMoeFP4Event{lStart, lStart + loadDur[s]}
		wStart := loads[s].end
		if s > 0 && wgmmas[s-1].end > wStart {
			wStart = wgmmas[s-1].end
		}
		wgmmas[s] = megaMoeFP4Event{wStart, wStart + wgmmaDur[s]}
		wEnds[s%megaMoeFP4PipelineDepth] = wgmmas[s].end
	}
	return loads, wgmmas, nil
}

// validateMegaMoeFP4Trace checks a proposed stage trace against the
// contracted dependencies and the double-buffer exclusion rule.
func validateMegaMoeFP4Trace(loads, wgmmas []megaMoeFP4Event) error {
	if len(loads) != len(wgmmas) {
		return fmt.Errorf("%w: mismatched stage counts", errMegaMoeFP4StageOrder)
	}
	for s := range loads {
		if wgmmas[s].start < loads[s].end {
			return fmt.Errorf("%w: stage %d wgmma starts before its load finishes", errMegaMoeFP4StageOrder, s)
		}
		if s >= megaMoeFP4PipelineDepth && loads[s].start < wgmmas[s-megaMoeFP4PipelineDepth].end {
			return fmt.Errorf("%w: stage %d load reuses a busy double buffer", errMegaMoeFP4StageOrder, s)
		}
	}
	for i := range loads {
		for j := i + 1; j < len(loads); j++ {
			if i%megaMoeFP4PipelineDepth == j%megaMoeFP4PipelineDepth &&
				loads[i].start < loads[j].end && loads[j].start < loads[i].end {
				return fmt.Errorf("%w: loads %d and %d overlap on buffer %d", errMegaMoeFP4StageOrder, i, j, i%megaMoeFP4PipelineDepth)
			}
		}
	}
	return nil
}

// megaMoeFP4Dispatcher models the dispatch warps' pull-buffer admission: at
// most megaMoeFP4MaxInFlightRows rows in flight, each admitted row taking an
// idle slot under the per-pass row rotation.
type megaMoeFP4Dispatcher struct {
	slots    int
	pass     int
	inFlight map[int]int
}

func newMegaMoeFP4Dispatcher(slots int) (*megaMoeFP4Dispatcher, error) {
	if slots < megaMoeFP4MaxInFlightRows {
		return nil, fmt.Errorf("%w: slots=%d cap=%d", errMegaMoeFP4Undersized, slots, megaMoeFP4MaxInFlightRows)
	}
	return &megaMoeFP4Dispatcher{slots: slots, inFlight: map[int]int{}}, nil
}

// admit places row into an idle pull-buffer slot; the per-pass row rotation
// advances the base offset by one slot each pass.
func (d *megaMoeFP4Dispatcher) admit(row int) (int, error) {
	if len(d.inFlight) >= megaMoeFP4MaxInFlightRows {
		return 0, fmt.Errorf("%w: row=%d", errMegaMoeFP4PullBufferFull, row)
	}
	if _, busy := d.inFlight[row]; busy {
		return 0, fmt.Errorf("%w: row=%d already in flight", errMegaMoeFP4PullBufferFull, row)
	}
	slot := (row + d.pass) % d.slots
	d.inFlight[row] = slot
	return slot, nil
}

func (d *megaMoeFP4Dispatcher) release(row int) { delete(d.inFlight, row) }

func (d *megaMoeFP4Dispatcher) nextPass() { d.pass++ }

type megaMoeFP4Envelope struct{ hidden, intermediate, experts, expertParallel int }

var megaMoeFP4ReferenceEnvelopes = []megaMoeFP4Envelope{
	{hidden: 7168, intermediate: 3072, experts: 384, expertParallel: 4},
	{hidden: 2560, intermediate: 640, experts: 512, expertParallel: 2},
}

// megaMoeFP4UpstreamReportedBand is the H200 wall-clock band upstream
// reported for untuned launches on the two reference envelopes. Provenance
// data only: the skeleton records it, never claims it as fak performance.
var megaMoeFP4UpstreamReportedBand = struct{ lo, hi float64 }{lo: 0.84, hi: 0.95}

// tokenBucket maps a launch token count onto the contracted untuned-launch
// buckets [1,16), [16,64), [64,256), [256,4096]; counts outside fail closed.
func tokenBucket(tokens int) (int, error) {
	switch {
	case tokens < 1 || tokens > megaMoeFP4MaxTabledTokens:
		return 0, fmt.Errorf("%w: tokens=%d", errMegaMoeFP4UnknownBucket, tokens)
	case tokens < 16:
		return 0, nil
	case tokens < 64:
		return 1, nil
	case tokens < 256:
		return 2, nil
	default:
		return 3, nil
	}
}

// tacticFor resolves the untuned-launch tactic for an envelope and token
// count. IDs are deterministic table indices, not tuned winners.
func tacticFor(env megaMoeFP4Envelope, tokens int) (uint16, error) {
	envIdx := -1
	for i, e := range megaMoeFP4ReferenceEnvelopes {
		if e == env {
			envIdx = i
			break
		}
	}
	if envIdx < 0 {
		return 0, fmt.Errorf("%w: %+v", errMegaMoeFP4UnknownEnvelope, env)
	}
	bucket, err := tokenBucket(tokens)
	if err != nil {
		return 0, err
	}
	return uint16(envIdx*4 + bucket), nil
}

func TestMegaMoeFP4TaskShapeContract(t *testing.T) {
	for rows, want := range map[int]int{1: 72, 65: 72, 72: 72, 73: 80, 80: 80} {
		got, err := selectTokenTileWidth(rows)
		if err != nil {
			t.Fatalf("selectTokenTileWidth(%d): %v", rows, err)
		}
		if got != want {
			t.Errorf("selectTokenTileWidth(%d) = %d, want %d", rows, got, want)
		}
	}
	for _, bad := range []int{0, 81, 256} {
		if _, err := selectTokenTileWidth(bad); !errors.Is(err, errMegaMoeFP4RowsOutOfRange) {
			t.Errorf("rows=%d: err=%v, want errors.Is %v", bad, err, errMegaMoeFP4RowsOutOfRange)
		}
	}
	// Scale pool is exact-row, never padded to a tile multiple.
	for _, rows := range []int{65, 73, 80} {
		if got := scalePoolRows(rows); got != rows {
			t.Errorf("scalePoolRows(%d) = %d, want %d (no tile-multiple padding)", rows, got, rows)
		}
	}
	if padded := (65 + 63) / 64 * 64; scalePoolRows(65) == padded {
		t.Error("scale pool fell back to the rejected N64 tile-multiple padding")
	}
}

func TestMegaMoeFP4StageOrderOracle(t *testing.T) {
	// Balanced K128 pipeline: loads fully hide behind WGMMA tiles.
	loads, wgmmas, err := scheduleMegaMoeFP4Stages([]int{10, 10, 10, 10}, []int{10, 10, 10, 10})
	if err != nil {
		t.Fatalf("schedule balanced: %v", err)
	}
	wantLoads := []megaMoeFP4Event{{0, 10}, {0, 10}, {20, 30}, {30, 40}}
	wantWgmmas := []megaMoeFP4Event{{10, 20}, {20, 30}, {30, 40}, {40, 50}}
	if !eventsEqual(loads, wantLoads) || !eventsEqual(wgmmas, wantWgmmas) {
		t.Errorf("balanced trace = (%v, %v), want (%v, %v)", loads, wgmmas, wantLoads, wantWgmmas)
	}
	if got := wgmmas[len(wgmmas)-1].end; got != 50 {
		t.Errorf("balanced makespan = %d, want 50", got)
	}
	if err := validateMegaMoeFP4Trace(loads, wgmmas); err != nil {
		t.Errorf("balanced trace fails validation: %v", err)
	}

	// Slow loads: double-buffer reuse serializes the pipeline.
	loads2, wgmmas2, err := scheduleMegaMoeFP4Stages([]int{20, 20, 20, 20}, []int{10, 10, 10, 10})
	if err != nil {
		t.Fatalf("schedule slow loads: %v", err)
	}
	wantLoads2 := []megaMoeFP4Event{{0, 20}, {0, 20}, {30, 50}, {40, 60}}
	wantWgmmas2 := []megaMoeFP4Event{{20, 30}, {30, 40}, {50, 60}, {60, 70}}
	if !eventsEqual(loads2, wantLoads2) || !eventsEqual(wgmmas2, wantWgmmas2) {
		t.Errorf("slow-load trace = (%v, %v), want (%v, %v)", loads2, wgmmas2, wantLoads2, wantWgmmas2)
	}
	if got := wgmmas2[len(wgmmas2)-1].end; got != 70 {
		t.Errorf("slow-load makespan = %d, want 70", got)
	}

	// Tampered trace: a WGMMA issued before its own load finishes.
	bad := append([]megaMoeFP4Event{}, wgmmas...)
	bad[0].start = loads[0].end - 1
	if err := validateMegaMoeFP4Trace(loads, bad); !errors.Is(err, errMegaMoeFP4StageOrder) {
		t.Errorf("tampered trace: err=%v, want errors.Is %v", err, errMegaMoeFP4StageOrder)
	}

	// Infeasible durations are refused up front.
	if _, _, err := scheduleMegaMoeFP4Stages([]int{0}, []int{10}); !errors.Is(err, errMegaMoeFP4StageOrder) {
		t.Errorf("zero duration: err=%v, want errors.Is %v", err, errMegaMoeFP4StageOrder)
	}
}

func eventsEqual(got, want []megaMoeFP4Event) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestMegaMoeFP4DispatchInFlightRotation(t *testing.T) {
	if _, err := newMegaMoeFP4Dispatcher(3); !errors.Is(err, errMegaMoeFP4Undersized) {
		t.Errorf("slots=3: err=%v, want errors.Is %v", err, errMegaMoeFP4Undersized)
	}
	d, err := newMegaMoeFP4Dispatcher(8)
	if err != nil {
		t.Fatalf("new dispatcher: %v", err)
	}
	for row := 0; row < megaMoeFP4MaxInFlightRows; row++ {
		if _, err := d.admit(row); err != nil {
			t.Fatalf("admit row %d: %v", row, err)
		}
	}
	if _, err := d.admit(megaMoeFP4MaxInFlightRows); !errors.Is(err, errMegaMoeFP4PullBufferFull) {
		t.Errorf("fifth row: err=%v, want errors.Is %v", err, errMegaMoeFP4PullBufferFull)
	}
	d.release(0)
	if _, err := d.admit(megaMoeFP4MaxInFlightRows); err != nil {
		t.Errorf("re-admit after release: %v", err)
	}

	// Per-pass row rotation: the same row lands on a different slot each pass.
	r, err := newMegaMoeFP4Dispatcher(8)
	if err != nil {
		t.Fatalf("rotation dispatcher: %v", err)
	}
	prev, err := r.admit(3)
	if err != nil {
		t.Fatalf("admit rotated row: %v", err)
	}
	for pass := 1; pass <= 8; pass++ {
		r.release(3)
		r.nextPass()
		cur, err := r.admit(3)
		if err != nil {
			t.Fatalf("pass %d admit: %v", pass, err)
		}
		if cur == prev {
			t.Fatalf("pass %d: rotation did not advance the slot (slot %d)", pass, cur)
		}
		prev = cur
	}

	// Within one pass, four contiguous rows occupy four distinct slots.
	f, err := newMegaMoeFP4Dispatcher(8)
	if err != nil {
		t.Fatalf("distinct-slot dispatcher: %v", err)
	}
	seen := map[int]bool{}
	for row := 0; row < megaMoeFP4MaxInFlightRows; row++ {
		slot, err := f.admit(row)
		if err != nil {
			t.Fatalf("admit row %d: %v", row, err)
		}
		if seen[slot] {
			t.Fatalf("rows collide on pull-buffer slot %d", slot)
		}
		seen[slot] = true
	}
}

func TestMegaMoeFP4TokenBucketTactics(t *testing.T) {
	// Closed coverage: every reference envelope x bucket resolves to exactly
	// one deterministic tactic, and the eight pairs are pairwise distinct.
	ids := map[uint16]int{}
	for _, env := range megaMoeFP4ReferenceEnvelopes {
		for _, tokens := range []int{8, 32, 128, 512} {
			id, err := tacticFor(env, tokens)
			if err != nil {
				t.Fatalf("tacticFor(%+v, %d): %v", env, tokens, err)
			}
			ids[id]++
		}
	}
	if len(ids) != 8 {
		t.Fatalf("tactic table covers %d distinct ids, want 8", len(ids))
	}
	a, _ := tacticFor(megaMoeFP4ReferenceEnvelopes[0], 8)
	b, _ := tacticFor(megaMoeFP4ReferenceEnvelopes[0], 8)
	if a != b {
		t.Errorf("tactic lookup not deterministic: %d vs %d", a, b)
	}
	if _, err := tacticFor(megaMoeFP4Envelope{hidden: 1, intermediate: 1, experts: 1, expertParallel: 1}, 8); !errors.Is(err, errMegaMoeFP4UnknownEnvelope) {
		t.Errorf("unknown envelope: err=%v, want errors.Is %v", err, errMegaMoeFP4UnknownEnvelope)
	}
	for _, tokens := range []int{0, 4097} {
		if _, err := tacticFor(megaMoeFP4ReferenceEnvelopes[0], tokens); !errors.Is(err, errMegaMoeFP4UnknownBucket) {
			t.Errorf("tokens=%d: err=%v, want errors.Is %v", tokens, err, errMegaMoeFP4UnknownBucket)
		}
	}
	for tokens, want := range map[int]int{1: 0, 15: 0, 16: 1, 63: 1, 64: 2, 255: 2, 256: 3, 4096: 3} {
		got, err := tokenBucket(tokens)
		if err != nil {
			t.Fatalf("tokenBucket(%d): %v", tokens, err)
		}
		if got != want {
			t.Errorf("tokenBucket(%d) = %d, want %d", tokens, got, want)
		}
	}
	// The upstream-reported H200 band is recorded as provenance data.
	if megaMoeFP4UpstreamReportedBand.lo != 0.84 || megaMoeFP4UpstreamReportedBand.hi != 0.95 {
		t.Errorf("upstream-reported band = %+v, want lo=0.84 hi=0.95", megaMoeFP4UpstreamReportedBand)
	}
}

func TestMegaMoeFP4SourceContract(t *testing.T) {
	cuh, err := os.ReadFile(filepath.Join("cuda", "mega_moe_fp4.cuh"))
	if err != nil {
		t.Fatalf("read cuda/mega_moe_fp4.cuh: %v", err)
	}
	text := string(cuh)
	for _, want := range []string{
		"kMegaMoeFP4M = 256",
		"kMegaMoeFP4TokenTileN72 = 72",
		"kMegaMoeFP4TokenTileN80 = 80",
		"kMegaMoeFP4MaxRowsPerTask = 80",
		"kMegaMoeFP4KStage = 128",
		"kMegaMoeFP4PipelineDepth = 2",
		"kMegaMoeFP4MaxInFlightRows = 4",
		"74c8d6bd0943165f8ec85c898908f3cebcef875a",
		"Apache-2.0",
		"static_assert(mega_moe_fp4_pipeline_makespan(kMegaMoeFP4OracleLoadsEven, kMegaMoeFP4OracleWgmmaEven, 4) == 50",
		"static_assert(mega_moe_fp4_pipeline_makespan(kMegaMoeFP4OracleLoadsSlow, kMegaMoeFP4OracleWgmmaFast, 4) == 70",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("cuda/mega_moe_fp4.cuh missing contracted text %q", want)
		}
	}
	kernels, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatalf("read cuda_kernels.cu: %v", err)
	}
	if !strings.Contains(string(kernels), `#include "cuda/mega_moe_fp4.cuh"`) {
		t.Error("cuda_kernels.cu does not include cuda/mega_moe_fp4.cuh")
	}
	// The fragment must be reachable through the translation unit's quoted
	// include chain, not just named.
	closure := cudaSourceClosure(t, "cuda_kernels.cu")
	if !strings.Contains(closure, "kMegaMoeFP4MaxRowsPerTask = 80") {
		t.Error("cuda/mega_moe_fp4.cuh is not part of the cuda_kernels.cu include closure")
	}
}
