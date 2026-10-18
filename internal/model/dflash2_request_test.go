package model

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func dflash2RequestFixture(t *testing.T, fullAccept bool) (*Qwen38DFlash2Request, *Session) {
	t.Helper()
	m := qwen35TargetFeatureTestModel(t)
	if fullAccept {
		m.Cfg.TieWordEmbeddings = false
		nbytes, offset := m.Cfg.HiddenSize*m.Cfg.VocabSize*4, len(m.raw)
		m.raw = append(m.raw, make([]byte, nbytes)...)
		m.manifest["lm_head.weight"] = tensorMeta{Dtype: "F32", Shape: []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}, Offset: offset, Nbytes: nbytes}
	}
	target, plain := m.NewSession(), m.NewSession()
	t.Cleanup(target.Close)
	t.Cleanup(plain.Close)
	target.captureTargetHidden, plain.captureTargetHidden = true, true
	g := dflash2TinyGeometry()
	w := dflash2TinyWeightsWithValue(t, g, func(name string, j int) float32 {
		if fullAccept && strings.HasPrefix(name, "candidate_selector.") {
			return 0
		}
		return dflash2OracleValue(g, name, j)
	})
	f, err := newDFlash2Reference(w, target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	q, err := newDFlash2Request(f)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := q.Close(); err != nil {
			t.Error(err)
		}
	})
	return q, plain
}

func dflash2RequestPlain(prompt []int, plain *Session) (logits []float32) {
	for _, token := range prompt {
		logits = append([]float32(nil), plain.Step(token)...)
	}
	return logits
}

func dflash2RequestEqual(t *testing.T, q *Qwen38DFlash2Request, plain *Session, logits []float32) {
	t.Helper()
	assertFloat32BitsEqual(t, "owned request boundary", logits, q.logits)
	normalizeSnapshotForTest(t, q.ref.target)
	normalizeSnapshotForTest(t, plain)
	assertQwen35MTPTargetStateEqual(t, q.ref.target, plain)
	if err := q.ref.proposalReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	if q.ref.target.qwen35TargetFeatureTxn != nil || q.ref.target.qwen35TargetLayerCapture != nil {
		t.Fatal("request leaked a target transaction or capture")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestBootstrapGenuineChunks(t *testing.T) {
	for _, size := range []int{1, 8, 9, 23} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			q, plain := dflash2RequestFixture(t, false)
			if _, err := q.Prepare(context.Background(), 1); err == nil {
				t.Fatal("unseeded request was ready")
			}
			prompt := make([]int, size)
			for i := range prompt {
				prompt[i] = 3 + i%7
			}
			if err := q.Bootstrap(context.Background(), prompt); err != nil {
				t.Fatal(err)
			}
			want := dflash2RequestPlain(prompt, plain)
			if reflect.DeepEqual(want, make([]float32, len(want))) {
				t.Fatal("bootstrap oracle must have genuine nonzero logits")
			}
			dflash2RequestEqual(t, q, plain, want)
			if q.ref.nextPosition != size || len(q.ref.cache[4].keys) != min(size, q.ref.g.window-1)*q.ref.g.kvHeads*q.ref.g.headDim {
				t.Fatal("bootstrap did not retain the complete bounded tail")
			}
			if err := q.Bootstrap(context.Background(), prompt); err == nil {
				t.Fatal("duplicate bootstrap accepted")
			}
			for i := 0; i < 3; i++ {
				round, err := q.Prepare(context.Background(), 8)
				if err != nil {
					t.Fatal(err)
				}
				tokens, err := round.Commit(context.Background(), len(round.Tokens()))
				if err != nil || len(tokens) == 0 {
					t.Fatalf("commit %v: %v", tokens, err)
				}
				for _, token := range tokens {
					if token != argmaxF32(want) {
						t.Fatal("generated token differs from ordinary target")
					}
					want = append([]float32(nil), plain.Step(token)...)
				}
				dflash2RequestEqual(t, q, plain, want)
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestStopPrefixEveryCount(t *testing.T) {
	for accepted := 0; accepted <= 8; accepted++ {
		t.Run(fmt.Sprint(accepted), func(t *testing.T) {
			q, plain := dflash2RequestFixture(t, true)
			prompt := []int{3, 7, 11}
			if err := q.Bootstrap(context.Background(), prompt); err != nil {
				t.Fatal(err)
			}
			want := dflash2RequestPlain(prompt, plain)
			round, err := q.Prepare(context.Background(), 8)
			if err != nil || len(round.Tokens()) != 8 {
				t.Fatalf("must genuinely verify all eight: %v", err)
			}
			if _, err := q.Prepare(context.Background(), 1); err == nil {
				t.Fatal("overlapping Prepare accepted")
			}
			prepared := round.Tokens()
			prepared[0] = 96
			if round.Tokens()[0] != 0 {
				t.Fatal("prepared IDs alias internal acceptance state")
			}
			// The stop policy picks a prefix before any publication. Zero models
			// an EOS/budget stop before the anchor; no rejected suffix is emitted.
			var published []int
			tokens, err := round.Commit(context.Background(), accepted)
			if err != nil || len(tokens) != accepted || len(published) != 0 {
				t.Fatalf("selected prefix %d: %v", accepted, err)
			}
			for _, token := range tokens {
				if token != argmaxF32(want) {
					t.Fatal("stop prefix contains an unverified token")
				}
				want = append([]float32(nil), plain.Step(token)...)
			}
			dflash2RequestEqual(t, q, plain, want)
			published = append(published, tokens...)
			if len(published) != accepted || q.ref.nextPosition != len(prompt)+accepted {
				t.Fatal("publication and committed prefix disagree")
			}
			if _, err := round.Commit(context.Background(), 0); err == nil || round.Tokens() != nil {
				t.Fatal("stale round can publish again")
			}
			if err := round.Abort(); err != nil {
				t.Fatal(err)
			}
			next, err := q.Prepare(context.Background(), 1)
			if err != nil {
				t.Fatal(err)
			}
			if err := next.Abort(); err != nil {
				t.Fatal(err)
			}
			dflash2RequestEqual(t, q, plain, want)
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestBootstrapRollback(t *testing.T) {
	for _, fault := range []string{"read", "cancel", "panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			q, plain := dflash2RequestFixture(t, false)
			reader := q.ref.w.r
			marker := errors.New("bootstrap fifth-layer fault")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			trigger := q.ref.w.tensors["layers.4.self_attn.v_proj.weight"].offset
			hit := false
			q.ref.w.r = dflash2ReadObserver{reader, func(offset int64) error {
				if offset != trigger || hit {
					return nil
				}
				hit = true
				if q.ref.target.Cache.Len() != 8 {
					panic("fault did not follow eight genuine target steps")
				}
				switch fault {
				case "cancel":
					cancel()
					return ctx.Err()
				case "panic":
					panic(marker)
				case "nil-panic":
					panic(nil)
				case "goexit":
					runtime.Goexit()
				}
				return marker
			}}
			var err error
			returned := false
			done := make(chan struct{})
			go func() {
				defer close(done)
				err = q.Bootstrap(ctx, []int{1, 2, 3, 4, 5, 6, 7, 8, 9})
				returned = true
			}()
			<-done
			if !hit || (fault == "goexit" && returned) || (fault != "goexit" && (!returned || err == nil)) {
				t.Fatalf("fault not witnessed: hit=%v returned=%v err=%v", hit, returned, err)
			}
			if fault == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation identity lost: %v", err)
			}
			if q.closed || q.bootstrapped || q.logits != nil || q.ref.nextPosition != 0 || len(q.ref.cache[4].keys) != 0 {
				t.Fatal("failed bootstrap published or lost a restorable owner")
			}
			normalizeSnapshotForTest(t, q.ref.target)
			normalizeSnapshotForTest(t, plain)
			assertQwen35MTPTargetStateEqual(t, q.ref.target, plain)
			q.ref.w.r = reader
			prompt := []int{3, 7, 11}
			if err := q.Bootstrap(context.Background(), prompt); err != nil {
				t.Fatal(err)
			}
			dflash2RequestEqual(t, q, plain, dflash2RequestPlain(prompt, plain))
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestLateCommitRollback(t *testing.T) {
	for _, fault := range []string{"read", "cancel", "invalid-count", "restore-failure"} {
		t.Run(fault, func(t *testing.T) {
			q, plain := dflash2RequestFixture(t, true)
			prompt := []int{3, 7, 11}
			if err := q.Bootstrap(context.Background(), prompt); err != nil {
				t.Fatal(err)
			}
			want := dflash2RequestPlain(prompt, plain)
			beforeCache := q.ref.cache
			f, target := q.ref, q.ref.target
			round, err := q.Prepare(context.Background(), 8)
			if err != nil || len(round.Tokens()) != 8 {
				t.Fatalf("full block: %v", err)
			}
			reader := f.w.r
			marker := errors.New("late draft append failure")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			hit := false
			if fault != "invalid-count" {
				f.w.r = dflash2ReadObserver{reader, func(int64) error {
					hit = true
					if target.Cache.Len() != len(prompt)+3 {
						panic("failure preceded target prefix commit")
					}
					if fault == "cancel" {
						cancel()
						return ctx.Err()
					}
					if fault == "restore-failure" {
						target.cacheGeometryEpoch++ // make the complete snapshot explicitly stale
					}
					return marker
				}}
			}
			count := 3
			if fault == "invalid-count" {
				count = 9
			}
			tokens, err := round.Commit(ctx, count)
			if err == nil || tokens != nil || (fault != "invalid-count" && !hit) || !round.done || q.pending != nil {
				t.Fatalf("late failure result: tokens=%v err=%v hit=%v", tokens, err, hit)
			}
			if fault == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation identity lost: %v", err)
			}
			if fault == "restore-failure" {
				if !q.closed || q.ref != nil || !f.closed || target.modelWeightsHeld {
					t.Fatal("failed restore left a reusable target/draft owner")
				}
				if _, err := q.Prepare(context.Background(), 1); err == nil {
					t.Fatal("retired request resumed")
				}
				return
			}
			f.w.r = reader
			if !reflect.DeepEqual(f.cache, beforeCache) {
				t.Fatal("late commit fault changed draft K/V")
			}
			dflash2RequestEqual(t, q, plain, want)
			next, err := q.Prepare(context.Background(), 1)
			if err != nil {
				t.Fatal(err)
			}
			if err := next.Abort(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestHALBootstrapAndStops(t *testing.T) {
	m := dflash2HALTestModel(t)
	target, backend := dflash2HALTestSession(t, m)
	plain, plainBackend := dflash2HALTestSession(t, m)
	g := dflash2TinyGeometry()
	f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	q, err := newDFlash2Request(f)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := q.Close(); err != nil {
			t.Error(err)
		}
	})
	backend.poison = true // bootstrap must copy the full logits before recycle
	prompt := []int{3, 7, 11, 13, 17, 19, 23, 29, 31}
	if err := q.Bootstrap(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	want := dflash2RequestPlain(prompt, plain)
	assertFloat32BitsEqual(t, "HAL genuine bootstrap", want, q.logits)
	if backend.tapReads != len(prompt)*len(g.taps) || backend.headMuls < len(prompt) || backend.q8Muls == 0 {
		t.Fatal("bootstrap did not execute native packed target/tap operations")
	}
	// Build the expected draft tail from independent post-FFN observations in
	// ordinary target execution, rather than reusing the capture implementation.
	oracle, err := newDFlash2Reference(dflash2TinyWeights(t, g), plain, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := oracle.Close(); err != nil {
			t.Error(err)
		}
	})
	for start := 0; start < len(prompt); start += 8 {
		var rows []qwen35TargetFeatureRow
		for i := start; i < min(start+8, len(prompt)); i++ {
			row := qwen35TargetFeatureRow{Position: i, Token: prompt[i], Layers: append([]int(nil), g.taps...)}
			for _, layer := range g.taps {
				row.Hidden = append(row.Hidden, append([]float32(nil), plainBackend.observed[i][layer]...))
			}
			rows = append(rows, row)
		}
		if err := oracle.AppendCommitted(context.Background(), rows); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(q.ref.cache, oracle.cache) {
		t.Fatal("bootstrap draft tail differs from independently observed target features")
	}
	for _, count := range []int{0, 1, 0, 1} {
		before := dflash2HALState(t, target)
		round, err := q.Prepare(context.Background(), 8)
		if err != nil {
			t.Fatal(err)
		}
		tokens, err := round.Commit(context.Background(), count)
		if err != nil || len(tokens) != count {
			t.Fatalf("HAL stop prefix: %v", err)
		}
		if count == 0 && !reflect.DeepEqual(before, dflash2HALState(t, target)) {
			t.Fatal("stop-before-anchor changed complete HAL state")
		}
		for _, token := range tokens {
			if token != argmaxF32(want) {
				t.Fatal("HAL prefix is not exact greedy")
			}
			want = append([]float32(nil), plain.Step(token)...)
		}
		assertFloat32BitsEqual(t, "HAL committed boundary", want, q.logits)
		gotState, wantState := dflash2HALState(t, target), dflash2HALState(t, plain)
		// The existing transaction counts committed HAL decode steps; ordinary
		// Step does not use those coordinator counters. Compare actual state.
		gotState.step, gotState.warm, wantState.step, wantState.warm = 0, false, 0, false
		if !reflect.DeepEqual(gotState, wantState) {
			t.Fatal("HAL KV/recurrent/conv/lineage differs from plain continuation")
		}
		dflash2HALReleased(t, backend)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestHALBootstrapFault(t *testing.T) {
	m := dflash2HALTestModel(t)
	target, backend := dflash2HALTestSession(t, m)
	g := dflash2TinyGeometry()
	f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	q, err := newDFlash2Request(f)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := q.Close(); err != nil {
			t.Error(err)
		}
	})
	before := dflash2HALState(t, target)
	hit := false
	backend.readHook = func(x compute.Tensor) bool {
		if len(x.Shape) == 1 && x.Shape[0] == g.vocab {
			hit = true
			return true
		}
		return false
	}
	if err := q.Bootstrap(context.Background(), []int{3, 7}); err == nil || !hit {
		t.Fatal("first real logits read fault was not witnessed")
	}
	backend.readHook = nil
	if q.closed || q.bootstrapped || !reflect.DeepEqual(before, dflash2HALState(t, target)) {
		t.Fatal("first-token failure did not restore complete fresh HAL state")
	}
	dflash2HALReleased(t, backend)
	if err := q.Bootstrap(context.Background(), []int{3, 7}); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestCloseAndConcurrency(t *testing.T) {
	q, _ := dflash2RequestFixture(t, false)
	peer, _ := dflash2RequestFixture(t, false)
	reader, target := q.ref.w.r, q.ref.target
	entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	hit := false
	q.ref.w.r = dflash2ReadObserver{reader, func(int64) error {
		if !hit {
			hit = true
			close(entered)
			<-release
		}
		return nil
	}}
	go func() { finished <- q.Bootstrap(context.Background(), []int{3, 7}) }()
	<-entered
	if q.mu.TryLock() {
		q.mu.Unlock()
		t.Fatal("in-flight bootstrap did not hold the request owner")
	}
	// A different request progresses while this one is held inside a weight read.
	if err := peer.Bootstrap(context.Background(), []int{11, 13}); err != nil {
		t.Fatal(err)
	}
	closeStarted, closed := make(chan struct{}), make(chan error, 1)
	go func() { close(closeStarted); closed <- q.Close() }()
	<-closeStarted
	select {
	case err := <-closed:
		t.Fatalf("Close crossed an in-flight reader: %v", err)
	default:
	}
	once.Do(func() { close(release) })
	if err := <-finished; err != nil {
		t.Fatal(err)
	}
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	if !q.closed || q.ref != nil || target.modelWeightsHeld {
		t.Fatal("Close did not dispose the transferred Session")
	}
	round, err := peer.Prepare(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	peerTarget := peer.ref.target
	if err := peer.Close(); err != nil || peerTarget.modelWeightsHeld || !round.done || round.Tokens() != nil {
		t.Fatalf("Close did not retire pending round: %v", err)
	}
	if _, err := round.Commit(context.Background(), 1); err == nil {
		t.Fatal("closed request's round committed")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2RequestRejectsInvalidBootstrap(t *testing.T) {
	q, _ := dflash2RequestFixture(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, prompt := range [][]int{nil, {-1}, {3, q.ref.g.vocab}} {
		if err := q.Bootstrap(context.Background(), prompt); err == nil {
			t.Fatal("invalid prompt accepted")
		}
	}
	if err := q.Bootstrap(ctx, []int{3}); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled bootstrap accepted")
	}
	if q.ref.target.Cache.Len() != 0 || q.ref.nextPosition != 0 || q.bootstrapped {
		t.Fatal("invalid prompt mutated owners")
	}
	if _, err := NewQwen38DFlash2Request(context.Background(), q.ref.target, "not-opened"); err == nil {
		t.Fatal("tiny target admitted as stock checkpoint geometry")
	}
	if !q.ref.target.modelWeightsHeld {
		t.Fatal("failed constructor took the caller's target")
	}
	q.ref.target.Step(3) // deliberately violate transfer contract to test refusal
	if err := q.Bootstrap(context.Background(), []int{7}); err == nil {
		t.Fatal("nonfresh target accepted")
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2RequestActualEOSStop(t *testing.T) {
	q, plain := dflash2RequestFixture(t, true)
	q.ref.target.M.Cfg.EOSTokenID = 0
	q.ref.target.M.Cfg.EOSTokenIDs = []int{0}
	prompt := []int{3, 7}
	if err := q.Bootstrap(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	want := dflash2RequestPlain(prompt, plain)
	if argmaxF32(want) != 0 || !q.ref.target.M.Cfg.IsEOS(0) {
		t.Fatal("EOS control is not active")
	}
	round, err := q.Prepare(context.Background(), 8)
	if err != nil || !reflect.DeepEqual(round.Tokens(), []int{0}) {
		t.Fatalf("EOS prepared suffix: %v", err)
	}
	tokens, err := round.Commit(context.Background(), 0)
	if err != nil || len(tokens) != 0 {
		t.Fatalf("EOS stop published tokens: %v %v", tokens, err)
	}
	dflash2RequestEqual(t, q, plain, want)
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestPrepareAndCommitInterruptions(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, phase := range []string{"prepare", "commit"} {
		for _, mode := range []string{"panic", "nil-panic", "goexit", "cancel"} {
			t.Run(phase+"-"+mode, func(t *testing.T) {
				q, plain := dflash2RequestFixture(t, true)
				prompt := []int{3, 7}
				if err := q.Bootstrap(context.Background(), prompt); err != nil {
					t.Fatal(err)
				}
				want := dflash2RequestPlain(prompt, plain)
				var round *Qwen38DFlash2Round
				var err error
				if phase == "commit" {
					round, err = q.Prepare(context.Background(), 8)
					if err != nil || len(round.Tokens()) != 8 {
						t.Fatalf("full block: %v", err)
					}
				}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				reader, hit := q.ref.w.r, false
				q.ref.w.r = dflash2ReadObserver{reader, func(int64) error {
					hit = true
					if phase == "commit" && q.ref.target.Cache.Len() != len(prompt)+3 {
						panic("not a late target commit")
					}
					if mode == "cancel" {
						cancel()
						return ctx.Err()
					}
					siblingFault(mode, errors.New("request phase interruption"))
					return nil
				}}
				var emitted []int
				out := siblingRun(func() {
					if phase == "prepare" {
						round, err = q.Prepare(ctx, 8)
					} else {
						emitted, err = round.Commit(ctx, 3)
					}
				})
				if !hit || len(emitted) != 0 || (mode == "goexit" && out.resumed) || (mode != "goexit" && (!out.returned || err == nil)) {
					t.Fatalf("phase control: %+v err=%v hit=%v", out, err, hit)
				}
				if mode == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatal("cancellation identity lost")
				}
				if q.closed || q.pending != nil {
					t.Fatal("restorable phase did not release its pending owner")
				}
				q.ref.w.r = reader
				dflash2RequestEqual(t, q, plain, want)
			})
		}
	}
}

type requestCleanupBackend struct {
	*dflash2HALTestBackend
	freeHook, trimHook   func()
	freeCalls, trimCalls int
}

func (b *requestCleanupBackend) Qwen35GDNDecode(
	normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj, convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	b.owned("request-gdn")
	b.gdnCalls++
	if b.failAt > 0 && b.gdnCalls == b.failAt {
		return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, errInjectedQwen35GDN
	}
	return b.dflash2HALTestBackend.Qwen35GDNDecode(
		normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel,
		rmsNormEpsilon,
	)
}

func (b *requestCleanupBackend) Free(x compute.Tensor) {
	b.freeCalls++
	if b.freeHook != nil {
		b.freeHook()
	}
	b.dflash2HALTestBackend.Free(x)
}
func (b *requestCleanupBackend) Trim() {
	b.trimCalls++
	if b.trimHook != nil {
		b.trimHook()
	}
}

func requestCleanupFixture(t *testing.T) (*Qwen38DFlash2Request, *Session, *requestCleanupBackend) {
	t.Helper()
	m := dflash2HALTestModel(t)
	base := &dflash2HALTestBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
	b := &requestCleanupBackend{dflash2HALTestBackend: base}
	t.Cleanup(base.reference.Close)
	target, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	base.target, target.Quant = target, true
	t.Cleanup(target.Close)
	g := dflash2TinyGeometry()
	f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	q, err := newDFlash2Request(f)
	if err != nil {
		t.Fatal(err)
	}
	// Failed-close tests explicitly retain and recover their quarantined owner.
	t.Cleanup(func() {
		b.freeHook, b.trimHook, b.readHook = nil, nil, nil
		_ = q.Close()
		if q.quarantinedFence.Buf() != nil {
			func() {
				scope := target.beginBackendExecution()
				defer scope.finish()
				b.Free(q.quarantinedFence)
				q.quarantinedFence = compute.Tensor{}
			}()
		}
		if q.quarantinedTarget != nil {
			_ = q.disposeTarget()
		} // explicit fixture recovery after disabling faults
		target.Close()
	})
	return q, target, b
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestCleanupFaultLedger(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, site := range []string{"free", "trim", "fence"} {
		for _, mode := range []string{"panic", "nil-panic", "goexit"} {
			t.Run(site+"-"+mode, func(t *testing.T) {
				q, target, b := requestCleanupFixture(t)
				if err := q.Bootstrap(context.Background(), []int{3, 7}); err != nil {
					t.Fatal(err)
				}
				marker, hit := errors.New("request cleanup fault"), false
				fault := func() {
					if !hit {
						hit = true
						siblingFault(mode, marker)
					}
				}
				switch site {
				case "free":
					b.freeHook = fault
				case "trim":
					b.trimHook = fault
				case "fence":
					b.readHook = func(x compute.Tensor) bool {
						if len(x.Shape) == 1 && x.Shape[0] == 1 {
							fault()
							return true
						}
						return false
					}
				}
				var err error
				out := siblingRun(func() { err = q.Close() })
				if !hit || !q.closed || q.closeErr == nil || (mode == "goexit" && out.resumed) || (mode != "goexit" && (!out.returned || err == nil)) {
					t.Fatalf("cleanup outcome: %+v err=%v retained=%v", out, err, q.closeErr)
				}
				if site == "fence" {
					if q.quarantinedTarget != target || !target.modelWeightsHeld || q.quarantinedFence.Buf() == nil {
						t.Fatal("unfenced ownership was dropped")
					}
				} else if target.modelWeightsHeld || b.trimCalls != 1 {
					t.Fatal("fenced failure stranded a target pin or skipped a cleanup sibling")
				}
				beforeFree, beforeTrim, retained := b.freeCalls, b.trimCalls, q.closeErr
				if repeated := q.Close(); repeated != retained || b.freeCalls != beforeFree || b.trimCalls != beforeTrim {
					t.Fatal("failed cleanup was silently retried or forgotten")
				}
				if !b.owner.TryLock() {
					t.Fatal("cleanup stranded backend execution owner")
				}
				b.owner.Unlock()
			})
		}
	}
}

type requestCloneFaultKV struct {
	compute.KVStore
	mode  string
	calls int
}

func (k *requestCloneFaultKV) Clone() compute.KVStore {
	k.calls++
	siblingFault(k.mode, errors.New("request snapshot acquisition"))
	panic("unreachable clone")
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2RequestCheckpointAcquisitionInterruption(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			q, target, b := requestCleanupFixture(t)
			kv := &requestCloneFaultKV{KVStore: target.halKV, mode: mode}
			target.halKV = kv
			var err error
			out := siblingRun(func() { err = q.Bootstrap(context.Background(), []int{3}) })
			if kv.calls != 1 || !q.closed || target.modelWeightsHeld || q.bootstrapped || q.pending != nil || q.ref != nil {
				t.Fatal("failed checkpoint acquisition left a usable or pinned request")
			}
			if mode == "goexit" {
				if out.resumed {
					t.Fatal("acquisition swallowed Goexit")
				}
			} else if !out.returned || err == nil {
				t.Fatalf("acquisition error lost: %+v %v", out, err)
			}
			if !b.owner.TryLock() {
				t.Fatal("acquisition stranded backend owner")
			}
			b.owner.Unlock()
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2RequestFailClosedFenceRetainsLastPin(t *testing.T) {
	for _, failFence := range []bool{false, true} {
		t.Run(fmt.Sprint(failFence), func(t *testing.T) {
			q, target, b := requestCleanupFixture(t)
			m := target.M
			b.reference.Close() // remove the test-only third pin before admission
			closer := &executionOwnerCloser{owner: &b.owner}
			m.SetWeightCloser(closer)
			var active *WeightSessionsActiveError
			if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 2 {
				t.Fatalf("expected target and reference pins: %v", err)
			}
			b.failAt = 1 // actual GDN dispatch -> failBackendForward -> closeBackendOwned
			fenceSeen := false
			b.readHook = func(x compute.Tensor) bool {
				if len(x.Shape) == 1 && x.Shape[0] == 1 {
					fenceSeen = true
					if closer.closed || target.modelWeightsHeld {
						panic("backing retired or target did not really fail closed")
					}
					return failFence
				}
				return false
			}
			err := q.Bootstrap(context.Background(), []int{3})
			if !errors.Is(err, errInjectedQwen35GDN) || b.gdnCalls != 1 || !fenceSeen || !q.closed || target.modelWeightsHeld {
				t.Fatalf("selected failure control not reached: %v", err)
			}
			m.weightCloser.mu.Lock()
			holds := m.weightCloser.sessions
			m.weightCloser.mu.Unlock()
			if failFence {
				if closer.closed || holds != 1 || q.quarantinedReference == nil || q.quarantinedReference.closed || q.quarantinedTarget != target || q.closeErr == nil {
					t.Fatal("failed fence discarded the last backing pin")
				}
			} else if !closer.closed || holds != 0 || q.quarantinedReference != nil || q.quarantinedTarget != nil {
				t.Fatal("successful fence did not retire backing after unlock")
			}
			b.readHook = nil
		})
	}
}

type dflash2RetirementFaultCloser struct {
	mode  string
	calls int
}

func (c *dflash2RetirementFaultCloser) Close() error {
	c.calls++
	siblingFault(c.mode, errors.New("final model retirement"))
	return nil
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceRetirementFailureStaysVisible(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			m := NewSynthetic(qwen35HybridTestCfg())
			t.Cleanup(func() { _ = m.CloseWeights() })
			target := m.NewSession()
			t.Cleanup(target.Close)
			g := dflash2TinyGeometry()
			f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = f.Close() })
			target.Close()
			closer := &dflash2RetirementFaultCloser{mode: mode}
			m.SetWeightCloser(closer)
			var active *WeightSessionsActiveError
			if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 1 {
				t.Fatalf("reference is not last pin: %v", err)
			}
			out := siblingRun(func() { _ = f.Close() })
			if out.returned || (mode == "goexit" && out.resumed) || (mode != "goexit" && !out.resumed) || closer.calls != 1 || f.closeErr == nil {
				t.Fatalf("retirement fault was lost: %+v", out)
			}
			retained := f.closeErr
			if err := f.Close(); err != retained || closer.calls != 1 {
				t.Fatal("reference falsely succeeded or retried final retirement")
			}
		})
	}
}

type requestKVLedger struct {
	nodes                                  []*requestLedgerKV
	mode                                   string
	failFree, failBytes, freeHit, bytesHit bool
}
type requestLedgerKV struct {
	compute.KVStore
	ledger               *requestKVLedger
	clone, freed, effect bool
}

func (l *requestKVLedger) wrap(kv compute.KVStore, clone bool) *requestLedgerKV {
	k := &requestLedgerKV{KVStore: kv, ledger: l, clone: clone}
	l.nodes = append(l.nodes, k)
	return k
}
func (k *requestLedgerKV) Clone() compute.KVStore { return k.ledger.wrap(k.KVStore.Clone(), true) }
func (k *requestLedgerKV) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}
func (k *requestLedgerKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	return compute.SnapshotKVToHost(k.KVStore)
}
func (k *requestLedgerKV) Free() {
	if k.freed {
		panic("KV handle released more than once")
	}
	k.freed = true
	k.KVStore.Free()
	k.effect = true
	if k.ledger.failFree && !k.ledger.freeHit {
		k.ledger.freeHit = true
		siblingFault(k.ledger.mode, errors.New("KV free after physical effect"))
	}
}
func (k *requestLedgerKV) ResidentBytes() int64 {
	if k.clone && k.ledger.failBytes && !k.ledger.bytesHit {
		k.ledger.bytesHit = true
		siblingFault(k.ledger.mode, errors.New("snapshot receipt after successful clone"))
	}
	return k.KVStore.ResidentBytes()
}
func (l *requestKVLedger) assertReleased(t *testing.T) {
	t.Helper()
	for i, k := range l.nodes {
		if !k.freed || !k.effect {
			t.Fatalf("KV owner %d was abandoned", i)
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2RequestPartialCommitFreeAfterEffect(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, accepted := range []int{0, 1} {
		for _, mode := range []string{"panic", "nil-panic", "goexit"} {
			t.Run(fmt.Sprint(accepted)+"-"+mode, func(t *testing.T) {
				q, target, _ := requestCleanupFixture(t)
				if err := q.Bootstrap(context.Background(), []int{3, 7}); err != nil {
					t.Fatal(err)
				}
				before, logits := dflash2HALState(t, target), append([]float32(nil), q.logits...)
				ledger := &requestKVLedger{mode: mode}
				target.halKV = ledger.wrap(target.halKV, false)
				round, err := q.Prepare(context.Background(), 4)
				if err != nil || len(round.Tokens()) < 1 || len(round.cp.tx.tx.draft) != 4 {
					t.Fatalf("partial commit control: %v", err)
				}
				ledger.failFree = true
				var emitted []int
				out := siblingRun(func() { emitted, err = round.Commit(context.Background(), accepted) })
				if !ledger.freeHit || len(ledger.nodes) != 4 || !ledger.nodes[0].effect || len(emitted) != 0 || q.closed || q.pending != nil {
					t.Fatalf("partial commit ownership failed: %+v err=%v owners=%d", out, err, len(ledger.nodes))
				}
				if mode == "goexit" {
					if out.resumed {
						t.Fatal("partial commit swallowed Goexit")
					}
				} else if !out.returned || err == nil {
					t.Fatalf("failed partial commit escaped: %+v %v", out, err)
				}
				assertFloat32BitsEqual(t, "partial commit rollback logits", q.logits, logits)
				if !reflect.DeepEqual(dflash2HALState(t, target), before) {
					t.Fatal("partial commit did not restore complete HAL state")
				}
				if err := q.Close(); err != nil {
					t.Fatal(err)
				}
				ledger.assertReleased(t)
			})
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2RequestTransactionReceiptAcquisitionFailure(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			q, target, _ := requestCleanupFixture(t)
			if err := q.Bootstrap(context.Background(), []int{3, 7}); err != nil {
				t.Fatal(err)
			}
			before, logits := dflash2HALState(t, target), append([]float32(nil), q.logits...)
			ledger := &requestKVLedger{mode: mode, failBytes: true}
			target.halKV = ledger.wrap(target.halKV, false)
			var round *Qwen38DFlash2Round
			var err error
			out := siblingRun(func() { round, err = q.Prepare(context.Background(), 4) })
			if !ledger.bytesHit || len(ledger.nodes) != 3 || !ledger.nodes[2].effect || round != nil || q.closed || q.pending != nil {
				t.Fatalf("constructor snapshot escaped/leaked: %+v %v", out, err)
			}
			if mode == "goexit" {
				if out.resumed {
					t.Fatal("constructor swallowed Goexit")
				}
			} else if !out.returned || err == nil {
				t.Fatalf("constructor fault escaped: %+v %v", out, err)
			}
			assertFloat32BitsEqual(t, "constructor rollback logits", q.logits, logits)
			if !reflect.DeepEqual(dflash2HALState(t, target), before) {
				t.Fatal("constructor did not restore complete HAL state")
			}
			if err := q.Close(); err != nil {
				t.Fatal(err)
			}
			ledger.assertReleased(t)
		})
	}
}
