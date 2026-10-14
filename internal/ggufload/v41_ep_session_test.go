package ggufload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=10s lane=default
func TestV41ExpertParallelSessionFactory(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	shards, digest := ds41AuthenticatedEPShards(t)
	ws, err := OpenWeightsAuthenticated(t.Context(), shards, digest)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	oracle, err := ws.QuantModelQ4KContext(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	oracle.SetWeightCloser(ws)
	defer oracle.CloseWeights()
	refBackend := newDS41EPRecorder(oracle)
	ref, err := oracle.NewBackendSessionChecked(refBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	tokens := []int{1, 2, 3}
	want := [][]float32{ref.Prefill(tokens[:2]), ref.Step(tokens[2])}
	ref.Close()
	const ranks = 3
	// This is caller-declared agreement for a synthetic software fixture.
	source := fmt.Sprintf("%x", sha256.Sum256([]byte("V4.1 owner factory software fixture")))
	errs := ds41EPGroup(t, ranks, func(g *model.DistComm) error {
		backend := newDS41EPRecorder(oracle)
		req := V41ExpertParallelSessionRequest{
			Shards: shards, ExpectedShardSetSHA256: strings.ToUpper(digest),
			SourceSHA256: source, Group: g, Backend: backend,
		}
		canceled, stop := context.WithCancel(t.Context())
		stop()
		if s, c, err := LoadV41ExpertParallelSession(canceled, V41ExpertParallelSessionRequest{}); !errors.Is(err, context.Canceled) || s != nil || c != nil {
			return fmt.Errorf("context refusal did not precede backend/source work: %v", err)
		}
		// These refusals precede source opening and native Session binding.
		// The later successful decode proves they leave the borrowed group usable.
		for _, mutate := range []func(*V41ExpertParallelSessionRequest){
			func(r *V41ExpertParallelSessionRequest) { r.Backend = nil },
			func(r *V41ExpertParallelSessionRequest) { r.Backend = compute.Default() },
			func(r *V41ExpertParallelSessionRequest) {
				r.Backend = &ds41FactoryCapabilityBackend{ds41EPRecorder: backend, missingQ3: true}
			},
			func(r *V41ExpertParallelSessionRequest) {
				r.Backend = &ds41FactoryCapabilityBackend{ds41EPRecorder: backend, missingRouted: true}
			},
			func(r *V41ExpertParallelSessionRequest) { r.ExpectedShardSetSHA256 = strings.Repeat("0", 64) },
			func(r *V41ExpertParallelSessionRequest) { r.SourceSHA256 = "unverified-label" },
			func(r *V41ExpertParallelSessionRequest) { r.Group = nil },
		} {
			bad := req
			mutate(&bad)
			opened := false
			s, c, err := loadV41ExpertParallelSession(t.Context(), bad, func(context.Context, []ShardIdentity, string) (*WeightSource, error) {
				opened = true
				return nil, fmt.Errorf("preflight opened the source")
			})
			if err == nil || s != nil || c != nil || opened {
				return fmt.Errorf("preflight failed closed contract: session=%v coordinator=%v opened=%t err=%v", s, c, opened, err)
			}
		}
		// Cancellation after authentication must release every opened reader,
		// before the factory can materialize owned weights or construct a Session.
		ctx, cancel := context.WithCancel(t.Context())
		var retained []*ds41EPSource
		s, c, err := loadV41ExpertParallelSession(ctx, req, func(ctx context.Context, manifest []ShardIdentity, set string) (*WeightSource, error) {
			w, err := OpenWeightsAuthenticated(ctx, manifest, set)
			if err != nil {
				return nil, err
			}
			for i, closer := range w.closers {
				tracked := &ds41EPSource{Closer: closer}
				w.closers[i] = tracked
				retained = append(retained, tracked)
			}
			cancel()
			return w, nil
		})
		cancel()
		if !errors.Is(err, context.Canceled) || s != nil || c != nil || len(retained) != len(shards) {
			return fmt.Errorf("post-authentication cancellation: %v", err)
		}
		for _, reader := range retained {
			if reader.closed.Load() != 1 {
				return fmt.Errorf("canceled source closed %d times", reader.closed.Load())
			}
		}
		s, coord, err := LoadV41ExpertParallelSession(t.Context(), req)
		if err != nil {
			return err
		}
		m := s.M
		defer m.CloseWeights()
		defer s.Close()
		band, err := ExpertShardForRank(oracle.Cfg.NumExperts, ranks, g.Rank())
		if err != nil {
			return err
		}
		for expert := range oracle.Cfg.NumExperts {
			for _, projection := range []string{"gate_proj", "up_proj", "down_proj"} {
				name := batchedExpertCanonicalName("deepseek41", 0, expert, projection)
				got, present := m.KQuantRaw(name)
				original, _ := oracle.KQuantRaw(name)
				owned := expert >= band.Lo && expert < band.Hi
				if present != owned || (owned && !bytes.Equal(got, original)) {
					return fmt.Errorf("factory rank %d changed ownership of %s", g.Rank(), name)
				}
				if !owned {
					delete(backend.allowed, sha256.Sum256(original))
				}
			}
		}
		// The factory owns readers through the existing model/session pin, with
		// no separate wrapper lifetime that can close them too early.
		var active *model.WeightSessionsActiveError
		if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 1 {
			return fmt.Errorf("factory lost its session lifetime pin: %v", err)
		}
		if g.Rank() == 0 {
			if coord == nil || !m.EPDecodeCoordinated() {
				return fmt.Errorf("factory omitted rank-0 decode coordinator")
			}
			got := [][]float32{s.Prefill(tokens[:2]), s.Step(tokens[2])}
			for step, row := range got {
				if len(row) != len(want[step]) || len(row) == 0 {
					return fmt.Errorf("factory operation %d changed logits width", step)
				}
				for i, value := range row {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value) != math.Float32bits(want[step][i]) {
						return fmt.Errorf("factory operation %d logit %d differs", step, i)
					}
				}
			}
			s.Close()
			if err := coord.Shutdown(); err != nil {
				return err
			}
		} else {
			if coord != nil || m.EPDecodeCoordinated() {
				return fmt.Errorf("factory bound a root coordinator on a follower")
			}
			used := false
			if err := model.RunEPFollower(g, func() *model.Session {
				if used {
					panic("single-request witness reused its closed factory session")
				}
				used = true
				return s
			}); err != nil {
				return err
			}
		}
		expected := ds41EPOwnedProjectionCounts(band, len(tokens))
		actual := make(map[string]int)
		for _, name := range backend.order {
			actual[name]++
		}
		wantCalls := 0
		for _, count := range expected {
			wantCalls += count
		}
		if len(backend.order) != wantCalls || !maps.Equal(actual, expected) || backend.routed != 6 {
			return fmt.Errorf("factory rank %d missed owned native work", g.Rank())
		}
		if g.Rank() == 0 {
			if len(backend.sums) != len(refBackend.sums) {
				return fmt.Errorf("factory root changed routed sum count")
			}
			for i, round := range backend.sums {
				if !slices.Equal(round, refBackend.sums[i]) {
					return fmt.Errorf("factory root position %d changed exact reduction", i)
				}
			}
		} else if len(backend.sums) != 0 {
			return fmt.Errorf("factory follower rank %d executed a routed sum", g.Rank())
		}
		if len(refBackend.finalRows) != 3 || !slices.Equal(backend.finalRows, refBackend.finalRows) {
			return fmt.Errorf("factory rank %d changed final routed F32 row bytes", g.Rank())
		}
		return backend.retired()
	})
	for rank, err := range errs {
		if err != nil {
			t.Fatalf("factory rank %d: %v", rank, err)
		}
	}
	// These failures happen after binding, so no group-reuse promise is made.
	// Native constructor cleanup and terminal transport semantics remain intact.
	for _, fault := range []bool{false, true} {
		errs := ds41EPGroup(t, ranks, func(g *model.DistComm) error {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			primary := errors.New("factory constructor fault")
			backend := &ds41FactoryBackend{ds41EPRecorder: newDS41EPRecorder(oracle)}
			backend.reset = func() {
				if fault {
					panic(primary)
				}
				cancel()
			}
			req := V41ExpertParallelSessionRequest{Shards: shards, ExpectedShardSetSHA256: digest, SourceSHA256: source, Group: g, Backend: backend}
			var retained []*ds41EPSource
			var closeBeforeKV bool
			var gotPanic any
			var session *model.Session
			var coord *model.EPDecodeCoordinator
			var cause error
			func() {
				defer func() { gotPanic = recover() }()
				session, coord, cause = loadV41ExpertParallelSession(ctx, req, func(ctx context.Context, manifest []ShardIdentity, set string) (*WeightSource, error) {
					w, err := OpenWeightsAuthenticated(ctx, manifest, set)
					if err != nil {
						return nil, err
					}
					for i, closer := range w.closers {
						tracked := &ds41EPSource{Closer: ds41FactoryCloser{Closer: closer, before: func() {
							if backend.kvFrees != 1 {
								closeBeforeKV = true
							}
						}}}
						w.closers[i] = tracked
						retained = append(retained, tracked)
					}
					return w, nil
				})
			}()
			if session != nil || coord != nil || closeBeforeKV || backend.kvFrees != 1 || len(retained) != len(shards) {
				return fmt.Errorf("factory failed-construction cleanup: sourceBeforeKV=%t KVfrees=%d readers=%d", closeBeforeKV, backend.kvFrees, len(retained))
			}
			if (fault && gotPanic != primary) || (!fault && (gotPanic != nil || !errors.Is(cause, context.Canceled))) {
				return fmt.Errorf("factory changed primary fault: panic=%v error=%v", gotPanic, cause)
			}
			for _, reader := range retained {
				if reader.closed.Load() != 1 {
					return fmt.Errorf("failed-construction source closes=%d", reader.closed.Load())
				}
			}
			return backend.retired()
		})
		for rank, err := range errs {
			if err != nil {
				t.Fatalf("factory fault=%t rank %d: %v", fault, rank, err)
			}
		}
	}
	if err := oracle.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if err := refBackend.retired(); err != nil {
		t.Fatal(err)
	}
}

type ds41FactoryCapabilityBackend struct {
	*ds41EPRecorder
	missingQ3, missingRouted bool
}

func (b *ds41FactoryCapabilityBackend) SupportsDeviceWeightDtype(dtype compute.Dtype) bool {
	return !(b.missingQ3 && dtype == compute.Q3_K) && b.ds41EPRecorder.SupportsDeviceWeightDtype(dtype)
}

func (b *ds41FactoryCapabilityBackend) V41RoutedSwiGLUAvailable() bool {
	return !b.missingRouted
}

type ds41FactoryBackend struct {
	*ds41EPRecorder
	reset   func()
	kvFrees int
}

func (b *ds41FactoryBackend) GraphReset() { b.reset() }

func (b *ds41FactoryBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return &ds41FactoryKV{KVStore: b.Backend.NewKV(cfg), owner: b}
}

type ds41FactoryKV struct {
	compute.KVStore
	owner *ds41FactoryBackend
}

func (kv *ds41FactoryKV) Free() {
	kv.owner.kvFrees++
	if owned, ok := kv.KVStore.(interface{ Free() }); ok {
		owned.Free()
	}
}

type ds41FactoryCloser struct {
	io.Closer
	before func()
}

func (c ds41FactoryCloser) Close() error {
	c.before()
	return c.Closer.Close()
}
