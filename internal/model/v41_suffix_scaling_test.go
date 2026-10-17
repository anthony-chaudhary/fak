package model

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime slow est=120s lane=optin
// Synthetic weights execute Session.Prefill, including its actual suffix route,
// followed by Session.Step. Full-profile means the production full-geometry
// branch, not the published 40-layer checkpoint or physical GPU qualification.
func TestV41SuffixScaling(t *testing.T) {
	for _, profile := range []string{"reduced_boundary", "full_compressed_engram", "streamed_experts"} {
		sizes := []int{16, 128}
		if profile == "reduced_boundary" {
			sizes = append(sizes, 1024, 1025)
		}
		for _, n := range sizes {
			t.Run(fmt.Sprintf("%s/suffix%d", profile, n), func(t *testing.T) {
				build := func() *Model {
					var m *Model
					switch profile {
					case "reduced_boundary":
						m = v41IncrementalPlainModel(t, 2)
					case "full_compressed_engram":
						m, _ = v41IncrementalEngramFixture(t, true)
					case "streamed_experts":
						m = v41IncrementalExpertFixture(t, false, true)
					}
					m.Cfg.Window = make([]int, m.Cfg.NumLayers)
					for layer := range m.Cfg.Window {
						m.Cfg.Window[layer] = 64
					}
					return m
				}
				type observation struct {
					logits, next   []float32
					chunks         []int
					steps, reads   int
					elapsed        time.Duration
					bytes, mallocs uint64
					full           bool
				}
				run := func(batched bool) observation {
					m := build()
					defer m.CloseWeights()
					full, err := v41ForwardGeometry(m.Cfg)
					if err != nil {
						t.Fatal(err)
					}
					if (profile == "full_compressed_engram") != full {
						t.Fatalf("profile %s full=%t", profile, full)
					}
					if batched {
						t.Setenv(v41SuffixUbatchEnv, "2")
					} else {
						t.Setenv(v41SuffixUbatchEnv, "")
					}
					prefix := v41ParityPrompt(m.Cfg.VocabSize, 7)
					suffix := v41SuffixTail(m.Cfg, len(prefix), n)
					s := m.NewSession()
					defer s.Close()
					s.Prefill(prefix)
					if !s.v41IncrementalEligible() {
						t.Fatal("suffix route not eligible")
					}
					o := observation{full: full}
					oldProbe, oldStep := v41SuffixUbatchProbe, v41PrefillStepProbe
					v41SuffixUbatchProbe = func(rows int) { o.chunks = append(o.chunks, rows) }
					v41PrefillStepProbe = func() { o.steps++ }
					defer func() { v41SuffixUbatchProbe = oldProbe; v41PrefillStepProbe = oldStep }()
					reads := func() int {
						if m.expertCheckpoint != nil {
							return m.expertCheckpoint.Stats().Reads
						}
						return 0
					}
					beforeReads := reads()
					runtime.GC()
					var before, after runtime.MemStats
					runtime.ReadMemStats(&before)
					start := time.Now()
					o.logits = append([]float32(nil), s.Prefill(suffix)...)
					o.elapsed = time.Since(start)
					runtime.ReadMemStats(&after)
					o.bytes, o.mallocs = after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs
					o.reads = reads() - beforeReads
					history := append(append([]int(nil), prefix...), suffix...)
					v41RequireCursors(t, s, history)
					next := v41SuffixTail(m.Cfg, len(history), 1)[0]
					o.next = append([]float32(nil), s.Step(next)...)
					v41RequireCursors(t, s, append(history, next))
					if batched {
						chunks := (n + v41SuffixUbatchMaxRows - 1) / v41SuffixUbatchMaxRows
						size := (n + chunks - 1) / chunks
						var want []int
						for done := 0; done < n; done += size {
							want = append(want, min(size, n-done))
						}
						if !reflect.DeepEqual(o.chunks, want) || o.steps != 0 {
							t.Fatalf("batch chunks=%v want=%v serial_steps=%d", o.chunks, want, o.steps)
						}
					} else if len(o.chunks) != 0 || o.steps != n {
						t.Fatalf("serial chunks=%v steps=%d want=%d", o.chunks, o.steps, n)
					}
					t.Logf("profile=%s full_geometry=%t suffix=%d batched=%t chunks=%v serial_steps=%d expert_reads=%d elapsed_ns=%d allocated_bytes=%d mallocs=%d digest=%s", profile, full, n, batched, o.chunks, o.steps, o.reads, o.elapsed.Nanoseconds(), o.bytes, o.mallocs, v41LogitsDigest(o.logits))
					return o
				}
				serial, batched := run(false), run(true)
				v41RequireSameLogits(t, "suffix serial vs batched", batched.logits, serial.logits)
				v41RequireSameLogits(t, "next step serial vs batched", batched.next, serial.next)
				if profile == "streamed_experts" && (batched.reads <= 0 || batched.reads > serial.reads) {
					t.Fatalf("expert reads batched=%d serial=%d", batched.reads, serial.reads)
				}
			})
		}
	}
}

var v41SuffixAdmissionSink []float32

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41SuffixAdmissionReuse(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	defer m.CloseWeights()
	m.Cfg.Window = []int{64, 64}
	prefix := v41ParityPrompt(m.Cfg.VocabSize, 7)
	suffix := v41SuffixTail(m.Cfg, len(prefix), 16)
	t.Setenv(v41SuffixUbatchEnv, "")
	run := func(legacy bool) []float32 {
		s := m.NewSession()
		defer s.Close()
		s.Prefill(prefix)
		var out []float32
		var err error
		if legacy {
			out, err = v41LegacySuffixUbatchBody(s, suffix, true)
		} else {
			out, err = s.v41SuffixUbatchBody(suffix, true)
		}
		if err != nil {
			panic(err)
		}
		return out
	}
	v41RequireSameLogits(t, "unchanged ubatch arithmetic", run(false), run(true))
	before := testing.AllocsPerRun(5, func() { v41SuffixAdmissionSink = run(true) })
	after := testing.AllocsPerRun(5, func() { v41SuffixAdmissionSink = run(false) })
	t.Logf("seed plus16-token owned ubatch allocations legacy=%g candidate=%g", before, after)
	if after >= before {
		t.Fatalf("same immutable admission repeated per token: legacy=%g candidate=%g", before, after)
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41SuffixAdmissionRefusals(t *testing.T) {
	for _, kind := range []string{"invalid_later_token", "malformed_first_model", "selected_second_embedding", "callback_mutates_and_removes_itself"} {
		t.Run(kind, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			defer m.CloseWeights()
			m.Cfg.Window = []int{64, 64}
			t.Setenv(v41SuffixUbatchEnv, "2")
			s := m.NewSession()
			defer s.Close()
			prefix := v41ParityPrompt(m.Cfg.VocabSize, 7)
			s.Prefill(prefix)
			suffix := []int{1, 2, 3}
			sentinel := fmt.Errorf("selected second embedding failed")
			calls := 0
			switch kind {
			case "invalid_later_token":
				suffix[1] = m.Cfg.VocabSize
			case "malformed_first_model":
				name := layerName(0, "attn_norm.weight")
				meta := m.manifest[name]
				meta.Shape = []int{m.Cfg.HiddenSize + 1}
				m.manifest[name] = meta
			case "callback_mutates_and_removes_itself":
				s.v41Forward.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
					calls++
					row, err := m.v41EmbeddingPanel(ids)
					name := layerName(1, "attn_norm.weight")
					meta := m.manifest[name]
					meta.Shape = []int{m.Cfg.HiddenSize + 1}
					m.manifest[name] = meta
					s.v41Forward.gatherOps = nil
					return row, err
				}}

			case "selected_second_embedding":
				s.v41Forward.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
					calls++
					if calls == 2 {
						return nil, sentinel
					}
					return m.v41EmbeddingPanel(ids)
				}}
			}
			_, err := s.v41SuffixUbatch(suffix, true)
			if err == nil {
				t.Fatal("expected refusal")
			}
			if kind == "selected_second_embedding" && (!errors.Is(err, sentinel) || calls != 2) {
				t.Fatalf("dynamic selected error=%v calls=%d", err, calls)
			}
			if kind != "selected_second_embedding" && !errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("typed admission error=%v", err)
			}
			if kind == "callback_mutates_and_removes_itself" && calls != 1 {
				t.Fatalf("mutating callback calls=%d", calls)
			}
			v41RequireCursors(t, s, prefix)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41SuffixAdmissionLaterChunkFailure(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	defer m.CloseWeights()
	m.Cfg.Window = []int{64, 64}
	t.Setenv(v41SuffixUbatchEnv, "2")
	s := m.NewSession()
	defer s.Close()
	prefix := v41ParityPrompt(m.Cfg.VocabSize, 7)
	s.Prefill(prefix)
	suffix := v41SuffixTail(m.Cfg, len(prefix), 1025)
	sentinel := errors.New("selected next-chunk embedding failure")
	calls := 0
	s.v41Forward.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
		calls++
		if calls == 514 {
			return nil, sentinel
		}
		return m.v41EmbeddingPanel(ids)
	}}
	err := panicAsError(func() { s.Prefill(suffix) })
	if !errors.Is(err, sentinel) || calls != 514 {
		t.Fatalf("next chunk failure=%v embedding calls=%d", err, calls)
	}
	// Balanced chunks are513+512. A selected second-chunk failure cannot replay
	// the prefix or erase the first chunk's already-committed positions.
	committed := append(append([]int(nil), prefix...), suffix[:513]...)
	v41RequireCursors(t, s, committed)
	s.v41Forward.gatherOps = nil
	want := m.NewSession()
	defer want.Close()
	want.Prefill(committed)
	v41RequireSameLogits(t, "step after refused chunk", s.Step(suffix[513]), want.Step(suffix[513]))
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41SuffixAdmissionClockMutation(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, true)
	defer m.CloseWeights()
	m.Cfg.Window = []int{64, 64, 64}
	s := m.NewSession()
	defer s.Close()
	prefix := v41ParityPrompt(m.Cfg.VocabSize, 7)
	s.Prefill(prefix)
	calls := 0
	ledger := v41ExpertFaultLedgerOf(m, true)
	m.v41SetExpertTimeClock(func() int64 {
		calls++
		name := layerName(1, "attn_norm.weight")
		meta := m.manifest[name]
		meta.Shape = []int{m.Cfg.HiddenSize + 1}
		m.manifest[name] = meta
		// nowLocked already holds the ledger lock; restore the default without
		// recursively entering the public test setter.
		ledger.nowNanos = nil
		return 1
	})
	_, err := s.v41SuffixUbatch([]int{1, 2}, true)
	if !errors.Is(err, ErrV41ForwardStage) || calls != 1 {
		t.Fatalf("clock mutation error=%v calls=%d", err, calls)
	}
	v41RequireCursors(t, s, prefix)
}

// Frozen pre-change ubatch assembly, receiver syntax renamed only. This
// isolates admission reuse from the expert batching that already existed.
func v41LegacySuffixUbatchBody(s *Session, ids []int, wantLogits bool) (logits []float32, err error) {
	m, st := s.M, s.v41State()
	txs := make([]*v41StepTransaction, 0, len(ids))
	// Uncommitted positions roll back newest first, so every layer's journal
	// restores the cursor and ring row it staged.
	defer func() {
		for k := len(txs) - 1; k >= 0; k-- {
			txs[k].abortAfterFailure()
		}
	}()
	base := st.history
	view := append(base[:len(base):len(base)], ids...)
	for k, id := range ids {
		// A step takes its position and Engram context from history; each
		// position begins as if the earlier suffix positions had committed.
		st.history = view[:len(base)+k]
		tx, berr := m.beginV41Step(id, st, &v41ProjScratch{})
		st.history = base
		if berr != nil {
			return nil, berr
		}
		txs = append(txs, tx)
	}
	scratch := &v41ProjScratch{}
	scratch.bindLayerSessionBindings(st)
	defer scratch.clearLayerSessionBindings()
	works := make([]*v41LayerStepWork, len(txs))
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for k, tx := range txs {
			if err = tx.prepareLayer(); err != nil {
				return nil, err
			}
			if err = tx.publishPendingKV(); err != nil {
				return nil, err
			}
			works[k] = tx.pendingLayer
		}
		routed, rerr := m.v41SuffixUbatchRouted(l, works, st, scratch)
		if rerr != nil {
			return nil, rerr
		}
		for k, tx := range txs {
			if err = tx.finishLayer(routed[k]); err != nil {
				return nil, err
			}
		}
	}
	if v41SuffixUbatchProbe != nil {
		v41SuffixUbatchProbe(len(txs))
	}
	for k, tx := range txs {
		if wantLogits && k == len(txs)-1 {
			logits, err = tx.finalize()
		} else {
			err = tx.finalizeNoLogits()
		}
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}
