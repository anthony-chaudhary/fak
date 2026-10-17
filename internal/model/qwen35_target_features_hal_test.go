package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is a software HAL witness, not device evidence. The ordinary HAL runs
// real Q8 primitives; GDN invokes native Q8 math with state reconstructed from
// the actual HAL-owned tensors on EVERY call. No private reference history can
// conceal a broken COW restore. F32 matrix entries are removed from the fixture.
type dflash2HALTestBackend struct {
	*recordingQwen35Backend
	owner                       sync.Mutex
	target                      *Session
	batch, poison               bool
	begins, flushes, recycles   int
	tapReads, q8Uploads, q8Muls int
	headMuls                    int
	aliases                     [][]float32
	readHook                    func(compute.Tensor) bool
	adds                        int
	current                     [][]float32
	observed                    [][][]float32
	protectHead                 bool
}

func (b *dflash2HALTestBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *dflash2HALTestBackend) owned(site string) {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned diagnostic backend operation: " + site)
	}
}
func (b *dflash2HALTestBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.UploadDtype = true
	return caps
}
func (b *dflash2HALTestBackend) Upload(x compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.owned("upload")
	if dtype == compute.Q8_0 {
		if x.Dtype != compute.Q8_0 {
			panic("Q8 upload expanded the source weight")
		}
		b.q8Uploads++
	}
	return b.Backend.Upload(x, dtype)
}
func (b *dflash2HALTestBackend) UploadClass(x compute.Tensor, dtype compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.owned("upload-class")
	return b.recordingQwen35Backend.UploadClass(x, dtype, class, site)
}
func (b *dflash2HALTestBackend) BeginBatch() {
	b.owned("begin")
	if b.batch {
		panic("nested diagnostic batch")
	}
	b.batch = true
	b.begins++
}
func (b *dflash2HALTestBackend) FlushBatch() {
	b.owned("flush")
	if b.batch {
		b.flushes++
		b.batch = false
	}
}
func (b *dflash2HALTestBackend) Recycle() {
	b.owned("recycle")
	if b.batch {
		panic("recycle before diagnostic fence")
	}
	b.recycles++
	if b.poison {
		for _, row := range b.aliases {
			for i := range row {
				row[i] = float32(math.NaN())
			}
		}
	}
	b.aliases = nil
}
func (b *dflash2HALTestBackend) Read(x compute.Tensor) []float32 {
	b.owned("read")
	if b.readHook != nil && b.readHook(x) {
		return nil
	}
	row := b.Backend.Read(x)
	if b.target != nil && len(x.Shape) == 1 && x.Shape[0] == b.model.Cfg.HiddenSize && b.target.qwen35TargetLayerCapture != nil {
		if b.batch {
			panic("target residual read before explicit fence")
		}
		b.tapReads++
	}
	if len(x.Shape) == 1 && x.Shape[0] == b.model.Cfg.VocabSize {
		if b.batch {
			panic("logit read before explicit fence")
		}
		b.aliases = append(b.aliases, row)
	}
	return row
}
func (b *dflash2HALTestBackend) Free(x compute.Tensor) {
	b.owned("free")
	if b.protectHead && b.target != nil {
		for _, weight := range b.target.halW {
			if weight.Buf() == x.Buf() {
				panic("shared head operation freed a borrowed target weight")
			}
		}
	}
	b.recordingQwen35Backend.Free(x)
}
func (b *dflash2HALTestBackend) CloneTensor(x compute.Tensor) (compute.Tensor, error) {
	b.owned("clone")
	return b.recordingQwen35Backend.CloneTensor(x)
}
func (b *dflash2HALTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.owned("matmul")
	if len(w.Shape) == 2 && w.Shape[0] == b.model.Cfg.VocabSize {
		head := b.target.halW["q8:lm_head.weight"]
		if w.Shape[1] != b.model.Cfg.HiddenSize || w.Dtype != compute.Q8_0 || head.Buf() == nil || w.Buf() != head.Buf() {
			panic("vocabulary projection did not use the resident untied Q8 head")
		}
		b.headMuls++
	}
	if w.Dtype == compute.Q8_0 {
		b.q8Muls++
	}
	return b.Backend.MatMul(w, x)
}
func (b *dflash2HALTestBackend) RMSNorm(x, w compute.Tensor, eps float32) compute.Tensor {
	b.owned("norm")
	return b.Backend.RMSNorm(x, w, eps)
}
func (b *dflash2HALTestBackend) RoPE(x compute.Tensor, pos, heads, dim int, theta float64) compute.Tensor {
	b.owned("rope")
	return b.Backend.RoPE(x, pos, heads, dim, theta)
}
func (b *dflash2HALTestBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.owned("swiglu")
	return b.Backend.SwiGLU(g, u)
}
func (b *dflash2HALTestBackend) AddInPlace(dst, src compute.Tensor) {
	b.owned("residual")
	b.Backend.AddInPlace(dst, src)
	// Independent observation of the two residual additions per layer. This
	// does not invoke the feature hook and observes the actual post-FFN value.
	b.adds++
	if b.adds%2 == 0 {
		b.current = append(b.current, append([]float32(nil), b.Backend.Read(dst)...))
	}
	if b.adds == 2*b.model.Cfg.NumLayers {
		b.observed = append(b.observed, b.current)
		b.adds, b.current = 0, nil
	}
}
func (b *dflash2HALTestBackend) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.owned("attention")
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}
func (b *dflash2HALTestBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.owned("new-kv")
	return &dflash2HALTestKV{KVStore: b.Backend.NewKV(cfg), backend: b}
}

type dflash2HALTestKV struct {
	compute.KVStore
	backend *dflash2HALTestBackend
}

func (k *dflash2HALTestKV) Clone() compute.KVStore {
	k.backend.owned("clone-kv")
	return &dflash2HALTestKV{KVStore: k.KVStore.Clone(), backend: k.backend}
}
func (k *dflash2HALTestKV) Free() {
	k.backend.owned("free-kv")
	k.KVStore.Free()
}
func (k *dflash2HALTestKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	k.backend.owned("snapshot-kv")
	return compute.SnapshotKVToHost(k.KVStore)
}
func (k *dflash2HALTestKV) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}

func (b *dflash2HALTestBackend) Qwen35GDNDecode(
	normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj, convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	b.owned("gdn")
	layer := -1
	for _, candidate := range b.linearLayers {
		weight := b.target.halW["q8:"+layerName(candidate, "linear_attn.in_proj_qkv.weight")]
		if weight.Buf() == inProjQKV.Buf() {
			layer = candidate
		}
	}
	if layer < 0 || inProjQKV.Dtype != compute.Q8_0 || outProj.Dtype != compute.Q8_0 {
		return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, errors.New("missing packed GDN input")
	}
	cfg := b.model.Cfg
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	state := newLinearAttnLayerState(cfg)
	conv := b.Backend.Read(convState)
	for start := 0; start < len(conv); start += convDim {
		state.conv = append(state.conv, append([]float32(nil), conv[start:start+convDim]...))
	}
	recurrent := b.Backend.Read(recurrentState)
	stride := keyHeadDim * valueHeadDim
	for head := range state.recurrent {
		copy(state.recurrent[head], recurrent[head*stride:(head+1)*stride])
	}
	b.reference.Cache.linear.layers[layer] = state
	out := b.reference.linearAttnStep(layer, append([]float32(nil), b.Backend.Read(normalizedInput)...), q8Kernel{b.model})
	state = b.reference.Cache.linear.layers[layer]
	for row := range state.conv {
		copy(conv[row*convDim:(row+1)*convDim], state.conv[row])
	}
	for head := range state.recurrent {
		copy(recurrent[head*stride:(head+1)*stride], state.recurrent[head])
	}
	return compute.NewF32(b.Backend, []int{cfg.HiddenSize}, out), convState, recurrentState, nil
}

func dflash2HALTestModel(t *testing.T) *Model {
	t.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.AttnOutputGate = false // isolate the new read/fence boundary from legacy bridges
	m := NewSynthetic(cfg)
	// NewSynthetic forces tied embeddings. Install a distinct, nonzero head
	// AFTER construction: [0.25, 0.5) cannot equal the [-0.2, 0.2) embedding.
	man, raw := synthBuildRaw([]synthTensor{{"lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}}}, func(_ string, next func() float32) float32 {
		return 0.375 + next()*0.125
	})
	meta := man["lm_head.weight"]
	meta.Offset += len(m.raw)
	m.raw = append(m.raw, raw...)
	m.manifest["lm_head.weight"] = meta
	m.Cfg.TieWordEmbeddings = false
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	m.Quantize()
	for name := range m.manifest {
		if m.q8w[name] != nil && name != "model.embed_tokens.weight" {
			delete(m.manifest, name)
		}
	}
	head := m.q8w["lm_head.weight"]
	_, f32Head := m.manifest["lm_head.weight"]
	if m.Cfg.TieWordEmbeddings || m.headName() != "lm_head.weight" || f32Head || head == nil || head.out != cfg.VocabSize || head.in != cfg.HiddenSize {
		t.Fatal("fixture must expose only its packed untied head")
	}
	for _, code := range head.q {
		if code <= 0 {
			t.Fatal("packed head must be nonzero and distinct from embeddings")
		}
	}
	return m
}

func dflash2HALTestSession(t *testing.T, m *Model) (*Session, *dflash2HALTestBackend) {
	t.Helper()
	b := &dflash2HALTestBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
	t.Cleanup(b.reference.Close)
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	b.target, s.Quant = s, true
	t.Cleanup(s.Close)
	return s, b
}

type dflash2HALImage struct {
	kv              compute.KVHostSnapshot
	conv, recurrent [][]float32
	lineage         []uint32
	step            int
	warm            bool
}

func dflash2HALState(t *testing.T, s *Session) dflash2HALImage {
	t.Helper()
	p, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	h, err := p.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	image := dflash2HALImage{kv: h.kv, lineage: append([]uint32(nil), h.halLineage.ids...), step: s.halStep, warm: s.halLogitsWarm}
	for _, layer := range h.qwen35.layers {
		image.conv = append(image.conv, layer.conv.data)
		image.recurrent = append(image.recurrent, layer.recurrent.data)
	}
	return image
}

func dflash2HALReleased(t *testing.T, b *dflash2HALTestBackend) {
	t.Helper()
	if !b.owner.TryLock() {
		t.Fatal("backend execution owner leaked")
	}
	b.owner.Unlock()
	if b.batch || b.target.backendExecutionScope != nil || b.target.qwen35TargetFeatureTxn != nil || b.target.qwen35TargetLayerCapture != nil {
		t.Fatal("diagnostic owner, batch, or capture leaked")
	}
}

// fak-test:runtime medium est=3s lane=default
func TestQwen35TargetFeaturesHALEveryAcceptedPrefix(t *testing.T) {
	for accepted := 0; accepted <= 8; accepted++ {
		t.Run(fmt.Sprintf("accepted-%d", accepted), func(t *testing.T) {
			m := dflash2HALTestModel(t)
			target, backend := dflash2HALTestSession(t, m)
			plain, plainBackend := dflash2HALTestSession(t, m)
			proposer, _ := dflash2HALTestSession(t, m)
			oracle, oracleBackend := dflash2HALTestSession(t, m)
			prompt, taps := []int{3, 7, 11}, []int{0, 2, 3}
			before, want := target.Prefill(prompt), plain.Prefill(prompt)
			proposals := proposer.Prefill(prompt)
			oracle.Prefill(prompt)
			draft := make([]int, 8)
			for i := range draft {
				draft[i] = argmaxF32(proposals)
				proposals = proposer.Step(draft[i])
			}
			if accepted < 8 {
				draft[accepted] = (draft[accepted] + 1) % m.Cfg.VocabSize
			}
			backend.poison = true
			tx, err := beginQwen35TargetFeatureTransaction(target, before, taps)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = tx.Abort() })
			verified, err := tx.Verify(context.Background(), draft)
			if err != nil {
				t.Fatal(err)
			}
			gotAccepted, current := 0, before
			for i, token := range draft {
				assertFloat32BitsEqual(t, "verified Q8 HAL row", oracle.Step(token), verified[i])
				if gotAccepted == i && token == argmaxF32(current) {
					gotAccepted++
					current = verified[i]
				}
			}
			if gotAccepted != accepted {
				t.Fatalf("accepted %d, want %d", gotAccepted, accepted)
			}
			got, features, err := tx.Commit(context.Background(), accepted)
			if err != nil || len(features) != accepted {
				t.Fatalf("commit rows=%d: %v", len(features), err)
			}
			for i, token := range draft[:accepted] {
				want = plain.Step(token)
				if features[i].Position != len(prompt)+i || features[i].Token != token || !reflect.DeepEqual(features[i].Layers, taps) {
					t.Fatal("accepted feature identity changed")
				}
				for tap, layer := range taps {
					assertFloat32BitsEqual(t, "independent post-FFN residual", oracleBackend.observed[len(prompt)+i][layer], features[i].Hidden[tap])
				}
			}
			assertFloat32BitsEqual(t, "owned commit logits after recycle", want, got)
			if !reflect.DeepEqual(dflash2HALState(t, target), dflash2HALState(t, plain)) {
				t.Fatal("accepted HAL KV, GDN/conv, lineage, or step differs from plain Q8 target")
			}
			if target.Cache.Len() != 0 || target.halKV.Len() != len(prompt)+accepted || backend.tapReads != 8*len(taps) || plainBackend.tapReads != 0 || backend.q8Uploads == 0 || backend.q8Muls == 0 || backend.headMuls == 0 {
				t.Fatal("wrong prefix authority, tap count, ordinary-route overhead, or packed-weight route")
			}
			backend.poison = false
			for i := 0; i < 2; i++ {
				token := argmaxF32(want)
				if token != argmaxF32(got) {
					t.Fatal("greedy continuation diverged")
				}
				want, got = plain.Step(token), target.Step(token)
				assertFloat32BitsEqual(t, "plain Q8 continuation", want, got)
			}
			dflash2HALReleased(t, backend)
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestQwen35TargetFeaturesHALFailureRestoresPrefix(t *testing.T) {
	for _, mode := range []string{"cancel-verify", "read-tap", "read-logits", "cancel-replay", "read-replay"} {
		t.Run(mode, func(t *testing.T) {
			m := dflash2HALTestModel(t)
			s, b := dflash2HALTestSession(t, m)
			before := s.Prefill([]int{3, 7})
			baseline := dflash2HALState(t, s)
			tx, err := beginQwen35TargetFeatureTransaction(s, before, []int{0, 2, 3})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = tx.Abort() })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			replay := mode == "cancel-replay" || mode == "read-replay"
			if replay {
				if _, err := tx.Verify(ctx, []int{11, 5, 17}); err != nil {
					t.Fatal(err)
				}
			}
			b.poison = true
			b.readHook = func(x compute.Tensor) bool {
				width := m.Cfg.HiddenSize
				if replay || mode == "read-logits" {
					width = m.Cfg.VocabSize
				}
				if b.batch || len(x.Shape) != 1 || x.Shape[0] != width {
					return false
				}
				b.readHook = nil
				if mode == "cancel-verify" || mode == "cancel-replay" {
					cancel()
					return false
				}
				return true
			}
			if replay {
				_, _, err = tx.Commit(ctx, 2)
			} else {
				_, err = tx.Verify(ctx, []int{11, 5, 17})
			}
			if err == nil || b.readHook != nil {
				t.Fatal("injected cancellation/read failure did not reach the diagnostic boundary")
			}
			if !reflect.DeepEqual(baseline, dflash2HALState(t, s)) {
				t.Fatal("failed diagnostic operation did not restore the complete prefix")
			}
			if err := tx.Abort(); err != nil {
				t.Fatal(err)
			}
			dflash2HALReleased(t, b)
			b.poison, b.adds, b.current = false, 0, nil
			reuse, err := beginQwen35TargetFeatureTransaction(s, before, []int{0, 3})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = reuse.Abort() })
			if _, err := reuse.Verify(context.Background(), []int{11}); err != nil {
				t.Fatal(err)
			}
			if _, _, err := reuse.Commit(context.Background(), 1); err != nil {
				t.Fatal(err)
			}
			dflash2HALReleased(t, b)
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2ReferenceQuantizedHALGreedyRounds(t *testing.T) {
	m := dflash2HALTestModel(t)
	target, b := dflash2HALTestSession(t, m)
	plain, _ := dflash2HALTestSession(t, m)
	g := dflash2TinyGeometry()
	f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	seed, err := beginQwen35TargetFeatureTransaction(target, make([]float32, g.vocab), g.taps)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = seed.Abort() })
	if _, err := seed.Verify(context.Background(), []int{3, 7, 11}); err != nil {
		t.Fatal(err)
	}
	before, features, err := seed.Commit(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.AppendCommitted(context.Background(), features); err != nil {
		t.Fatal(err)
	}
	want := plain.Prefill([]int{3, 7, 11})
	b.poison = true
	for round := 0; round < 3; round++ {
		tokens, next, err := f.GreedyRound(context.Background(), before, 8)
		if err != nil || len(tokens) == 0 {
			t.Fatalf("round %d: tokens=%v err=%v", round, tokens, err)
		}
		for _, token := range tokens {
			if token != argmaxF32(want) {
				t.Fatal("pretrained-fixture draft escaped exact Q8 target verification")
			}
			want = plain.Step(token)
		}
		assertFloat32BitsEqual(t, "DFlash Q8 boundary", want, next)
		if !reflect.DeepEqual(dflash2HALState(t, target), dflash2HALState(t, plain)) || f.nextPosition != target.halKV.Len() {
			t.Fatal("draft and target committed boundaries differ")
		}
		before = next
	}
	// The shared head must copy its borrowed Read before Free/Recycle and must
	// not retire the target's immutable weight. Cancellation/read failure must
	// leave its execution owner available for the next call.
	hidden := make([]float32, g.hidden)
	hidden[0] = 1
	b.protectHead = true
	headMuls := b.headMuls
	first, err := target.dflash2TargetHead(context.Background(), hidden)
	if err != nil {
		t.Fatal(err)
	}
	second, err := target.dflash2TargetHead(context.Background(), hidden)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "owned shared-head rows", first, second)
	if b.headMuls != headMuls+2 || first[0] <= 0 {
		t.Fatal("shared-head probe must use the nonzero resident Q8 head twice")
	}
	ctx, cancel := context.WithCancel(context.Background())
	b.readHook = func(x compute.Tensor) bool {
		if len(x.Shape) == 1 && x.Shape[0] == g.vocab {
			b.readHook = nil
			cancel()
		}
		return false
	}
	if row, err := target.dflash2TargetHead(ctx, hidden); !errors.Is(err, context.Canceled) || row != nil {
		t.Fatalf("shared-head cancellation row=%v err=%v", row, err)
	}
	cancel()
	b.readHook = func(compute.Tensor) bool { b.readHook = nil; return true }
	if row, err := target.dflash2TargetHead(context.Background(), hidden); err == nil || row != nil {
		t.Fatal("shared-head read failure was published")
	}
	b.protectHead = false
	dflash2HALReleased(t, b)
}
