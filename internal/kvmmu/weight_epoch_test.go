package kvmmu_test

// weight_epoch_test.go — pins the span-level weight-epoch accounting: Append stamps
// the live weight state onto every segment through the optional abi.KVBackend seam,
// backends without the seam keep their legacy unstamped behavior, and the serve-time
// check fails closed across an epoch_bump while honoring compatible windows and
// revert-restored digests. [SW-VERIFIED]: synthetic model, CPU reference backend, no
// hardware claims.

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const weightEpochTestTarget = "model.layers.0.self_attn.q_proj.weight"

// lcgFill32 fills a slice with deterministic pseudo-random values in [-scale,scale]
// (the same LCG the model-package LoRA witnesses use).
func lcgFill32(n int, seed uint64, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		out[i] = (u*2 - 1) * scale
	}
	return out
}

// loRADeltaForTest builds a ready-to-apply additive-LoRA delta over the tiny
// synthetic q_proj ([Out=32, In=32] under synthCfg).
func loRADeltaForTest(kvCompat model.WeightDeltaKVCompat, seed uint64) model.WeightDelta {
	a := &model.LoRAAdapter{
		Adapter: "math", Target: weightEpochTestTarget,
		In: 32, Out: 32, Rank: 4, Alpha: 8,
		A: lcgFill32(4*32, seed, 0.1), B: lcgFill32(32*4, seed^0xabcdef, 0.1),
	}
	return model.WeightDelta{Kind: model.WeightDeltaLoRAAdapterAdd, Adapters: []*model.LoRAAdapter{a}, KVCompat: kvCompat}
}

// liveEpochKV fronts a real session backend and publishes the LIVE weight-state
// stamp through the optional WeightEpoch seam, exactly like model.kvBackend does.
type liveEpochKV struct {
	inner abi.KVBackend
	wds   *model.WeightDeltaSource
}

func (e *liveEpochKV) Len() int                    { return e.inner.Len() }
func (e *liveEpochKV) Prefill(ids []int) []float32 { return e.inner.Prefill(ids) }
func (e *liveEpochKV) Evict(from, n int) int       { return e.inner.Evict(from, n) }
func (e *liveEpochKV) ModelID() string             { return e.inner.ModelID() }
func (e *liveEpochKV) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return e.inner.StageSpan(ctx, digest, from, n)
}
func (e *liveEpochKV) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	return e.inner.RestoreSpan(ctx, digest)
}
func (e *liveEpochKV) WeightEpoch() model.WeightEpochStamp { return e.wds.Stamp() }

// TestWeightEpochStampedOnSegments pins the stack threading end to end: the span
// ledger records the generation AND the content digest of the weight state each span
// was computed under, and both move with committed transitions. A backend WITHOUT
// the seam keeps its legacy unstamped spans.
func TestWeightEpochStampedOnSegments(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	s := m.NewSession()
	inner, ok := model.KVBackend(s)
	if !ok {
		t.Fatal("model.KVBackend(session): ok=false")
	}
	wds := model.NewWeightDeltaSource(m, nil)
	m.SetWeightDeltaSource(wds)

	rec := &liveEpochKV{inner: inner, wds: wds}
	c := kvmmu.NewBackendWithGate(rec, ctxmmu.New())

	baseDigest := wds.Digest()
	if _, seg := c.Append("sys", "system", []int{1, 2, 3}); seg.WeightEpoch != 0 || seg.WeightDigest != baseDigest {
		t.Fatalf("first span stamped (gen=%d digest=%q), want generation 0 / the base digest %q",
			seg.WeightEpoch, seg.WeightDigest, baseDigest)
	}

	d := loRADeltaForTest(model.KVCompatEpochBump, 7)
	d.BaseDigest = baseDigest
	if _, err := wds.Apply(d); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if _, seg := c.Append("t1", "tool", []int{4, 5, 6}); seg.WeightEpoch != 1 || seg.WeightDigest != wds.Digest() {
		t.Fatalf("post-apply span stamped (gen=%d digest=%q), want generation 1 / the new digest %q",
			seg.WeightEpoch, seg.WeightDigest, wds.Digest())
	}
	if got, _ := c.SegmentByID("sys"); got.WeightEpoch != 0 || got.WeightDigest != baseDigest {
		t.Fatalf("ledger lost the first span's stamp: %+v", got)
	}

	// Legacy behavior: a backend without the seam leaves its spans unstamped.
	plain := &recordingKV{inner: inner}
	pc := kvmmu.NewBackendWithGate(plain, ctxmmu.New())
	if _, seg := pc.Append("legacy", "system", []int{7, 8}); seg.WeightEpoch != 0 || seg.WeightDigest != "" {
		t.Fatalf("unseamed backend stamped (gen=%d digest=%q), want the legacy zero stamp",
			seg.WeightEpoch, seg.WeightDigest)
	}
}

// TestWeightEpochMismatchRefusalFailClosed pins the never-silent-recompute rule at
// the span level: serving across an epoch_bump refuses with the errors.Is sentinel;
// compatible windows survive; a revert restores servability of earlier-digest spans;
// rewound or incompletely accounted windows fail closed.
func TestWeightEpochMismatchRefusalFailClosed(t *testing.T) {
	const H, H1, H2 = "H", "H1", "H2"
	bump := model.KVCompatEpochBump
	compat := model.KVCompatCompatible

	cases := []struct {
		name    string
		stamped model.WeightEpochStamp
		current model.WeightEpochStamp
		decls   []model.WeightDeltaKVCompat
		wantErr bool
	}{
		{"fresh span, same digest", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 0, Digest: H}, nil, false},
		{"revert restored digest at higher generation", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 2, Digest: H}, []model.WeightDeltaKVCompat{bump, bump}, false},
		{"epoch bump in window", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 1, Digest: H1}, []model.WeightDeltaKVCompat{bump}, true},
		{"compatible window survives", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 2, Digest: H2}, []model.WeightDeltaKVCompat{compat, compat}, false},
		{"rewound stamp", model.WeightEpochStamp{Generation: 5, Digest: H}, model.WeightEpochStamp{Generation: 2, Digest: H2}, nil, true},
		{"accounting gap: window too short", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 2, Digest: H2}, []model.WeightDeltaKVCompat{compat}, true},
		{"accounting gap: window too long", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 1, Digest: H1}, []model.WeightDeltaKVCompat{compat, compat}, true},
		{"different digest, empty window", model.WeightEpochStamp{Generation: 0, Digest: H}, model.WeightEpochStamp{Generation: 1, Digest: H1}, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := kvmmu.CheckSpanEpoch(tc.stamped, tc.current, tc.decls)
			if tc.wantErr && (!errors.Is(err, kvmmu.ErrWeightEpochMismatch)) {
				t.Fatalf("CheckSpanEpoch: err=%v, want ErrWeightEpochMismatch", err)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("CheckSpanEpoch: err=%v, want nil", err)
			}
		})
	}

	// Integration through the real weight layer: the declaration window the check
	// consumes comes from WeightDeltaSource.DeclarationsSince, and a revert restores
	// servability of spans stamped under the pre-apply digest.
	m := model.NewSynthetic(synthCfg())
	wds := model.NewWeightDeltaSource(m, nil)
	stamp0 := wds.Stamp()

	d := loRADeltaForTest(model.KVCompatEpochBump, 11)
	d.BaseDigest = stamp0.Digest
	if _, err := wds.Apply(d); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	cur := wds.Stamp()
	if err := kvmmu.CheckSpanEpoch(stamp0, cur, wds.DeclarationsSince(stamp0.Generation)); !errors.Is(err, kvmmu.ErrWeightEpochMismatch) {
		t.Fatalf("span under the old epoch after an epoch_bump apply: err=%v, want ErrWeightEpochMismatch (fail-closed)", err)
	}

	if _, err := wds.Revert(); err != nil {
		t.Fatalf("Revert: %v", err)
	}
	restored := wds.Stamp()
	if restored.Digest != stamp0.Digest {
		t.Fatalf("revert did not restore the pre-apply digest (%q vs %q)", restored.Digest, stamp0.Digest)
	}
	if err := kvmmu.CheckSpanEpoch(stamp0, restored, wds.DeclarationsSince(stamp0.Generation)); err != nil {
		t.Fatalf("span under the reverted-to digest must be servable again: err=%v", err)
	}
}
