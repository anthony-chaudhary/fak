//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

// TestProjectionGraphQwenDeviceKVByteParityL100L300 is the #13087 criterion-2
// witness: for L=100 and L=300 the device-resident KV walk appends the SAME
// Kraw/Kpost/V rows to the host KV cache as the historical host-append walk, and
// it does so BYTE-FOR-BYTE (exact float32 equality, not a tolerance).
//
// Why exact equality is the honest bar here: the native
// mg_qwen35_graph_attention_dkv path computes this panel's pre-norm key
// (`kr`), post-norm key (`kpo`) and raw value (`vp`) with the SAME qgQK kernel
// the host path uses, then BLITS those three device buffers into the persistent
// pair with `copyFromBuffer` (qwen35_graph.m) — a bit-preserving device memmove,
// not a re-quantization. The only difference from the host path is WHERE the
// prefix lives (device pair vs host memcpy); the appended rows are the identical
// bytes. So a `!=` comparison is not stricter than the transport: it is the
// transport's own guarantee, and any drift means the append landed at the wrong
// offset or a row was clobbered.
//
// Panelization matches the production walk owner
// (tryPrefillQwen35HybridQ4K, internal/model/qwen35_prefill_q4k.go): the panel
// quantum is 32 (`panelCover = (L/32)*32`) and each panel is capped at
// PromptPanelMaxTokens (128). So:
//
//	L=100 -> cover 96  -> panels [96]        (a 4-row host remainder, not device)
//	L=300 -> cover 288 -> panels [128,128,32]
//
// The multi-panel walk exercises the crossover the change exists for: panel p+1
// reads its Kpost/V prefix on the device, from the rows panel p appended, with no
// host prefix re-upload in between.
//
// GPU-gated: t.Skip on a host without a Metal device, a real GPU check on this
// Apple M3 Pro.
func TestProjectionGraphQwenDeviceKVByteParityL100L300(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const input, nH, nKV, hd, rotary = 256, 2, 1, 32, 16
	const qwidth, kvwidth = nH * hd, nKV * hd
	scale, qkEps := float32(1/math.Sqrt(float64(hd))), float32(1e-6)

	// productionPanels mirrors the walk owner's panelizer for a prompt that is
	// already panel-quantum aligned. It asserts the two production constants it
	// depends on so a future widening of PromptPanelMaxTokens (or a change of the
	// 32 quantum) turns this test red instead of silently testing a stale shape.
	const panelQuantum = 32
	productionPanels := func(t *testing.T, promptLen int) (cover int, panels []int) {
		t.Helper()
		if PromptPanelMaxTokens != 128 {
			t.Fatalf("PromptPanelMaxTokens=%d, witness pinned to the production 128", PromptPanelMaxTokens)
		}
		cover = (promptLen / panelQuantum) * panelQuantum
		for start := 0; start < cover; start += PromptPanelMaxTokens {
			rows := PromptPanelMaxTokens
			if start+rows > cover {
				rows = cover - start
			}
			panels = append(panels, rows)
		}
		return cover, panels
	}

	// Weight set shared by both arms. One full-attention layer's worth of
	// q|gate, k and v projections.
	qgateWeight := UploadQ4K(q4kTestRaw(2*qwidth, input, 1308781), 2*qwidth, input)
	kWeight := UploadQ4K(q4kTestRaw(kvwidth, input, 1308782), kvwidth, input)
	vWeight := UploadQ4K(q4kTestRaw(kvwidth, input, 1308783), kvwidth, input)
	if qgateWeight == nil || kWeight == nil || vWeight == nil {
		t.Fatal("device-KV L100/L300 parity Q4_K upload")
	}
	qnorm, knorm := make([]float32, hd), make([]float32, hd)
	for i := range qnorm {
		qnorm[i], knorm[i] = 0.87+float32(i%7)*0.05, 0.91+float32(i%5)*0.04
	}
	// ropePhases builds deterministic per-row cos/sin for `rows` positions starting
	// at absolute position `base`.
	ropePhases := func(rows, base int) ([]float32, []float32) {
		cosv, sinv := make([]float32, rows*(rotary/2)), make([]float32, rows*(rotary/2))
		for row := 0; row < rows; row++ {
			for dim := 0; dim < rotary/2; dim++ {
				angle := float64((base+row+1)*(dim+1)) * 0.003
				cosv[row*(rotary/2)+dim], sinv[row*(rotary/2)+dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
			}
		}
		return cosv, sinv
	}
	// panelX is the panel's activation rows, seeded by (promptLen, panelIndex) so
	// each panel carries distinct tokens the way a real prompt does.
	panelX := func(promptLen, panel int, rows int) []float32 {
		return q4kTestVector(rows*input, int64(1308790+promptLen+panel))
	}
	// hostWalk runs the historical host-append path: every panel seeds the host
	// prefix accumulated so far (Kpost/V only — Kraw has no attention role) and its
	// readback rows are appended, exactly as Qwen35MetalForwardSequence does.
	hostWalk := func(t *testing.T, panels []int) (kraw, kpost, v []float32) {
		t.Helper()
		pos := 0
		for i, rows := range panels {
			g, err := BeginProjectionGraph(panelX(0, i, rows), nil, nil, rows, input)
			if err != nil {
				t.Fatal(err)
			}
			q, err := g.EncodeQ4K(qgateWeight)
			if err != nil {
				t.Fatal(err)
			}
			k, err := g.EncodeQ4K(kWeight)
			if err != nil {
				t.Fatal(err)
			}
			vres, err := g.EncodeQ4K(vWeight)
			if err != nil {
				t.Fatal(err)
			}
			q2, gate, err := g.SplitGatedQ(q, qwidth, hd)
			if err != nil {
				t.Fatal(err)
			}
			cosv, sinv := ropePhases(rows, pos)
			att, err := g.FullAttention(q2, k, vres, gate, qnorm, knorm, cosv, sinv,
				kpost[:pos*kvwidth], v[:pos*kvwidth], pos, nH, nKV, hd, rotary, scale, qkEps, true, true)
			if err != nil {
				t.Fatal(err)
			}
			out, _, err := g.FinishRead(att.KRaw, att.KPost, att.V)
			if err != nil {
				t.Fatal(err)
			}
			g.Free()
			kraw = append(kraw, out[0]...)
			kpost = append(kpost, out[1]...)
			v = append(v, out[2]...)
			pos += rows
		}
		return kraw, kpost, v
	}
	// deviceWalk runs the SAME panels through one persistent DeviceKV pair, seeding
	// nothing (a fresh walk appends every row on the device) and downloading the
	// pair's rows ONCE at the end — the exact shape ReconcileDeviceKV implements.
	deviceWalk := func(t *testing.T, panels []int, cover int) (kraw, kpost, v []float32) {
		t.Helper()
		kv := NewDeviceKV(1, cover, kvwidth)
		if kv == nil {
			t.Fatal("NewDeviceKV returned nil")
		}
		defer kv.Close()
		pos := 0
		for i, rows := range panels {
			g, err := BeginProjectionGraph(panelX(0, i, rows), nil, nil, rows, input)
			if err != nil {
				t.Fatal(err)
			}
			q, err := g.EncodeQ4K(qgateWeight)
			if err != nil {
				t.Fatal(err)
			}
			k, err := g.EncodeQ4K(kWeight)
			if err != nil {
				t.Fatal(err)
			}
			vres, err := g.EncodeQ4K(vWeight)
			if err != nil {
				t.Fatal(err)
			}
			q2, gate, err := g.SplitGatedQ(q, qwidth, hd)
			if err != nil {
				t.Fatal(err)
			}
			cosv, sinv := ropePhases(rows, pos)
			att, err := g.FullAttentionDevice(q2, k, vres, gate, kv, 0, qnorm, knorm, cosv, sinv, pos, nH, nKV, hd, rotary, scale, qkEps, true, true)
			if err != nil {
				t.Fatal(err)
			}
			// The production panel always commits its graph with a terminal read
			// (Qwen35MetalForwardSequence's FinishRead of the hidden); the device
			// append blit rides that same command buffer. Read this panel's output
			// back so the append is committed before the next panel reads the pair.
			if _, _, err := g.FinishRead(att.Output); err != nil {
				t.Fatal(err)
			}
			g.Free()
			pos += rows
		}
		// One terminal download per side, exactly the rows the host cache gains.
		download := func(side int) []float32 {
			dst := make([]float32, cover*kvwidth)
			if err := kv.DownloadRegion(side, 0, dst); err != nil {
				t.Fatalf("download side %d: %v", side, err)
			}
			return dst
		}
		return download(0), download(1), download(2)
	}

	for _, tc := range []struct {
		name      string
		promptLen int
		wantCover int
		wantPanel []int
	}{
		{name: "L100", promptLen: 100, wantCover: 96, wantPanel: []int{96}},
		{name: "L300", promptLen: 300, wantCover: 288, wantPanel: []int{128, 128, 32}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cover, panels := productionPanels(t, tc.promptLen)
			if cover != tc.wantCover {
				t.Fatalf("cover=%d, want %d (production panelizer for L=%d)", cover, tc.wantCover, tc.promptLen)
			}
			if len(panels) != len(tc.wantPanel) {
				t.Fatalf("panels=%v, want %v", panels, tc.wantPanel)
			}
			sum := 0
			for i, rows := range panels {
				if rows != tc.wantPanel[i] {
					t.Fatalf("panels=%v, want %v", panels, tc.wantPanel)
				}
				sum += rows
			}
			if sum != cover {
				t.Fatalf("panels sum to %d, want cover %d", sum, cover)
			}

			wantKRaw, wantKPost, wantV := hostWalk(t, panels)
			gotKRaw, gotKPost, gotV := deviceWalk(t, panels, cover)

			if len(gotKPost) != cover*kvwidth || len(gotV) != cover*kvwidth || len(gotKRaw) != cover*kvwidth {
				t.Fatalf("device cache rows=%d/%d/%d, want 3x%d", len(gotKRaw), len(gotKPost), len(gotV), cover*kvwidth)
			}
			// Exact byte parity. Any float drift is a real transport defect, not a
			// tolerance question (see the file comment), so assert float32 identity.
			compare := func(name string, got, want []float32) {
				t.Helper()
				if len(got) != len(want) {
					t.Fatalf("%s rows: got %d want %d", name, len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%s[%d]: device=%v host=%v (not byte-identical)", name, i, got[i], want[i])
					}
				}
			}
			compare("Kraw", gotKRaw, wantKRaw)
			compare("Kpost", gotKPost, wantKPost)
			compare("V", gotV, wantV)

			// The host cache the decode path consumes has exactly `cover` rows per
			// side after reconcile — the device pair did not drop or duplicate a row.
			if wantCoverRows := cover * kvwidth; len(gotKPost) != wantCoverRows {
				t.Fatalf("reconciled host cache rows=%d, want %d", len(gotKPost), wantCoverRows)
			}
		})
	}
}
