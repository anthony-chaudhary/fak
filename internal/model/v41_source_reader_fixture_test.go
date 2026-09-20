package model

// v41_source_reader_fixture_test.go - the deterministic source/reader fixture and
// the independent scalar expected values for
// TestV41SourceReaderNumericOracle (fak#13323).
//
// Why this file exists. Before it, the only numeric witnesses for the V4.1
// compressed-attention SOURCE step (CED/CSA2 pooling then lightning-index
// selection) and the READER step (a later layer resolving the source's published
// rows) were (a) each stage in isolation against the pinned reference
// (v41_compress_index_ref_test.go, fak#13151) and (b) the reduced token-path
// parity arm, which exercises a source as the LAST layer so no later reader ever
// resolves it (v41_parity_test.go:306-373). Neither drives a source followed by a
// reader through the production publication/consumption seam. This file closes
// that gap: it fixes a tiny ratio-2 source/reader case and computes every expected
// value by INDEPENDENTLY TRANSCRIBED SCALAR MATH, so a correlated mis-read of the
// same equations in both the production path and the fixture cannot pass
// silently.
//
// Independence discipline. Every expected value below is produced by the
// from-scratch scalar functions in this file (v41SRRefProject, v41SRRefPool,
// v41SRRefScores, v41SRRefSelectRows). They call NO production helper: not
// v41CompressedRows, not matRows, not V41IndexerScore, not v41RoundBF16, not
// rmsnormCfg, and they never consume a production-generated intermediate. They
// re-implement the projection dot, the pinned compressor pooling and the
// lightning-indexer reduction from the equation text in float64.
//
// Provenance. Target: deepseek-ai/DeepSeek-V4.1-Flash inference/model.py
// (Compressor + lightning indexer), revision
// dba1be0a40aa45a94ad051997016db3960a90277 (MIT), the same pinned revision as the
// sibling oracles (v41_compress_index_ref_test.go, v41_oracle_test.go).
//
// Scope honesty. Test-only; no production behavior changes; no checkpoint, no
// hardware, no full-model generation. It exercises the source publication and
// reader reuse seams at the boundaries v41_forward.go / v41_compress_index.go
// declare.

import (
	"math"
	"testing"
)

// v41SRFixture is the deterministic source/reader fixture. The reduced config
// fixes HiddenSize=64 and HeadDim=32, so the compressor latent width is 32 and
// every projected row is 32 wide, projected from a 64-wide hidden input.
type v41SRFixture struct {
	Ratio      int // compressor group width (2: one compression boundary)
	Width      int // compressor latent width (v41CompressorWidth == HeadDim)
	Hidden     int // model hidden width (the projection input width)
	Eps        float64
	NormWeight []float32 // length Width

	// Per-position hidden inputs (length Hidden), oldest first. Distinct so a
	// wrong row identity is observable.
	Inputs [][]float32

	// Projection weights (row-major [Width x Hidden]) that turn each hidden input
	// into the projected KV and score rows the compressor pools. They are
	// explicit so the reference can reproduce the same projection independently.
	WKV   []float32
	WGate []float32

	// Source indexer inputs: one head of width Width.
	SrcQ       []float32
	SrcWeights []float32 // per-head reduction weight, length 1
	SrcTopK    int
	SrcOffset  int

	// Reader indexer inputs over the SAME compressed key stream, but its own
	// query and per-head weight so the reader's selection is distinguishable
	// from the source's.
	ReaderQ       []float32
	ReaderWeights []float32
	ReaderTopK    int
}

// v41SRFixtureValues builds the fixed fixture. The hidden inputs carry
// non-symmetric values so the pooling order and the softmax weights both matter.
func v41SRFixtureValues() v41SRFixture {
	H := 64
	W := 32
	// A deterministic, non-degenerate projection: weight[r][c] = ((r+1)*(c+2) mod 7 - 3) / 8.
	proj := func(seed int) []float32 {
		out := make([]float32, W*H)
		for r := 0; r < W; r++ {
			for c := 0; c < H; c++ {
				v := ((r+seed+1)*(c+2))%7 - 3
				out[r*H+c] = float32(v) / 8
			}
		}
		return out
	}
	mkInput := func(base float64) []float32 {
		in := make([]float32, H)
		for c := 0; c < H; c++ {
			in[c] = float32(math.Sin(base + float64(c)*0.17))
		}
		return in
	}
	return v41SRFixture{
		Ratio:      2,
		Width:      W,
		Hidden:     H,
		Eps:        1e-6,
		NormWeight: []float32{1.25, 0.75, 1.5, 1.1, 0.9, 1.35, 1.05, 0.8, 1.4, 0.95, 1.2, 1.15, 0.85, 1.3, 1.0, 1.25, 0.7, 1.45, 1.05, 0.9, 1.15, 1.3, 0.8, 1.2, 1.1, 1.35, 0.75, 1.25, 1.0, 1.4, 0.95, 1.05},
		Inputs: [][]float32{
			mkInput(0.1),
			mkInput(0.7),
			mkInput(1.3),
			mkInput(1.9),
		},
		WKV:   proj(0),
		WGate: proj(1),
		// SrcQ / ReaderQ are length Width (IndexHeadDim == Width).
		SrcQ:          v41SRMakeVec(W, 0.31),
		SrcWeights:    []float32{1.3},
		SrcTopK:       2,
		SrcOffset:     0,
		ReaderQ:       v41SRMakeVec(W, 0.83),
		ReaderWeights: []float32{0.8},
		ReaderTopK:    2,
	}
}

// v41SRMakeVec builds a deterministic non-degenerate vector of length n.
func v41SRMakeVec(n int, phase float64) []float32 {
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float32(math.Cos(phase + float64(i)*0.23))
	}
	return out
}

// v41SRRefBF16 is an independent round-to-nearest-even BF16 cast in float64,
// written from the dtype boundary; it shares no code with v41RoundBF16 /
// v41RefBF16.
func v41SRRefBF16(v float64) float64 {
	f := float32(v)
	bits := math.Float32bits(f)
	bits += 0x7fff + ((bits >> 16) & 1)
	return float64(math.Float32frombits(bits & 0xffff0000))
}

// v41SRRefProject independently reproduces one row of a row-major [out x in]
// projection: a float64 dot of the weight row with the input. It shares no code
// with matRows / fdot.
func v41SRRefProject(w, x []float32, out, in int, row int) []float64 {
	y := make([]float64, out)
	for o := 0; o < out; o++ {
		var acc float64
		for i := 0; i < in; i++ {
			acc += float64(w[o*in+i]) * float64(x[i])
		}
		y[o] = acc
	}
	return y
}

// v41SRRefPool executes the reference compressor pooling independently over the
// (already projected) KV/score rows: for every complete group of `ratio`
// positions, a per-dimension stable softmax over the group's scores, the weighted
// sum of the KV rows, then the BF16 cast + FP32 RMSNorm + learned-weight + BF16
// cast tail. It returns one pooled row per complete group, in causal order.
// Written from the equation text in float64; reuses nothing from production.
func v41SRRefPool(ratio, width int, kv, score [][]float64, normWeight []float32, eps float64) [][]float32 {
	var out [][]float32
	for start := 0; start+ratio <= len(kv); start += ratio {
		pooled := make([]float64, width)
		for dim := 0; dim < width; dim++ {
			maxScore := math.Inf(-1)
			for t := 0; t < ratio; t++ {
				if s := score[start+t][dim]; s > maxScore {
					maxScore = s
				}
			}
			weights := make([]float64, ratio)
			var denom float64
			for t := 0; t < ratio; t++ {
				weights[t] = math.Exp(score[start+t][dim] - maxScore)
				denom += weights[t]
			}
			var acc float64
			for t := 0; t < ratio; t++ {
				acc += kv[start+t][dim] * (weights[t] / denom)
			}
			pooled[dim] = acc
		}
		var meanSquare float64
		for dim := 0; dim < width; dim++ {
			pooled[dim] = v41SRRefBF16(pooled[dim])
			meanSquare += pooled[dim] * pooled[dim]
		}
		rstd := 1 / math.Sqrt(meanSquare/float64(width)+eps)
		row := make([]float32, width)
		for dim := 0; dim < width; dim++ {
			row[dim] = float32(v41SRRefBF16(pooled[dim] * rstd * float64(normWeight[dim])))
		}
		out = append(out, row)
	}
	return out
}

// v41SRRefScores executes the reference lightning-indexer reduction
// independently: for each compressed key, a per-head dot with the query, a
// per-element ReLU, then the learned per-head weighted sum. Written from the
// equation text in float64; reuses nothing from production.
func v41SRRefScores(q, keys []float32, weights []float32, nHeads, headDim, compressLen int) []float64 {
	out := make([]float64, compressLen)
	for t := 0; t < compressLen; t++ {
		var acc float64
		for h := 0; h < nHeads; h++ {
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(q[h*headDim+d]) * float64(keys[t*headDim+d])
			}
			if dot < 0 {
				dot = 0
			}
			acc += dot * float64(weights[h])
		}
		out[t] = acc
	}
	return out
}

// v41SRRefSelectRows executes the reference final row selection independently
// from the equation text: one candidate entry per compressed position; a position
// that is beyond the causal length or excluded by the candidate mask carries the
// invisible score; the best min(topK, positions) entries by score are kept, then
// returned in ascending position order and shifted by offset. An invisible entry
// that survives the top-k is emitted as -1 (the reference's empty-slot marker).
// Reuses nothing from production.
//
// Output length contract. The selection emits exactly min(topK, len(scores))
// entries — it does NOT pad a short stream up to a fixed topK width. Padding a
// short selection to the consumer's fixed stride with -1 is the CONSUMER's job
// (production's v41AttentionIndexList.one normalizes each resolved row to
// plan.TopKWidth). A masked position that survives the top-k is still a selected
// slot and is emitted as -1 here, matching the production selector; it is not
// silently dropped.
func v41SRRefSelectRows(scores []float64, compressLen, topK, offset int, mask []bool) []int32 {
	invisible := math.Inf(-1)
	type ranked struct {
		pos   int
		score float64
	}
	all := make([]ranked, len(scores))
	for t := range scores {
		s := scores[t]
		if t >= compressLen || (mask != nil && !mask[t]) {
			s = invisible
		}
		all[t] = ranked{pos: t, score: s}
	}
	// Stable-ish descending selection: higher score first, ties by lower position.
	for i := 1; i < len(all); i++ {
		for j := i; j > 0 && all[j].score > all[j-1].score; j-- {
			all[j], all[j-1] = all[j-1], all[j]
		}
	}
	keep := topK
	if keep > len(all) {
		keep = len(all)
	}
	if keep < 0 {
		keep = 0
	}
	chosen := all[:keep]
	for i := 1; i < len(chosen); i++ {
		for j := i; j > 0 && chosen[j].pos < chosen[j-1].pos; j-- {
			chosen[j], chosen[j-1] = chosen[j-1], chosen[j]
		}
	}
	out := make([]int32, len(chosen))
	for i, sel := range chosen {
		if sel.pos >= compressLen || sel.score == invisible {
			out[i] = -1
			continue
		}
		out[i] = int32(sel.pos + offset)
	}
	return out
}

// v41SRRefCompressedRows is the full independent source reference: project each
// hidden input through wkv/wgate, then pool. It returns the emitted compressed
// rows for a prefix of `seq` positions.
func v41SRRefCompressedRows(f v41SRFixture, seq int) [][]float32 {
	var kv, score [][]float64
	for pos := 0; pos < seq; pos++ {
		kv = append(kv, v41SRRefProject(f.WKV, f.Inputs[pos], f.Width, f.Hidden, pos))
		score = append(score, v41SRRefProject(f.WGate, f.Inputs[pos], f.Width, f.Hidden, pos))
	}
	return v41SRRefPool(f.Ratio, f.Width, kv, score, f.NormWeight, f.Eps)
}

// v41SRFlattenRows flattens pooled rows into the flat key vector the indexer
// consumes. It is a pure reshape and asserts no arithmetic.
func v41SRFlattenRows(rows [][]float32) []float32 {
	var flat []float32
	for _, row := range rows {
		flat = append(flat, row...)
	}
	return flat
}

// v41SRCloseF32 reports whether two fp32 values agree within an absolute
// tolerance.
func v41SRCloseF32(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// v41SRSameInt32 reports whether two int32 slices are elementwise equal.
func v41SRSameInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// v41SRSameF32 reports whether two fp32 slices are elementwise equal.
func v41SRSameF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// v41SRMustParity fails the test when two int32 row selections disagree.
func v41SRMustParity(t *testing.T, label string, got, want []int32) {
	t.Helper()
	if !v41SRSameInt32(got, want) {
		t.Fatalf("%s: production rows %v, independent reference %v", label, got, want)
	}
}
