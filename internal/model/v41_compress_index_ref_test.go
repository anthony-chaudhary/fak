package model

// v41_compress_index_ref_test.go - the REFERENCE-EXECUTED pinned-vector arm for
// the DeepSeek V4.1 CED/CSA2 compressor and lightning indexer (fak#13151).
//
// Why this file exists. Before it, the only numeric witnesses for these two
// stages were (a) the in-package port's own hand-computed constants and (b) the
// independent transcription in `internal/model/v41`, which is a sibling port
// rather than the pinned artifact itself. Both are useful, but neither produces
// its expected values by EXECUTING the pinned reference: the fixture bytes and
// the code under test descend from the same reading of the same equation, so a
// correlated mis-read of the official equations would pass silently. This file
// closes that gap for the compressor and the indexer.
//
// What "reference-executed" means here. The expected values in
// testdata/v41_oracle/compress_index_ref.json are produced by v41RefExecFixture,
// a from-scratch scalar transcription of the pinned reference
// (deepseek-ai/DeepSeek-V4.1-Flash inference/model.py Compressor +
// lightning indexer, revision
// dba1be0a40aa45a94ad051997016db3960a90277, MIT) that deliberately shares NO
// helper with the production path under test, nor with the `model/v41`
// transcription: it re-implements the BF16 round-to-nearest-even cast, the
// stable-softmax pooling, the learned-RMSNorm tail, the rectified per-head
// reduction and the block/top-k selection from the equation text, in float64.
// The test re-derives the fixture at run time and compares the PRODUCTION
// V41CompressorPool / V41IndexerScore / V41SelectCandidateBlocks /
// V41SelectIndexRows against the independently executed reference numbers.
//
// Provenance. Revision: dba1be0a40aa45a94ad051997016db3960a90277. Equation
// source: the Compressor and lightning-indexer stages of the reference
// inference/model.py at that revision. The fixture records that revision so a
// drift is caught by the revision check, exactly like the sibling oracle.
//
// Scope honesty. Test-only; no production behavior changes; no checkpoint, no
// hardware, no full-model generation. It exercises the two stages in isolation,
// at the boundaries v41_compress_index.go declares (projected inputs in,
// pooled latent / selected rows out).

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const v41CompressIndexRefPath = "testdata/v41_oracle/compress_index_ref.json"

// v41CompressIndexRefFixture is the checked-in reference-executed fixture:
// one compressor case (pooled latent through the BF16 + learned-RMSNorm tail)
// and one indexer case (rectified scores, candidate mask, selected rows).
type v41CompressIndexRefFixture struct {
	Revision string `json:"revision"`

	// Compressor case: ratio, width, per-position projected KV/score rows, the
	// learned norm weight, eps, and the reference-executed pooled-latent output
	// for every emitted group.
	Compress struct {
		Ratio       int         `json:"ratio"`
		Width       int         `json:"width"`
		NormWeight  []float32   `json:"norm_weight"`
		Eps         float32     `json:"eps"`
		ProjectedKV [][]float32 `json:"projected_kv"`
		Score       [][]float32 `json:"projected_score"`
		Pooled      [][]float32 `json:"pooled"`
	} `json:"compress"`

	// Indexer case: geometry, projected query/keys/head weights, and the
	// reference-executed rectified score vector, candidate mask and rows.
	Index struct {
		NHeads      int       `json:"n_heads"`
		HeadDim     int       `json:"head_dim"`
		CompressLen int       `json:"compress_len"`
		TopKBlocks  int       `json:"top_k_blocks"`
		BlockSize   int       `json:"block_size"`
		TopK        int       `json:"top_k"`
		Offset      int       `json:"offset"`
		Q           []float32 `json:"q"`
		Keys        []float32 `json:"keys"`
		Weights     []float32 `json:"weights"`
		Score       []float32 `json:"score"`
		Mask        []bool    `json:"mask"`
		Rows        []int32   `json:"rows"`
	} `json:"index"`
}

// v41RefBF16 is an independent round-to-nearest-even BF16 cast written in
// float64, matching the reference dtype boundary. It shares no code with
// v41RoundBF16 / v41BF16.
func v41RefBF16(v float64) float64 {
	f := float32(v)
	bits := math.Float32bits(f)
	bits += 0x7fff + ((bits >> 16) & 1)
	return float64(math.Float32frombits(bits & 0xffff0000))
}

// v41RefCompressorGroups executes the reference compressor pooling on the
// projected rows: per-dimension stable softmax over the token group, weighted
// sum, then the BF16 cast + FP32 RMSNorm + learned weight + BF16 cast tail.
// Written from the equation text in float64; reuses nothing from production.
func v41RefCompressorGroups(ratio, width int, kv, score [][]float32, normWeight []float32, eps float64) [][]float32 {
	var out [][]float32
	for start := 0; start+ratio <= len(kv); start += ratio {
		pooled := make([]float64, width)
		for dim := 0; dim < width; dim++ {
			maxScore := float64(score[start][dim])
			for t := 1; t < ratio; t++ {
				if s := float64(score[start+t][dim]); s > maxScore {
					maxScore = s
				}
			}
			weights := make([]float64, ratio)
			var denom float64
			for t := 0; t < ratio; t++ {
				weights[t] = math.Exp(float64(score[start+t][dim]) - maxScore)
				denom += weights[t]
			}
			var acc float64
			for t := 0; t < ratio; t++ {
				acc += float64(kv[start+t][dim]) * (weights[t] / denom)
			}
			pooled[dim] = acc
		}
		var meanSquare float64
		for dim := 0; dim < width; dim++ {
			pooled[dim] = v41RefBF16(pooled[dim])
			meanSquare += pooled[dim] * pooled[dim]
		}
		rstd := 1 / math.Sqrt(meanSquare/float64(width)+eps)
		row := make([]float32, width)
		for dim := 0; dim < width; dim++ {
			row[dim] = float32(v41RefBF16(pooled[dim] * rstd * float64(normWeight[dim])))
		}
		out = append(out, row)
	}
	return out
}

// v41RefIndexerScoreExec executes the reference lightning-indexer reduction:
// per-head dot, per-element ReLU, then the learned per-head weighted sum.
// Written from the equation text in float64; reuses nothing from production.
func v41RefIndexerScoreExec(q, keys []float32, weights []float32, nHeads, headDim, compressLen int) []float32 {
	out := make([]float32, compressLen)
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
		out[t] = float32(acc)
	}
	return out
}

// v41RefCandidateMaskExec executes the reference block selection: top-k blocks
// by block max, with the newest block pinned on ties. Written from the equation
// text; reuses nothing from production.
func v41RefCandidateMaskExec(scores []float32, compressLen, topKBlocks, blockSize int) []bool {
	if topKBlocks <= 0 || blockSize <= 0 || compressLen <= 0 {
		return nil
	}
	nBlocks := (compressLen + blockSize - 1) / blockSize
	type blockScore struct {
		idx int
		max float64
	}
	blocks := make([]blockScore, 0, nBlocks)
	for b := 0; b < nBlocks; b++ {
		best := math.Inf(-1)
		start := b * blockSize
		end := start + blockSize
		if end > compressLen {
			end = compressLen
		}
		for t := start; t < end; t++ {
			if s := float64(scores[t]); s > best {
				best = s
			}
		}
		blocks = append(blocks, blockScore{idx: b, max: best})
	}
	for i := 1; i < len(blocks); i++ {
		for j := i; j > 0 && (blocks[j].max > blocks[j-1].max ||
			(blocks[j].max == blocks[j-1].max && blocks[j].idx > blocks[j-1].idx)); j-- {
			blocks[j], blocks[j-1] = blocks[j-1], blocks[j]
		}
	}
	keep := topKBlocks
	if keep > len(blocks) {
		keep = len(blocks)
	}
	mask := make([]bool, compressLen)
	for b := 0; b < keep; b++ {
		start := blocks[b].idx * blockSize
		end := start + blockSize
		if end > compressLen {
			end = compressLen
		}
		for t := start; t < end; t++ {
			mask[t] = true
		}
	}
	return mask
}

// v41RefIndexRowsExec executes the reference final row selection: the top-k
// causal-visible positions by score, returned in position order and shifted by
// offset, padded with -1 for invisible slots. Written from the equation text;
// reuses nothing from production.
func v41RefIndexRowsExec(scores []float32, compressLen, topK, offset int, mask []bool) []int32 {
	if topK < 0 {
		return nil
	}
	type ranked struct {
		pos   int
		score float64
	}
	var visible []ranked
	for t := 0; t < compressLen; t++ {
		vis := true
		if mask != nil {
			vis = mask[t]
		}
		if vis {
			visible = append(visible, ranked{pos: t, score: float64(scores[t])})
		}
	}
	for i := 1; i < len(visible); i++ {
		for j := i; j > 0 && (visible[j].score > visible[j-1].score ||
			(visible[j].score == visible[j-1].score && visible[j].pos > visible[j-1].pos)); j-- {
			visible[j], visible[j-1] = visible[j-1], visible[j]
		}
	}
	keep := topK
	if keep > len(visible) {
		keep = len(visible)
	}
	chosen := make([]int, 0, keep)
	for i := 0; i < keep; i++ {
		chosen = append(chosen, visible[i].pos)
	}
	for i := 1; i < len(chosen); i++ {
		for j := i; j > 0 && chosen[j] < chosen[j-1]; j-- {
			chosen[j], chosen[j-1] = chosen[j-1], chosen[j]
		}
	}
	out := make([]int32, 0, len(chosen)+(topK-len(chosen)))
	for i := 0; i < topK-len(chosen); i++ {
		out = append(out, -1)
	}
	for _, pos := range chosen {
		out = append(out, int32(pos+offset))
	}
	return out
}

// v41RefExecFixture builds the deterministic inputs and executes the reference
// transcription above to produce every expected value in the fixture. It is the
// generator the checked-in JSON is produced from; the test calls it again and
// also checks the checked-in bytes, so a hand-edited or stale fixture is caught.
func v41RefExecFixture() v41CompressIndexRefFixture {
	f := v41CompressIndexRefFixture{Revision: v41OracleRevision}

	// ---- compressor: ratio 2, width 3, four positions -> two groups ----
	f.Compress.Ratio = 2
	f.Compress.Width = 3
	f.Compress.Eps = 1e-6
	f.Compress.NormWeight = []float32{1.25, 0.75, 1.5}
	f.Compress.ProjectedKV = [][]float32{
		{1.003, 2.007, -0.5},
		{3.011, 4.015, 0.25},
		{-2.5, 1.5, 3.25},
		{0.5, -1.25, 2.75},
	}
	f.Compress.Score = [][]float32{
		{0, 1, -1},
		{2, 0, 1},
		{-1, 3, 0},
		{0.5, -0.5, 2},
	}
	f.Compress.Pooled = v41RefCompressorGroups(
		f.Compress.Ratio, f.Compress.Width,
		f.Compress.ProjectedKV, f.Compress.Score,
		f.Compress.NormWeight, float64(f.Compress.Eps))

	// ---- indexer: 2 heads x width 2, 5 compressed positions, 2 blocks of 3 ----
	f.Index.NHeads = 2
	f.Index.HeadDim = 2
	f.Index.CompressLen = 5
	f.Index.TopKBlocks = 1
	f.Index.BlockSize = 3
	f.Index.TopK = 3
	f.Index.Offset = 100
	f.Index.Q = []float32{1, 1, 1, -1}
	f.Index.Keys = []float32{
		1, 1,
		1, -1,
		2, 2,
		3, -1,
		-1, -2,
	}
	f.Index.Weights = []float32{0.5, 2.0}
	f.Index.Score = v41RefIndexerScoreExec(
		f.Index.Q, f.Index.Keys, f.Index.Weights,
		f.Index.NHeads, f.Index.HeadDim, f.Index.CompressLen)
	f.Index.Mask = v41RefCandidateMaskExec(
		f.Index.Score, f.Index.CompressLen, f.Index.TopKBlocks, f.Index.BlockSize)
	f.Index.Rows = v41RefIndexRowsExec(
		f.Index.Score, f.Index.CompressLen, f.Index.TopK, f.Index.Offset, f.Index.Mask)
	return f
}

// TestV41CompressIndexReferenceExecuted is the #13151 reference-executed arm for
// the CED/CSA2 compressor and the lightning indexer. It pins the production
// stages against values produced by executing an independent transcription of
// the pinned reference, with a revision check and non-vacuity controls so a
// correlated mis-read or a stale fixture cannot pass.
func TestV41CompressIndexReferenceExecuted(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(v41CompressIndexRefPath))
	if err != nil {
		t.Fatalf("read reference-executed fixture: %v", err)
	}
	var want v41CompressIndexRefFixture
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("decode reference-executed fixture: %v", err)
	}
	if want.Revision != v41OracleRevision {
		t.Fatalf("fixture revision %q, want pinned %q", want.Revision, v41OracleRevision)
	}

	// Re-execute the reference: the checked-in bytes must be exactly what the
	// pinned-equation transcription produces now.
	got := v41RefExecFixture()
	if !v41RefFixtureEqual(t, got, want) {
		t.Fatal("fixture is stale: re-executing the pinned reference no longer reproduces it")
	}

	t.Run("compressor matches the reference-executed pooled latent", func(t *testing.T) {
		c, err := NewV41CompressorPool(want.Compress.Ratio, want.Compress.Width)
		if err != nil {
			t.Fatal(err)
		}
		var emitted [][]float32
		for pos := range want.Compress.ProjectedKV {
			pooled, ok, err := c.PushNormalized(
				pos, want.Compress.ProjectedKV[pos], want.Compress.Score[pos],
				want.Compress.NormWeight, want.Compress.Eps)
			if err != nil {
				t.Fatalf("PushNormalized(%d): %v", pos, err)
			}
			if ok {
				emitted = append(emitted, pooled)
			}
		}
		if len(emitted) != len(want.Compress.Pooled) {
			t.Fatalf("emitted %d groups, reference produced %d", len(emitted), len(want.Compress.Pooled))
		}
		for g := range emitted {
			for dim := range emitted[g] {
				if emitted[g][dim] != want.Compress.Pooled[g][dim] {
					t.Fatalf("group %d dim %d = %g, reference %g",
						g, dim, emitted[g][dim], want.Compress.Pooled[g][dim])
				}
			}
		}
	})

	t.Run("indexer score matches the reference-executed reduction", func(t *testing.T) {
		score, err := V41IndexerScore(
			want.Index.Q, want.Index.Keys, want.Index.Weights,
			want.Index.NHeads, want.Index.HeadDim, want.Index.CompressLen)
		if err != nil {
			t.Fatal(err)
		}
		for i := range score {
			if score[i] != want.Index.Score[i] {
				t.Fatalf("score[%d] = %g, reference %g", i, score[i], want.Index.Score[i])
			}
		}
	})

	t.Run("indexer selection matches the reference-executed mask and rows", func(t *testing.T) {
		mask, err := V41SelectCandidateBlocks(
			want.Index.Score, want.Index.CompressLen, want.Index.TopKBlocks, want.Index.BlockSize)
		if err != nil {
			t.Fatal(err)
		}
		if !v41RefMaskEqual(mask, want.Index.Mask) {
			t.Fatalf("candidate mask = %v, reference %v", mask, want.Index.Mask)
		}
		rows, err := V41SelectIndexRows(
			want.Index.Score, want.Index.CompressLen, want.Index.TopK, want.Index.Offset, mask)
		if err != nil {
			t.Fatal(err)
		}
		if !v41RefRowsEqual(rows, want.Index.Rows) {
			t.Fatalf("selected rows = %v, reference %v", rows, want.Index.Rows)
		}
	})

	t.Run("reference arm is non-vacuous", func(t *testing.T) {
		movedNorm := v41RefCompressorGroups(
			want.Compress.Ratio, want.Compress.Width,
			want.Compress.ProjectedKV, want.Compress.Score,
			[]float32{9.5, 0.75, 1.5}, float64(want.Compress.Eps))
		sameNorm := true
		for g := range movedNorm {
			for dim := range movedNorm[g] {
				if movedNorm[g][dim] != want.Compress.Pooled[g][dim] {
					sameNorm = false
				}
			}
		}
		if sameNorm {
			t.Fatal("changing the norm weight did not move the pooled latent; fixture is vacuous")
		}

		// Selection non-vacuity needs MORE visible positions than the top-k, so
		// a score change can actually reorder the picked set. With the fixture's
		// mask {3,4} and topK=1, raising position 4 above position 3 must flip
		// the single pick from 103 to 104.
		baseRows := v41RefIndexRowsExec(
			want.Index.Score, want.Index.CompressLen, 1, want.Index.Offset, want.Index.Mask)
		if !v41RefRowsEqual(baseRows, []int32{103}) {
			t.Fatalf("topK=1 baseline rows = %v, want [103]", baseRows)
		}
		flipped := make([]float32, len(want.Index.Score))
		copy(flipped, want.Index.Score)
		flipped[4] = want.Index.Score[4] + 1000
		flippedRows := v41RefIndexRowsExec(
			flipped, want.Index.CompressLen, 1, want.Index.Offset, want.Index.Mask)
		if v41RefRowsEqual(flippedRows, baseRows) {
			t.Fatal("raising a competing score did not move the selected rows; fixture is vacuous")
		}
	})
}

func v41RefFixtureEqual(t *testing.T, a, b v41CompressIndexRefFixture) bool {
	t.Helper()
	if a.Revision != b.Revision ||
		a.Compress.Ratio != b.Compress.Ratio || a.Compress.Width != b.Compress.Width ||
		a.Index.NHeads != b.Index.NHeads || a.Index.CompressLen != b.Index.CompressLen {
		return false
	}
	if !v41RefF32Equal(a.Compress.NormWeight, b.Compress.NormWeight) ||
		!v41RefF32Equal(a.Index.Score, b.Index.Score) ||
		!v41RefF32Equal(a.Index.Q, b.Index.Q) ||
		!v41RefF32Equal(a.Index.Keys, b.Index.Keys) ||
		!v41RefF32Equal(a.Index.Weights, b.Index.Weights) {
		return false
	}
	if !v41RefRowsEqual(a.Index.Rows, b.Index.Rows) || !v41RefMaskEqual(a.Index.Mask, b.Index.Mask) {
		return false
	}
	if len(a.Compress.Pooled) != len(b.Compress.Pooled) {
		return false
	}
	for i := range a.Compress.Pooled {
		if !v41RefF32Equal(a.Compress.Pooled[i], b.Compress.Pooled[i]) {
			return false
		}
	}
	return true
}

func v41RefF32Equal(a, b []float32) bool {
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

func v41RefMaskEqual(a, b []bool) bool {
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

func v41RefRowsEqual(a, b []int32) bool {
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
