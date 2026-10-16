package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"
)

const v41ParityPromptTokens = 512

func v41LogitsDigest(row []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, v := range row {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func v41ParityPrompt(vocab, n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*7 + 3) % vocab
	}
	return ids
}

// fak-test:runtime slow est=60s lane=default
func TestV41GroupedPrefillMatchesTokenMajor512(t *testing.T) {
	t.Cleanup(func() { v41ForceTokenMajor = false })
	grouped, _ := v41TierOnlyBudgetModel(t, 0)
	ids := v41ParityPrompt(grouped.Cfg.VocabSize, v41ParityPromptTokens)

	v41ForceTokenMajor = false
	g, err := grouped.forwardV41WithHeadSelection(ids, nil, true)
	if err != nil {
		t.Fatalf("grouped prefill: %v", err)
	}
	tokenMajor, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = true
	tm, err := tokenMajor.forwardV41WithHeadSelection(ids, nil, true)
	v41ForceTokenMajor = false
	if err != nil {
		t.Fatalf("token-major prefill: %v", err)
	}
	gRow, tmRow := lastLogits(g), lastLogits(tm)
	if len(gRow) == 0 || len(gRow) != len(tmRow) {
		t.Fatalf("logit widths grouped=%d token-major=%d", len(gRow), len(tmRow))
	}
	gd, td := v41LogitsDigest(gRow), v41LogitsDigest(tmRow)
	if gd != td {
		t.Fatalf("512-token last-logits digest grouped=%s token-major=%s", gd, td)
	}
	gReads, tmReads := grouped.expertCheckpoint.Stats().Reads, tokenMajor.expertCheckpoint.Stats().Reads
	if gReads == 0 || gReads > tmReads {
		t.Fatalf("routed projection reads grouped=%d token-major=%d", gReads, tmReads)
	}
	t.Logf("digest=%s reads grouped=%d token-major=%d", gd, gReads, tmReads)
}

// fak-test:runtime medium est=5s lane=default
func TestV41SuffixPrefillMatchesColdPrefill512(t *testing.T) {
	const prefixLen, tol = v41ParityPromptTokens / 2, 1e-5
	m := v41IncrementalPlainModel(t, 2)
	m.Cfg.Window = []int{64, 64}
	ids := v41ParityPrompt(m.Cfg.VocabSize, v41ParityPromptTokens)

	cold := m.NewSession()
	want := cold.Prefill(ids)

	warm := m.NewSession()
	warm.Prefill(ids[:prefixLen])
	if !warm.v41IncrementalEligible() {
		t.Fatal("seeded session is not incrementally eligible")
	}
	read, restore := v41SuffixPrefillCalls(t)
	got := warm.Prefill(ids[prefixLen:])
	calls := read()
	restore()
	if calls != len(ids)-prefixLen {
		t.Fatalf("suffix advances = %d, want %d", calls, len(ids)-prefixLen)
	}
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("logit widths suffix=%d cold=%d", len(got), len(want))
	}
	maxDelta := 0.0
	for j := range want {
		d := math.Abs(float64(got[j] - want[j]))
		if math.IsNaN(d) || d > tol {
			t.Fatalf("logits[%d] suffix=%g cold=%g", j, got[j], want[j])
		}
		maxDelta = math.Max(maxDelta, d)
	}
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("argmax suffix=%d cold=%d", argmaxF32(got), argmaxF32(want))
	}
	t.Logf("cold=%s suffix=%s max_abs_delta=%g", v41LogitsDigest(want), v41LogitsDigest(got), maxDelta)
}
