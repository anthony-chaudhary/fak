package model

import (
	"fmt"
	"reflect"
	"testing"
)

func v41RequireSameLogits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("%s: logit widths %d and %d", label, len(got), len(want))
	}
	if gd, wd := v41LogitsDigest(got), v41LogitsDigest(want); gd != wd {
		t.Fatalf("%s: last-logits digest %s, want %s", label, gd, wd)
	}
}

func v41RequireCursors(t *testing.T, s *Session, want []int) {
	t.Helper()
	if !reflect.DeepEqual(s.v41Forward.history, want) {
		t.Fatalf("history has %d tokens, want %d", len(s.v41Forward.history), len(want))
	}
	for l, pos := range v41AllNextWindowPos(s.v41Forward) {
		if pos != len(want) {
			t.Fatalf("layer %d cursor=%d, want %d", l, pos, len(want))
		}
	}
}

// v41SuffixPrefill prefills prefix then suffix on a fresh session and returns
// the suffix logits plus how many per-token suffix steps ran.
func v41SuffixPrefill(t *testing.T, m *Model, prefix, suffix []int, ubatch string) (*Session, []float32, int) {
	t.Helper()
	t.Setenv(v41SuffixUbatchEnv, ubatch)
	s := m.NewSession()
	s.Prefill(prefix)
	if !s.v41IncrementalEligible() {
		t.Fatal("seeded session is not incrementally eligible")
	}
	read, restore := v41SuffixPrefillCalls(t)
	got := s.Prefill(suffix)
	calls := read()
	restore()
	return s, got, calls
}

// fak-test:runtime medium est=5s lane=default
func TestV41SuffixUbatchMatchesColdPrefill512(t *testing.T) {
	const prefixLen = v41ParityPromptTokens / 2
	m := v41IncrementalPlainModel(t, 2)
	m.Cfg.Window = []int{64, 64}
	ids := v41ParityPrompt(m.Cfg.VocabSize, v41ParityPromptTokens)

	t.Setenv(v41SuffixUbatchEnv, "")
	want := m.NewSession().Prefill(ids)

	s, got, calls := v41SuffixPrefill(t, m, ids[:prefixLen], ids[prefixLen:], "2")
	if calls != 0 {
		t.Fatalf("suffix ran %d per-token steps, want one batched ubatch", calls)
	}
	v41RequireSameLogits(t, "batched suffix vs cold", got, want)
	v41RequireCursors(t, s, ids)
	t.Logf("digest=%s", v41LogitsDigest(got))
}

// fak-test:runtime medium est=10s lane=default
func TestV41SuffixUbatchMatchesPerTokenSuffix(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("engram/compressed=%t", compressed), func(t *testing.T) {
			m, _ := v41IncrementalEngramFixture(t, compressed)
			prefix, _ := v41IncrementalPrefix(m.Cfg, 7)
			suffix := v41SuffixTail(m.Cfg, len(prefix), 24)

			serial, want, serialCalls := v41SuffixPrefill(t, m, prefix, suffix, "")
			if serialCalls != len(suffix) {
				t.Fatalf("serial suffix ran %d steps, want %d", serialCalls, len(suffix))
			}
			batched, got, calls := v41SuffixPrefill(t, m, prefix, suffix, "2")
			if calls != 0 {
				t.Fatalf("batched suffix ran %d per-token steps, want 0", calls)
			}
			v41RequireSameLogits(t, "batched vs serial suffix", got, want)
			history := append(append([]int(nil), prefix...), suffix...)
			v41RequireCursors(t, batched, history)
			v41RequireCursors(t, serial, history)
			next := v41SuffixTail(m.Cfg, len(history), 1)[0]
			v41RequireSameLogits(t, "step after batched suffix", batched.Step(next), serial.Step(next))
		})
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41SuffixUbatchBoundsRoutedExpertReads(t *testing.T) {
	t.Cleanup(func() { v41TestLayerCacheBudgetOverride = 0 })
	const slabBytes = int64(v41TierWidth) * (v41TierWidth / qkK) * q4kBlockBytes
	prefix := []int{1, 2, 3, 4}
	suffix := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

	reads := func(ubatch string) (int, []float32) {
		m := v41IncrementalExpertFixture(t, false, true)
		v41TestLayerCacheBudgetOverride = slabBytes
		t.Setenv(v41SuffixUbatchEnv, ubatch)
		s := m.NewSession()
		s.Prefill(prefix)
		before := m.expertCheckpoint.Stats().Reads
		got := s.Prefill(suffix)
		return m.expertCheckpoint.Stats().Reads - before, got
	}
	serialReads, want := reads("")
	batchedReads, got := reads("2")
	v41RequireSameLogits(t, "batched vs serial suffix", got, want)
	if batchedReads == 0 || batchedReads >= serialReads {
		t.Fatalf("routed projection reads batched=%d serial=%d, want batched below serial", batchedReads, serialReads)
	}
	t.Logf("routed projection reads batched=%d serial=%d", batchedReads, serialReads)
}

// fak-test:runtime medium est=5s lane=default
func TestV41SuffixUbatchStructuralRefusalRefolds(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	ids := v41ParityPrompt(m.Cfg.VocabSize, 160)

	t.Setenv(v41SuffixUbatchEnv, "")
	want := m.NewSession().Prefill(ids)

	// The full-causal layers retain a 128-row ring, so position 128 of the
	// ubatch refuses mid-layer and the whole suffix re-folds.
	s, got, calls := v41SuffixPrefill(t, m, ids[:100], ids[100:], "2")
	if calls != 0 {
		t.Fatalf("suffix ran %d per-token steps after a batched refusal, want a re-fold", calls)
	}
	v41RequireSameLogits(t, "re-folded suffix vs cold", got, want)
	v41RequireCursors(t, s, ids)
}
