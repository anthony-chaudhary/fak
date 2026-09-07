package model

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestNgramDrafterDefaultOff(t *testing.T) {
	if got := (NgramDrafter{}).Draft([]int{1, 2, 3, 4, 1, 2, 3}); got != nil {
		t.Fatalf("default-off draft = %v, want nil", got)
	}
	if got := (NgramDrafter{}).DraftTree([]int{1, 2, 3, 4, 1, 2, 3}, 2); got != nil {
		t.Fatalf("default-off draft tree = %v, want nil", got)
	}
}

func TestNgramDrafterLongestSuffixAndBoundedCopy(t *testing.T) {
	history := []int{1, 2, 3, 4, 5, 1, 2, 3}
	d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 2}
	got := d.Draft(history)
	want := []int{4, 5}
	if !ngramEqualInts(got, want) {
		t.Fatalf("draft = %v, want %v", got, want)
	}
	got[0] = 99
	if history[3] != 4 {
		t.Fatal("draft aliases committed history")
	}
}

func TestNgramDrafterNoRepeatedSuffixIsInert(t *testing.T) {
	d := NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 8, MaxDraft: 4}
	if got := d.Draft([]int{1, 2, 3, 4, 5, 6}); got != nil {
		t.Fatalf("non-repeating history draft = %v, want nil", got)
	}
}

func TestFirstTokenSubsequenceHandlesOverlap(t *testing.T) {
	if got := firstTokenSubsequence([]int{1, 1, 1, 1, 2}, []int{1, 1, 2}); got != 2 {
		t.Fatalf("overlap match = %d, want 2", got)
	}
}

func TestSpecDecodeGreedyWithNgramDrafterMatchesPlainGreedy(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 1, 2, 3}
	d := NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 3, MaxDraft: 4}
	want := m.NewSession().Generate(prompt, 20)
	run, err := SpecDecodeGreedyWithDrafter(m.NewSession(), prompt, 20, d.MaxDraft, d.Draft)
	if err != nil {
		t.Fatalf("SpecDecodeGreedyWithDrafter: %v", err)
	}
	if !ngramEqualInts(run.Output, want) {
		t.Fatalf("prompt-lookup output = %v, plain greedy = %v", run.Output, want)
	}
	if run.DraftedTokens == 0 {
		t.Fatal("prompt-lookup proposed no tokens; verify path was not exercised")
	}
}

func TestNgramFirstTokenSubsequenceAVX512Parity(t *testing.T) {
	TestFirstTokenSubsequenceAVX512Parity(t)
}

func TestFirstTokenSubsequenceAVX512Parity(t *testing.T) {
	testSizes := []int{0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 256, 1024}

	for _, size := range testSizes {
		tokens := make([]int, size)
		tokensI32 := make([]int32, size)
		for i := range tokens {
			tokens[i] = i*3 + 1
			tokensI32[i] = int32(i*3 + 1)
		}

		// Test target match at every valid position
		for pos := 0; pos < size; pos++ {
			patternLen := 1
			if pos+3 <= size {
				patternLen = 3
			}
			pattern := tokens[pos : pos+patternLen]
			patternI32 := tokensI32[pos : pos+patternLen]

			wantScalar := firstTokenSubsequenceScalar(tokens, pattern)
			got := firstTokenSubsequence(tokens, pattern)
			if got != wantScalar {
				t.Fatalf("size=%d pos=%d patternLen=%d: got %d, wantScalar %d", size, pos, patternLen, got, wantScalar)
			}

			wantScalarI32 := firstTokenSubsequenceI32Scalar(tokensI32, patternI32)
			gotI32 := firstTokenSubsequenceI32(tokensI32, patternI32)
			if gotI32 != wantScalarI32 {
				t.Fatalf("I32 size=%d pos=%d patternLen=%d: got %d, wantScalar %d", size, pos, patternLen, gotI32, wantScalarI32)
			}
		}

		// Test target not present
		missingPattern := []int{999999, 999998}
		missingPatternI32 := []int32{999999, 999998}
		if got := firstTokenSubsequence(tokens, missingPattern); got != -1 {
			t.Fatalf("size=%d missing pattern: got %d, want -1", size, got)
		}
		if got := firstTokenSubsequenceI32(tokensI32, missingPatternI32); got != -1 {
			t.Fatalf("I32 size=%d missing pattern: got %d, want -1", size, got)
		}
	}

	// Overlap edge cases
	overlapTokens := []int{1, 2, 1, 2, 1, 2, 3, 4}
	overlapPattern := []int{1, 2, 3}
	if got := firstTokenSubsequence(overlapTokens, overlapPattern); got != 4 {
		t.Fatalf("overlap tokens got %d, want 4", got)
	}

	overlapTokensI32 := []int32{1, 2, 1, 2, 1, 2, 3, 4}
	overlapPatternI32 := []int32{1, 2, 3}
	if got := firstTokenSubsequenceI32(overlapTokensI32, overlapPatternI32); got != 4 {
		t.Fatalf("I32 overlap tokens got %d, want 4", got)
	}

	// Direct testing of scanNextMatchAVX512_I64 and scanNextMatchAVX512_I32 when supported
	if hasAVX512() {
		for _, size := range []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100} {
			tokens := make([]int, size)
			tokens32 := make([]int32, size)
			for i := range tokens {
				tokens[i] = i * 2
				tokens32[i] = int32(i * 2)
			}

			// Direct bounds and negative start tests
			if size == 0 {
				if got := scanNextMatchAVX512_I64(nil, 0, 0, 10); got != -1 {
					t.Fatalf("expected -1 for empty slice, got %d", got)
				}
				if got := scanNextMatchAVX512_I32(nil, 0, 0, 10); got != -1 {
					t.Fatalf("expected -1 for empty slice, got %d", got)
				}
				continue
			}

			if got := scanNextMatchAVX512_I64(&tokens[0], -5, size, 0); got != -1 {
				t.Fatalf("expected -1 for negative start, got %d", got)
			}
			if got := scanNextMatchAVX512_I32(&tokens32[0], -5, size, 0); got != -1 {
				t.Fatalf("expected -1 for negative start, got %d", got)
			}
			if got := scanNextMatchAVX512_I64(&tokens[0], size, size, 0); got != -1 {
				t.Fatalf("expected -1 for start == size, got %d", got)
			}
			if got := scanNextMatchAVX512_I32(&tokens32[0], size, size, 0); got != -1 {
				t.Fatalf("expected -1 for start == size, got %d", got)
			}

			for pos := 0; pos < size; pos++ {
				target := tokens[pos]
				target32 := tokens32[pos]
				for start := 0; start <= size; start++ {
					// Scalar expectation
					want := -1
					for i := start; i < size; i++ {
						if tokens[i] == target {
							want = i
							break
						}
					}

					got := scanNextMatchAVX512_I64(&tokens[0], start, size, target)
					if got != want {
						t.Fatalf("I64 size=%d pos=%d start=%d: got %d, want %d", size, pos, start, got, want)
					}

					got32 := scanNextMatchAVX512_I32(&tokens32[0], start, size, target32)
					if got32 != want {
						t.Fatalf("I32 size=%d pos=%d start=%d: got %d, want %d", size, pos, start, got32, want)
					}
				}
			}
		}
	}
}

func TestNgramDraftTreeExtraction(t *testing.T) {
	// 1. Short history (< 2) returns nil
	d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 3}
	if tree := d.DraftTree([]int{1}, 2); tree != nil {
		t.Fatalf("history len < 2 expected nil tree, got %+v", tree)
	}
	if tree := d.DraftTree(nil, 2); tree != nil {
		t.Fatalf("nil history expected nil tree, got %+v", tree)
	}

	// 2. No match found returns CandidateDraftTree with 0 counts and nil slices
	noMatchHistory := []int{10, 20, 30, 40, 50, 60, 70, 80}
	treeNoMatch := d.DraftTree(noMatchHistory, 2)
	if treeNoMatch == nil {
		t.Fatal("expected non-nil tree for no-match")
	}
	if treeNoMatch.MatchCount != 0 || treeNoMatch.MatchLength != 0 {
		t.Fatalf("expected MatchCount=0, MatchLength=0, got count=%d len=%d", treeNoMatch.MatchCount, treeNoMatch.MatchLength)
	}
	if treeNoMatch.Tokens != nil || treeNoMatch.Branches != nil {
		t.Fatalf("expected nil Tokens and Branches, got Tokens=%v Branches=%v", treeNoMatch.Tokens, treeNoMatch.Branches)
	}

	// 3. Basic single branch match
	// History has suffix [1, 2, 3] at end, and [1, 2, 3] earlier followed by [4, 5, 6, 7]
	history := []int{1, 2, 3, 4, 5, 6, 7, 100, 1, 2, 3}
	tree := d.DraftTree(history, 2)
	if tree == nil {
		t.Fatal("expected non-nil tree for basic match")
	}
	if tree.MatchLength != 3 {
		t.Errorf("expected MatchLength=3, got %d", tree.MatchLength)
	}
	if tree.MatchCount != 1 {
		t.Errorf("expected MatchCount=1, got %d", tree.MatchCount)
	}
	expectedTokens := []int{4, 5, 6}
	if !reflect.DeepEqual(tree.Tokens, expectedTokens) {
		t.Errorf("expected Tokens=%v, got %v", expectedTokens, tree.Tokens)
	}
	if len(tree.Branches) != 1 {
		t.Fatalf("expected 1 branch, got %d", len(tree.Branches))
	}
	if !reflect.DeepEqual(tree.Branches[0], expectedTokens) {
		t.Errorf("expected Branches[0]=%v, got %v", expectedTokens, tree.Branches[0])
	}

	// 4. Multiple branches
	// Suffix [1, 2] appears 3 times before the end
	multiHistory := []int{
		1, 2, 10, 11, // match at 0 -> [10, 11]
		1, 2, 20, 21, // match at 4 -> [20, 21]
		1, 2, 30, 31, // match at 8 -> [30, 31]
		1, 2, // suffix at 12
	}
	dMulti := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 2, MaxDraft: 2}
	treeMulti := dMulti.DraftTree(multiHistory, 2)
	if treeMulti == nil {
		t.Fatal("expected non-nil tree for multiple branches")
	}
	if treeMulti.MatchCount != 3 {
		t.Errorf("expected MatchCount=3, got %d", treeMulti.MatchCount)
	}
	if len(treeMulti.Branches) != 2 {
		t.Fatalf("expected 2 branches (bounded by maxBranches), got %d", len(treeMulti.Branches))
	}
	if !reflect.DeepEqual(treeMulti.Branches[0], []int{10, 11}) {
		t.Errorf("expected Branch[0]=[10, 11], got %v", treeMulti.Branches[0])
	}
	if !reflect.DeepEqual(treeMulti.Branches[1], []int{20, 21}) {
		t.Errorf("expected Branch[1]=[20, 21], got %v", treeMulti.Branches[1])
	}
	if !reflect.DeepEqual(treeMulti.Tokens, []int{10, 11}) {
		t.Errorf("expected Tokens=[10, 11], got %v", treeMulti.Tokens)
	}

	// 5. Longest suffix match priority
	longestHistory := []int{10, 20, 30, 40, 99, 98, 5, 6, 30, 40, 77, 76, 10, 20, 30, 40}
	dLongest := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 2}
	treeLongest := dLongest.DraftTree(longestHistory, 2)
	if treeLongest == nil {
		t.Fatal("expected non-nil tree for longest suffix test")
	}
	if treeLongest.MatchLength != 4 {
		t.Errorf("expected MatchLength=4, got %d", treeLongest.MatchLength)
	}
	if !reflect.DeepEqual(treeLongest.Tokens, []int{99, 98}) {
		t.Errorf("expected Tokens=[99, 98], got %v", treeLongest.Tokens)
	}

	// 6. Copy safety
	tree.Tokens[0] = 999
	if history[3] != 4 {
		t.Fatal("DraftTree mutated input history")
	}

	// 7. Package-level DraftTree and DraftTreeScalar functions
	pkgTree := DraftTree(history, 2)
	if pkgTree == nil || pkgTree.MatchLength != 3 {
		t.Fatalf("package DraftTree failed: %+v", pkgTree)
	}
	pkgScalarTree := DraftTreeScalar(history, 2)
	if pkgScalarTree == nil || pkgScalarTree.MatchLength != 3 {
		t.Fatalf("package DraftTreeScalar failed: %+v", pkgScalarTree)
	}
}

func TestNgramAVX512BitIdenticalToScalar(t *testing.T) {
	rnd := rand.New(rand.NewSource(12345))

	sizes := []int{10, 32, 64, 128, 512, 2048, 8192}
	for _, size := range sizes {
		history := make([]int, size)
		for i := range history {
			history[i] = rnd.Intn(50) + 1
		}

		// Inject repeated n-gram suffix
		suffixLen := 4
		if size > 16 {
			for i := 0; i < suffixLen; i++ {
				history[size-suffixLen+i] = 100 + i
			}
			// Copy earlier at 1 or 2 locations
			pos1 := size / 4
			for i := 0; i < suffixLen; i++ {
				history[pos1+i] = 100 + i
			}
			pos2 := size / 2
			for i := 0; i < suffixLen; i++ {
				history[pos2+i] = 100 + i
			}
		}

		d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 6, MaxDraft: 4}

		treeAVX := d.DraftTree(history, 3)
		treeScalar := d.DraftTreeScalar(history, 3)

		if (treeAVX == nil) != (treeScalar == nil) {
			t.Fatalf("size=%d nil mismatch: avx=%v, scalar=%v", size, treeAVX == nil, treeScalar == nil)
		}
		if treeAVX != nil {
			if treeAVX.MatchCount != treeScalar.MatchCount {
				t.Fatalf("size=%d MatchCount mismatch: avx=%d, scalar=%d", size, treeAVX.MatchCount, treeScalar.MatchCount)
			}
			if treeAVX.MatchLength != treeScalar.MatchLength {
				t.Fatalf("size=%d MatchLength mismatch: avx=%d, scalar=%d", size, treeAVX.MatchLength, treeScalar.MatchLength)
			}
			if !reflect.DeepEqual(treeAVX.Tokens, treeScalar.Tokens) {
				t.Fatalf("size=%d Tokens mismatch: avx=%v, scalar=%v", size, treeAVX.Tokens, treeScalar.Tokens)
			}
			if len(treeAVX.Branches) != len(treeScalar.Branches) {
				t.Fatalf("size=%d Branches len mismatch: avx=%d, scalar=%d", size, len(treeAVX.Branches), len(treeScalar.Branches))
			}
			for b := range treeAVX.Branches {
				if !reflect.DeepEqual(treeAVX.Branches[b], treeScalar.Branches[b]) {
					t.Fatalf("size=%d Branch[%d] mismatch: avx=%v, scalar=%v", size, b, treeAVX.Branches[b], treeScalar.Branches[b])
				}
			}

			// Verify Draft(history) matches the most recent branch
			draftTokens := d.Draft(history)
			var latestBranch []int
			if len(treeAVX.Branches) > 0 {
				latestBranch = treeAVX.Branches[len(treeAVX.Branches)-1]
			}
			if !reflect.DeepEqual(draftTokens, latestBranch) {
				t.Fatalf("size=%d Draft() tokens %v != latest branch %v", size, draftTokens, latestBranch)
			}
		}

		// Also test findContinuationBranchesAVX512 vs findContinuationBranchesScalar
		pattern := history[size-suffixLen:]
		branchesAVX := findContinuationBranchesAVX512(history, pattern, 4, 3)
		branchesScalar := findContinuationBranchesScalar(history, pattern, 4, 3)
		if !reflect.DeepEqual(branchesAVX, branchesScalar) {
			t.Fatalf("size=%d continuation branches mismatch: avx=%v, scalar=%v", size, branchesAVX, branchesScalar)
		}
	}

	// Test environment variable override FAK_NGRAM_AVX512=0
	t.Run("EnvOverrideDisableAVX512", func(t *testing.T) {
		t.Setenv("FAK_NGRAM_AVX512", "0")
		if hasAVX512() {
			t.Fatal("expected hasAVX512() to be false when FAK_NGRAM_AVX512=0")
		}
		hist := []int{1, 2, 3, 4, 5, 1, 2, 3}
		tree := DraftTree(hist, 2)
		scalarTree := DraftTreeScalar(hist, 2)
		if !reflect.DeepEqual(tree, scalarTree) {
			t.Fatalf("tree mismatch under FAK_NGRAM_AVX512=0: %+v vs %+v", tree, scalarTree)
		}
	})
}

func BenchmarkNgramScanner32k(b *testing.B) {
	rnd := rand.New(rand.NewSource(999))
	size := 32768
	history := make([]int, size)
	for i := range history {
		history[i] = rnd.Intn(1000)
	}

	// Insert repeated n-grams periodically
	suffix := []int{42, 43, 44, 45, 46}
	copy(history[size-len(suffix):], suffix)
	for i := 1; i <= 8; i++ {
		pos := size * i / 10
		copy(history[pos:pos+len(suffix)], suffix)
		// Give some varied continuations
		history[pos+len(suffix)] = i * 10
	}

	d := NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 8, MaxDraft: 4}

	b.Run("AVX512_DraftTree", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tree := d.DraftTree(history, 4)
			if tree == nil || tree.MatchCount == 0 {
				b.Fatal("expected match")
			}
		}
	})

	b.Run("Scalar_DraftTree", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tree := d.DraftTreeScalar(history, 4)
			if tree == nil || tree.MatchCount == 0 {
				b.Fatal("expected match")
			}
		}
	})

	b.Run("AVX512_FirstTokenSubsequence", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := firstTokenSubsequence(history[:size-1], suffix)
			if idx < 0 {
				b.Fatal("expected match")
			}
		}
	})

	b.Run("Scalar_FirstTokenSubsequence", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := firstTokenSubsequenceScalar(history[:size-1], suffix)
			if idx < 0 {
				b.Fatal("expected match")
			}
		}
	})
}

func ngramEqualInts(a, b []int) bool {
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

func TestNgramDrafterBackwardMatchPreference(t *testing.T) {
	// A phrase appears multiple times in context with different continuations:
	// Occurrence 1 (earliest): [10, 20, 30] followed by [100, 101, 102]
	// Occurrence 2 (middle):   [10, 20, 30] followed by [200, 201, 202]
	// Occurrence 3 (latest):   [10, 20, 30] followed by [300, 301, 302]
	// Current suffix:          [10, 20, 30]
	history := []int{
		10, 20, 30, 100, 101, 102,
		99,
		10, 20, 30, 200, 201, 202,
		98,
		10, 20, 30, 300, 301, 302,
		97,
		10, 20, 30,
	}

	d := NgramDrafter{
		Enabled:  true,
		MinMatch: 3,
		MaxMatch: 3,
		MaxDraft: 3,
	}

	got := d.Draft(history)
	want := []int{300, 301, 302}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Draft got %v, want continuation from most recent match %v", got, want)
	}

	// Also verify with MinN / MaxN aliases
	dAliases := NgramDrafter{
		Enabled:  true,
		MinN:     3,
		MaxN:     3,
		MaxDraft: 3,
	}
	gotAliases := dAliases.Draft(history)
	if !reflect.DeepEqual(gotAliases, want) {
		t.Fatalf("Draft with MinN/MaxN got %v, want %v", gotAliases, want)
	}
}

func TestNgramDrafterAdaptiveBlockExpansion(t *testing.T) {
	d := NgramDrafter{
		Enabled:           true,
		MinN:              3,
		MaxN:              5,
		MaxDraft:          4,
		AdaptiveExpansion: true,
		MaxAdaptiveDraft:  16,
	}

	// Baseline draft length should be 4
	if eff := d.EffectiveDraftLen(); eff != 4 {
		t.Fatalf("initial EffectiveDraftLen = %d, want 4", eff)
	}

	// Saturated round 1: 4 proposed, 4 accepted -> expands to 8
	d.RecordVerificationResult(4, 4)
	if d.ConsecutiveHits != 1 || d.ConsecutiveSaturatedRounds != 1 {
		t.Fatalf("after round 1, ConsecutiveHits = %d, want 1", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 8 {
		t.Fatalf("after round 1, EffectiveDraftLen = %d, want 8", eff)
	}

	// Saturated round 2: 8 proposed, 8 accepted -> expands to 12
	d.RecordVerificationResult(8, 8)
	if d.ConsecutiveHits != 2 || d.ConsecutiveSaturatedRounds != 2 {
		t.Fatalf("after round 2, ConsecutiveHits = %d, want 2", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 12 {
		t.Fatalf("after round 2, EffectiveDraftLen = %d, want 12", eff)
	}

	// Saturated round 3: 12 proposed, 12 accepted -> expands to 16
	d.RecordVerificationResult(12, 12)
	if d.ConsecutiveHits != 3 || d.ConsecutiveSaturatedRounds != 3 {
		t.Fatalf("after round 3, ConsecutiveHits = %d, want 3", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 16 {
		t.Fatalf("after round 3, EffectiveDraftLen = %d, want 16", eff)
	}

	// Saturated round 4: 16 proposed, 16 accepted -> capped at MaxAdaptiveDraft (16)
	d.RecordVerificationResult(16, 16)
	if d.ConsecutiveHits != 4 || d.ConsecutiveSaturatedRounds != 4 {
		t.Fatalf("after round 4, ConsecutiveHits = %d, want 4", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 16 {
		t.Fatalf("after round 4 capped, EffectiveDraftLen = %d, want 16", eff)
	}

	// Verify Draft() actually proposes up to 16 tokens when context allows
	history := make([]int, 0, 50)
	history = append(history, 1, 2, 3)
	for i := 100; i < 120; i++ {
		history = append(history, i)
	}
	history = append(history, 999, 1, 2, 3)

	draft := d.Draft(history)
	if len(draft) != 16 {
		t.Fatalf("Draft length = %d, want 16 (expanded)", len(draft))
	}
	for i := 0; i < 16; i++ {
		if draft[i] != 100+i {
			t.Fatalf("draft[%d] = %d, want %d", i, draft[i], 100+i)
		}
	}
}

func TestNgramDrafterAdaptiveContractionUponRejection(t *testing.T) {
	d := NgramDrafter{
		Enabled:           true,
		MinN:              3,
		MaxN:              5,
		MaxDraft:          4,
		AdaptiveExpansion: true,
		MaxAdaptiveDraft:  16,
	}

	// Expand to 16
	d.RecordVerificationResult(4, 4)
	d.RecordVerificationResult(8, 8)
	d.RecordVerificationResult(12, 12)
	if eff := d.EffectiveDraftLen(); eff != 16 {
		t.Fatalf("EffectiveDraftLen before rejection = %d, want 16", eff)
	}

	// Case 1: Partial acceptance (short match): 10 accepted out of 16 -> resets to baseline 4
	d.RecordVerificationResult(10, 16)
	if d.ConsecutiveHits != 0 || d.ConsecutiveSaturatedRounds != 0 {
		t.Fatalf("after partial match, ConsecutiveHits = %d, want 0", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 4 {
		t.Fatalf("after partial match, EffectiveDraftLen = %d, want baseline 4", eff)
	}

	// Expand again to 8
	d.RecordVerificationResult(4, 4)
	if eff := d.EffectiveDraftLen(); eff != 8 {
		t.Fatalf("EffectiveDraftLen = %d, want 8", eff)
	}

	// Case 2: Zero acceptance: 0 accepted out of 8 -> resets to baseline 4
	d.RecordVerificationResult(0, 8)
	if d.ConsecutiveHits != 0 || d.ConsecutiveSaturatedRounds != 0 {
		t.Fatalf("after zero acceptance, ConsecutiveHits = %d, want 0", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 4 {
		t.Fatalf("after zero acceptance, EffectiveDraftLen = %d, want baseline 4", eff)
	}

	// Case 3: Draft proposed with baseline
	history := make([]int, 0, 50)
	history = append(history, 1, 2, 3)
	for i := 100; i < 120; i++ {
		history = append(history, i)
	}
	history = append(history, 999, 1, 2, 3)

	draft := d.Draft(history)
	if len(draft) != 4 {
		t.Fatalf("Draft length after reset = %d, want 4 (baseline)", len(draft))
	}
}

func TestNgramDrafterAdaptiveDisabled(t *testing.T) {
	d := NgramDrafter{
		Enabled:           true,
		MinN:              3,
		MaxN:              5,
		MaxDraft:          4,
		AdaptiveExpansion: false,
		MaxAdaptiveDraft:  16,
	}

	// When adaptive expansion is disabled, EffectiveDraftLen stays at baseline
	if eff := d.EffectiveDraftLen(); eff != 4 {
		t.Fatalf("EffectiveDraftLen = %d, want 4", eff)
	}

	d.RecordVerificationResult(4, 4)
	if d.ConsecutiveHits != 1 {
		t.Fatalf("ConsecutiveHits = %d, want 1", d.ConsecutiveHits)
	}
	if eff := d.EffectiveDraftLen(); eff != 4 {
		t.Fatalf("EffectiveDraftLen = %d, want 4 when adaptive is disabled", eff)
	}
}
