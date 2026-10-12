package model

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

// This is a source-derived contract witness, not an executed Python/TileLang
// differential. Only the named deterministic-tie case exercises Fak's tie
// policy. The newest-block pin makes the source distinction observable with
// only four keys.
// fak-test:runtime fast est=10ms lane=default
func TestV41CandidatePublicationContract(t *testing.T) {
	t.Parallel()
	source := v41CandidateSource{layer: 20, kvSourceLayer: 20, ratio: 1, queryEnd: 4}
	scores := [][]float32{
		{10, 9, 2, 1},
		{10, 9, 2, 1},
		{10, 9, 2, 1},
		{10, 9, 2, 1},
	}
	p, sourceRows, err := v41SelectCandidateSource(source, scores, 1, 2, 1, 7)
	if err != nil {
		t.Fatal(err)
	}
	if want := [][]int32{{7}, {7}, {7}, {7}}; !reflect.DeepEqual(sourceRows, want) {
		t.Fatalf("candidate source must select unmasked rows: got %v, want %v", sourceRows, want)
	}
	for position, want := range [][]bool{
		{true, true, false, false},
		{true, true, false, false},
		{false, false, true, true},
		{false, false, true, true},
	} {
		got, ok := p.mask(position)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Fatalf("query %d mask %v present=%v, want %v", position, got, ok, want)
		}
		got[0] = !got[0]
		again, _ := p.mask(position)
		if !reflect.DeepEqual(again, want) {
			t.Fatalf("query %d mask accessor aliases publication", position)
		}
	}
	// Later layers prefer excluded key 0. Query 2 cannot see key 3 despite its
	// high score and its true bit within the pinned, partly visible block.
	for _, tc := range []struct {
		position int
		logits   []float32
		want     []int32
	}{
		{0, []float32{100, 99, 7, 1000}, []int32{7}},
		{2, []float32{100, 99, 7, 1000}, []int32{9}},
		{3, []float32{100, 99, 7, 8}, []int32{10}},
		{3, []float32{100, 99, 9, 8}, []int32{9}},
	} {
		got, err := p.selectRows(source, 24, tc.position, tc.logits, 1, 7)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("consumer query %d: rows %v err=%v, want %v", tc.position, got, err, tc.want)
		}
	}
	// Mutating source inputs cannot alter retained masks. Final source IDs stay
	// separate from masks and can be handed directly to an ordinary reader.
	scores[0][0], scores[3][0] = -100, -100
	sourceRows[3][0] = -1
	if got, _ := p.mask(3); !reflect.DeepEqual(got, []bool{false, false, true, true}) {
		t.Fatalf("publication aliases input or final IDs: %v", got)
	}

	t.Run("identity and absent are refused", func(t *testing.T) {
		for _, wrong := range []v41CandidateSource{
			{layer: 19, kvSourceLayer: 19, ratio: 1, queryEnd: 4},
			{layer: 20, kvSourceLayer: 14, ratio: 1, queryEnd: 4},
			{layer: 20, kvSourceLayer: 20, ratio: 2, queryEnd: 4},
			{layer: 20, kvSourceLayer: 20, ratio: 1, queryStart: 1, queryEnd: 4},
			{layer: 20, kvSourceLayer: 20, ratio: 1, queryEnd: 5},
		} {
			if _, err := p.selectRows(wrong, 24, 3, scores[3], 1, 0); err == nil {
				t.Fatalf("accepted stale identity %+v", wrong)
			}
		}
		for _, consumer := range [][2]int{{20, 3}, {19, 3}, {24, -1}, {24, 4}} {
			if _, err := p.selectRows(source, consumer[0], consumer[1], scores[3], 1, 0); err == nil {
				t.Fatalf("accepted source/earlier layer or stale position %v", consumer)
			}
		}
		if _, err := p.selectRows(source, 24, 3, []float32{1}, 1, 0); err == nil {
			t.Fatal("accepted wrong consumer key extent")
		}
		var absent *v41CandidatePublication
		if mask, ok := absent.mask(0); mask != nil || ok {
			t.Fatal("absent publication became a present mask")
		}
		if _, err := absent.selectRows(source, 24, 0, scores[0], 1, 0); err == nil {
			t.Fatal("missing source became unrestricted selection")
		}
	})

	t.Run("empty extent is present", func(t *testing.T) {
		emptySource := v41CandidateSource{layer: 2, kvSourceLayer: 2, ratio: 2, queryEnd: 1}
		empty, rows, err := v41SelectCandidateSource(emptySource, [][]float32{{}}, 1, 2, 1, 0)
		if err != nil || empty == nil || len(rows) != 1 || rows[0] == nil || len(rows[0]) != 0 {
			t.Fatalf("zero-key source: publication=%v rows=%v err=%v", empty, rows, err)
		}
		mask, ok := empty.mask(0)
		if !ok || mask == nil || len(mask) != 0 {
			t.Fatalf("present empty mask became absent: %v %v", mask, ok)
		}
		got, err := empty.selectRows(emptySource, 3, 0, nil, 1, 0)
		if err != nil || got == nil || len(got) != 0 {
			t.Fatalf("zero-key consumer: %v %v", got, err)
		}
	})

	t.Run("decode is position bound", func(t *testing.T) {
		decode := v41CandidateSource{layer: 20, kvSourceLayer: 20, ratio: 1, queryStart: 3, queryEnd: 4}
		p, rows, err := v41SelectCandidateSource(decode, [][]float32{{10, 9, 2, 1}}, 1, 2, 1, 0)
		if err != nil || !reflect.DeepEqual(rows, [][]int32{{0}}) {
			t.Fatalf("decode source: %v %v", rows, err)
		}
		got, err := p.selectRows(decode, 24, 3, []float32{100, 99, 7, 8}, 1, 0)
		if err != nil || !reflect.DeepEqual(got, []int32{3}) {
			t.Fatalf("decode consumer: %v %v", got, err)
		}
		if _, err := p.selectRows(decode, 24, 4, []float32{100, 99, 7, 8}, 1, 0); err == nil {
			t.Fatal("accepted stale decode mask on the next query")
		}
	})

	t.Run("invalid source publishes nothing", func(t *testing.T) {
		badScores := [][]float32{{10, 9}, {10, float32(math.NaN())}}
		badSource := source
		badSource.queryEnd = 2
		p, rows, err := v41SelectCandidateSource(badSource, badScores, 1, 2, 1, 0)
		if err == nil || p != nil || rows != nil {
			t.Fatalf("partial result escaped failed publication: %v %v %v", p, rows, err)
		}
		for _, bad := range []v41CandidateSource{
			{layer: -1, kvSourceLayer: 0, ratio: 1, queryEnd: 1},
			{layer: 2, kvSourceLayer: 3, ratio: 1, queryEnd: 1},
			{layer: 2, kvSourceLayer: 2, ratio: 0, queryEnd: 1},
			{layer: 2, kvSourceLayer: 2, ratio: 1, queryStart: -1, queryEnd: 1},
			{layer: 2, kvSourceLayer: 2, ratio: 1, queryStart: int(^uint(0) >> 1), queryEnd: int(^uint(0) >> 1)},
		} {
			if p, rows, err := v41SelectCandidateSource(bad, [][]float32{{1}}, 1, 2, 1, 0); err == nil || p != nil || rows != nil {
				t.Fatalf("invalid source escaped: %+v %v %v %v", bad, p, rows, err)
			}
		}
	})

	t.Run("compressed boundaries and clipped final block", func(t *testing.T) {
		for _, ratio := range []int{2, 128} {
			t.Run(fmt.Sprintf("ratio-%d", ratio), func(t *testing.T) {
				ref := v41CandidateSource{layer: 2, kvSourceLayer: 2, ratio: ratio, queryEnd: 3 * ratio}
				logits := make([][]float32, ref.queryEnd)
				for i := range logits {
					logits[i] = []float32{10, 9, 2}
				}
				p, rows, err := v41SelectCandidateSource(ref, logits, 1, 2, 1, 0)
				if err != nil {
					t.Fatal(err)
				}
				for _, tc := range []struct {
					position int
					mask     []bool
					sourceID int32
					readerID int32
				}{
					{ratio - 2, []bool{false, false, false}, -1, -1},
					{ratio - 1, []bool{true, true, false}, 0, 0},
					{2*ratio - 2, []bool{true, true, false}, 0, 0},
					{2*ratio - 1, []bool{true, true, false}, 0, 1},
					{3*ratio - 2, []bool{true, true, false}, 0, 1},
					{3*ratio - 1, []bool{false, false, true}, 0, 2},
				} {
					mask, ok := p.mask(tc.position)
					if !ok || !reflect.DeepEqual(mask, tc.mask) {
						t.Fatalf("position %d mask=%v present=%v, want %v", tc.position, mask, ok, tc.mask)
					}
					if !reflect.DeepEqual(rows[tc.position], []int32{tc.sourceID}) {
						t.Fatalf("position %d source rows %v, want [%d]", tc.position, rows[tc.position], tc.sourceID)
					}
					got, err := p.selectRows(ref, 3, tc.position, []float32{99, 100, 7}, 1, 0)
					if err != nil || !reflect.DeepEqual(got, []int32{tc.readerID}) {
						t.Fatalf("position %d reader rows %v err=%v, want [%d]", tc.position, got, err, tc.readerID)
					}
				}
			})
		}
	})

	t.Run("multiple rows sort by position and ties use lower IDs", func(t *testing.T) {
		ref := v41CandidateSource{layer: 20, kvSourceLayer: 20, ratio: 1, queryEnd: 5}
		logits := make([][]float32, ref.queryEnd)
		for i := range logits {
			logits[i] = []float32{1, 5, 3, 2, 4}
		}
		p, rows, err := v41SelectCandidateSource(ref, logits, 2, 2, 3, 11)
		if err != nil || !reflect.DeepEqual(rows[4], []int32{12, 13, 15}) {
			t.Fatalf("source position order: %v %v", rows, err)
		}
		if mask, _ := p.mask(4); !reflect.DeepEqual(mask, []bool{true, true, false, false, true}) {
			t.Fatalf("clipped source mask: %v", mask)
		}
		for _, tc := range []struct {
			scores []float32
			want   []int32
		}{
			{[]float32{4, 1, 99, 100, 9}, []int32{11, 15}},
			{[]float32{7, 7, 99, 100, 7}, []int32{11, 12}},
		} {
			got, err := p.selectRows(ref, 24, 4, tc.scores, 2, 11)
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("consumer rows %v err=%v, want %v", got, err, tc.want)
			}
		}
	})

	t.Run("source refusals publish neither output", func(t *testing.T) {
		ref := v41CandidateSource{layer: 2, kvSourceLayer: 2, ratio: 1, queryEnd: 3}
		good := [][]float32{{10, 9, 2}, {10, 9, 2}, {10, 9, 2}}
		continuation := ref
		continuation.queryStart = 1
		for _, tc := range []struct {
			name      string
			ref       v41CandidateSource
			scores    [][]float32
			blocks    int
			blockSize int
			topK      int
			offset    int
		}{
			{"continuation", continuation, good[:2], 1, 2, 1, 0},
			{"late-width", ref, [][]float32{{10, 9, 2}, {10, 9, 2}, {10, 9}}, 1, 2, 1, 0},
			{"zero-blocks", ref, good, 0, 2, 1, 0},
			{"negative-blocks", ref, good, -1, 2, 1, 0},
			{"zero-block-size", ref, good, 1, 0, 1, 0},
			{"negative-block-size", ref, good, 1, -1, 1, 0},
			{"negative-top-k", ref, good, 1, 2, -1, 0},
			{"negative-offset", ref, good, 1, 2, 1, -1},
			{"late-offset-overflow", ref, good, 1, 2, 1, math.MaxInt32},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, rows, err := v41SelectCandidateSource(tc.ref, tc.scores, tc.blocks, tc.blockSize, tc.topK, tc.offset)
				if err == nil || p != nil || rows != nil {
					t.Fatalf("refusal published partial data: %v %v %v", p, rows, err)
				}
			})
		}
		// A zero selection budget is valid and distinct from a disabled source.
		p, rows, err := v41SelectCandidateSource(ref, good, 1, 2, 0, 0)
		if err != nil || p == nil || len(rows) != 3 {
			t.Fatalf("zero top-k rejected: %v %v %v", p, rows, err)
		}
		for _, row := range rows {
			if row == nil || len(row) != 0 {
				t.Fatalf("zero top-k did not return owned empty rows: %v", rows)
			}
		}
	})

	t.Run("consumer errors preserve publication and retry", func(t *testing.T) {
		wantMask, _ := p.mask(3)
		for _, tc := range []struct {
			scores []float32
			topK   int
			offset int
		}{
			{[]float32{100, 99, 7, 8}, -1, 7},
			{[]float32{100, 99, 7, 8}, 1, -1},
			{[]float32{100, 99, 7, 8}, 1, math.MaxInt32},
			{[]float32{100, 99, 7, float32(math.NaN())}, 1, 7},
		} {
			if got, err := p.selectRows(source, 24, 3, tc.scores, tc.topK, tc.offset); err == nil || got != nil {
				t.Fatalf("consumer refusal returned rows: %v %v", got, err)
			}
			if mask, _ := p.mask(3); !reflect.DeepEqual(mask, wantMask) {
				t.Fatalf("consumer refusal mutated mask: %v, want %v", mask, wantMask)
			}
			got, err := p.selectRows(source, 24, 3, []float32{100, 99, 7, 8}, 1, 7)
			if err != nil || !reflect.DeepEqual(got, []int32{10}) {
				t.Fatalf("valid retry after consumer refusal: %v %v", got, err)
			}
			got[0] = -1
			if mask, _ := p.mask(3); !reflect.DeepEqual(mask, wantMask) {
				t.Fatalf("consumer result aliases mask: %v, want %v", mask, wantMask)
			}
		}
	})
}
