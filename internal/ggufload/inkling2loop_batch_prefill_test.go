package ggufload

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// Ending a row at its cursor drops the rest of its prompt from every later chunk,
// and the rows after it close up into the room it leaves.
func TestInkling2PrefillNextChunkDropsEndedRow(t *testing.T) {
	end, cursor := []int{5, 7, 9}, make([]int, 3)
	var got [][]ink2PrefillSeg
	for c := 0; ; c++ {
		if c == 1 {
			end[1] = cursor[1]
		}
		segs := inkling2PrefillNextChunk(end, cursor, 4)
		if len(segs) == 0 {
			break
		}
		got = append(got, segs)
	}
	want := [][]ink2PrefillSeg{
		{{row: 0, pos0: 0, off: 0, n: 4}},
		{{row: 0, pos0: 4, off: 0, n: 1, last: true}, {row: 2, pos0: 0, off: 1, n: 3}},
		{{row: 2, pos0: 3, off: 0, n: 4}},
		{{row: 2, pos0: 7, off: 0, n: 2, last: true}},
	}
	if !reflect.DeepEqual(got, want) || cursor[1] != 0 {
		t.Fatalf("chunks %v cursor %v, want %v with row 1 never packed", got, cursor, want)
	}
}

// prefill refuses an empty prompt instead of reading it as a skipped row, so a caller
// that passes nil for a row it means to skip loses the guard it relied on.
func TestInkling2PrefillRejectsEmptyPrompt(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	b, err := bt.acquire(2, 64)
	if err != nil {
		t.Fatal(err)
	}
	err = b.prefill([][]int{{1, 2, 3}, {}}, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "empty prompt") {
		t.Fatalf("prefill with an empty prompt: err=%v, want an empty-prompt error", err)
	}
}

// Every prompt position lands in exactly one chunk segment, in order, each chunk
// packs at most T tokens contiguously, and only a row's final segment is last.
func TestInkling2PrefillChunksCoverEveryPromptOnce(t *testing.T) {
	for _, tc := range []struct {
		lens []int
		T    int
	}{
		{[]int{3}, 512},
		{[]int{82, 88, 85, 84, 86, 83, 87, 88, 82, 88, 85, 84, 86, 83, 87, 88}, 512},
		{[]int{600, 1, 1}, 512},
		{[]int{5, 7, 9}, 4},
		{[]int{4, 4}, 4},
	} {
		chunks := inkling2PrefillChunks(tc.lens, tc.T)
		next := make([]int, len(tc.lens))
		for c, segs := range chunks {
			off := 0
			for _, sg := range segs {
				if sg.off != off || sg.n < 1 || sg.pos0 != next[sg.row] {
					t.Fatalf("lens %v T %d chunk %d: segment %+v at offset %d, row at %d", tc.lens, tc.T, c, sg, off, next[sg.row])
				}
				off += sg.n
				next[sg.row] += sg.n
				if sg.last != (next[sg.row] == tc.lens[sg.row]) {
					t.Fatalf("lens %v T %d: segment %+v last flag wrong", tc.lens, tc.T, sg)
				}
			}
			if off > tc.T || (c < len(chunks)-1 && off != tc.T) {
				t.Fatalf("lens %v T %d: chunk %d packs %d tokens", tc.lens, tc.T, c, off)
			}
		}
		if !reflect.DeepEqual(next, tc.lens) {
			t.Fatalf("lens %v T %d: covered %v", tc.lens, tc.T, next)
		}
	}
}

// The mixed planner gives a long row its cap and the short row its whole prompt in the
// first chunk, then holds the chunk's prompt share to the cap beside a decode segment.
func TestInkling2MixedPrefillPlanLiteral(t *testing.T) {
	const T, capTok = inkling2PrefillChunk, 64
	lens := []int{1536, 20}
	got := inkling2MixedPrefillPlan(lens, []int{0, 0}, nil, T, capTok)
	want := []ink2PrefillSeg{{row: 0, pos0: 0, off: 0, n: 492}, {row: 1, pos0: 0, off: 492, n: 20, last: true}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunk 0 plan %+v, want %+v", got, want)
	}
	got = inkling2MixedPrefillPlan(lens, []int{492, 20}, []int{1}, T, capTok)
	want = []ink2PrefillSeg{{row: 0, pos0: 492, off: 0, n: 64}, {row: 1, pos0: 20, off: 64, n: 1, last: true}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunk 1 plan %+v, want %+v", got, want)
	}
}

// With one prefilling row and nothing decoding, the mixed planner packs exactly as the
// prompt-only packer does, chunk by chunk.
func TestInkling2MixedPrefillPlanSingleRowMatchesPromptPacking(t *testing.T) {
	const T, capTok = 300, 64
	lens := []int{1000}
	want := inkling2PrefillChunks(lens, T)
	cursor := []int{0}
	for k := range want {
		got := inkling2MixedPrefillPlan(lens, cursor, nil, T, capTok)
		if !reflect.DeepEqual(got, want[k]) {
			t.Fatalf("chunk %d plan %+v, want %+v", k, got, want[k])
		}
		cursor[0] += got[0].n
	}
	if cursor[0] != lens[0] {
		t.Fatalf("planned %d of %d prompt tokens", cursor[0], lens[0])
	}
}

// Unset/empty defaults to 512 (continuous admission on); "0", negative, or unparsable disables it.
func TestInkling2CohortLongPrefillCapParsesKnob(t *testing.T) {
	const name = "FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD"
	for _, tc := range []struct {
		name string
		set  bool
		v    string
		want int
	}{
		{"unset", false, "", 512},
		{"empty", true, "", 512},
		{"blank", true, "   ", 512},
		{"zero", true, "0", 0},
		{"negative", true, "-5", 0},
		{"unparsable", true, "abc", 0},
		{"positive", true, "64", 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := inkling2CohortLongPrefillCap(func(k string) (string, bool) {
				if k != name {
					t.Errorf("looked up %q, want %q", k, name)
				}
				return tc.v, tc.set
			})
			if got != tc.want {
				t.Fatalf("cap %d, want %d", got, tc.want)
			}
		})
	}
}

// A cohort prefilled as one ragged token stream emits, row for row, what the lockstep
// prompt path and the single-sequence decode emit. The rows span several prefill
// chunks, and a long-prompt row with a short budget finishes while a short-prompt row
// still decodes past the long row's cache bound unless the finished row is parked.
func TestInkling2CohortPrefillMatchesLockstepAndScalar(t *testing.T) {
	type row struct{ prompt, maxNew int }
	rows := []row{{90, 2}, {1, 60}, {70, 8}, {85, 1}, {3, 30}}
	for i := 0; len(rows) < Inkling2LoopMaxBatch; i++ {
		rows = append(rows, row{40 + 3*i, 4 + i})
	}
	run := func(prefill string) ([][]int, []Inkling2LoopBatchResult, int) {
		t.Helper()
		t.Setenv("FAK_INKLING2_COHORT_PREFILL", prefill)
		t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "0")
		d := newFakeInk2Cohort()
		bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
		defer bt.Close()
		got := make([][]int, len(rows))
		reqs := make([]Inkling2LoopBatchRequest, len(rows))
		for i, r := range rows {
			prompt := make([]int, r.prompt)
			for j := range prompt {
				prompt[j] = 1000*i + j
			}
			i := i
			reqs[i] = Inkling2LoopBatchRequest{Prompt: prompt, MaxTokens: r.maxNew,
				Emit: func(tok int) bool { got[i] = append(got[i], tok); return true }}
		}
		results, err := bt.Generate(reqs)
		if err != nil {
			t.Fatalf("prefill=%s: %v", prefill, err)
		}
		return got, results, d.heads
	}
	lock, lockRes, lockHeads := run("0")
	pre, preRes, preHeads := run("1")
	if lockHeads != 0 || preHeads != len(rows) {
		t.Fatalf("prefill heads: lockstep %d, prefilled %d, want 0 and %d", lockHeads, preHeads, len(rows))
	}
	for i, r := range rows {
		prompt := make([]int, r.prompt)
		for j := range prompt {
			prompt[j] = 1000*i + j
		}
		var want []int
		turn := Inkling2LoopTurn{MaxTokens: r.maxNew, Emit: func(tok int) bool { want = append(want, tok); return true }}
		if _, err := Inkling2LoopSessionDecode(context.Background(), fakeInk2Session{}, prompt, &turn); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(pre[i], want) || !reflect.DeepEqual(lock[i], want) {
			t.Fatalf("row %d: prefilled %v, lockstep %v, scalar %v", i, pre[i], lock[i], want)
		}
		if preRes[i].Generated != lockRes[i].Generated || preRes[i].Err != nil {
			t.Fatalf("row %d: prefilled %+v, lockstep %+v", i, preRes[i], lockRes[i])
		}
	}
}
