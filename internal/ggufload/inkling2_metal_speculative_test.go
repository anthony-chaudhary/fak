package ggufload

import (
	"context"
	"errors"
	"slices"
	"testing"
)

type inklingMetalTestTarget struct {
	want, committed, staged []int32
	kept                    []int
	short                   bool
	cancel                  context.CancelFunc
}

func (m *inklingMetalTestTarget) Prefill(_ []int32) ([]float32, error) {
	logits := make([]float32, 17)
	logits[m.want[0]] = 1
	logits[16] = 9 // padded vocabulary row
	return logits, nil
}

func (m *inklingMetalTestTarget) BeginBlockIDs(in []int32) ([]int32, error) {
	m.staged = slices.Clone(in)
	rows := make([]int32, len(in))
	valid := true
	for i, id := range in {
		p := len(m.committed) + i
		valid = valid && p < len(m.want) && id == m.want[p]
		rows[i] = 15
		if valid && p+1 < len(m.want) {
			rows[i] = m.want[p+1]
		}
	}
	if m.cancel != nil {
		m.cancel()
	}
	if m.short {
		return rows[:len(rows)-1], nil
	}
	return rows, nil
}

func (m *inklingMetalTestTarget) Commit(n int) error {
	m.committed = append(m.committed, m.staged[:n]...)
	m.kept = append(m.kept, n)
	m.staged = nil
	return nil
}

type inklingMetalTestDraft struct{ proposal []int32 }

func (d inklingMetalTestDraft) Draft(_ int32, n int) ([]int32, error) {
	return slices.Clone(d.proposal[:min(n, len(d.proposal))]), nil
}

func TestInkling2MetalGreedyOnlyExposesVerifiedTokens(t *testing.T) {
	tests := []struct {
		name            string
		proposal, truth []int32
		prompt          []int
		eos, limit      int
		want            []int
	}{
		{"wrong-middle", []int32{2, 9, 9}, []int32{1, 2, 3, 4, 5, 6}, []int{12}, -1, 6, []int{1, 2, 3, 4, 5, 6}},
		{"full-accept", []int32{2, 3, 4}, []int32{1, 2, 3, 4, 5}, []int{12}, -1, 5, []int{1, 2, 3, 4, 5}},
		{"confirmed-eos", []int32{2, 3, 4}, []int32{1, 2, 3, 4}, []int{12}, 3, 8, []int{1, 2}},
		{"wrong-eos", []int32{3}, []int32{1, 2, 4, 5}, []int{12}, 3, 4, []int{1, 2, 4, 5}},
		{"lookup", nil, []int32{3, 4, 5, 6}, []int{1, 2, 3, 4, 5, 6, 1, 2}, -1, 4, []int{3, 4, 5, 6}},
		{"first-eos", nil, []int32{3}, []int{12}, 3, 8, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &inklingMetalTestTarget{want: tc.truth}
			got, st, err := Inkling2MetalGreedy(context.Background(), m, inklingMetalTestDraft{tc.proposal}, Inkling2MetalGreedyRequest{Prompt: tc.prompt, MaxTokens: tc.limit, Vocab: 16, EOS: tc.eos, VerifyRows: 5, DraftDepth: 4})
			if err != nil || !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v, error %v", got, tc.want, err)
			}
			if !slices.Equal(m.committed, tc.truth[:len(m.committed)]) {
				t.Fatalf("committed rejected rows %v", m.committed)
			}
			if tc.name == "lookup" && st.Accepted == 0 {
				t.Fatal("lookup didn't advance a verified row")
			}
			if tc.name == "wrong-middle" && (len(m.kept) == 0 || m.kept[0] != 2) {
				t.Fatalf("partial commit counts %v", m.kept)
			}
		})
	}
}

func TestInkling2MetalGreedyDiscardsUnusableBlock(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		ctx, stop := context.WithCancel(context.Background())
		m := &inklingMetalTestTarget{want: []int32{1, 2, 3}, short: !cancel}
		if cancel {
			m.cancel = stop
		}
		_, _, err := Inkling2MetalGreedy(ctx, m, nil, Inkling2MetalGreedyRequest{Prompt: []int{1}, MaxTokens: 3, Vocab: 16, EOS: -1, VerifyRows: 4})
		stop()
		if err == nil || cancel && !errors.Is(err, context.Canceled) || !slices.Equal(m.kept, []int{0}) {
			t.Fatalf("cancel=%v, committed %v, error %v", cancel, m.kept, err)
		}
	}
}

type inklingMetalTreeFixture struct {
	inklingMetalTestTarget
	keptLeaf int
	invalid  bool
}

func (m *inklingMetalTreeFixture) DraftTree(anchor int32, _ int, _ int) ([][3]int32, error) {
	return [][3]int32{{anchor, -1, 0}, {15, 0, 1}, {2, 0, 1}, {3, 2, 2}}, nil
}
func (m *inklingMetalTreeFixture) BeginTreeIDs(_ [][3]int32) ([]int32, error) {
	if m.invalid {
		return []int32{2}, nil
	}
	return []int32{2, 14, 3, 4}, nil
}
func (m *inklingMetalTreeFixture) CommitTree(leaf int) error { m.keptLeaf = leaf; return nil }

func TestInkling2MetalBranchKeepsReachablePrefix(t *testing.T) {
	for _, tc := range []struct {
		name    string
		eos     int
		invalid bool
		leaf    int
		want    []int
	}{
		{"noncontiguous", -1, false, 3, []int{1, 2, 3, 4}},
		{"eos", 3, false, 2, []int{1, 2}},
		{"missing-score", -1, true, -1, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &inklingMetalTreeFixture{inklingMetalTestTarget: inklingMetalTestTarget{want: []int32{1, 2, 3, 4}}, invalid: tc.invalid, keptLeaf: -2}
			out, _, err := Inkling2MetalGreedy(context.Background(), m, inklingMetalTestDraft{}, Inkling2MetalGreedyRequest{Prompt: []int{12}, MaxTokens: 4, Vocab: 16, EOS: tc.eos, VerifyRows: 5, DraftDepth: 3, TreeWidth: 4})
			if (err != nil) != tc.invalid || !slices.Equal(out, tc.want) || m.keptLeaf != tc.leaf {
				t.Fatalf("output %v leaf %d err %v", out, m.keptLeaf, err)
			}
		})
	}
}

type inklingMetalCommittedTreeFixture struct {
	inklingMetalTreeFixture
	mask uint32
}

func (m *inklingMetalCommittedTreeFixture) ProposeVerifyCommit(anchor int32, depth, width, maximum int, eos int32) ([][3]int32, []int32, uint32, error) {
	if maximum != 3 || depth != 2 || width != 4 {
		return nil, nil, 0, errors.New("unexpected committed row limit")
	}
	rows, err := m.DraftTree(anchor, depth, width)
	if err != nil {
		return nil, nil, 0, err
	}
	ids, err := m.BeginTreeIDs(rows)
	return rows, ids, m.mask, err
}

func TestInkling2MetalCommittedTreeChecksRetainedAncestors(t *testing.T) {
	for _, tc := range []struct {
		name string
		eos  int
		mask uint32
		want []int
	}{
		{"noncontiguous", -1, 0b1101, []int{1, 2, 3, 4}},
		{"eos", 3, 0b0101, []int{1, 2}},
		{"wrong-branch", -1, 0b1011, nil},
		{"kept-eos", 3, 0b1101, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &inklingMetalCommittedTreeFixture{mask: tc.mask}
			m.want = []int32{1, 2, 3, 4}
			m.keptLeaf = -2
			got, _, err := Inkling2MetalGreedy(context.Background(), m, inklingMetalTestDraft{}, Inkling2MetalGreedyRequest{Prompt: []int{12}, MaxTokens: 4, Vocab: 16, EOS: tc.eos, VerifyRows: 5, DraftDepth: 3, TreeWidth: 4})
			if (err != nil) != (tc.want == nil) || !slices.Equal(got, tc.want) || m.keptLeaf != -2 {
				t.Fatalf("output %v, extra commit leaf %d, error %v", got, m.keptLeaf, err)
			}
		})
	}
}
