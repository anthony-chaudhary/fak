package adjudicator

import (
	"math/rand"
	"reflect"
	"testing"
)

// TestComputeDeltaLintExcludesPreExisting is the issue's named witness: a finding
// present in the pre-edit baseline (after projection) is excluded, and only the
// genuinely new finding is emitted.
func TestComputeDeltaLintExcludesPreExisting(t *testing.T) {
	pre := []LintError{
		{Code: "G1", File: "a.go", Line: 2, Col: 1, Msg: "old"},
		{Code: "G2", File: "a.go", Line: 10, Col: 5, Msg: "old"},
	}
	// The edit inserted 3 lines at line 5, so the old line-10 finding now sits
	// at line 13 in post-edit space and must be excluded; line 2 is untouched.
	post := []LintError{
		{Code: "G1", File: "a.go", Line: 2, Col: 1, Msg: "old"},
		{Code: "G2", File: "a.go", Line: 13, Col: 5, Msg: "old"},
		{Code: "G3", File: "a.go", Line: 20, Col: 2, Msg: "new"},
	}
	got := ComputeDeltaLint(pre, post, 5, 3)
	want := []LintError{{Code: "G3", File: "a.go", Line: 20, Col: 2, Msg: "new"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("newly introduced = %+v, want %+v", got, want)
	}
}

func TestComputeDeltaLintTable(t *testing.T) {
	tests := []struct {
		name       string
		pre, post  []LintError
		edit, dlt  int
		want       []LintError
		wantNilOut bool
	}{
		{
			name: "no pre errors: all post are new",
			pre:  nil,
			post: []LintError{{Code: "A", File: "f.go", Line: 1}},
			edit: 1, dlt: 1,
			want: []LintError{{Code: "A", File: "f.go", Line: 1}},
		},
		{
			name: "all pre errors persist at shifted lines: none new",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 10}},
			post: []LintError{{Code: "A", File: "f.go", Line: 12}},
			edit: 1, dlt: 2,
			wantNilOut: true,
		},
		{
			name: "deletion shifts upward",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 10}},
			post: []LintError{{Code: "A", File: "f.go", Line: 7}},
			edit: 4, dlt: -3,
			wantNilOut: true,
		},
		{
			name: "pre error before the edit line is not shifted",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 2}},
			post: []LintError{{Code: "A", File: "f.go", Line: 5}},
			edit: 4, dlt: 3,
			want: []LintError{{Code: "A", File: "f.go", Line: 5}},
		},
		{
			name: "same code different column is a new finding",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 5, Col: 1}},
			post: []LintError{{Code: "A", File: "f.go", Line: 5, Col: 9}},
			edit: 1, dlt: 0,
			want: []LintError{{Code: "A", File: "f.go", Line: 5, Col: 9}},
		},
		{
			name: "same code different file is a new finding",
			pre:  []LintError{{Code: "A", File: "a.go", Line: 5}},
			post: []LintError{{Code: "A", File: "b.go", Line: 5}},
			edit: 1, dlt: 0,
			want: []LintError{{Code: "A", File: "b.go", Line: 5}},
		},
		{
			name: "empty post yields nil",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 1}},
			post: nil,
			edit: 1, dlt: 1,
			wantNilOut: true,
		},
		{
			name: "non-positive edit line means no projection",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 10}},
			post: []LintError{{Code: "A", File: "f.go", Line: 10}},
			edit: 0, dlt: 5,
			wantNilOut: true,
		},
		{
			name: "duplicate pre entries still exclude a single post twin",
			pre:  []LintError{{Code: "A", File: "f.go", Line: 3}, {Code: "A", File: "f.go", Line: 3}},
			post: []LintError{{Code: "A", File: "f.go", Line: 3}},
			edit: 1, dlt: 0,
			wantNilOut: true,
		},
		{
			name: "order of newly introduced findings is preserved",
			pre:  nil,
			post: []LintError{{Code: "B", File: "f.go", Line: 9}, {Code: "A", File: "f.go", Line: 2}},
			edit: 1, dlt: 1,
			want: []LintError{{Code: "B", File: "f.go", Line: 9}, {Code: "A", File: "f.go", Line: 2}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeDeltaLint(tt.pre, tt.post, tt.edit, tt.dlt)
			if tt.wantNilOut {
				if len(got) != 0 {
					t.Fatalf("got %+v, want none", got)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

// TestComputeDeltaLintMatchesReference is an independent brute-force oracle: it
// re-derives the expected result with a literal membership predicate and checks
// ComputeDeltaLint agrees across randomized inputs (including negative deltas
// and duplicate codes).
func TestComputeDeltaLintMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(10973))
	for i := 0; i < 2000; i++ {
		nPre := rng.Intn(4)
		nPost := rng.Intn(4)
		pre := make([]LintError, nPre)
		for j := range pre {
			pre[j] = LintError{
				Code: string(rune('A' + rng.Intn(3))),
				File: string(rune('a'+rng.Intn(2))) + ".go",
				Line: 1 + rng.Intn(12),
				Col:  1 + rng.Intn(3),
			}
		}
		post := make([]LintError, nPost)
		for j := range post {
			post[j] = LintError{
				Code: string(rune('A' + rng.Intn(3))),
				File: string(rune('a'+rng.Intn(2))) + ".go",
				Line: 1 + rng.Intn(12),
				Col:  1 + rng.Intn(3),
			}
		}
		edit := rng.Intn(14)
		dlt := rng.Intn(7) - 3

		want := referenceDelta(pre, post, edit, dlt)
		got := ComputeDeltaLint(pre, post, edit, dlt)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("case %d: got %+v, want %+v (pre=%+v post=%+v edit=%d dlt=%d)",
				i, got, want, pre, post, edit, dlt)
		}
	}
}

// referenceDelta is a deliberately literal re-implementation used only as a test
// oracle; it never shares code with the production join.
func referenceDelta(pre, post []LintError, editLine, linesDelta int) []LintError {
	projected := map[[4]any]bool{}
	for _, e := range pre {
		line := e.Line
		if editLine > 0 && line >= editLine {
			line += linesDelta
		}
		projected[[4]any{e.File, line, e.Col, e.Code}] = true
	}
	var out []LintError
	for _, p := range post {
		if projected[[4]any{p.File, p.Line, p.Col, p.Code}] {
			continue
		}
		out = append(out, p)
	}
	return out
}
