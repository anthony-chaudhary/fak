package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestRaggedVerifyScatter(t *testing.T) {
	t.Parallel()
	// Four requests packed into one dense verify batch of six tokens. Request 1
	// drafted nothing and request 3 drafted one token, so no fixed stride maps
	// a packed index to its owner.
	draftLens := []int{3, 0, 2, 1}
	pack, err := NewRaggedVerifyPack(draftLens)
	if err != nil {
		t.Fatal(err)
	}
	if pack.Requests() != 4 || pack.Tokens() != 6 {
		t.Fatalf("pack = %d requests, %d tokens; want 4, 6", pack.Requests(), pack.Tokens())
	}

	t.Run("each packed token maps to its owner and position", func(t *testing.T) {
		want := []RaggedVerifyOwner{{0, 0}, {0, 1}, {0, 2}, {2, 0}, {2, 1}, {3, 0}}
		for token, w := range want {
			got, ok := pack.Owner(token)
			if !ok || got != w {
				t.Fatalf("Owner(%d) = %+v, %v; want %+v", token, got, ok, w)
			}
		}
		for _, token := range []int{-1, 6, 100} {
			if got, ok := pack.Owner(token); ok {
				t.Fatalf("Owner(%d) = %+v, want no owner", token, got)
			}
		}
		// The mapping and the spans are inverses of each other.
		for r := 0; r < pack.Requests(); r++ {
			start, end, ok := pack.Span(r)
			if !ok || end-start != draftLens[r] {
				t.Fatalf("Span(%d) = [%d,%d) %v, want length %d", r, start, end, ok, draftLens[r])
			}
			for tok := start; tok < end; tok++ {
				if o, _ := pack.Owner(tok); o.Request != r || o.Position != tok-start {
					t.Fatalf("token %d in span of request %d resolves to %+v", tok, r, o)
				}
			}
		}
		if _, _, ok := pack.Span(4); ok {
			t.Fatal("Span past the last request admitted")
		}
	})

	t.Run("packed results return to the owning requests", func(t *testing.T) {
		packed := []string{"a0", "a1", "a2", "c0", "c1", "d0", "pad", "pad"}
		got, err := ScatterRaggedVerify(pack, packed)
		if err != nil {
			t.Fatal(err)
		}
		want := [][]string{{"a0", "a1", "a2"}, {}, {"c0", "c1"}, {"d0"}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("ScatterRaggedVerify = %q, want %q", got, want)
		}
		// A uniform two-per-request stride, the positional assumption this
		// replaces, would hand request 1 tokens that belong to requests 0 and 2.
		if positional := packed[2:4]; reflect.DeepEqual(got[1], positional) {
			t.Fatalf("request 1 received positionally sliced tokens %q", positional)
		}
		got[0][0] = "mutated"
		if packed[0] != "a0" {
			t.Fatal("scattered slice aliases the packed batch")
		}
	})

	t.Run("acceptances commit per request", func(t *testing.T) {
		// Request 0: accept, reject, accept -> only the leading token commits.
		// Request 2: both accepted. Request 3: rejected. Padding reads accepted
		// and must reach nobody.
		accepted := []bool{true, false, true, true, true, false, true, true}
		got, err := pack.CommitAcceptances(accepted)
		if err != nil {
			t.Fatal(err)
		}
		want := []RaggedVerifyCommit{
			{Request: 0, Accepted: 1, PackedStart: 0, PackedEnd: 1},
			{Request: 1, Accepted: 0, PackedStart: 3, PackedEnd: 3},
			{Request: 2, Accepted: 2, PackedStart: 3, PackedEnd: 5},
			{Request: 3, Accepted: 0, PackedStart: 5, PackedEnd: 5},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("CommitAcceptances = %+v, want %+v", got, want)
		}
		// Every committed cache range lies inside its own request's span.
		for _, c := range got {
			start, end, _ := pack.Span(c.Request)
			if c.PackedStart != start || c.PackedEnd > end || c.PackedEnd-c.PackedStart != c.Accepted {
				t.Fatalf("commit %+v escapes span [%d,%d)", c, start, end)
			}
		}
	})

	t.Run("an accepted run stops at the request boundary", func(t *testing.T) {
		allAccepted := []bool{true, true, true, true, true, true}
		got, err := pack.CommitAcceptances(allAccepted)
		if err != nil {
			t.Fatal(err)
		}
		for r, c := range got {
			if c.Accepted != draftLens[r] {
				t.Fatalf("request %d accepted %d of %d drafts", r, c.Accepted, draftLens[r])
			}
		}
		// Trailing zero-draft request: it owns no token and commits nothing even
		// though the token before its empty span was accepted.
		tail, err := NewRaggedVerifyPack([]int{2, 0})
		if err != nil {
			t.Fatal(err)
		}
		commits, err := tail.CommitAcceptances([]bool{true, true, true})
		if err != nil {
			t.Fatal(err)
		}
		if commits[0].Accepted != 2 || commits[1] != (RaggedVerifyCommit{Request: 1, PackedStart: 2, PackedEnd: 2}) {
			t.Fatalf("trailing empty request commits = %+v", commits)
		}
	})

	t.Run("graph padding maps to the ghost row", func(t *testing.T) {
		rows, err := pack.TokenRows(8)
		if err != nil {
			t.Fatal(err)
		}
		if want := []int{0, 0, 0, 2, 2, 3, 4, 4}; !reflect.DeepEqual(rows, want) {
			t.Fatalf("TokenRows = %v, want %v", rows, want)
		}
		exact, err := pack.TokenRows(6)
		if err != nil || !reflect.DeepEqual(exact, []int{0, 0, 0, 2, 2, 3}) {
			t.Fatalf("TokenRows with no padding = %v, %v", exact, err)
		}
		if _, err := pack.TokenRows(5); !errors.Is(err, ErrRaggedVerifyScatter) {
			t.Fatalf("TokenRows with too few slots admitted: %v", err)
		}
	})

	t.Run("a batch with no draft tokens at all", func(t *testing.T) {
		empty, err := NewRaggedVerifyPack([]int{0, 0})
		if err != nil {
			t.Fatal(err)
		}
		commits, err := empty.CommitAcceptances(nil)
		if err != nil || len(commits) != 2 || commits[0].Accepted != 0 || commits[1].Accepted != 0 {
			t.Fatalf("all-empty batch commits = %+v, %v", commits, err)
		}
		scattered, err := ScatterRaggedVerify(empty, []int{})
		if err != nil || !reflect.DeepEqual(scattered, [][]int{{}, {}}) {
			t.Fatalf("all-empty batch scatter = %v, %v", scattered, err)
		}
	})

	t.Run("fail closed", func(t *testing.T) {
		if _, err := NewRaggedVerifyPack(nil); !errors.Is(err, ErrRaggedVerifyScatter) {
			t.Fatalf("pack with no requests admitted: %v", err)
		}
		if _, err := NewRaggedVerifyPack([]int{2, -1}); !errors.Is(err, ErrRaggedVerifyScatter) {
			t.Fatalf("negative draft length admitted: %v", err)
		}
		if _, err := ScatterRaggedVerify(pack, []int{1, 2, 3, 4, 5}); !errors.Is(err, ErrRaggedVerifyScatter) {
			t.Fatalf("short packed result admitted: %v", err)
		}
		if _, err := pack.CommitAcceptances([]bool{true}); !errors.Is(err, ErrRaggedVerifyScatter) {
			t.Fatalf("short verdict vector admitted: %v", err)
		}
		var zero RaggedVerifyPack
		var nilPack *RaggedVerifyPack
		for _, p := range []*RaggedVerifyPack{&zero, nilPack} {
			if p.Requests() != 0 || p.Tokens() != 0 {
				t.Fatal("empty pack reports contents")
			}
			if _, err := p.CommitAcceptances(nil); !errors.Is(err, ErrRaggedVerifyScatter) {
				t.Fatalf("empty pack committed: %v", err)
			}
			if _, err := ScatterRaggedVerify(p, []int{}); !errors.Is(err, ErrRaggedVerifyScatter) {
				t.Fatalf("empty pack scattered: %v", err)
			}
			if _, err := p.TokenRows(0); !errors.Is(err, ErrRaggedVerifyScatter) {
				t.Fatalf("empty pack produced token rows: %v", err)
			}
			if _, ok := p.Owner(0); ok {
				t.Fatal("empty pack resolved an owner")
			}
		}
	})
}
