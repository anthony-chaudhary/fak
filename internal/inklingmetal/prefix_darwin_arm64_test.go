//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"math"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

var prefixHeader = []int32{200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 107750, 149928, 152627, 13, 149917}
var prefixPassage = []int32{220, 36598, 157269, 220, 34541, 169899, 150156, 149878, 167166, 158808, 150419, 220, 137375, 1500, 150109, 32426, 151371, 82, 193784, 154046, 302, 167627, 82, 149880, 220, 50348, 82, 220, 143948, 150749, 149876, 151594, 152397, 150207, 152504, 43538, 154366, 302, 154258, 150040, 52866, 149880, 149878, 158842, 86828, 176015, 149876, 151346, 170850, 158842, 149890, 150105, 189459, 150710, 196904, 149887, 150396, 156258, 150391, 150471, 154621, 149878, 150971, 153962, 149879, 149876, 150317, 177903, 156999, 149917}

// prefixPrompt repeats the passage after the header. From fork on it reads the
// passage backwards, starting at a token the forward prompt does not have there.
func prefixPrompt(n, fork int) []int32 {
	out := slices.Clone(prefixHeader)
	for i := 0; len(out) < n; i++ {
		id := prefixPassage[i%len(prefixPassage)]
		if len(out) >= fork {
			id = prefixPassage[len(prefixPassage)-1-i%len(prefixPassage)]
			if len(out) == fork && id == prefixPassage[i%len(prefixPassage)] {
				id = 13
			}
		}
		out = append(out, id)
	}
	return out
}

func prefixLoad() string {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// prefixSlack leaves room for the rows one verify pass stages past the reply.
const prefixSlack = 16

type prefixRun struct {
	head   []float32
	out    []int32
	trees  [][][3]int32
	first  time.Duration
	reused int
}

// prefixRequest serves one greedy request with a fixed tree depth and width,
// so two runs from equal states choose equal tokens.
func prefixRequest(t *testing.T, m *Model, prompt []int32, tokens int, reuse bool) prefixRun {
	t.Helper()
	var r prefixRun
	capacity := len(prompt) + tokens + prefixSlack
	start := time.Now()
	var s *Session
	var err error
	if reuse {
		s, err = m.NewSessionFor(capacity, "prefix-test", prompt)
	} else {
		s, err = m.NewSession(capacity)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	r.reused = s.Reused()
	if r.head, err = s.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	r.first = time.Since(start)
	next := int32(0)
	for i, x := range r.head {
		if x > r.head[next] {
			next = int32(i)
		}
	}
	for len(r.out) < tokens {
		r.out = append(r.out, next)
		tree, err := s.DraftTree(next, 7, 2)
		if err != nil {
			t.Fatal(err)
		}
		if len(tree) == 0 {
			ids, err := s.BeginBlockIDs([]int32{next})
			if err != nil {
				t.Fatal(err)
			}
			if err = s.Commit(1); err != nil {
				t.Fatal(err)
			}
			next = ids[0]
			continue
		}
		r.trees = append(r.trees, tree)
		ids, err := s.BeginTreeIDs(tree)
		if err != nil {
			t.Fatal(err)
		}
		leaf := 0
		for row := 1; row < len(tree) && len(r.out) < tokens; row++ {
			if int(tree[row][1]) == leaf && tree[row][0] == ids[leaf] {
				leaf = row
				r.out = append(r.out, tree[row][0])
			}
		}
		if err = s.CommitTree(leaf); err != nil {
			t.Fatal(err)
		}
		next = ids[leaf]
	}
	return r
}

func prefixCompare(t *testing.T, name string, got, want prefixRun) {
	t.Helper()
	differing, worst := 0, 0.0
	for i := range want.head {
		if math.Float32bits(want.head[i]) != math.Float32bits(got.head[i]) {
			differing++
			worst = math.Max(worst, math.Abs(float64(want.head[i]-got.head[i])))
		}
	}
	if differing != 0 {
		t.Errorf("%s: %d first-row scores differ from a new session's, by at most %g", name, differing, worst)
	}
	if !slices.Equal(got.out, want.out) {
		t.Errorf("%s: greedy tokens differ from a new session's:\n got %v\nwant %v", name, got.out, want.out)
	}
	if !slices.EqualFunc(got.trees, want.trees, func(a, b [][3]int32) bool { return slices.Equal(a, b) }) {
		t.Errorf("%s: draft trees differ from a new session's", name)
	}
}

// fak-test:runtime slow est=150s
func TestPrefixReuseMatchesNewSession(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	draft, err := adjacentDraft(path)
	if err != nil {
		t.Fatal(err)
	}
	const reply = 32
	for _, variant := range []struct {
		name, draft string
	}{{"target", ""}, {"draft", draft}} {
		if variant.name == "draft" && draft == "" {
			continue
		}
		t.Run(variant.name, func(t *testing.T) {
			m, err := OpenWithDraft(path, variant.draft)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			key, _, err := m.prefixKey("prefix-test")
			if err != nil {
				t.Fatal(err)
			}
			rows, window := int(key.ChunkRows), int(key.DraftWindow)
			// The first request also compiles the kernels; it is not timed.
			prefixRequest(t, m, prefixPrompt(64, 64), 2, false)
			for _, tc := range []struct {
				name               string
				first, added, fork int
			}{
				{"below a chunk", 2000, 50, 1000},
				{"past two chunks", 2*rows + 104, 50, rows + 952},
			} {
				a := prefixPrompt(tc.first, tc.first)
				b := prefixPrompt(tc.first+tc.added, tc.first+tc.added)
				forked := prefixPrompt(tc.first+tc.added, tc.fork)
				if !slices.Equal(a, b[:len(a)]) || !slices.Equal(a[:tc.fork], forked[:tc.fork]) || a[tc.fork] == forked[tc.fork] {
					t.Fatal("prompts do not share the intended prefixes")
				}
				wantB := prefixRequest(t, m, b, reply, false)
				wantFork := prefixRequest(t, m, forked, reply, false)
				if again := prefixRequest(t, m, b, reply, false); !slices.Equal(again.out, wantB.out) {
					t.Fatalf("%s: two new sessions disagree, so nothing can be compared", tc.name)
				}
				if first := prefixRequest(t, m, a, reply, true); first.reused != 0 {
					t.Fatalf("%s: unrelated idle session lent %d tokens", tc.name, first.reused)
				}
				gotB := prefixRequest(t, m, b, reply, true)
				gotFork := prefixRequest(t, m, forked, reply, true)
				extend := reuseChunks(len(a), len(b), 1, len(a)/rows, rows, window) * rows
				diverge := reuseChunks(tc.fork, len(forked), 1, len(b)/rows, rows, window) * rows
				if gotB.reused != extend || gotFork.reused != diverge {
					t.Errorf("%s: reused %d and %d tokens, want %d and %d", tc.name, gotB.reused, gotFork.reused, extend, diverge)
				}
				prefixCompare(t, tc.name+" extended", gotB, wantB)
				prefixCompare(t, tc.name+" diverging", gotFork, wantFork)
				t.Logf("%s: %d-token prompt after a %d-token one: first token %s reusing %d tokens, %s from a new session; diverging at %d: %s reusing %d tokens, %s new; load %s",
					tc.name, len(b), len(a), gotB.first, gotB.reused, wantB.first, tc.fork, gotFork.first, gotFork.reused, wantFork.first, prefixLoad())
			}
			// A prompt that differs at its first token starts over.
			other := prefixPrompt(300, 300)
			other[0] = 13
			if r := prefixRequest(t, m, other, 1, true); r.reused != 0 {
				t.Errorf("prompt differing at token 0 reused %d tokens", r.reused)
			}
		})
	}
}

// fak-test:runtime slow est=60s
func TestPrefixReuseLongPrompt(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	tokens, _ := strconv.Atoi(os.Getenv("FAK_INKLING2_PREFIX_LONG"))
	if path == "" || tokens < 1 {
		t.Skip("set FAK_INKLING2_TEST_GGUF and FAK_INKLING2_PREFIX_LONG to a prompt length to time a long prompt")
	}
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const added, reply = 50, 16
	prefixRequest(t, m, prefixPrompt(64, 64), 2, false)
	a, b := prefixPrompt(tokens, tokens), prefixPrompt(tokens+added, tokens+added)
	want := prefixRequest(t, m, b, reply, false)
	prefixRequest(t, m, a, reply, true)
	got := prefixRequest(t, m, b, reply, true)
	prefixCompare(t, "long prompt", got, want)
	t.Logf("%d-token prompt after a %d-token one, draft %v: first token %s reusing %d tokens, %s from a new session; load %s",
		len(b), len(a), m.hasDraft, got.first, got.reused, want.first, prefixLoad())
}
