package tokenizer

import (
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestBPESpansPreserveAllocatingSymbols(t *testing.T) {
	mr := map[tokenPair]int{
		{"a", "b"}: 0, {"ab", "c"}: 1, {"a", "a"}: 2, {"aa", "aa"}: 3,
		{"▁", "▁"}: 4, {"é", "😀"}: 5, {"�", "a"}: 6, {"�", "�"}: 7,
	}
	for _, tc := range []struct{ name, input string }{
		{"empty", ""}, {"single", "�"}, {"merge_chain", "abcabc"},
		{"overlapping_ties", "aaaaa"}, {"multibyte", "▁▁é😀▁"},
		{"invalid_single", "a\xffb"}, {"invalid_adjacent", "\xff\xfea"},
		{"invalid_truncated", "a\xf0\x9f"}, {"no_merges", strings.Repeat("z", 1024)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tok := &Tokenizer{mergeRank: mr}
			got, err := tok.bpe(tc.input)
			want, refErr := allocatingBPEReference(tok, tc.input)
			if err != nil || refErr != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("span symbols=%q err=%v; allocating symbols=%q err=%v", got, err, want, refErr)
			}
			if naive := bpeNaive(mr, tc.input); !reflect.DeepEqual(got, naive) {
				t.Fatalf("span symbols=%q; naive symbols=%q", got, naive)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestBPESpansReachEncodeConsumers(t *testing.T) {
	tok, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},"merges":["a b","ab c"]},"added_tokens":[{"id":9,"content":"<x>","special":true}],"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func() ([]int, error)
		want []int
	}{
		{"encode", func() ([]int, error) { return tok.Encode("abc<x>abc") }, []int{4, 9, 4}},
		{"trusted", func() ([]int, error) { return tok.EncodeSegments([]Segment{{Text: "abc<x>abc", Trusted: true}}) }, []int{4, 9, 4}},
		{"ordinary_seam", func() ([]int, error) { return tok.EncodeSegments([]Segment{{Text: "a"}, {Text: "bc"}}) }, []int{4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.run()
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ids=%v err=%v; want %v", got, err, tc.want)
			}
		})
	}
	gemma, err := FromGGML([]string{"a", "b", "▁", "�", "ab", "�a"}, []string{"� a", "a b"}, nil, "gemma4")
	if err != nil {
		t.Fatal(err)
	}
	t.Run("metaspace_invalid_utf8", func(t *testing.T) {
		got, err := gemma.Encode("\xff\xfeab")
		if err != nil || !reflect.DeepEqual(got, []int{3, 5, 1}) {
			t.Fatalf("invalid-byte ids=%v err=%v; want [3 5 1]", got, err)
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestBPESpansReduceSymbolAllocations(t *testing.T) {
	tok := &Tokenizer{mergeRank: doublingMergeRank()}
	for _, fixture := range []struct{ name, input string }{
		{"short", "defoo"}, {"no_merge", strings.Repeat("z", 1024)}, {"metaspace", benchInput(1024)},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			measure := func(encode func(*Tokenizer, string) ([]string, error)) float64 {
				return testing.AllocsPerRun(16, func() {
					got, err := encode(tok, fixture.input)
					if err != nil || len(got) == 0 {
						t.Fatalf("empty or failed encoding: %v", err)
					}
				})
			}
			old := measure(allocatingBPEReference)
			spans := measure((*Tokenizer).bpe)
			t.Logf("fixture-local allocations: allocating=%.0f spans=%.0f", old, spans)
			if spans >= old {
				t.Fatalf("symbol allocation reduction missing: spans=%v allocating=%v", spans, old)
			}
		})
	}
}

// The exact pre-span heap algorithm is retained as the paired allocation and
// symbol oracle. Only function/type names and omitted comments differ from the base.
func allocatingBPEReference(t *Tokenizer, encoded string) ([]string, error) {
	if len(encoded) == 0 {
		return nil, nil
	}
	nodes := make([]allocatingBPENode, 0, len(encoded))
	for _, r := range encoded {
		nodes = append(nodes, allocatingBPENode{text: string(r)})
	}
	n := len(nodes)
	for i := range nodes {
		nodes[i].prev = i - 1
		nodes[i].next = i + 1
	}
	nodes[n-1].next = -1
	h := make(mergeHeap, 0, n)
	push := func(pos int) {
		nx := nodes[pos].next
		if nx < 0 {
			return
		}
		if rank, ok := t.mergeRank[tokenPair{left: nodes[pos].text, right: nodes[nx].text}]; ok {
			h.push(mergeCandidate{rank: rank, pos: pos})
		}
	}
	for i := 0; i < n; i++ {
		push(i)
	}
	for len(h) > 0 {
		c := h.pop()
		left := &nodes[c.pos]
		if left.dead || left.next < 0 {
			continue
		}
		right := &nodes[left.next]
		rank, ok := t.mergeRank[tokenPair{left: left.text, right: right.text}]
		if !ok || rank != c.rank {
			continue
		}
		left.text += right.text
		right.dead = true
		left.next = right.next
		if right.next >= 0 {
			nodes[right.next].prev = c.pos
		}
		if left.prev >= 0 {
			push(left.prev)
		}
		push(c.pos)
	}
	syms := make([]string, 0, n)
	for i := 0; i >= 0; i = nodes[i].next {
		syms = append(syms, nodes[i].text)
	}
	return syms, nil
}

type allocatingBPENode struct {
	text string
	prev int
	next int
	dead bool
}

// These paired helper measurements report raw allocations and time, not request
// latency. Alternating arm order exposes order effects; preserve every row.
func BenchmarkBPESpanPairs(b *testing.B) {
	tok := &Tokenizer{mergeRank: doublingMergeRank()}
	for _, fixture := range []struct{ name, input string }{
		{"short", "defoo"}, {"no_merge", strings.Repeat("z", 1024)}, {"metaspace", benchInput(1024)},
	} {
		for round := 0; round < 4; round++ {
			arms := []string{"allocating", "spans"}
			if round%2 != 0 {
				arms[0], arms[1] = arms[1], arms[0]
			}
			for _, arm := range arms {
				b.Run(fixture.name+"/round_"+strconv.Itoa(round)+"/"+arm, func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						var got []string
						var err error
						if arm == "spans" {
							got, err = tok.bpe(fixture.input)
						} else {
							got, err = allocatingBPEReference(tok, fixture.input)
						}
						if err != nil || len(got) == 0 {
							b.Fatalf("empty or failed encoding: %v", err)
						}
					}
				})
			}
		}
	}
}
