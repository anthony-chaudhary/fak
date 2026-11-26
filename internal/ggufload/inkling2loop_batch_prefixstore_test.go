package ggufload

import (
	"strings"
	"testing"
)

func storeCkpt(pos, floats int) *ink2RowCkpt {
	return &ink2RowCkpt{pos: pos, slots: []ink2SlotCkpt{{k: make([]float32, floats)}}}
}

func seqTokens(n, base int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = base + i
	}
	return out
}

// A hit is the longest entry that is a prefix of the prompt and still leaves the
// prompt's last token to compute; an entry as long as the prompt, or one that departs
// from it, is no hit.
func TestInkling2PrefixStoreLookupLongestProperPrefix(t *testing.T) {
	st := &inkling2PrefixStore{capBytes: 1 << 20, minTokens: 2}
	sys := seqTokens(8, 100)
	turn1 := append(append([]int(nil), sys...), 7, 8, 9)
	st.put(sys, storeCkpt(len(sys), 4))
	st.put(turn1, storeCkpt(len(turn1), 4))
	if e := st.lookup(append(append([]int(nil), turn1...), 1, 2)); e == nil || len(e.tokens) != len(turn1) {
		t.Fatalf("next turn: hit %v, want the %d-token turn entry", e, len(turn1))
	}
	if e := st.lookup(append(append([]int(nil), sys...), 5)); e == nil || len(e.tokens) != len(sys) {
		t.Fatalf("other conversation on the system prompt: hit %v, want the system prompt", e)
	}
	if e := st.lookup(turn1); e == nil || len(e.tokens) != len(sys) {
		t.Fatalf("prompt equal to an entry: hit %v, want the shorter system prompt so its last token is computed", e)
	}
	if e := st.lookup(seqTokens(20, 0)); e != nil {
		t.Fatalf("unrelated prompt hit %d tokens", len(e.tokens))
	}
	if s := st.snapshot(); s.Lookups != 4 || s.Hits != 3 || s.Entries != 2 {
		t.Fatalf("stats %+v, want 4 lookups, 3 hits, 2 entries", s)
	}
}

// The branch point is where a prompt leaves a stored entry or a peer's prompt,
// capped one short of its end, past the row's restored cursor, and long enough;
// a branch point some entry already ends at is no branch.
func TestInkling2PrefixStoreBranchPoint(t *testing.T) {
	st := &inkling2PrefixStore{capBytes: 1 << 20, minTokens: 4}
	sys := seqTokens(10, 100)
	a := append(append([]int(nil), sys...), 1, 2, 3)
	b := append(append([]int(nil), sys...), 4, 5)
	st.put(a, storeCkpt(len(a), 1))
	if k := st.branch(b, 0, nil, -1); k != len(sys) {
		t.Fatalf("branch vs a stored conversation = %d, want %d", k, len(sys))
	}
	if k := st.branch(b, len(sys), nil, -1); k != 0 {
		t.Fatalf("branch at the restored cursor = %d, want none", k)
	}
	st.put(sys, storeCkpt(len(sys), 1))
	if k := st.branch(b, 0, nil, -1); k != 0 {
		t.Fatalf("branch an entry ends at = %d, want none (lookup serves it)", k)
	}
	empty := &inkling2PrefixStore{capBytes: 1 << 20, minTokens: 4}
	if k := empty.branch(b, 0, [][]int{a, b}, 1); k != len(sys) {
		t.Fatalf("branch vs a peer prompt = %d, want %d", k, len(sys))
	}
	if k := empty.branch(b, 0, [][]int{b, b}, 0); k != len(b)-1 {
		t.Fatalf("branch vs an identical peer = %d, want one short of the prompt (%d)", k, len(b)-1)
	}
	if k := empty.branch(seqTokens(3, 100), 0, [][]int{sys}, -1); k != 0 {
		t.Fatalf("branch shorter than the minimum = %d, want none", k)
	}
}

// The store never holds more than its cap: the least recently used entry goes first,
// a hit refreshes an entry, a checkpoint larger than the cap or shorter than the
// minimum is not kept, and a second save of the same tokens replaces the first.
func TestInkling2PrefixStoreEvictsToCap(t *testing.T) {
	per := storeCkpt(4, 100).Bytes() + 8*4
	st := &inkling2PrefixStore{capBytes: 2*per + per/2, minTokens: 4}
	e1, e2, e3 := seqTokens(4, 10), seqTokens(4, 20), seqTokens(4, 30)
	st.put(e1, storeCkpt(4, 100))
	st.put(e2, storeCkpt(4, 100))
	st.lookup(append(append([]int(nil), e1...), 0))
	st.put(e3, storeCkpt(4, 100))
	if st.held > st.capBytes || len(st.entries) != 2 {
		t.Fatalf("held %d of cap %d over %d entries", st.held, st.capBytes, len(st.entries))
	}
	for _, e := range st.entries {
		if e.tokens[0] == 20 {
			t.Fatal("the least recently used entry survived; the refreshed one should have")
		}
	}
	st.put(e3, storeCkpt(4, 100))
	if len(st.entries) != 2 || st.held != 2*per {
		t.Fatalf("re-save of the same tokens: %d entries, held %d, want 2 and %d", len(st.entries), st.held, 2*per)
	}
	st.put(seqTokens(4, 40), storeCkpt(4, 1000))
	st.put(seqTokens(3, 50), storeCkpt(3, 1))
	if len(st.entries) != 2 {
		t.Fatalf("an oversized or too-short checkpoint was kept: %d entries", len(st.entries))
	}
	if s := st.snapshot(); s.Evictions != 1 || s.HeldBytes != st.held {
		t.Fatalf("stats %+v", s)
	}
}

// The knob is off unless set, a MiB count when set, and refuses a malformed value
// rather than serving silently without the store.
func TestInkling2PrefixStoreFromEnv(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	if st, err := inkling2PrefixStoreFromEnv(env(nil)); st != nil || err != nil {
		t.Fatalf("unset: %v %v, want off", st, err)
	}
	if st, err := inkling2PrefixStoreFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "0"})); st != nil || err != nil {
		t.Fatalf("0: %v %v, want off", st, err)
	}
	st, err := inkling2PrefixStoreFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "512", "FAK_INKLING2_BATCH_PREFIX_MIN": "32"}))
	if err != nil || st == nil || st.capBytes != 512<<20 || st.minTokens != 32 {
		t.Fatalf("512/32: %+v %v", st, err)
	}
	for _, kv := range []map[string]string{
		{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "lots"},
		{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64", "FAK_INKLING2_BATCH_PREFIX_MIN": "0"},
	} {
		if _, err := inkling2PrefixStoreFromEnv(env(kv)); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_BATCH_PREFIX") {
			t.Errorf("%v: err %v, want a refusal naming the knob", kv, err)
		}
	}
}
