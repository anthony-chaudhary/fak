package ggufload

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2PrefixStore keeps cohort row checkpoints across cohorts, keyed by the exact
// tokens a row had fed, so a later request whose prompt starts with those tokens
// restores the row instead of prefilling them. A checkpoint of a looped, sliding-window
// row cannot be cut shorter (its conv histories and local rings hold only the state at
// its own end), so entries are whole: a hit is the longest entry that is a prefix of
// the prompt, held one token short of it so the row still computes its own first
// token. Where a prompt shares a long prefix with an entry but no entry ends there,
// the row prefills to that branch point and checkpoints it, so the next request of
// the same family (a shared system prompt) hits it. Entries are host memory, capped
// in bytes, and evicted least recently used.
type inkling2PrefixStore struct {
	capBytes  int64
	minTokens int
	// loopSlots is the KV slot count of the loop config the store is bound to; a
	// checkpoint at another loop depth has a different slot layout and is refused.
	// Zero leaves the store unbound.
	loopSlots int
	held      int64
	tick      uint64
	entries   []*ink2PrefixEntry
	stats     Inkling2PrefixStoreStats
}

type ink2PrefixEntry struct {
	tokens []int
	ck     *ink2RowCkpt
	bytes  int64
	used   uint64
}

// Inkling2PrefixStoreStats counts what the cross-request prefix store did.
type Inkling2PrefixStoreStats struct {
	Lookups, Hits    uint64
	HitTokens        uint64
	Saves, Evictions uint64
	Entries          int
	HeldBytes        int64
}

// inkling2PrefixMinDefault is the shortest prefix worth a checkpoint: below it the
// copy costs about what the prefill it saves does.
const inkling2PrefixMinDefault = 64

// inkling2PrefixStoreFromEnv reads FAK_INKLING2_BATCH_PREFIX_CACHE_MIB, the host
// memory the cross-request prefix store may hold (unset or 0 keeps it off), and
// FAK_INKLING2_BATCH_PREFIX_MIN, the shortest prefix it checkpoints. A malformed value
// is an error, so an operator who set the knob is never silently served without it.
func inkling2PrefixStoreFromEnv(lookup func(string) (string, bool)) (*inkling2PrefixStore, error) {
	v, _ := lookup("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB")
	v = strings.TrimSpace(v)
	if v == "" || v == "0" {
		return nil, nil
	}
	mib, err := strconv.Atoi(v)
	if err != nil || mib < 0 {
		return nil, fmt.Errorf("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB=%q is not a MiB count", v)
	}
	least := inkling2PrefixMinDefault
	if m, ok := lookup("FAK_INKLING2_BATCH_PREFIX_MIN"); ok && strings.TrimSpace(m) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(m))
		if err != nil || n < 1 {
			return nil, fmt.Errorf("FAK_INKLING2_BATCH_PREFIX_MIN=%q is not a positive token count", m)
		}
		least = n
	}
	return &inkling2PrefixStore{capBytes: int64(mib) << 20, minTokens: least}, nil
}

// Inkling2BatchPrefixCacheMiB reports the host MiB FAK_INKLING2_BATCH_PREFIX_CACHE_MIB
// gives the cross-request prefix store on be and whether that turns it on. It refuses
// a malformed value, and a store the cohorts could never use: restoring a row
// mid-prompt needs cohort prefill, which FAK_INKLING2_COHORT_PREFILL=0 turns off and
// f16 batch KV leaves off unless FAK_INKLING2_COHORT_PREFILL_KV=f16.
func Inkling2BatchPrefixCacheMiB(be any, lookup func(string) (string, bool)) (int, bool, error) {
	st, err := inkling2PrefixStoreFromEnv(lookup)
	if st == nil || err != nil {
		return 0, false, err
	}
	if err := inkling2PrefixStoreUsable(be, lookup); err != nil {
		return 0, false, err
	}
	return int(st.capBytes >> 20), true, nil
}

func inkling2PrefixStoreUsable(be any, lookup func(string) (string, bool)) error {
	half, err := inkling2BatchKVHalf(lookup)
	if err != nil {
		return err
	}
	on, _, err := inkling2CohortPrefillKV(be, half, lookup)
	if err != nil {
		return err
	}
	if !on {
		return fmt.Errorf("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB needs cohort prefill: FAK_INKLING2_COHORT_PREFILL is 0, or f16 batch KV runs without FAK_INKLING2_COHORT_PREFILL_KV=f16")
	}
	return nil
}

// UsePrefixStore turns on the cross-request prefix store FAK_INKLING2_BATCH_PREFIX_CACHE_MIB
// sizes; it reports whether the store is on. Cohorts then restore rows from it, record
// their branch points and finished rows in it, and run as narrow as one row. A store
// Inkling2BatchPrefixCacheMiB refuses stays off.
func (bt *Inkling2LoopCohorts) UsePrefixStore(lookup func(string) (string, bool)) (bool, error) {
	st, err := inkling2PrefixStoreFromEnv(lookup)
	if err == nil && st != nil {
		err = inkling2PrefixStoreUsable(bt.be, lookup)
	}
	if err != nil {
		return false, err
	}
	if st != nil && bt.m != nil {
		st.loopSlots = inkling2LoopSlots(bt.m.HP)
	}
	bt.store = st
	return st != nil, nil
}

// inkling2LoopSlots is the KV slot count one row holds at hp's loop depth: every layer
// visit gets a slot, and each extra core pass adds its core slots again.
func inkling2LoopSlots(hp model.Inkling2LoopHParams) int {
	return hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
}

// PrefixStats reads the prefix store's counters; zero when the store is off.
func (bt *Inkling2LoopCohorts) PrefixStats() Inkling2PrefixStoreStats {
	if bt.store == nil {
		return Inkling2PrefixStoreStats{}
	}
	return bt.store.snapshot()
}

func inkling2HasPrefix(prompt, tokens []int) bool {
	if len(tokens) > len(prompt) {
		return false
	}
	for i, t := range tokens {
		if prompt[i] != t {
			return false
		}
	}
	return true
}

// lookup returns the longest entry that is a prefix of prompt and leaves at least its
// last token to compute, or nil.
func (st *inkling2PrefixStore) lookup(prompt []int) *ink2PrefixEntry {
	st.stats.Lookups++
	var best *ink2PrefixEntry
	for _, e := range st.entries {
		if len(e.tokens) < len(prompt) && (best == nil || len(e.tokens) > len(best.tokens)) && inkling2HasPrefix(prompt, e.tokens) {
			best = e
		}
	}
	if best != nil {
		st.tick++
		best.used = st.tick
		st.stats.Hits++
		st.stats.HitTokens += uint64(len(best.tokens))
	}
	return best
}

// branch is the position past from at which prompt leaves the tokens of a stored
// entry or of another cohort lane's prompt (peers, skipping self), one token short of
// the prompt's end at most, when that point is long enough to checkpoint and no
// entry already ends there; 0 otherwise.
func (st *inkling2PrefixStore) branch(prompt []int, from int, peers [][]int, self int) int {
	k := 0
	common := func(other []int) {
		n := min(len(other), len(prompt)-1)
		j := 0
		for j < n && other[j] == prompt[j] {
			j++
		}
		k = max(k, j)
	}
	for _, e := range st.entries {
		common(e.tokens)
	}
	for i, p := range peers {
		if i != self {
			common(p)
		}
	}
	if k < st.minTokens || k <= from {
		return 0
	}
	for _, e := range st.entries {
		if len(e.tokens) == k && inkling2HasPrefix(prompt, e.tokens) {
			return 0
		}
	}
	return k
}

// put records ck as the state after tokens, replacing an entry for the same tokens,
// then evicts least recently used entries until the store is within its cap. A
// checkpoint shorter than the minimum, larger than the whole cap, or at another loop
// depth than the store is bound to is not kept.
func (st *inkling2PrefixStore) put(tokens []int, ck *ink2RowCkpt) {
	if ck == nil || len(tokens) != ck.pos || len(tokens) < st.minTokens {
		return
	}
	if st.loopSlots != 0 && len(ck.slots) != st.loopSlots {
		return
	}
	bytes := ck.Bytes() + int64(8*len(tokens))
	if bytes > st.capBytes {
		return
	}
	for i, e := range st.entries {
		if len(e.tokens) == len(tokens) && inkling2HasPrefix(tokens, e.tokens) {
			st.held -= e.bytes
			st.entries = append(st.entries[:i], st.entries[i+1:]...)
			break
		}
	}
	st.tick++
	st.entries = append(st.entries, &ink2PrefixEntry{tokens: append([]int(nil), tokens...), ck: ck, bytes: bytes, used: st.tick})
	st.held += bytes
	st.stats.Saves++
	for st.held > st.capBytes && len(st.entries) > 0 {
		lru := 0
		for i, e := range st.entries {
			if e.used < st.entries[lru].used {
				lru = i
			}
		}
		st.held -= st.entries[lru].bytes
		st.entries = append(st.entries[:lru], st.entries[lru+1:]...)
		st.stats.Evictions++
	}
}

func (st *inkling2PrefixStore) snapshot() Inkling2PrefixStoreStats {
	s := st.stats
	s.Entries, s.HeldBytes = len(st.entries), st.held
	return s
}
