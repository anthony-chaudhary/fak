package main

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func inkling2Extend(prompt []int, more ...int) []int {
	return append(append([]int(nil), prompt...), more...)
}

// Two conversations taking turns each keep their own prefix; with one slot every
// turn evicts the other's checkpoint and re-prefills its whole history.
func TestInkling2AdapterPrefixCacheInterleavedConversationsReuseOwnSlots(t *testing.T) {
	canned := []int{'a', 'b', 'c', 'd', 'e'}
	a1, b1 := []int{'s', 'y', 's', 'A', '1'}, []int{'s', 'y', 's', 'B', 'x', '1'}
	a2, b2 := inkling2Extend(a1, 'a', 'b', 'A', '2'), inkling2Extend(b1, 'a', 'b', 'B', '2')
	a3 := inkling2Extend(a2, 'a', 'A', '3')
	for _, tc := range []struct {
		slots                int
		wantA2, wantB2, want int
	}{
		{slots: 1},
		{slots: 2, wantA2: len(a1) - 1, wantB2: len(b1) - 1, want: len(a2) - 1},
	} {
		eng, opened := inkling2SlotEngine(canned, tc.slots, 1<<40, 0)
		inkling2Gen(t, eng, a1, 2)
		inkling2Gen(t, eng, b1, 2)
		if res, _ := inkling2Gen(t, eng, a2, 2); res.CachedPromptTokens != tc.wantA2 {
			t.Fatalf("slots=%d: A turn 2 cached %d, want %d", tc.slots, res.CachedPromptTokens, tc.wantA2)
		}
		if res, _ := inkling2Gen(t, eng, b2, 2); res.CachedPromptTokens != tc.wantB2 {
			t.Fatalf("slots=%d: B turn 2 cached %d, want %d", tc.slots, res.CachedPromptTokens, tc.wantB2)
		}
		res, got := inkling2Gen(t, eng, a3, 2)
		if res.CachedPromptTokens != tc.want || !inkling2EqualInts(got, []int{'a', 'b'}) {
			t.Fatalf("slots=%d: A turn 3 cached %d emitted %v, want %d", tc.slots, res.CachedPromptTokens, got, tc.want)
		}
		if len(*opened) != tc.slots {
			t.Fatalf("slots=%d: opened %d sessions", tc.slots, len(*opened))
		}
		for i, s := range *opened {
			if s.overlap.Load() {
				t.Fatalf("slots=%d: session %d ran two requests at once", tc.slots, i)
			}
		}
		if tc.slots == 2 {
			if a, b := (*opened)[0], (*opened)[1]; !inkling2EqualInts(a.prompt, a3) || !inkling2EqualInts(b.prompt, b2) {
				t.Fatalf("slot histories %v / %v, want A's and B's own", a.prompt, b.prompt)
			}
		}
		if err := eng.Close(); err != nil {
			t.Fatal(err)
		}
		for i, s := range *opened {
			if !s.closed {
				t.Fatalf("slots=%d: session %d left open by Close", tc.slots, i)
			}
		}
	}
}

// A third conversation evicts the least recently used slot, by count and by bytes.
func TestInkling2AdapterPrefixCacheEvictsLeastRecentlyUsedAtCap(t *testing.T) {
	canned := []int{'a', 'b', 'c'}
	a1, b1, c1 := []int{'A', 'p', 'q'}, []int{'B', 'p', 'q'}, []int{'C', 'p', 'q'}
	for name, build := range map[string]func() (*inkling2LoopEngine, *[]*inkling2FakeSession){
		"slot count": func() (*inkling2LoopEngine, *[]*inkling2FakeSession) { return inkling2SlotEngine(canned, 2, 1<<40, 0) },
		// two sessions of about 100 bytes fit under 250, a third does not
		"byte cap": func() (*inkling2LoopEngine, *[]*inkling2FakeSession) { return inkling2SlotEngine(canned, 8, 250, 100) },
	} {
		eng, opened := build()
		inkling2Gen(t, eng, a1, 1)
		inkling2Gen(t, eng, b1, 1)
		inkling2Gen(t, eng, c1, 1) // A is the least recently used
		if len(*opened) != 2 {
			t.Fatalf("%s: opened %d sessions, want 2", name, len(*opened))
		}
		if res, _ := inkling2Gen(t, eng, inkling2Extend(b1, 'a', 'B'), 1); res.CachedPromptTokens != len(b1)-1 {
			t.Fatalf("%s: B cached %d, want %d", name, res.CachedPromptTokens, len(b1)-1)
		}
		if res, _ := inkling2Gen(t, eng, inkling2Extend(c1, 'a', 'C'), 1); res.CachedPromptTokens != len(c1)-1 {
			t.Fatalf("%s: C cached %d, want %d", name, res.CachedPromptTokens, len(c1)-1)
		}
		if res, _ := inkling2Gen(t, eng, inkling2Extend(a1, 'a', 'A'), 1); res.CachedPromptTokens != 0 {
			t.Fatalf("%s: evicted A cached %d, want 0", name, res.CachedPromptTokens)
		}
		if len(*opened) != 2 {
			t.Fatalf("%s: opened %d sessions after eviction, want the two slots reused", name, len(*opened))
		}
		if err := eng.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// A prefill cancelled mid-flight drops its own slot and leaves the others' prefixes.
func TestInkling2AdapterPrefixCacheCancelInvalidatesOnlyItsSlot(t *testing.T) {
	entered := make(chan struct{}, 1)
	var opened []*inkling2GatedSession
	eng := &inkling2LoopEngine{slots: 2, capBytes: 1 << 40, open: func() (ggufload.Inkling2LoopSession, error) {
		g := &inkling2GatedSession{inkling2FakeSession: &inkling2FakeSession{canned: []int{'a', 'b', 'c'}}, entered: entered}
		opened = append(opened, g)
		return g, nil
	}}
	defer eng.Close()
	a1, b1 := []int{'A', 'p', 'q'}, []int{'B', 'p', 'q'}
	inkling2Gen(t, eng, a1, 1)
	inkling2Gen(t, eng, b1, 1)
	if len(opened) != 2 {
		t.Fatalf("opened %d sessions, want one per conversation", len(opened))
	}
	opened[0].block = true
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := eng.Generate(ctx, inkling2Extend(a1, 'a', 'x', 'y'), 1, nil, func(int) bool { return true })
		errc <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("A's suffix prefill never started")
	}
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled turn = %v, want context.Canceled", err)
	}
	if !opened[0].closed || opened[1].closed {
		t.Fatalf("closed A=%v B=%v, want only the cancelled slot dropped", opened[0].closed, opened[1].closed)
	}
	if res, _ := inkling2Gen(t, eng, inkling2Extend(b1, 'a', 'B'), 1); res.CachedPromptTokens != len(b1)-1 {
		t.Fatalf("B after A's cancel cached %d, want %d", res.CachedPromptTokens, len(b1)-1)
	}
	res, _ := inkling2Gen(t, eng, inkling2Extend(a1, 'a', 'x', 'y'), 1)
	if res.CachedPromptTokens != 0 || len(opened) != 3 {
		t.Fatalf("A after its cancel cached %d opened %d, want a full prefill on a fresh session", res.CachedPromptTokens, len(opened))
	}
	if s := opened[2]; len(s.prefills) == 0 || !inkling2EqualInts(s.prefills[0], []int{'A', 'p', 'q', 'a', 'x'}) {
		t.Fatalf("fresh A slot prefilled %v, want the whole prompt", s.prefills)
	}
}

// However many conversations arrive, the bytes held by more than one slot stay
// under the cap; a lone slot is the default single resident session and always fits.
func TestInkling2AdapterPrefixCacheHeldBytesNeverExceedCap(t *testing.T) {
	const capBytes = 300
	eng, opened := inkling2SlotEngine([]int{'a', 'b', 'c', 'd'}, 8, capBytes, 60)
	defer eng.Close()
	convs := make([][]int, 6)
	for i := range convs {
		convs[i] = []int{'c', 'a' + i}
	}
	for turn := 0; turn < 5; turn++ {
		for i := range convs {
			convs[i] = inkling2Extend(convs[i], 'a', 'b', 'x'+turn%3)
			inkling2Gen(t, eng, convs[i], 2)
			if held, n := eng.prefixHeldBytes(), eng.prefixSlotCount(); n > 1 && held > capBytes {
				t.Fatalf("turn %d conv %d: %d slots hold %d bytes over the %d cap", turn, i, n, held, capBytes)
			}
		}
	}
	live := 0
	for _, s := range *opened {
		if !s.closed {
			live++
		}
	}
	if live != eng.prefixSlotCount() || live < 2 {
		t.Fatalf("live sessions %d, slot table %d, want the table to own every open session and hold more than one", live, eng.prefixSlotCount())
	}

	one, _ := inkling2SlotEngine([]int{'a', 'b'}, 1, 10, 1000)
	defer one.Close()
	inkling2Gen(t, one, []int{'p', 'q'}, 1)
	if res, _ := inkling2Gen(t, one, []int{'p', 'q', 'r'}, 1); res.CachedPromptTokens != 1 || one.prefixSlotCount() != 1 {
		t.Fatalf("one slot over the cap: cached %d slots %d, want reuse on the lone slot", res.CachedPromptTokens, one.prefixSlotCount())
	}
}

// A session that cannot report its footprint keeps the cache at one slot.
func TestInkling2AdapterPrefixCacheUnmeteredSessionStaysSingleSlot(t *testing.T) {
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{slots: 4, capBytes: 1 << 40, open: func() (ggufload.Inkling2LoopSession, error) {
		s := &inkling2FakeSession{canned: []int{'a', 'b'}}
		opened = append(opened, s)
		return inkling2UnmeteredSession{s}, nil
	}}
	defer eng.Close()
	inkling2Gen(t, eng, []int{'A', 'p'}, 1)
	inkling2Gen(t, eng, []int{'B', 'p'}, 1)
	if len(opened) != 1 || eng.prefixSlotCount() != 1 {
		t.Fatalf("opened %d slots %d, want one session without a byte meter", len(opened), eng.prefixSlotCount())
	}
}

// inkling2UnmeteredSession is a rewinder without HeldBytes.
type inkling2UnmeteredSession struct{ s *inkling2FakeSession }

func (u inkling2UnmeteredSession) Prefill(ctx context.Context, p []int) (int, error) {
	return u.s.Prefill(ctx, p)
}
func (u inkling2UnmeteredSession) Step(t int) (int, error) { return u.s.Step(t) }
func (u inkling2UnmeteredSession) Close() error            { return u.s.Close() }
func (u inkling2UnmeteredSession) Pos() int                { return u.s.Pos() }
func (u inkling2UnmeteredSession) Checkpoint(h int) error  { return u.s.Checkpoint(h) }
func (u inkling2UnmeteredSession) Restore() error          { return u.s.Restore() }
func (u inkling2UnmeteredSession) Reset() error            { return u.s.Reset() }

func TestInkling2PrefixCacheEnvKnobs(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	const defCap = int64(inkling2PrefixCacheMiBDefault) << 20
	for name, tc := range map[string]struct {
		kv    map[string]string
		slots int
		cap   int64
	}{
		"default":    {nil, 1, defCap},
		"four slots": {map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "4", "FAK_INKLING2_PREFIX_CACHE_MIB": "8192"}, 4, 8192 << 20},
		"clamped":    {map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "99"}, inkling2PrefixSlotsMax, defCap},
		"invalid":    {map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "two", "FAK_INKLING2_PREFIX_CACHE_MIB": "-1"}, 1, defCap},
		"zero slots": {map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "0"}, 1, defCap},
	} {
		slots, capBytes := inkling2PrefixCacheFromEnv(env(tc.kv))
		if slots != tc.slots || capBytes != tc.cap {
			t.Fatalf("%s: slots=%d cap=%d, want %d %d", name, slots, capBytes, tc.slots, tc.cap)
		}
	}
}

func TestInkling2WarmUpHonoursContext(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{0})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := warmInkling2Session(ctx, eng.open, inkling2FakeTokenizer(t)); !errors.Is(err, context.Canceled) {
		t.Fatalf("warm-up with a cancelled ctx = %v, want context.Canceled", err)
	}
	if len(*opened) != 0 {
		t.Fatalf("a cancelled warm-up opened %d sessions", len(*opened))
	}
}

// The longest checkpointed prefix wins when two slots both prefix the prompt.
func TestInkling2AdapterPrefixCachePicksLongestPrefix(t *testing.T) {
	eng, opened := inkling2SlotEngine([]int{'a', 'b'}, 2, 1<<40, 0)
	defer eng.Close()
	short, long := []int{'s', 'y', 's'}, []int{'s', 'y', 's', 'u', 'v', 'w'}
	inkling2Gen(t, eng, long, 1)
	inkling2Gen(t, eng, short, 1) // shorter than long's checkpoint: a second slot
	res, _ := inkling2Gen(t, eng, inkling2Extend(long, 'z'), 1)
	if res.CachedPromptTokens != len(long)-1 || len(*opened) != 2 {
		t.Fatalf("cached %d opened %d, want the %d-token slot reused", res.CachedPromptTokens, len(*opened), len(long)-1)
	}
	if s := (*opened)[0]; s.calls[len(s.calls)-2] != "restore" {
		t.Fatalf("long slot calls %v, want a restore", s.calls)
	}
}

// A slot the device cannot open caps the cache at the slots it already holds; the
// request still runs, on the least recently used slot.
func TestInkling2AdapterPrefixCacheOpenFailureCapsSlots(t *testing.T) {
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{slots: 4, capBytes: 1 << 40, open: func() (ggufload.Inkling2LoopSession, error) {
		if len(opened) == 2 {
			return nil, errors.New("out of device memory")
		}
		s := &inkling2FakeSession{canned: []int{'a', 'b'}}
		opened = append(opened, s)
		return s, nil
	}}
	defer eng.Close()
	a1, b1, c1 := []int{'A', 'p'}, []int{'B', 'p'}, []int{'C', 'p'}
	inkling2Gen(t, eng, a1, 1)
	inkling2Gen(t, eng, b1, 1)
	if res, got := inkling2Gen(t, eng, c1, 1); res.CachedPromptTokens != 0 || len(got) != 1 {
		t.Fatalf("request after a failed slot open: cached %d emitted %v", res.CachedPromptTokens, got)
	}
	if eng.prefixSlotCount() != 2 || eng.slots != 2 || opened[0].closed {
		t.Fatalf("slots %d limit %d, want the two open slots kept and the limit capped at them", eng.prefixSlotCount(), eng.slots)
	}
	if res, _ := inkling2Gen(t, eng, inkling2Extend(b1, 'q'), 1); res.CachedPromptTokens != len(b1)-1 {
		t.Fatalf("B after the cap cached %d, want its slot kept", res.CachedPromptTokens)
	}
}

// A caller whose emit panics loses its own slot only.
func TestInkling2AdapterPrefixCacheEmitPanicDropsOnlyItsSlot(t *testing.T) {
	eng, opened := inkling2SlotEngine([]int{'a', 'b', 'c'}, 2, 1<<40, 0)
	defer eng.Close()
	a1, b1 := []int{'A', 'p', 'q'}, []int{'B', 'p', 'q'}
	inkling2Gen(t, eng, a1, 1)
	inkling2Gen(t, eng, b1, 1)
	if _, err := eng.Generate(context.Background(), inkling2Extend(a1, 'r'), 2, nil, func(int) bool { panic("client gone") }); err == nil {
		t.Fatal("emit panic returned no error")
	}
	if !(*opened)[0].closed || (*opened)[1].closed || eng.prefixSlotCount() != 1 {
		t.Fatalf("closed A=%v B=%v slots %d, want only A's slot dropped", (*opened)[0].closed, (*opened)[1].closed, eng.prefixSlotCount())
	}
	if res, _ := inkling2Gen(t, eng, inkling2Extend(b1, 'r'), 1); res.CachedPromptTokens != len(b1)-1 {
		t.Fatalf("B after A's panic cached %d, want %d", res.CachedPromptTokens, len(b1)-1)
	}
}

// The serialized engine takes its slot knobs from the generator's lookup, like
// every other serve knob, and names them in the startup message.
func TestInkling2ServeGeneratorReadsPrefixSlotsFromLookup(t *testing.T) {
	t.Setenv("FAK_INKLING2_PREFIX_SLOTS", "7")
	t.Setenv("FAK_INKLING2_PREFIX_REUSE", "0")
	lookup := func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_PREFIX_SLOTS":
			return "3", true
		case "FAK_INKLING2_PREFIX_CACHE_MIB":
			return "6144", true
		}
		return "", false
	}
	gen, msg, err := newServeInkling2Generator(nil, nil, lookup)
	eng, ok := gen.(*inkling2LoopEngine)
	if err != nil || !ok {
		t.Fatalf("gen=%T err=%v, want the serialized engine", gen, err)
	}
	if eng.noReuse || eng.slots != 3 || eng.capBytes != 6144<<20 {
		t.Fatalf("slots=%d cap=%d, want the lookup's 3 slots and 6144 MiB, not the process env", eng.slots, eng.capBytes)
	}
	if !strings.Contains(msg, "prefix slots=3") {
		t.Fatalf("startup message %q does not name the prefix slots", msg)
	}
	if _, msg, _ := newServeInkling2Generator(nil, nil, func(string) (string, bool) { return "", false }); strings.Contains(msg, "prefix slots") {
		t.Fatalf("default startup message %q names slots the default does not add", msg)
	}
}

// Behind the cohort engine every lone request runs on a fresh session whatever
// FAK_INKLING2_PREFIX_SLOTS says, and the startup message says it is ignored.
func TestInkling2PrefixSlotsOffBehindCohorts(t *testing.T) {
	eng, opened := inkling2SlotEngine([]int{'a', 'b'}, 4, 1<<40, 0)
	defer eng.Close()
	note := inkling2ScalarBehindCohorts(eng)
	if !strings.Contains(note, "FAK_INKLING2_PREFIX_SLOTS=4 ignored") {
		t.Fatalf("note %q does not name the ignored slots", note)
	}
	a1, b1 := []int{'A', 'p'}, []int{'B', 'p'}
	inkling2Gen(t, eng, a1, 1)
	inkling2Gen(t, eng, b1, 1)
	if res, _ := inkling2Gen(t, eng, inkling2Extend(a1, 'q'), 1); res.CachedPromptTokens != 0 {
		t.Fatalf("lone request behind cohorts cached %d, want a fresh session", res.CachedPromptTokens)
	}
	for i, s := range *opened {
		if !s.closed {
			t.Fatalf("session %d stayed resident beside the cohorts", i)
		}
	}
	if eng.prefixSlotCount() != 0 {
		t.Fatalf("%d slots resident behind cohorts", eng.prefixSlotCount())
	}
	one, _ := inkling2SlotEngine([]int{'a'}, 1, 1<<40, 0)
	defer one.Close()
	if note := inkling2ScalarBehindCohorts(one); note != "" {
		t.Fatalf("default one slot note %q, want none", note)
	}
}
