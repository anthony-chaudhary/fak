package main

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
	if eng.prefixReuse.disabled() || eng.slots != 3 || eng.capBytes != 6144<<20 {
		t.Fatalf("slots=%d cap=%d, want the lookup's 3 slots and 6144 MiB, not the process env", eng.slots, eng.capBytes)
	}
	if !strings.Contains(msg, "prefix slots=3") {
		t.Fatalf("startup message %q does not name the prefix slots", msg)
	}
	if _, msg, _ := newServeInkling2Generator(nil, nil, func(string) (string, bool) { return "", false }); strings.Contains(msg, "prefix slots") {
		t.Fatalf("default startup message %q names slots the default does not add", msg)
	}
}

// F16 prefix reuse is admitted by the session's checkpoint capability. The global
// half-KV rows stay in place, so a growing turn reuses one session without opening a
// second full-window allocation.
func TestInkling2ServeGeneratorF16ReusesCapableCheckpoint(t *testing.T) {
	lookup := func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_KV":
			return "fp16", true
		case "FAK_INKLING2_PREFIX_REUSE":
			return "1", true
		case "FAK_INKLING2_PREFIX_SLOTS":
			return "8", true
		case "FAK_INKLING2_PREFIX_CACHE_MIB":
			return "65536", true
		}
		return "", false
	}
	gen, msg, err := newServeInkling2Generator(nil, nil, lookup)
	eng, ok := gen.(*inkling2LoopEngine)
	if err != nil || !ok {
		t.Fatalf("gen=%T err=%v, want serialized engine", gen, err)
	}
	if eng.prefixReuse != inkling2PrefixReuseOn || strings.Contains(msg, "prefix reuse disabled") {
		t.Fatalf("mode=%v startup=%q, want enabled f16 prefix reuse", eng.prefixReuse, msg)
	}
	var opened []*inkling2FakeSession
	eng.open = inkling2FakeOpener([]int{'x', 'y'}, false, &opened)
	defer eng.Close()
	prompt := []int{'a', 'b'}
	inkling2Gen(t, eng, prompt, 1)
	res, _ := inkling2Gen(t, eng, inkling2Extend(prompt, 'c'), 1)
	if eng.prefixReuse != inkling2PrefixReuseOn || len(opened) != 1 || opened[0].closed || res.CachedPromptTokens != len(prompt)-1 {
		t.Fatalf("mode=%v opened=%d closed=%v cached=%d, want one reused f16 session", eng.prefixReuse, len(opened), len(opened) > 0 && opened[0].closed, res.CachedPromptTokens)
	}
	if held, slots := eng.prefixHeldBytes(), eng.prefixSlotCount(); held == 0 || slots != 1 {
		t.Fatalf("post-request prefix budget holds %d bytes in %d slots, want one metered session", held, slots)
	}
}

func TestInkling2ServeGeneratorF16CheckpointUnavailableResetsAndRetries(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'x', 'y'})
	defer eng.Close()
	realOpen := eng.open
	eng.open = func() (ggufload.Inkling2LoopSession, error) {
		session, err := realOpen()
		if err == nil {
			session.(*inkling2FakeSession).ckErr = ggufload.ErrInkling2CheckpointUnavailable
		}
		return session, err
	}
	inkling2Gen(t, eng, []int{'A', 'a'}, 1)
	session := (*opened)[0]
	session.ckErr = nil
	unrelated := []int{'B', 'b'}
	cold, _ := inkling2Gen(t, eng, unrelated, 1)
	if cold.CachedPromptTokens != 0 || !slices.Equal(session.prompt, unrelated) {
		t.Fatalf("after unavailable checkpoint: cached=%d prompt=%v", cold.CachedPromptTokens, session.prompt)
	}
	warm, _ := inkling2Gen(t, eng, inkling2Extend(unrelated, 'c'), 1)
	if warm.CachedPromptTokens != len(unrelated)-1 || len(*opened) != 1 {
		t.Fatalf("after retry: cached=%d sessions=%d", warm.CachedPromptTokens, len(*opened))
	}
}

func TestInkling2ScalarPrefixNear128KToolTurnKeepsStableRenderedBoundary(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	turn := `<|message_model|><|content_thinking|>inspect first<|end_message|>` +
		`<|message_model|>read<|content_invoke_tool_json|>{"name":"read","args":{"path":"main.go"}}<|end_message|>` +
		`<|content_model_end_sampling|>`
	canned, err := tok.Encode(turn)
	if err != nil {
		t.Fatal(err)
	}
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{prefixReuse: inkling2PrefixReuseOn, slots: 2, capBytes: 1 << 40,
		open: inkling2FakeOpener(canned, false, &opened)}
	defer eng.Close()
	planner, err := newServeInkling2LoopPlanner(eng, tok, "inkling2", 131072)
	if err != nil {
		t.Fatal(err)
	}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "read", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)}}}
	messages := []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("x", 126000)}}
	first, err := planner.Complete(context.Background(), messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(opened) != 1 || len(first.Message.ToolCalls) != 1 {
		t.Fatalf("opened=%d completion=%+v, want one session and one tool call", len(opened), first.Message)
	}
	session := opened[0]
	firstPrompt := append([]int(nil), session.prompt...)
	prefills := len(session.prefills)
	messages = append(messages, first.Message, agent.Message{Role: agent.RoleTool, ToolCallID: first.Message.ToolCalls[0].ID, Content: "package main"})
	second, err := planner.Complete(context.Background(), messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	secondPrompt := append([]int(nil), session.prompt...)
	cached := 0
	if second.Usage.PromptTokensDetails != nil {
		cached = second.Usage.PromptTokensDetails.CachedTokens
	}
	if len(firstPrompt) < 120000 || len(secondPrompt) >= 131072 || cached != len(firstPrompt)-1 || !slices.Equal(firstPrompt[:cached], secondPrompt[:cached]) {
		t.Fatalf("first=%d second=%d cached=%d, want near-128K exact n-1 rendered prefix", len(firstPrompt), len(secondPrompt), cached)
	}
	var suffix []int
	for _, p := range session.prefills[prefills:] {
		suffix = append(suffix, p...)
	}
	if !slices.Equal(session.prompt, secondPrompt) || !slices.Equal(suffix, secondPrompt[cached:]) {
		t.Fatalf("restored token identity or suffix prefill mismatch: prompt=%d suffix=%d want=%d", len(session.prompt), len(suffix), len(secondPrompt)-cached)
	}
	if second.Usage.PromptTokens != len(secondPrompt) || second.Usage.TotalTokens != second.Usage.PromptTokens+second.Usage.CompletionTokens || len(opened) != 1 {
		t.Fatalf("usage=%+v opened=%d, want exact full-prompt accounting on one session", second.Usage, len(opened))
	}
}

func TestInkling2ServeGeneratorF32PreservesExplicitPrefixReuse(t *testing.T) {
	lookup := func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_KV":
			return "fp32", true
		case "FAK_INKLING2_PREFIX_REUSE", "FAK_INKLING2_PREFIX_SLOTS":
			return "1", true
		}
		return "", false
	}
	gen, _, err := newServeInkling2Generator(nil, nil, lookup)
	eng, ok := gen.(*inkling2LoopEngine)
	if err != nil || !ok {
		t.Fatalf("gen=%T err=%v, want serialized engine", gen, err)
	}
	if eng.prefixReuse != inkling2PrefixReuseOn {
		t.Fatalf("mode=%v, want f32 reuse on", eng.prefixReuse)
	}
	var opened []*inkling2FakeSession
	eng.open = inkling2FakeOpener([]int{'x', 'y'}, false, &opened)
	defer eng.Close()
	prompt := []int{'a', 'b'}
	inkling2Gen(t, eng, prompt, 1)
	res, _ := inkling2Gen(t, eng, inkling2Extend(prompt, 'c'), 1)
	if len(opened) != 1 || res.CachedPromptTokens != len(prompt)-1 {
		t.Fatalf("opened=%d cached=%d, want one reused f32 session and %d tokens", len(opened), res.CachedPromptTokens, len(prompt)-1)
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

func TestInkling2PrefixHitMatchKeyTable(t *testing.T) {
	baseKey := inkling2PrefixMatchKey{
		ModelFile:         "/models/inkling2-small.gguf|size=1024|mtime=100",
		TokenizerID:       "tok-sha256-aaa",
		PositionConfig:    "arch=inkling2_loop;ctx=131072;kv=fp16",
		CheckpointVersion: inkling2CheckpointFormatVersion,
		TenantSalt:        "tenant-a",
	}
	withKey := func(fn func(*inkling2PrefixMatchKey)) inkling2PrefixMatchKey {
		k := baseKey
		fn(&k)
		return k
	}
	slot := func(ckpt []int, key inkling2PrefixMatchKey, convKey string, used uint64) *inkling2PrefixSlot {
		return &inkling2PrefixSlot{
			hasCkpt: true,
			ckpt:    ckpt,
			key:     key,
			convKey: convKey,
			used:    used,
		}
	}

	tests := []struct {
		name      string
		slots     []*inkling2PrefixSlot
		prompt    []int
		wantKey   inkling2PrefixMatchKey
		wantIndex int // -1 means miss
	}{
		{
			name:      "salt mismatch misses",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.TenantSalt = "tenant-b" }),
			wantIndex: -1,
		},
		{
			name:      "empty shared namespace does not match salted slot",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.TenantSalt = "" }),
			wantIndex: -1,
		},
		{
			name:      "salted request does not match empty shared namespace slot",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, withKey(func(k *inkling2PrefixMatchKey) { k.TenantSalt = "" }), "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   baseKey,
			wantIndex: -1,
		},
		{
			name:      "empty shared namespace matches empty shared namespace",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, withKey(func(k *inkling2PrefixMatchKey) { k.TenantSalt = "" }), "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.TenantSalt = "" }),
			wantIndex: 0,
		},
		{
			name:      "same prefix with different conversation keys still hits",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-alpha", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   baseKey,
			wantIndex: 0,
		},
		{
			name:      "model file mismatch fails closed",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.ModelFile = "/models/inkling2-other.gguf" }),
			wantIndex: -1,
		},
		{
			name:      "tokenizer id mismatch fails closed",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.TokenizerID = "tok-sha256-bbb" }),
			wantIndex: -1,
		},
		{
			name:      "rope and context config mismatch fails closed",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.PositionConfig = "arch=inkling2_loop;ctx=65536;kv=fp16" }),
			wantIndex: -1,
		},
		{
			name:      "checkpoint version mismatch fails closed",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, baseKey, "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.CheckpointVersion = "inkling2-scalar-ckpt-v2" }),
			wantIndex: -1,
		},
		{
			name:      "empty checkpoint version fails closed",
			slots:     []*inkling2PrefixSlot{slot([]int{1, 2, 3}, withKey(func(k *inkling2PrefixMatchKey) { k.CheckpointVersion = "" }), "conv-1", 10)},
			prompt:    []int{1, 2, 3, 4},
			wantKey:   withKey(func(k *inkling2PrefixMatchKey) { k.CheckpointVersion = "" }),
			wantIndex: -1,
		},
		{
			name: "longest proper prefix wins and exact length is skipped",
			slots: []*inkling2PrefixSlot{
				slot([]int{1, 2}, baseKey, "conv-1", 50),
				slot([]int{1, 2, 3, 4, 5}, baseKey, "conv-2", 10),
				slot([]int{1, 2, 3, 4, 5, 6}, baseKey, "conv-3", 99), // equal length, not a proper prefix
			},
			prompt:    []int{1, 2, 3, 4, 5, 6},
			wantKey:   baseKey,
			wantIndex: 1,
		},
		{
			name: "ties go to most recently used slot when newer is later",
			slots: []*inkling2PrefixSlot{
				slot([]int{1, 2, 3, 4}, baseKey, "conv-old", 10),
				slot([]int{1, 2, 3, 4}, baseKey, "conv-new", 25),
			},
			prompt:    []int{1, 2, 3, 4, 5},
			wantKey:   baseKey,
			wantIndex: 1,
		},
		{
			name: "ties go to most recently used slot when newer is earlier",
			slots: []*inkling2PrefixSlot{
				slot([]int{1, 2, 3, 4}, baseKey, "conv-new", 25),
				slot([]int{1, 2, 3, 4}, baseKey, "conv-old", 10),
			},
			prompt:    []int{1, 2, 3, 4, 5},
			wantKey:   baseKey,
			wantIndex: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			eng := &inkling2LoopEngine{cache: tc.slots}
			got := eng.prefixHit(tc.prompt, tc.wantKey)
			if tc.wantIndex < 0 {
				if got != nil {
					t.Fatalf("prefixHit() matched slot %+v, want nil (fail closed)", got)
				}
				return
			}
			want := tc.slots[tc.wantIndex]
			if got != want {
				t.Fatalf("prefixHit() = %p (used=%d len=%d), want slot %d %p (used=%d len=%d)",
					got, slotUsed(got), slotLen(got), tc.wantIndex, want, want.used, len(want.ckpt))
			}
		})
	}
}

func slotUsed(sl *inkling2PrefixSlot) uint64 {
	if sl == nil {
		return 0
	}
	return sl.used
}

func slotLen(sl *inkling2PrefixSlot) int {
	if sl == nil {
		return -1
	}
	return len(sl.ckpt)
}

func TestInkling2PrefixMatchKeyEndToEndAndConversationVictim(t *testing.T) {
	eng, opened := inkling2SlotEngine([]int{'a', 'b'}, 2, 1<<40, 0)
	defer eng.Close()
	mat := inkling2PrefixMaterialization{
		ModelFile:         "/models/inkling2.gguf|size=100|mtime=1",
		TokenizerID:       "tok-v1",
		PositionConfig:    "ctx=131072;kv=fp16",
		CheckpointVersion: inkling2CheckpointFormatVersion,
	}
	eng.setPrefixMaterialization(mat)

	genWith := func(ctx context.Context, prompt []int) agent.Inkling2LoopGeneration {
		t.Helper()
		res, err := eng.Generate(ctx, prompt, 1, nil, func(int) bool { return true })
		if err != nil {
			t.Fatal(err)
		}
		return res
	}

	basePrompt := []int{'s', 'y', 's', '1'}
	extPrompt := inkling2Extend(basePrompt, 'x')

	// Seed slot under tenant-a with conv-1.
	ctxA1 := withInkling2ConversationKey(agent.WithPrefixCacheIdentity(context.Background(), "tenant-a", ""), "conv-1", false)
	if res := genWith(ctxA1, basePrompt); res.CachedPromptTokens != 0 {
		t.Fatalf("cold tenant-a cached %d, want 0", res.CachedPromptTokens)
	}

	// Different conversation key under same tenant-a + same exact prefix MUST hit.
	ctxA2 := withInkling2ConversationKey(agent.WithPrefixCacheIdentity(context.Background(), "tenant-a", ""), "conv-2", false)
	if res := genWith(ctxA2, extPrompt); res.CachedPromptTokens != len(basePrompt)-1 {
		t.Fatalf("same tenant different conv cached %d, want %d", res.CachedPromptTokens, len(basePrompt)-1)
	}

	// Different tenant salt (tenant-b) with same conversation key and same prefix MUST miss.
	ctxB := withInkling2ConversationKey(agent.WithPrefixCacheIdentity(context.Background(), "tenant-b", ""), "conv-2", false)
	if res := genWith(ctxB, inkling2Extend(extPrompt, 'y')); res.CachedPromptTokens != 0 {
		t.Fatalf("different tenant same conv cached %d, want 0 (fail closed)", res.CachedPromptTokens)
	}

	// Empty shared namespace MUST miss against tenant-a / tenant-b slots.
	if res := genWith(context.Background(), inkling2Extend(extPrompt, 'z')); res.CachedPromptTokens != 0 {
		t.Fatalf("empty shared namespace cached %d against salted slots, want 0", res.CachedPromptTokens)
	}

	// Materialization change (e.g. tokenizer/RoPE reload) MUST fail closed.
	mat2 := mat
	mat2.PositionConfig = "ctx=65536;kv=fp16"
	eng.setPrefixMaterialization(mat2)
	if res := genWith(ctxA2, inkling2Extend(extPrompt, 'm')); res.CachedPromptTokens != 0 {
		t.Fatalf("materialization mismatch cached %d, want 0 (fail closed)", res.CachedPromptTokens)
	}
	if len(*opened) != 2 {
		t.Fatalf("opened %d sessions, want 2 bounded by slot cap", len(*opened))
	}

	// Conversation pinning protects a pinned slot from victim selection on a miss.
	engPin, _ := inkling2SlotEngine([]int{'a', 'b'}, 2, 1<<40, 0)
	defer engPin.Close()
	engPin.setPrefixMaterialization(mat)
	pA := []int{'A', '1', '2'}
	pB := []int{'B', '1', '2'}
	pC := []int{'C', '1', '2'}
	ctxPinA := withInkling2ConversationKey(withInkling2TenantSalt(context.Background(), "t1"), "conv-A", true)
	ctxUnpinB := withInkling2ConversationKey(withInkling2TenantSalt(context.Background(), "t1"), "conv-B", false)
	ctxUnpinC := withInkling2ConversationKey(withInkling2TenantSalt(context.Background(), "t1"), "conv-C", false)

	if _, err := engPin.Generate(ctxPinA, pA, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if _, err := engPin.Generate(ctxUnpinB, pB, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	// conv-A is older (LRU) than conv-B, but pinned: conv-C miss must evict conv-B, keeping conv-A warm.
	if _, err := engPin.Generate(ctxUnpinC, pC, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	resA, err := engPin.Generate(ctxPinA, inkling2Extend(pA, '3'), 1, nil, func(int) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if resA.CachedPromptTokens != len(pA)-1 {
		t.Fatalf("pinned conv-A cached %d after conv-C miss, want %d", resA.CachedPromptTokens, len(pA)-1)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2SlotMemoryStatusMatchesLedgerTotals(t *testing.T) {
	const slotBytes = int64(120)
	eng, opened := inkling2SlotEngine([]int{'a', 'b', 'c'}, 4, 1<<20, slotBytes)
	defer eng.Close()

	st := eng.prefixState()
	hitAt := time.Date(2026, 11, 27, 8, 0, 0, 0, time.UTC)
	st.now = func() time.Time { return hitAt }

	ctxPin := withInkling2ConversationKey(context.Background(), "conv-pin", true)
	ctxUnpin := withInkling2ConversationKey(context.Background(), "conv-unpin", false)

	pPin := []int{'P', '1', '2'}
	pUnpin := []int{'U', '1', '2'}
	if _, err := eng.Generate(ctxPin, pPin, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Generate(ctxUnpin, pUnpin, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}

	// Before any prefix hit, LastHitTime must be zero.
	pinBytes := (*opened)[0].HeldBytes()
	unpinBytes := (*opened)[1].HeldBytes()
	got := eng.SlotMemoryStatus()
	snap := st.ledger.Snapshot()
	if got.Slots != 2 || got.PinnedCount != 1 {
		t.Fatalf("status Slots=%d PinnedCount=%d, want 2 and 1", got.Slots, got.PinnedCount)
	}
	if got.HeldBytes != snap.Total || got.HeldBytes != pinBytes+unpinBytes {
		t.Fatalf("status HeldBytes=%d, ledger Total=%d, want %d", got.HeldBytes, snap.Total, pinBytes+unpinBytes)
	}
	if got.IdleBytes != unpinBytes {
		t.Fatalf("status IdleBytes=%d, want %d (unpinned idle slot only)", got.IdleBytes, unpinBytes)
	}
	if !got.LastHitTime.IsZero() {
		t.Fatalf("status LastHitTime=%v before any hit, want zero", got.LastHitTime)
	}

	// Trigger a prefix hit on the unpinned slot and add cohort_arena bytes to the shared ledger.
	res, _ := inkling2Gen(t, eng, inkling2Extend(pUnpin, '3'), 1)
	if res.CachedPromptTokens != len(pUnpin)-1 {
		t.Fatalf("expected prefix hit on unpinned slot, cached=%d", res.CachedPromptTokens)
	}
	unpinBytes = (*opened)[1].HeldBytes()
	st.ledger.Set("cohort_arena", inkling2LedgerCohortArena, 80)

	batch := &inkling2LoopBatchEngine{
		ledger:           st.ledger,
		slotMemoryStatus: eng.SlotMemoryStatus,
		shedIdleSlots:    eng.ShedIdleSlots,
	}
	got = batch.SlotMemoryStatus()
	snap = st.ledger.Snapshot()
	if got.HeldBytes != snap.Total || got.HeldBytes != pinBytes+unpinBytes+80 {
		t.Fatalf("batch status HeldBytes=%d, ledger Total=%d, want %d", got.HeldBytes, snap.Total, pinBytes+unpinBytes+80)
	}
	if got.IdleBytes != unpinBytes {
		t.Fatalf("batch status IdleBytes=%d, want %d", got.IdleBytes, unpinBytes)
	}
	if !got.LastHitTime.Equal(hitAt) {
		t.Fatalf("status LastHitTime=%v, want %v", got.LastHitTime, hitAt)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2ShedIdleSlotsFreesLRUUnpinnedNeverPinned(t *testing.T) {
	const slotBytes = int64(100)
	eng, opened := inkling2SlotEngine([]int{'a', 'b'}, 4, 350, slotBytes)
	defer eng.Close()

	ctxPin := withInkling2ConversationKey(context.Background(), "conv-pin", true)
	ctxB := withInkling2ConversationKey(context.Background(), "conv-B", false)
	ctxC := withInkling2ConversationKey(context.Background(), "conv-C", false)

	// Open in order: A (pinned, LRU), B (unpinned, older), C (unpinned, newest).
	if _, err := eng.Generate(ctxPin, []int{'A', '1', '2'}, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Generate(ctxB, []int{'B', '1', '2'}, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Generate(ctxC, []int{'C', '1', '2'}, 1, nil, func(int) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if len(*opened) != 3 {
		t.Fatalf("opened %d sessions, want 3", len(*opened))
	}
	bytesA := (*opened)[0].HeldBytes()
	bytesB := (*opened)[1].HeldBytes()
	bytesC := (*opened)[2].HeldBytes()

	// Shedding 100 bytes must evict B (LRU unpinned), leaving A (pinned) and C (MRU unpinned).
	freed := eng.ShedIdleSlots(100)
	if freed != bytesB {
		t.Fatalf("ShedIdleSlots(100) = %d, want %d", freed, bytesB)
	}
	if (*opened)[0].closed || !(*opened)[1].closed || (*opened)[2].closed {
		t.Fatalf("closed states after 100B shed = [%v %v %v], want [false true false]",
			(*opened)[0].closed, (*opened)[1].closed, (*opened)[2].closed)
	}
	st := eng.prefixState()
	if snap := st.ledger.Snapshot(); snap.Total != bytesA+bytesC {
		t.Fatalf("ledger total after first shed = %d, want %d", snap.Total, bytesA+bytesC)
	}

	// Cohort admission wired to ShedIdleSlots requesting 250 bytes (total > cap 350)
	// must shed C and never touch pinned A, even though bytesA+250 > 350.
	batch := &inkling2LoopBatchEngine{
		ledger:        st.ledger,
		shedIdleSlots: eng.ShedIdleSlots,
	}
	batch.admitCohortMemory(250)
	if (*opened)[0].closed || !(*opened)[2].closed {
		t.Fatalf("closed states after cohort admission = [%v %v], want [false true]",
			(*opened)[0].closed, (*opened)[2].closed)
	}

	// Requesting more bytes than remain must return 0 and never touch pinned A.
	if extra := batch.ShedIdleSlots(500); extra != 0 {
		t.Fatalf("ShedIdleSlots(500) with only pinned slot = %d, want 0", extra)
	}
	if (*opened)[0].closed {
		t.Fatal("pinned slot was evicted by ShedIdleSlots")
	}
	status := eng.SlotMemoryStatus()
	if status.Slots != 1 || status.PinnedCount != 1 || status.HeldBytes != bytesA || status.IdleBytes != 0 {
		t.Fatalf("final status = %+v, want Slots=1 PinnedCount=1 HeldBytes=%d IdleBytes=0", status, bytesA)
	}
}
