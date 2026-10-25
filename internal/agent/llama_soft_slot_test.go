package agent

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeLlamaSlots struct {
	mu      sync.Mutex
	busy    map[int]bool
	noSlots bool
	gone    map[int]bool // ids /slots no longer lists
	delay   time.Duration
	props   atomic.Int64 // GET /props count
	gets    atomic.Int64 // GET /slots count
}

func (f *fakeLlamaSlots) setBusy(ids ...int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.busy = map[int]bool{}
	for _, id := range ids {
		f.busy[id] = true
	}
}

// waitFreshSnapshot blocks until the background poller has read /slots after since.
func waitFreshSnapshot(t *testing.T, p *HTTPPlanner, since time.Time) {
	t.Helper()
	st := llamaSoftSlotsFor(p.BaseURL)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		st.mu.Lock()
		fresh := st.snapAt.After(since)
		st.mu.Unlock()
		if fresh {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no /slots snapshot after %s", since)
}

func newSoftSlotPlanner(t *testing.T, f *fakeLlamaSlots) *HTTPPlanner {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			f.props.Add(1)
			_, _ = w.Write([]byte(`{"total_slots":4}`))
		case "/slots":
			f.gets.Add(1)
			time.Sleep(f.delay)
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.noSlots {
				http.NotFound(w, r)
				return
			}
			var parts []string
			for i := 0; i < 4; i++ {
				if !f.gone[i] {
					parts = append(parts, fmt.Sprintf(`{"id":%d,"is_processing":%v}`, i, f.busy[i]))
				}
			}
			_, _ = w.Write([]byte("[" + strings.Join(parts, ",") + "]"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(ts.Close)
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: ts.URL + "/v1", ModelID: "m", LlamaSlotAffinity: true, LlamaSoftSlot: true}
	llamaSlotStateFor(p.BaseURL).slots.Store(4)
	// A later test's server can reuse this port; do not hand it this upstream's state
	// (a /slots 404 backoff would outlive the test by a minute).
	t.Cleanup(func() {
		llamaSoftSlotRegistry.Delete(p.BaseURL)
		llamaSlotRegistry.Delete(p.BaseURL)
	})
	return p
}

// softSlotTurn prepares one request and returns its id_slot (-1 when omitted) and the call,
// whose releaseSlot ends the turn.
func softSlotTurn(t *testing.T, p *HTTPPlanner, msgs []Message) (int, *upstreamCall) {
	t.Helper()
	call, err := p.prepareUpstream(msgs, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(call.body, &body); err != nil {
		t.Fatal(err)
	}
	raw, ok := body["id_slot"]
	if !ok {
		return -1, call
	}
	var id int
	if err := json.Unmarshal(raw, &id); err != nil {
		t.Fatal(err)
	}
	return id, call
}

func setSoftSlotUseOrder(t *testing.T, p *HTTPPlanner, ids ...int) {
	t.Helper()
	st := llamaSoftSlotsFor(p.BaseURL)
	st.mu.Lock()
	defer st.mu.Unlock()
	base := time.Unix(1, 0)
	for i, id := range ids {
		st.slotUsed[id] = base.Add(time.Duration(i) * time.Second)
	}
}

func conv(user string, extra ...string) []Message {
	msgs := []Message{{Role: RoleSystem, Content: "shared harness prompt"}, {Role: RoleUser, Content: user}}
	for _, e := range extra {
		msgs = append(msgs, Message{Role: RoleAssistant, Content: "ok"}, Message{Role: RoleUser, Content: e})
	}
	return msgs
}

// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotReturnsConversationToItsIdleSlot(t *testing.T) {
	f := &fakeLlamaSlots{}
	p := newSoftSlotPlanner(t, f)

	a1, c := softSlotTurn(t, p, conv("agent A"))
	c.releaseSlot()
	b1, c := softSlotTurn(t, p, conv("agent B"))
	c.releaseSlot()
	if a1 < 0 || b1 < 0 || a1 == b1 {
		t.Fatalf("A=%d B=%d: two conversations sharing a system prompt must take distinct idle slots", a1, b1)
	}
	a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
	c.releaseSlot()
	b2, c := softSlotTurn(t, p, conv("agent B", "turn 2"))
	c.releaseSlot()
	if a2 != a1 || b2 != b1 {
		t.Fatalf("second turns went A %d->%d, B %d->%d; want each back on its own slot", a1, a2, b1, b2)
	}
}

// fak-test:runtime fast est=600ms lane=default
func TestLlamaSoftSlotNeverNamesABusySlot(t *testing.T) {
	f := &fakeLlamaSlots{}
	p := newSoftSlotPlanner(t, f)
	a1, c := softSlotTurn(t, p, conv("agent A"))
	c.releaseSlot()

	f.setBusy(a1)
	waitFreshSnapshot(t, p, time.Now())
	a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
	c.releaseSlot()
	if a2 == a1 || a2 < 0 {
		t.Fatalf("A's slot %d is busy upstream; got id_slot %d, want another idle slot", a1, a2)
	}

	f.setBusy(0, 1, 2, 3)
	waitFreshSnapshot(t, p, time.Now())
	if got, c := softSlotTurn(t, p, conv("agent A", "turn 3")); got != -1 {
		c.releaseSlot()
		t.Fatalf("every slot busy: id_slot %d sent, want omitted so no turn queues", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotCountsItsOwnInflightPins(t *testing.T) {
	f := &fakeLlamaSlots{}
	p := newSoftSlotPlanner(t, f)
	first, held := softSlotTurn(t, p, conv("agent A"))
	second, c := softSlotTurn(t, p, conv("agent A", "parallel"))
	c.releaseSlot()
	held.releaseSlot()
	if first < 0 || second == first {
		t.Fatalf("concurrent turns pinned %d and %d; an in-flight pin must keep its slot busy before /slots shows it", first, second)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotLeavesBodyAloneWhenOffOrUnreadable(t *testing.T) {
	f := &fakeLlamaSlots{noSlots: true}
	p := newSoftSlotPlanner(t, f)
	if got, _ := softSlotTurn(t, p, conv("agent A")); got != -1 {
		t.Fatalf("/slots unreadable: id_slot %d sent, want omitted", got)
	}
	if n := llamaSoftSlotCounts[SoftSlotUnavailable].Load(); n == 0 {
		t.Fatalf("unreadable /slots not counted as unavailable")
	}

	p2 := newSoftSlotPlanner(t, &fakeLlamaSlots{})
	p2.LlamaSoftSlot = false
	if got, _ := softSlotTurn(t, p2, conv("agent A")); got != -1 {
		t.Fatalf("soft slot off: id_slot %d sent", got)
	}

	p3 := newSoftSlotPlanner(t, &fakeLlamaSlots{})
	p3.ExtraBody = json.RawMessage(`{"id_slot":3}`)
	if got, _ := softSlotTurn(t, p3, conv("agent A")); got != 3 {
		t.Fatalf("operator id_slot overridden: %d", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestConversationKeyIgnoresLaterTurns(t *testing.T) {
	k1, ok1 := conversationKey(conv("agent A"))
	k2, ok2 := conversationKey(conv("agent A", "turn 2", "turn 3"))
	k3, _ := conversationKey(conv("agent B"))
	if !ok1 || !ok2 || k1 != k2 || k1 == k3 {
		t.Fatalf("keys A1=%x A3=%x B=%x", k1, k2, k3)
	}
	if _, ok := conversationKey([]Message{{Role: RoleSystem, Content: "only system"}}); ok {
		t.Fatalf("a transcript with no user turn has no conversation key")
	}
}

// A snapshot taken while our own pinned turn was still running shows its slot busy; once
// that pin is released the conversation must still go back to it, not be moved.
//
// fak-test:runtime fast est=50ms lane=default
func TestLlamaSoftSlotIgnoresStaleBusyFromItsOwnFinishedPin(t *testing.T) {
	p := newSoftSlotPlanner(t, &fakeLlamaSlots{})
	a1, c := softSlotTurn(t, p, conv("agent A"))
	st := llamaSoftSlotsFor(p.BaseURL)
	st.mu.Lock()
	st.snapBusy = map[int]bool{0: false, 1: false, 2: false, 3: false, a1: true}
	st.snapAt = time.Now()
	st.snapSent = st.snapAt
	st.mu.Unlock()
	c.releaseSlot()
	a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
	c.releaseSlot()
	if a2 != a1 {
		t.Fatalf("turn 2 moved %d -> %d on a snapshot that predates the release", a1, a2)
	}
}

// Policy wait keeps a known conversation on its own slot even while that slot is busy,
// where policy lru would move it into another conversation's slot and evict that KV.
//
// fak-test:runtime fast est=600ms lane=default
func TestLlamaSoftSlotWaitPolicyNeverMovesAConversation(t *testing.T) {
	f := &fakeLlamaSlots{}
	p := newSoftSlotPlanner(t, f)
	p.LlamaSoftSlotPolicy = SoftSlotPolicyWait
	a1, c := softSlotTurn(t, p, conv("agent A"))
	c.releaseSlot()

	f.setBusy(a1)
	waitFreshSnapshot(t, p, time.Now())
	before := llamaSoftSlotCounts[SoftSlotWaited].Load()
	a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
	c.releaseSlot()
	if a2 != a1 {
		t.Fatalf("wait policy moved A from busy slot %d to %d", a1, a2)
	}
	if llamaSoftSlotCounts[SoftSlotWaited].Load() != before+1 {
		t.Fatalf("pin on a busy own slot not counted as waited")
	}

	if b, c := softSlotTurn(t, p, conv("agent B")); b == a1 || b < 0 {
		c.releaseSlot()
		t.Fatalf("new conversation B got slot %d; want an idle slot other than busy %d", b, a1)
	} else {
		c.releaseSlot()
	}
}

// An unused slot may have a newer timestamp than an owned one. Only wait gives
// ownership priority over recency; the default LRU choice stays unchanged.
//
// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotPrefersUnownedOnlyUnderWait(t *testing.T) {
	for _, tc := range []struct {
		policy string
		fourth int
	}{{SoftSlotPolicyWait, 3}, {SoftSlotPolicyLRU, 0}} {
		t.Run(tc.policy, func(t *testing.T) {
			p := newSoftSlotPlanner(t, &fakeLlamaSlots{})
			p.LlamaSoftSlotPolicy = tc.policy
			st := llamaSoftSlotsFor(p.BaseURL)
			st.mu.Lock()
			st.slotUsed[3] = time.Now().Add(time.Hour)
			st.mu.Unlock()
			for i, want := range []int{0, 1, 2, tc.fourth} {
				got, call := softSlotTurn(t, p, conv(fmt.Sprintf("agent %d", i)))
				call.releaseSlot()
				if got != want {
					t.Fatalf("conversation %d got slot %d, want %d", i, got, want)
				}
			}
		})
	}
}

// More conversations than slots can displace a recorded owner under either
// policy. Count that assignment explicitly, including the displaced owner's
// next turn, but never count a request that could not name a slot.
//
// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotReportsOwnershipSteals(t *testing.T) {
	for _, policy := range []string{SoftSlotPolicyWait, SoftSlotPolicyLRU} {
		t.Run(policy, func(t *testing.T) {
			p := newSoftSlotPlanner(t, &fakeLlamaSlots{})
			p.LlamaSoftSlotPolicy = policy
			before := LlamaSoftSlotCounts()["stolen"]
			for i, name := range []string{"A", "B", "C", "D"} {
				got, call := softSlotTurn(t, p, conv(name))
				call.releaseSlot()
				if got != i {
					t.Fatalf("conversation %s got slot %d, want unowned slot %d", name, got, i)
				}
			}
			if got := LlamaSoftSlotCounts()["stolen"] - before; got != 0 {
				t.Fatalf("assigning unowned slots counted %d steals", got)
			}
			setSoftSlotUseOrder(t, p, 0, 1, 2, 3)
			for i, name := range []string{"E", "A"} {
				got, call := softSlotTurn(t, p, conv(name, "next turn"))
				call.releaseSlot()
				if got != i {
					t.Fatalf("conversation %s got slot %d, want oldest owned slot %d", name, got, i)
				}
				if stolen := LlamaSoftSlotCounts()["stolen"] - before; stolen != uint64(i+1) {
					t.Fatalf("conversation %s replaced an owner: stolen=%d, want %d", name, stolen, i+1)
				}
			}
			for _, name := range []string{"C", "D", "E", "A"} {
				_, call := softSlotTurn(t, p, conv(name, "held turn"))
				t.Cleanup(call.releaseSlot)
			}
			before = LlamaSoftSlotCounts()["stolen"]
			got, call := softSlotTurn(t, p, conv("overflow"))
			call.releaseSlot()
			if got != -1 || LlamaSoftSlotCounts()["stolen"] != before {
				t.Fatalf("all slots in flight: slot=%d, stolen changed=%v", got, LlamaSoftSlotCounts()["stolen"] != before)
			}
		})
	}
}

// After an idle spell the poller has exited and the last snapshot is older than
// llamaSlotsSnapshotMaxAge. The next turn must re-read /slots and pin, not go unpinned
// as unavailable forever.
//
// fak-test:runtime fast est=50ms lane=default
func TestLlamaSoftSlotRereadsStaleSnapshotAfterIdle(t *testing.T) {
	p := newSoftSlotPlanner(t, &fakeLlamaSlots{})
	a1, c := softSlotTurn(t, p, conv("agent A"))
	c.releaseSlot()
	if a1 < 0 {
		t.Fatalf("first turn not pinned")
	}
	st := llamaSoftSlotsFor(p.BaseURL)
	st.mu.Lock()
	st.snapAt = st.snapAt.Add(-40 * time.Second)
	st.lastDemand = st.lastDemand.Add(-40 * time.Second)
	st.mu.Unlock()
	before := llamaSoftSlotCounts[SoftSlotUnavailable].Load()
	a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
	c.releaseSlot()
	if a2 != a1 {
		t.Fatalf("turn after idle got slot %d, want %d from a re-read snapshot", a2, a1)
	}
	if llamaSoftSlotCounts[SoftSlotUnavailable].Load() != before {
		t.Fatalf("turn after idle counted unavailable")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestValidateSoftSlotPolicy(t *testing.T) {
	for _, ok := range []string{"", SoftSlotPolicyLRU, SoftSlotPolicyWait} {
		if err := ValidateSoftSlotPolicy(ok); err != nil {
			t.Fatalf("policy %q rejected: %v", ok, err)
		}
	}
	if err := ValidateSoftSlotPolicy("wiat"); err == nil || !strings.Contains(err.Error(), `"wiat"`) {
		t.Fatalf("typo policy error = %v, want a rejection naming it", err)
	}
}

// A fresh gateway hit by 16 concurrent first turns must probe /props once, share one GET
// /slots between them (#13785), and pin turn 1 so turn 2 can find its KV.
//
// fak-test:runtime fast est=100ms lane=default
func TestLlamaSoftSlotColdStampedeSharesOneRead(t *testing.T) {
	f := &fakeLlamaSlots{delay: 50 * time.Millisecond}
	p := newSoftSlotPlanner(t, f)
	llamaSlotStateFor(p.BaseURL).slots.Store(0) // cold: /props not yet probed
	const n = 16
	var (
		wg    sync.WaitGroup
		start = make(chan struct{})
		got   [n]int
		calls [n]*upstreamCall
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			got[i], calls[i] = softSlotTurn(t, p, conv(fmt.Sprintf("agent %d", i)))
		}(i)
	}
	close(start)
	wg.Wait()
	gets, props := f.gets.Load(), f.props.Load()
	pinned := map[int]bool{}
	for i := 0; i < n; i++ {
		if got[i] >= 0 {
			if pinned[got[i]] {
				t.Fatalf("slot %d pinned twice by concurrent first turns", got[i])
			}
			pinned[got[i]] = true
		}
		calls[i].releaseSlot()
	}
	if gets > 2 {
		t.Fatalf("cold stampede of %d turns made %d GET /slots, want at most 2", n, gets)
	}
	if props != 1 {
		t.Fatalf("cold stampede made %d GET /props, want 1", props)
	}
	if len(pinned) != 4 {
		t.Fatalf("cold stampede pinned slots %v, want all 4 idle slots used on turn 1", pinned)
	}
}

// /slots answering 404 (llama-server without --slots) backs off like /props instead of
// costing every turn a GET (#13786).
//
// fak-test:runtime fast est=50ms lane=default
func TestLlamaSoftSlotBacksOffAfterSlots404(t *testing.T) {
	f := &fakeLlamaSlots{noSlots: true}
	p := newSoftSlotPlanner(t, f)
	before := llamaSoftSlotCounts[SoftSlotUnavailable].Load()
	for i := 0; i < 10; i++ {
		if got, _ := softSlotTurn(t, p, conv("agent A", fmt.Sprint(i))); got != -1 {
			t.Fatalf("turn %d: id_slot %d sent with /slots 404", i, got)
		}
	}
	if gets := f.gets.Load(); gets != 1 {
		t.Fatalf("10 turns against a /slots 404 made %d GETs, want 1 then backoff", gets)
	}
	if d := llamaSoftSlotCounts[SoftSlotUnavailable].Load() - before; d < 10 {
		t.Fatalf("backed-off turns counted unavailable %d times, want 10", d)
	}
	st := llamaSoftSlotsFor(p.BaseURL)
	st.mu.Lock()
	until := time.Until(st.unsupportedUntil)
	st.mu.Unlock()
	if until < 50*time.Second || until > llamaSlotRetryAfter {
		t.Fatalf("backoff ends in %s, want about %s", until, llamaSlotRetryAfter)
	}
}

// A slot id the latest snapshot no longer lists is gone, not idle (#13788).
//
// fak-test:runtime fast est=600ms lane=default
func TestLlamaSoftSlotNeverNamesAVanishedSlot(t *testing.T) {
	for _, policy := range []string{SoftSlotPolicyLRU, SoftSlotPolicyWait} {
		t.Run(policy, func(t *testing.T) {
			f := &fakeLlamaSlots{}
			p := newSoftSlotPlanner(t, f)
			p.LlamaSoftSlotPolicy = policy
			a1, c := softSlotTurn(t, p, conv("agent A"))
			c.releaseSlot()
			if a1 < 0 {
				t.Fatalf("first turn not pinned")
			}
			f.mu.Lock()
			f.gone = map[int]bool{a1: true}
			f.mu.Unlock()
			waitFreshSnapshot(t, p, time.Now())
			a2, c := softSlotTurn(t, p, conv("agent A", "turn 2"))
			c.releaseSlot()
			if a2 == a1 || a2 < 0 {
				t.Fatalf("slot %d vanished from /slots; got id_slot %d, want another listed idle slot", a1, a2)
			}
		})
	}
}
