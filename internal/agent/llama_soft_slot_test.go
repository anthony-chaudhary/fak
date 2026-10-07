package agent

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeLlamaSlots struct {
	mu      sync.Mutex
	busy    map[int]bool
	noSlots bool
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
			_, _ = w.Write([]byte(`{"total_slots":4}`))
		case "/slots":
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.noSlots {
				http.NotFound(w, r)
				return
			}
			parts := make([]string, 4)
			for i := range parts {
				parts[i] = fmt.Sprintf(`{"id":%d,"is_processing":%v}`, i, f.busy[i])
			}
			_, _ = w.Write([]byte("[" + strings.Join(parts, ",") + "]"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(ts.Close)
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: ts.URL + "/v1", ModelID: "m", LlamaSlotAffinity: true, LlamaSoftSlot: true}
	llamaSlotStateFor(p.BaseURL).slots.Store(4)
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
	st.snapBusy = map[int]bool{a1: true}
	st.snapAt = time.Now()
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
