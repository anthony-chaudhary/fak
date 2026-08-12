package session

// pins_test.go — the operator-declared keep-set on the drive record (issue #2211,
// control-plane epic #2208). SetPins only RECORDS the set: it must bump Rev like any
// write, replace (never merge) the previous set, defend against caller aliasing,
// clear on nil/empty, be rejected by a terminal session, and stay off the wire when
// unused so a pre-#2211 State marshals byte-identically.

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSetPinsRecordsReplacesAndClears(t *testing.T) {
	tbl := NewTable()
	st, ok := tbl.SetPins("s", []string{"span:2", "span:7"})
	if !ok || st.Rev != 1 {
		t.Fatalf("SetPins ok=%v Rev=%d, want true/1", ok, st.Rev)
	}
	if len(st.Pins) != 2 || st.Pins[0] != "span:2" || st.Pins[1] != "span:7" {
		t.Fatalf("recorded pins = %v, want [span:2 span:7]", st.Pins)
	}

	// A later write REPLACES the whole set — pin/unpin merge discipline lives with the
	// CLI's read-modify-write, never in the table.
	st, _ = tbl.SetPins("s", []string{"span:7"})
	if len(st.Pins) != 1 || st.Pins[0] != "span:7" {
		t.Fatalf("replaced pins = %v, want [span:7]", st.Pins)
	}

	// nil clears every operator pin.
	st, _ = tbl.SetPins("s", nil)
	if len(st.Pins) != 0 {
		t.Fatalf("cleared pins = %v, want empty", st.Pins)
	}
}

func TestSetPinsCopiesTheCallerSlice(t *testing.T) {
	tbl := NewTable()
	pins := []string{"span:1"}
	tbl.SetPins("s", pins)
	pins[0] = "span:CLOBBERED"
	if got := tbl.Get("s").Pins[0]; got != "span:1" {
		t.Fatalf("stored pin = %q after caller mutation, want the copied span:1", got)
	}
}

func TestSetPinsTerminalRejected(t *testing.T) {
	tbl := NewTable()
	if _, ok := tbl.Transition("s", Stopped, "done"); !ok {
		t.Fatal("setup: could not stop session")
	}
	if _, ok := tbl.SetPins("s", []string{"span:1"}); ok {
		t.Fatal("SetPins on stopped ok=true, want false")
	}
}

func TestStateJSONPinsRoundTripAndOmitWhenUnused(t *testing.T) {
	// Unused ⇒ absent: the pre-#2211 wire shape is unchanged.
	raw, err := json.Marshal(DefaultState("s"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "pins") {
		t.Fatalf("empty pin-set must be absent from the wire, got %s", raw)
	}

	st := DefaultState("s")
	st.Pins = []string{"span:2", "fact:release-tag"}
	raw, err = json.Marshal(st)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"pins":["span:2","fact:release-tag"]`) {
		t.Fatalf("pin-set missing from JSON: %s", raw)
	}
	var back State
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if len(back.Pins) != 2 || back.Pins[0] != "span:2" || back.Pins[1] != "fact:release-tag" {
		t.Fatalf("round-tripped pins = %v, want the set verbatim", back.Pins)
	}
}
