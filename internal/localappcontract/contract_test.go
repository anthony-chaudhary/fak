package localappcontract

import (
	"bytes"
	"testing"
)

func TestReceiptTerminalMappingReplayAndPrivacy(t *testing.T) {
	for _, term := range []Terminal{Completed, Cancelled, Refused, Failed, HandedOff} {
		r := Receipt{Schema: Schema, TaskID: "t", Engine: "fak-native", Location: "local", Revision: "r1", Attempts: 1, Authority: "app", Terminal: term}
		raw, e := r.Marshal()
		if e != nil || ContainsSensitiveField(raw) || bytes.Contains(raw, []byte("secret")) {
			t.Fatalf("term=%s raw=%s err=%v", term, raw, e)
		}
	}
	events := []Event{{Schema: Schema, Sequence: 1, TaskID: "t", Kind: "admitted"}, {Schema: Schema, Sequence: 2, TaskID: "t", Kind: "completed"}}
	if err := Replay(events); err != nil {
		t.Fatal(err)
	}
	if err := Replay([]Event{events[1], events[0]}); err == nil {
		t.Fatal("unordered replay accepted")
	}
}
func TestAdditiveCompatibility(t *testing.T) {
	if err := CheckAdditiveCompatibility([]byte(`{"schema":"v1","task":"x"}`), []byte(`{"schema":"v1","task":"x","new":1}`)); err != nil {
		t.Fatal(err)
	}
	if err := CheckAdditiveCompatibility([]byte(`{"schema":"v1","task":"x"}`), []byte(`{"schema":"v1"}`)); err == nil {
		t.Fatal("removed field accepted")
	}
}
