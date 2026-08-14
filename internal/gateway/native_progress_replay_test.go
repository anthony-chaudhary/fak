package gateway

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeProgressReplayAfterCursorIsOrderedAndExclusive(t *testing.T) {
	var r nativeProgressReplay
	for seq := uint64(1); seq <= 4; seq++ {
		kind := agent.ProgressTurnStarted
		if seq == 3 {
			kind = agent.ProgressCallAdjudicated
		}
		r.append("session-a", agent.ProgressEvent{Seq: seq, Kind: kind})
	}
	got, err := r.after("session-a", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Seq != 3 || got[0].Kind != agent.ProgressCallAdjudicated || got[1].Seq != 4 {
		t.Fatalf("tail=%+v", got)
	}
}
func TestNativeProgressReplayMakesShedGapObservable(t *testing.T) {
	var r nativeProgressReplay
	for seq := uint64(1); seq <= nativeProgressReplayLimit+3; seq++ {
		r.append("session-a", agent.ProgressEvent{Seq: seq, Kind: agent.ProgressTurnStarted})
	}
	if _, err := r.after("session-a", 1); !errors.Is(err, errNativeProgressCursorTooOld) {
		t.Fatalf("old cursor err=%v", err)
	}
	tail, err := r.after("session-a", nativeProgressReplayLimit+1)
	if err != nil {
		t.Fatal(err)
	}
	if len(tail) != 2 || tail[0].Seq != nativeProgressReplayLimit+2 {
		t.Fatalf("tail=%+v", tail)
	}
}
func TestNativeProgressCursor(t *testing.T) {
	if got, err := nativeProgressCursor("42"); err != nil || got != 42 {
		t.Fatalf("got=%d err=%v", got, err)
	}
	if _, err := nativeProgressCursor("bad"); err == nil {
		t.Fatal("invalid cursor accepted")
	}
}
