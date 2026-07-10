package wipref

import (
	"reflect"
	"testing"
)

func TestValidSession(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		{"selfcheck", true},
		{"sess-01_abc.2", true},
		{"", false},
		{"a/b", false},       // no path separator
		{"has space", false}, // no whitespace
		{"tilde~1", false},   // no git refname metachar
		{"star*", false},     // no glob
		{"colon:", false},    // no ref metachar
	}
	for _, c := range cases {
		if got := ValidSession(c.id); got != c.want {
			t.Errorf("ValidSession(%q) = %v, want %v", c.id, got, c.want)
		}
	}
}

func TestSessionRefRoundTrip(t *testing.T) {
	for _, id := range []string{"selfcheck", "abc-123", "x.y_z"} {
		ref := SessionRef(id)
		if want := "refs/fak/wip/" + id; ref != want {
			t.Errorf("SessionRef(%q) = %q, want %q", id, ref, want)
		}
		if got := SessionFromRef(ref); got != id {
			t.Errorf("SessionFromRef(%q) = %q, want %q", ref, got, id)
		}
	}
}

func TestStampRoundTrip(t *testing.T) {
	in := Stamp{
		SessionID:      "selfcheck",
		StartSHA:       "0123456789abcdef",
		Leaves:         []string{"cmd/fak", "internal/wipref"},
		Buildable:      true,
		CheckpointedAt: 1_700_000_000,
	}
	msg, err := EncodeStamp(in)
	if err != nil {
		t.Fatalf("EncodeStamp: %v", err)
	}
	got, ok := DecodeStamp(msg)
	if !ok {
		t.Fatalf("DecodeStamp(%q) reported not-ok", msg)
	}
	if !reflect.DeepEqual(got, in) {
		t.Errorf("round-trip mismatch:\n got  %+v\n want %+v", got, in)
	}
}

func TestDecodeStampToleratesProse(t *testing.T) {
	stamp, err := EncodeStamp(Stamp{SessionID: "s1", StartSHA: "deadbeef", Buildable: false})
	if err != nil {
		t.Fatal(err)
	}
	// A commit-tree message with a human subject line above the marker line.
	msg := "WIP snapshot\n\n" + stamp + "\n"
	got, ok := DecodeStamp(msg)
	if !ok || got.SessionID != "s1" || got.StartSHA != "deadbeef" {
		t.Fatalf("DecodeStamp(prose+marker) = %+v, ok=%v", got, ok)
	}
}

func TestDecodeStampMissingMarker(t *testing.T) {
	if _, ok := DecodeStamp("just an ordinary commit message\n"); ok {
		t.Error("DecodeStamp on a markerless message reported ok=true")
	}
}

func TestFoldSortsAndComputesAge(t *testing.T) {
	recs := []RefRecord{
		{Ref: "refs/fak/wip/zeta", Object: "obj-z", Stamp: Stamp{SessionID: "zeta", StartSHA: "z", CheckpointedAt: 100}},
		{Ref: "refs/fak/wip/alpha", Object: "obj-a", Stamp: Stamp{SessionID: "alpha", StartSHA: "a", Leaves: []string{"cmd/fak"}, Buildable: true, CheckpointedAt: 90}},
	}
	rep := Fold(recs, 120)
	if rep.Count != 2 {
		t.Fatalf("Count = %d, want 2", rep.Count)
	}
	// Deterministic order: sorted by session, so alpha precedes zeta.
	if rep.Sessions[0].Session != "alpha" || rep.Sessions[1].Session != "zeta" {
		t.Fatalf("not sorted by session: %q, %q", rep.Sessions[0].Session, rep.Sessions[1].Session)
	}
	if rep.Sessions[0].AgeSeconds != 30 { // 120 - 90
		t.Errorf("alpha age = %d, want 30", rep.Sessions[0].AgeSeconds)
	}
	if rep.Sessions[1].AgeSeconds != 20 { // 120 - 100
		t.Errorf("zeta age = %d, want 20", rep.Sessions[1].AgeSeconds)
	}
	// Nil leaves normalize to a non-nil empty slice (stable JSON).
	if rep.Sessions[1].Leaves == nil {
		t.Error("zeta leaves should be non-nil empty, got nil")
	}
}

func TestFoldClampsNegativeAgeAndLabelsFromRef(t *testing.T) {
	recs := []RefRecord{
		// Stamp lost its session id -> labelled from the ref; future stamp -> age clamps to 0.
		{Ref: "refs/fak/wip/orphan", Object: "o1", Stamp: Stamp{CheckpointedAt: 500}},
	}
	rep := Fold(recs, 100)
	if rep.Sessions[0].Session != "orphan" {
		t.Errorf("session label = %q, want %q (from ref)", rep.Sessions[0].Session, "orphan")
	}
	if rep.Sessions[0].AgeSeconds != 0 {
		t.Errorf("negative age not clamped: %d", rep.Sessions[0].AgeSeconds)
	}
}
