package hostresurrect

import "testing"

func TestRequestCodecRoundTripAndRejectsIncomplete(t *testing.T) {
	want := Request{Schema: Schema, EventID: "event", Session: "g1", CWD: `C:\work`, Command: []string{"claude", "--continue"}, ResumeHandle: "g1"}
	encoded, err := EncodeRequest(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeRequest(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got.Session != want.Session || got.CWD != want.CWD || len(got.Command) != 2 {
		t.Fatalf("got=%+v", got)
	}
	if _, err := EncodeRequest(Request{Schema: Schema}); err == nil {
		t.Fatal("incomplete request accepted")
	}
}
