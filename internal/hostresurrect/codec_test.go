package hostresurrect

import "testing"

func TestRequestCodecRoundTripAndRejectsIncomplete(t *testing.T) {
	want := Request{Schema: Schema, EventID: "event", Session: "g1", CWD: t.TempDir(), Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	encoded, err := EncodeRequest(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeRequest(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got.Session != want.Session || got.CWD != want.CWD || len(got.Command) != 3 {
		t.Fatalf("got=%+v", got)
	}
	if _, err := EncodeRequest(Request{Schema: Schema}); err == nil {
		t.Fatal("incomplete request accepted")
	}
}

func TestRequestCodecAllowsCodexResume(t *testing.T) {
	want := Request{Schema: Schema, EventID: "event", Session: "c1", CWD: t.TempDir(), Command: []string{"codex", "--resume", "c1"}, ResumeHandle: "c1"}
	encoded, err := EncodeRequest(want)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := DecodeRequest(encoded); err != nil || got.Session != want.Session {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
func TestEncodeRequestRejectsArbitraryExecutable(t *testing.T) {
	req := Request{Schema: Schema, EventID: "evt", Session: "g1", CWD: t.TempDir(), Command: []string{"powershell.exe", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := EncodeRequest(req); err == nil {
		t.Fatal("arbitrary executable admitted")
	}
}
