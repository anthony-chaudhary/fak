package hostresurrect

import (
	"path/filepath"
	"testing"
)

func TestSpoolPersistsTypedRequestUntilCompleted(t *testing.T) {
	dir := t.TempDir()
	want := Request{Schema: Schema, EventID: "evt:1", Session: "g1", CWD: t.TempDir(), Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	path, err := Enqueue(dir, want)
	if err != nil {
		t.Fatal(err)
	}
	pending, err := Pending(dir)
	if err != nil || len(pending) != 1 || pending[0] != path {
		t.Fatalf("pending=%v err=%v", pending, err)
	}
	got, err := ReadQueued(path)
	if err != nil || got.Session != "g1" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if err := CompleteQueued(path); err != nil {
		t.Fatal(err)
	}
	pending, _ = Pending(dir)
	if len(pending) != 0 {
		t.Fatalf("pending after complete=%v", pending)
	}
	if _, err := filepath.Abs(path); err != nil {
		t.Fatal(err)
	}
}
