package ultracodebench

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestActivationStoreRoundTripsRedactedReceipt(t *testing.T) {
	root := t.TempDir()
	r, err := BeforeSpawn(BeforeSpawnInput{RunID: "run-store", ChildID: "worker-1", Harness: "claude", Requested: SettingOn, Resolved: SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteActivation(root, r); err != nil {
		t.Fatal(err)
	}
	got, err := ReadActivations(root, r.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !reflect.DeepEqual(got[0], r) {
		t.Fatalf("round trip=%+v want=%+v", got, r)
	}
	path, _ := ActivationReceiptPath(root, r.RunID, r.ChildID)
	raw, _ := os.ReadFile(path)
	for _, forbidden := range []string{"prompt", "account", "host", "settings", "argv"} {
		if strings.Contains(string(raw), forbidden) {
			t.Fatalf("stored receipt contains %q: %s", forbidden, raw)
		}
	}
}

func TestActivationStoreRejectsPathEscapeIdentifiers(t *testing.T) {
	for _, id := range []string{"../escape", `..\\escape`, "a/b", ".", ".."} {
		if _, err := ActivationReceiptPath(t.TempDir(), id, "child"); err == nil {
			t.Fatalf("run id %q accepted", id)
		}
	}
}
