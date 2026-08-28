package selfinstall

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func TestBeginLaunchTransactionPublishesCompletePriorBeforeMarker(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak")
	if err := os.WriteFile(target, []byte("old-version"), 0o755); err != nil {
		t.Fatal(err)
	}
	finish, err := BeginLaunchTransaction(target)
	if err != nil {
		t.Fatal(err)
	}
	marker := launchshim.UpdateStatePath(target)
	raw, err := os.ReadFile(marker)
	if err != nil {
		t.Fatal(err)
	}
	var state struct {
		Target string `json:"target"`
		Prior  string `json:"prior"`
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if state.Target != target {
		t.Fatalf("target=%q want %q", state.Target, target)
	}
	prior, err := os.ReadFile(state.Prior)
	if err != nil {
		t.Fatal(err)
	}
	if string(prior) != "old-version" {
		t.Fatalf("prior=%q", prior)
	}
	finish()
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("marker remains: %v", err)
	}
	if got, err := os.ReadFile(state.Prior); err != nil || string(got) != "old-version" {
		t.Fatalf("prior must survive read-to-exec race: %q err=%v", got, err)
	}
}
