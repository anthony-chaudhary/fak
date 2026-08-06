package launchshim

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRoundTripAndDirect(t *testing.T) {
	p := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", p)
	in := Config{Default: "claude", Providers: map[string]Provider{"claude": {Command: "/real/claude"}}}
	if err := Save(in); err != nil {
		t.Fatal(err)
	}
	got, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Default != "claude" || got.Providers["claude"].Command != "/real/claude" {
		t.Fatalf("got %+v", got)
	}
	t.Setenv("FAK_DIRECT", "1")
	if !EffectiveDirect(got, false) {
		t.Fatal("FAK_DIRECT must bypass fak")
	}
	_ = os.Remove(p)
}
