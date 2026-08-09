package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

func TestTUILaunchCapturedRenderShowsStateAndKeyHint(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	provider := filepath.Join(dir, "provider")
	if err := os.WriteFile(provider, []byte("provider"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := launchshim.Save(launchshim.Config{Default: "third", Providers: map[string]launchshim.Provider{"third": {Command: provider}}}); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runTUILaunch(&out, &errOut, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"Provider Launch", "ENABLED (fak guard intercepts)", "[d] direct next launch (one-shot)", "persisted disable"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("captured render missing %q:\n%s", want, out.String())
		}
	}
	out.Reset()
	if code := runTUILaunch(&out, &errOut, []string{"--direct-next"}); code != 0 {
		t.Fatal(code)
	}
	if !strings.Contains(out.String(), "DIRECT NEXT (one-shot; persisted setting unchanged)") {
		t.Fatalf("one-shot render=%s", out.String())
	}
	c, _ := launchshim.Load()
	if c.Disabled {
		t.Fatal("one-shot toggle persisted disable")
	}
}

func TestTUILaunchUsesPersistedLaunchshimState(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	provider := filepath.Join(dir, "provider")
	_ = os.WriteFile(provider, []byte("provider"), 0o755)
	if err := launchshim.Save(launchshim.Config{Disabled: true, Providers: map[string]launchshim.Provider{"third": {Command: provider}}}); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runTUILaunch(&out, &errOut, nil); code != 0 {
		t.Fatal(code)
	}
	if !strings.Contains(out.String(), "DISABLED (persisted pass-through)") {
		t.Fatalf("render=%s", out.String())
	}
	pane, ok := tuiplugin.Lookup("launch")
	if !ok || len(pane.Controls) != 2 {
		t.Fatalf("pane=%+v ok=%v", pane, ok)
	}
}
