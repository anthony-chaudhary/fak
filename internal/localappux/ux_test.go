package localappux

import (
	"os"
	"strings"
	"testing"
)

func TestCapturedStateRenders(t *testing.T) {
	states := []View{{State: StateFirstRun, Mode: ModeAutomatic, AssetBytes: 6 << 30, FreeBytes: 20 << 30, PendingTasks: []string{"draft", "summarize"}}, {State: StatePartial, Mode: ModePreferLocal, ReadyTasks: []string{"summarize"}, PendingTasks: []string{"draft"}}, {State: StateNoSpace, Mode: ModeLocalOnly, AssetBytes: 6 << 30, FreeBytes: 2 << 30}, {State: StateNoNetwork, Mode: ModeAutomatic}, {State: StatePressure, Mode: ModeAutomatic}, {State: StateBattery, Mode: ModePreferLocal}, {State: StateThermal, Mode: ModeAutomatic}, {State: StateHelperRestart, Mode: ModeAutomatic}, {State: StateHandoffAsk, Mode: ModeAutomatic, LocalData: "the document text", Destination: "your selected provider"}, {State: StateHandoffDenied, Mode: ModeLocalOnly}, {State: StateRollback, Mode: ModeAutomatic}}
	var out strings.Builder
	for _, v := range states {
		out.WriteString("=== " + string(v.State) + " ===\n")
		out.WriteString(Render(v))
	}
	want, err := os.ReadFile("testdata/states.golden.txt")
	if err != nil {
		t.Fatal(err)
	}
	normalized := strings.ReplaceAll(string(want), "\r\n", "\n")
	if out.String() != normalized {
		t.Fatalf("render mismatch\n--- got ---\n%s\n--- want ---\n%s", out.String(), normalized)
	}
}
func TestDiagnosticPreviewIsConsentSafe(t *testing.T) {
	b, err := PreviewDiagnostic(Diagnostic{AppVersion: "1.2", State: StateHelperRestart, Mode: ModeLocalOnly, Engine: "fak-native", ErrorCode: "HELPER_EXIT", Paths: []string{"/Users/alice/private"}, Prompt: "private application", Token: "secret"})
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, secret := range []string{"alice", "private application", "secret", "paths", "prompt", "token"} {
		if strings.Contains(s, secret) {
			t.Fatalf("preview leaked %q: %s", secret, s)
		}
	}
	for _, required := range []string{"fak.local-app-diagnostic/1", "fak-native", "HELPER_EXIT"} {
		if !strings.Contains(s, required) {
			t.Fatalf("preview omitted %q", required)
		}
	}
}

func TestReadyStateRendersReadyCopy(t *testing.T) {
	got := Render(View{State: StateReady, Mode: ModeAutomatic})
	for _, want := range []string{"Local features are ready", "Your tasks run on this Mac.", "Mode: Automatic"} {
		if !strings.Contains(got, want) {
			t.Fatalf("ready render missing %q: %s", want, got)
		}
	}
}
