package projectassets

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The fragile form is what shipped in a hand-edited ~/.pi/agent/models.json and what produced
// the operator-visible failure:
//
//	Error: API key auth failed for provider hive-ai: Failed to resolve API key for provider
//	"hive-ai" from shell command: powershell -NoProfile -ExecutionPolicy Bypass -Command
//	"(Get-Content -LiteralPath 'C:\Users\USER\.fak\keys\hive.key' -Raw).Trim("
const fragileHiveKeyCommand = `!powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content -LiteralPath 'C:\Users\USER\.fak\keys\hive.key' -Raw).Trim()"`

func TestPiProviderKeyCommandIsFragile(t *testing.T) {
	cases := []struct {
		name   string
		apiKey string
		want   bool
	}{
		{"powershell get-content", fragileHiveKeyCommand, true},
		{"pwsh get-content", `!pwsh -NoProfile -Command "(Get-Content -LiteralPath 'C:\k.key' -Raw).Trim()"`, true},
		{"already normalized", `!cat 'C:\Users\USER\.fak\keys\hive.key'`, false},
		{"env template", `${HIVE_API_KEY}`, false},
		{"literal", "fak", false},
		{"empty", "", false},
		{"plain command", "!op read op://vault/hive/key", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := PiProviderKeyCommandIsFragile(tc.apiKey); got != tc.want {
				t.Errorf("PiProviderKeyCommandIsFragile(%q) = %v, want %v", tc.apiKey, got, tc.want)
			}
		})
	}
}

func TestNormalizePiProviderKeyCommand(t *testing.T) {
	got, changed := NormalizePiProviderKeyCommand(fragileHiveKeyCommand)
	want := `!cat 'C:\Users\USER\.fak\keys\hive.key'`
	if !changed {
		t.Fatalf("expected the fragile form to be rewritten; got %q, changed=false", got)
	}
	if got != want {
		t.Errorf("NormalizePiProviderKeyCommand = %q, want %q", got, want)
	}

	// The normalized form must not mention powershell and must keep the path verbatim, so no
	// shell expansion can mangle it.
	if strings.Contains(strings.ToLower(got), "powershell") {
		t.Errorf("normalized value still shells out to powershell: %q", got)
	}
	if !strings.Contains(got, `'C:\Users\USER\.fak\keys\hive.key'`) {
		t.Errorf("normalized value lost or requoted the key path: %q", got)
	}

	// Idempotent: normalizing the already-normalized value is a no-op.
	if again, changedAgain := NormalizePiProviderKeyCommand(got); changedAgain || again != got {
		t.Errorf("normalization is not idempotent: %q (changed=%v)", again, changedAgain)
	}
}

// TestNormalizePiProviderKeyCommandPreservesOperatorValues pins the non-goal: an operator's
// deliberate value is never rewritten.
func TestNormalizePiProviderKeyCommandPreservesOperatorValues(t *testing.T) {
	for _, in := range []string{
		"fak",
		"",
		"${HIVE_API_KEY}",
		"!op read op://vault/hive/key",
		`!cat 'C:\already\normalized.key'`,
		`!powershell -Command "(Get-Content $env:USERPROFILE\k.key -Raw).Trim()"`,
	} {
		got, changed := NormalizePiProviderKeyCommand(in)
		if changed || got != in {
			t.Errorf("NormalizePiProviderKeyCommand(%q) = (%q, %v); want unchanged", in, got, changed)
		}
	}
}

// TestPiFragileKeyCommandFailsUnderMinimalPath is the executable root-cause witness: the
// fragile form cannot resolve when the resolving process's PATH lacks the WindowsPowerShell
// directory, which is exactly the condition that produced the operator-visible error. The
// normalized form must survive the same environment.
//
// It is skipped off Windows and when no bash is available, since the failure mode is a
// Windows PATH/PowerShell interaction.
func TestPiFragileKeyCommandFailsUnderMinimalPath(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live-shell witness in -short mode")
	}
	bash := findBashForTest(t)
	if bash == "" {
		t.Skip("no bash on PATH; cannot exercise the configured-shell leg")
	}

	keyDir := t.TempDir()
	keyPath := filepath.Join(keyDir, "hive.key")
	const keyValue = "test-key-material-not-a-real-credential"
	if err := os.WriteFile(keyPath, []byte(keyValue), 0o600); err != nil {
		t.Fatalf("write key fixture: %v", err)
	}

	// A PATH that has bash's own directory but NOT System32\WindowsPowerShell\v1.0 — the
	// trimmed-PATH condition that made `powershell` unresolvable.
	minimalPATH := filepath.Dir(bash)

	run := func(command string) (string, int) {
		cmd := exec.Command(bash, "-c", command)
		cmd.Env = []string{"PATH=" + minimalPATH}
		out, err := cmd.Output()
		code := 0
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				code = exitErr.ExitCode()
			} else {
				code = -1
			}
		}
		return strings.TrimSpace(string(out)), code
	}

	fragile := strings.TrimPrefix(fragileHiveKeyCommand, "!")
	fragileOut, fragileCode := run(strings.Replace(fragile, `C:\Users\USER\.fak\keys\hive.key`, keyPath, 1))
	if fragileCode == 0 && fragileOut == keyValue {
		t.Skip("powershell IS resolvable on this PATH; the trimmed-PATH failure cannot be reproduced here")
	}
	if fragileOut == keyValue {
		t.Errorf("fragile form unexpectedly returned the key under a minimal PATH")
	}

	normalized, changed := NormalizePiProviderKeyCommand(
		strings.Replace(fragileHiveKeyCommand, `C:\Users\USER\.fak\keys\hive.key`, keyPath, 1),
	)
	if !changed {
		t.Fatalf("expected the fragile fixture to be rewritten")
	}
	normalizedOut, normalizedCode := run(strings.TrimPrefix(normalized, "!"))
	if normalizedCode != 0 || normalizedOut != keyValue {
		t.Errorf("normalized form failed under minimal PATH: code=%d out=%q (command %q)", normalizedCode, normalizedOut, normalized)
	}
}

func findBashForTest(t *testing.T) string {
	t.Helper()
	if p, err := exec.LookPath("bash"); err == nil {
		return p
	}
	for _, candidate := range []string{
		`C:\Program Files\Git\bin\bash.exe`,
		`C:\Program Files (x86)\Git\bin\bash.exe`,
	} {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return ""
}

// TestEnsurePiProviderConfigRepairsFragileProviderKey is the end-to-end witness: a config
// carrying the fragile hive-ai apiKey is repaired by the canonical writer, while unrelated
// providers and models are preserved byte-for-byte.
func TestEnsurePiProviderConfigRepairsFragileProviderKey(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "models.json")

	existing := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:18101/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash"}]
    },
    "hive-ai": {
      "api": "openai-completions",
      "apiKey": ` + jsonQuote(fragileHiveKeyCommand) + `,
      "baseUrl": "https://api-cdn.thehive.ai/api/v3",
      "models": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash"}]
    }
  }
}`
	if err := os.WriteFile(target, []byte(existing), 0o644); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	_, modified, err := EnsurePiProviderConfig(target, "http://127.0.0.1:18101/v1", "deepseek-ai/DeepSeek-V4.1-Flash")
	if err != nil {
		t.Fatalf("EnsurePiProviderConfig: %v", err)
	}
	if !modified {
		t.Fatalf("expected the fragile apiKey to be repaired (modified=true)")
	}

	raw, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if strings.Contains(string(raw), "powershell") {
		t.Errorf("fragile PowerShell apiKey survived the repair:\n%s", raw)
	}
	if !strings.Contains(string(raw), `"!cat 'C:\\Users\\USER\\.fak\\keys\\hive.key'"`) {
		t.Errorf("expected the normalized !cat form in the repaired config:\n%s", raw)
	}
	if !strings.Contains(string(raw), `"hive-ai"`) {
		t.Errorf("repair dropped the hive-ai provider:\n%s", raw)
	}
	// An operator's explicit local literal must not be rewritten.
	if !strings.Contains(string(raw), `"apiKey": "fak"`) {
		t.Errorf("repair clobbered the fak provider literal:\n%s", raw)
	}
}

func jsonQuote(s string) string {
	out, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(out)
}
