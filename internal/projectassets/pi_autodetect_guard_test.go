package projectassets

import (
	"os"
	"path/filepath"
	"testing"
)

// TestPiSettingsDefaultModel covers the read half of the auto-detect guard: a
// deliberate defaultModel is reported, and every "no deliberate default" shape
// (missing file, missing key, wrong type, blank) reports "" with no error.
func TestPiSettingsDefaultModel(t *testing.T) {
	t.Run("reads a configured default", func(t *testing.T) {
		dir := t.TempDir()
		settings := filepath.Join(dir, "settings.json")
		if err := os.WriteFile(settings, []byte(`{"defaultModel":"deepseek-ai/DeepSeek-V4.1-Flash"}`), 0644); err != nil {
			t.Fatal(err)
		}
		got, err := PiSettingsDefaultModel(settings)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "deepseek-ai/DeepSeek-V4.1-Flash" {
			t.Fatalf("got %q, want the configured default", got)
		}
	})

	t.Run("missing file reports empty with no error", func(t *testing.T) {
		got, err := PiSettingsDefaultModel(filepath.Join(t.TempDir(), "settings.json"))
		if err != nil {
			t.Fatalf("a missing settings.json must not error: %v", err)
		}
		if got != "" {
			t.Fatalf("got %q, want \"\"", got)
		}
	})

	t.Run("missing key and wrong type report empty", func(t *testing.T) {
		for name, body := range map[string]string{
			"missing-key": `{"theme":"light"}`,
			"wrong-type":  `{"defaultModel":42}`,
			"blank":       `{"defaultModel":"   "}`,
		} {
			settings := filepath.Join(t.TempDir(), "settings.json")
			if err := os.WriteFile(settings, []byte(body), 0644); err != nil {
				t.Fatal(err)
			}
			got, err := PiSettingsDefaultModel(settings)
			if err != nil {
				t.Fatalf("%s: unexpected error: %v", name, err)
			}
			if got != "" {
				t.Fatalf("%s: got %q, want \"\"", name, got)
			}
		}
	})

	t.Run("BOM-prefixed file still parses", func(t *testing.T) {
		settings := filepath.Join(t.TempDir(), "settings.json")
		body := append([]byte{0xEF, 0xBB, 0xBF}, []byte(`{"defaultModel":"qwen38:27b-q4"}`)...)
		if err := os.WriteFile(settings, body, 0644); err != nil {
			t.Fatal(err)
		}
		got, err := PiSettingsDefaultModel(settings)
		if err != nil {
			t.Fatalf("BOM-prefixed settings.json must parse: %v", err)
		}
		if got != "qwen38:27b-q4" {
			t.Fatalf("got %q", got)
		}
	})
}

// TestShouldAdoptDetectedPiModel pins the precedence that fixes the clobber:
// an explicit --model always wins, a deliberate settings.json defaultModel is
// never replaced by the backend's /healthz label, and auto-detect only seeds
// the value when nothing is configured yet.
func TestShouldAdoptDetectedPiModel(t *testing.T) {
	writeDefault := func(t *testing.T, model string) string {
		t.Helper()
		settings := filepath.Join(t.TempDir(), "settings.json")
		if err := os.WriteFile(settings, []byte(`{"defaultModel":"`+model+`"}`), 0644); err != nil {
			t.Fatal(err)
		}
		return settings
	}
	absent := func(t *testing.T) string {
		t.Helper()
		return filepath.Join(t.TempDir(), "settings.json")
	}

	cases := []struct {
		name      string
		settings  func(*testing.T) string
		explicit  string
		detected  string
		wantAdopt bool
	}{
		{
			name:      "explicit model always wins",
			settings:  absent,
			explicit:  "deepseek-ai/DeepSeek-V4.1-Flash",
			detected:  "Qwen3.8-27B-UD-Q2_K_XL",
			wantAdopt: false,
		},
		{
			name:      "deliberate default is not clobbered",
			settings:  func(t *testing.T) string { return writeDefault(t, "deepseek-ai/DeepSeek-V4.1-Flash") },
			explicit:  "",
			detected:  "Qwen3.8-27B-UD-Q2_K_XL",
			wantAdopt: false,
		},
		{
			name:      "auto-detect seeds a first-ever default",
			settings:  absent,
			explicit:  "",
			detected:  "Qwen3.8-27B-UD-Q2_K_XL",
			wantAdopt: true,
		},
		{
			name:      "mock sentinel is never adopted",
			settings:  absent,
			explicit:  "",
			detected:  "mock",
			wantAdopt: false,
		},
		{
			name:      "blank detection is never adopted",
			settings:  absent,
			explicit:  "",
			detected:  "   ",
			wantAdopt: false,
		},
		{
			name:      "explicit model plus deliberate default still refuses",
			settings:  func(t *testing.T) string { return writeDefault(t, "qwen38:27b-q4") },
			explicit:  "deepseek-ai/DeepSeek-V4.1-Flash",
			detected:  "Qwen3.8-27B-UD-Q2_K_XL",
			wantAdopt: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ShouldAdoptDetectedPiModel(tc.settings(t), tc.explicit, tc.detected)
			if got != tc.wantAdopt {
				t.Fatalf("ShouldAdoptDetectedPiModel(explicit=%q, detected=%q) = %v, want %v",
					tc.explicit, tc.detected, got, tc.wantAdopt)
			}
		})
	}
}

// TestShouldAdoptDetectedPiModelUnreadableDegradesFalse: a settings file we
// cannot read is not license to clobber whatever default it may hold.
func TestShouldAdoptDetectedPiModelUnreadableDegradesFalse(t *testing.T) {
	dir := t.TempDir()
	// A directory where the settings file is expected: read fails with a
	// non-not-exist error.
	settings := filepath.Join(dir, "settings.json")
	if err := os.Mkdir(settings, 0755); err != nil {
		t.Fatal(err)
	}
	if ShouldAdoptDetectedPiModel(settings, "", "Qwen3.8-27B-UD-Q2_K_XL") {
		t.Fatal("an unreadable settings file must degrade to false (never clobber)")
	}
}
