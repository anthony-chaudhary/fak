package slackquiet

import (
	"os"
	"path/filepath"
	"testing"
)

// TestEngagedValueVocabulary pins the opt-in vocabulary and, more importantly, the
// polarity: unset/blank is NOT quiet, and the fleet's off-words are NOT quiet either. A
// regression that flipped the default would silently mute the whole fleet on a fresh
// clone, so it is asserted first.
func TestEngagedValueVocabulary(t *testing.T) {
	quiet := []string{"1", "on", "true", "yes", "enable", "enabled", "ON", " On ", "TRUE", "Enabled"}
	for _, v := range quiet {
		if !EngagedValue(v) {
			t.Errorf("EngagedValue(%q) = false, want true (on-ish means quiet)", v)
		}
	}
	loud := []string{"", " ", "0", "off", "false", "no", "disable", "disabled", "quiet", "0x0", "nope"}
	for _, v := range loud {
		if EngagedValue(v) {
			t.Errorf("EngagedValue(%q) = true, want false (only the opt-in vocabulary quiets)", v)
		}
	}
}

// TestEngagedUnsetIsNotQuiet is the default-on contract: with the variable absent the
// brake is disengaged, so a fresh clone and a fresh CI runner post exactly as before.
func TestEngagedUnsetIsNotQuiet(t *testing.T) {
	os.Unsetenv(EnvVar)
	if Engaged() {
		t.Fatal("Engaged() = true with FAK_SLACK_QUIET unset; the brake must be opt-in")
	}
}

// TestEngagedReadsEnv covers the environment half of the resolver.
func TestEngagedReadsEnv(t *testing.T) {
	t.Setenv(EnvVar, "off")
	if Engaged() {
		t.Error("Engaged() = true for FAK_SLACK_QUIET=off; off is not in the opt-in vocabulary")
	}
	t.Setenv(EnvVar, "1")
	if !Engaged() {
		t.Error("Engaged() = false for FAK_SLACK_QUIET=1; want quiet")
	}
}

// TestEngagedReadsEnvFile covers the .env.slack.local half: the operator's existing
// gitignored file must be able to quiet a workspace, which is the whole point of
// routing through slackenv instead of reading os.Getenv directly.
func TestEngagedReadsEnvFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(EnvVar+"=1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	os.Unsetenv(EnvVar)
	t.Chdir(dir)
	if !Engaged() {
		t.Fatalf("Engaged() = false with %s=1 in .env.slack.local; the file half of the resolver is not wired", EnvVar)
	}
	if got := Source(); got != "file" {
		t.Errorf("Source() = %q, want %q", got, "file")
	}
}

// TestSourceNamesTheSeam keeps the diagnostic honest: a health report must be able to
// say whether the operator set the switch in the environment or in the file.
func TestSourceNamesTheSeam(t *testing.T) {
	t.Setenv(EnvVar, "1")
	if got := Source(); got != "env" {
		t.Errorf("Source() = %q, want %q", got, "env")
	}
}
