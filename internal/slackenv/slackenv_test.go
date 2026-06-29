package slackenv

import (
	"os"
	"path/filepath"
	"testing"
)

// writeEnv writes a .env.slack.local with the given body into dir.
func writeEnv(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, EnvFileName), []byte(body), 0o600); err != nil {
		t.Fatalf("write env file: %v", err)
	}
}

func TestFileValueFrom(t *testing.T) {
	cases := []struct {
		name string
		body string
		key  string
		want string
	}{
		{"plain", "FAK_X_TOKEN=bottok-abc\n", "FAK_X_TOKEN", "bottok-abc"},
		{"export prefix", "export FAK_X_TOKEN=bottok-exp\n", "FAK_X_TOKEN", "bottok-exp"},
		{"leading whitespace", "   FAK_X_TOKEN=ws\n", "FAK_X_TOKEN", "ws"},
		{"value trimmed", "FAK_X_TOKEN=  trimmed  \n", "FAK_X_TOKEN", "trimmed"},
		{"first match wins", "FAK_X_TOKEN=first\nFAK_X_TOKEN=second\n", "FAK_X_TOKEN", "first"},
		{"value contains equals", "FAK_X_TOKEN=a=b=c\n", "FAK_X_TOKEN", "a=b=c"},
		{"key absent", "FAK_OTHER=z\n", "FAK_X_TOKEN", ""},
		{"blank value", "FAK_X_TOKEN=\n", "FAK_X_TOKEN", ""},
		{"prefix is not a false match", "FAK_X_TOKEN_EXTRA=no\n", "FAK_X_TOKEN", ""},
		{"crlf tolerated", "FAK_X_TOKEN=crlf\r\n", "FAK_X_TOKEN", "crlf"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeEnv(t, dir, tc.body)
			if got := fileValueFrom(dir, tc.key); got != tc.want {
				t.Fatalf("fileValueFrom(%q) = %q, want %q", tc.key, got, tc.want)
			}
		})
	}
}

func TestFileValueFromMissingFile(t *testing.T) {
	if got := fileValueFrom(t.TempDir(), "FAK_X_TOKEN"); got != "" {
		t.Fatalf("missing file: got %q, want \"\"", got)
	}
}

// TestFileValueWalksUp confirms a key set in an ancestor .env.slack.local resolves from a
// nested working directory, and that a nearer file's blank override wins over an ancestor.
func TestFileValueWalksUp(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "a", "b", "c")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeEnv(t, root, "FAK_X_CHANNEL=C-ROOT\n")
	if got := fileValueFrom(nested, "FAK_X_CHANNEL"); got != "C-ROOT" {
		t.Fatalf("walk-up: got %q, want C-ROOT", got)
	}

	// A nearer file that blanks the key stops the walk before the ancestor value.
	mid := filepath.Join(root, "a")
	writeEnv(t, mid, "FAK_X_CHANNEL=\n")
	if got := fileValueFrom(nested, "FAK_X_CHANNEL"); got != "" {
		t.Fatalf("nearer blank should win: got %q, want \"\"", got)
	}
}

// TestFileValueWalkBounded confirms the walk stops after maxWalkUp directories: a key set
// further up than the bound does not resolve.
func TestFileValueWalkBounded(t *testing.T) {
	root := t.TempDir()
	writeEnv(t, root, "FAK_X_TOKEN=too-far\n")
	deep := root
	for i := 0; i < maxWalkUp+1; i++ {
		deep = filepath.Join(deep, "d")
	}
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if got := fileValueFrom(deep, "FAK_X_TOKEN"); got != "" {
		t.Fatalf("beyond walk bound: got %q, want \"\" (resolver climbed too far)", got)
	}
}

func TestLookupEnvWins(t *testing.T) {
	dir := t.TempDir()
	writeEnv(t, dir, "FAK_X_TOKEN=from-file\n")
	t.Chdir(dir)
	t.Setenv("FAK_X_TOKEN", "from-env")

	got := Lookup("FAK_X_TOKEN")
	if got.Value != "from-env" || got.Source != SourceEnv {
		t.Fatalf("env must win over file: got %+v", got)
	}
	if !got.Set() {
		t.Fatalf("Set() should be true for a resolved value")
	}
}

func TestLookupFallsToFile(t *testing.T) {
	dir := t.TempDir()
	writeEnv(t, dir, "FAK_X_TOKEN=from-file\n")
	t.Chdir(dir)
	t.Setenv("FAK_X_TOKEN", "") // explicitly empty env => not set

	got := Lookup("FAK_X_TOKEN")
	if got.Value != "from-file" || got.Source != SourceFile {
		t.Fatalf("should fall to file: got %+v", got)
	}
}

func TestLookupUnset(t *testing.T) {
	dir := t.TempDir() // no env file
	t.Chdir(dir)
	t.Setenv("FAK_X_TOKEN", "")

	got := Lookup("FAK_X_TOKEN")
	if got.Set() || got.Source != SourceUnset {
		t.Fatalf("unset key: got %+v, want unset", got)
	}
	if got.Key != "FAK_X_TOKEN" {
		t.Fatalf("unset Resolved should still carry the key it tried: got %q", got.Key)
	}
}
