package toolprocgate

import "testing"

// TestEnvFromStringsDropsWindowsDrivePseudoVars pins the fix for the guard-child
// launch failure on Windows: os.Environ() there can carry per-drive current-
// directory pseudo-variables ("=C:=...", "=ExitCode=...", "=::=...") whose name is
// empty once split on the first '='. Before the skip, EnvFromStrings handed the
// empty name to normalizeEnv, which fail-closed the whole spawn with EMPTY_ENV_NAME
// and surfaced as `fak guard: could not run "claude": EMPTY_ENV_NAME`. The benign
// pseudo-vars must be dropped, leaving the genuinely-named entries intact.
func TestEnvFromStringsDropsWindowsDrivePseudoVars(t *testing.T) {
	in := []string{
		`=C:=C:\work\fak`,
		"PATH=/usr/bin",
		"=ExitCode=00000000",
		"ANTHROPIC_BASE_URL=http://127.0.0.1:8080",
		`=::=::\`,
	}
	got, err := EnvFromStrings(in)
	if err != nil {
		t.Fatalf("EnvFromStrings returned %v; the '=C:'-style pseudo-vars must be dropped, not error", err)
	}
	want := map[string]string{
		"PATH":               "/usr/bin",
		"ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
	}
	if len(got) != len(want) {
		t.Fatalf("got %d env vars %+v, want %d (%v)", len(got), got, len(want), want)
	}
	for _, kv := range got {
		if kv.Name == "" {
			t.Fatalf("an empty-named entry survived: %+v", kv)
		}
		w, ok := want[kv.Name]
		if !ok {
			t.Fatalf("unexpected env var %q survived the filter", kv.Name)
		}
		if kv.Value != w {
			t.Errorf("%s = %q, want %q", kv.Name, kv.Value, w)
		}
	}
}

// TestEnvFromStringsStillNormalizesNamedEntries guards against the skip weakening
// normalizeEnv for real, named variables: a last-wins duplicate is still folded,
// and a NUL-bearing value is still rejected.
func TestEnvFromStringsStillNormalizesNamedEntries(t *testing.T) {
	got, err := EnvFromStrings([]string{"FOO=first", "FOO=second"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 1 || got[0].Name != "FOO" || got[0].Value != "second" {
		t.Fatalf("duplicate fold broke: got %+v, want single FOO=second", got)
	}
	if _, err := EnvFromStrings([]string{"FOO=a\x00b"}); err == nil {
		t.Fatal("a NUL-bearing value must still be rejected as INVALID_ENV")
	}
}
