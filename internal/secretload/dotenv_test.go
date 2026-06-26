package secretload

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDotEnvParseDialect(t *testing.T) {
	content := "" +
		"# a comment\n" +
		"\n" +
		"PLAIN=value1\n" +
		"export EXPORTED=value2\n" +
		"QUOTED=\"with spaces\"\n" +
		"SQUOTED='single quoted'\n" +
		"  SPACED  =  trimmed  \n"
	d, err := parseDotEnv("test.env", []byte(content))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	cases := map[string]string{
		"PLAIN":    "value1",
		"EXPORTED": "value2",
		"QUOTED":   "with spaces",
		"SQUOTED":  "single quoted",
		"SPACED":   "trimmed",
	}
	for k, want := range cases {
		if v, ok := d.Lookup(k); !ok || v != want {
			t.Errorf("%s = (%q,%v), want %q", k, v, ok, want)
		}
	}
}

func TestDotEnvMalformedLine(t *testing.T) {
	if _, err := parseDotEnv("bad.env", []byte("NOEQUALS\n")); err == nil {
		t.Error("a line without '=' must error")
	}
	if _, err := parseDotEnv("bad.env", []byte("=novalue\n")); err == nil {
		t.Error("an empty key must error")
	}
}

func TestDotEnvExpiresPreflight(t *testing.T) {
	past := time.Now().Add(-1 * time.Hour).UTC().Format(time.RFC3339)
	d, err := parseDotEnv("e.env", []byte("K=v-value\n"+ExpiresAtKey+"="+past+"\n"))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// The expiry key is preflight metadata, never a returned secret.
	if _, ok := d.Lookup(ExpiresAtKey); ok {
		t.Error("ExpiresAtKey must not be returned from Lookup")
	}
	if _, ok := d.ExpiresAt(); !ok {
		t.Error("ExpiresAt should report a declared expiry")
	}
	if err := d.Preflight(time.Now()); err == nil {
		t.Error("Preflight on an expired file must error")
	}

	future := time.Now().Add(1 * time.Hour).UTC().Format(time.RFC3339)
	d2, _ := parseDotEnv("f.env", []byte("K=v\n"+ExpiresAtKey+"="+future+"\n"))
	if err := d2.Preflight(time.Now()); err != nil {
		t.Errorf("Preflight on a fresh file must pass, got %v", err)
	}

	d3, _ := parseDotEnv("n.env", []byte("K=v\n")) // no expiry declared
	if err := d3.Preflight(time.Now()); err != nil {
		t.Errorf("no-expiry file must always pass, got %v", err)
	}
}

func TestDotEnvBadExpiresFormat(t *testing.T) {
	if _, err := parseDotEnv("e.env", []byte(ExpiresAtKey+"=not-a-time\n")); err == nil {
		t.Error("a non-RFC3339 expiry must error at parse")
	}
}

func TestLoadDotEnvMissingFileIsNoError(t *testing.T) {
	d, existed, err := LoadDotEnv(filepath.Join(t.TempDir(), "nope.env"))
	if err != nil {
		t.Fatalf("missing file must not error: %v", err)
	}
	if existed {
		t.Error("existed should be false for a missing file")
	}
	if _, ok := d.Lookup("ANY"); ok {
		t.Error("missing .env should supply nothing")
	}
}

func TestDiscoverDotEnvPrecedence(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(".env", "K=from-base\nBASE_ONLY=b\n")
	write(".env.local", "K=from-local\n")

	got, err := DiscoverDotEnv(dir, "")
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("found %d files, want 2 (.env.local, .env)", len(got))
	}
	// Wire into a loader in discovered order; .env.local must win for K.
	l := New()
	for _, d := range got {
		l.AddSource(d)
	}
	if v, _ := l.Lookup("K"); v != "from-local" {
		t.Errorf("precedence: K = %q, want from-local", v)
	}
	if v, _ := l.Lookup("BASE_ONLY"); v != "b" {
		t.Errorf("base-only key not resolved: %q", v)
	}
}
