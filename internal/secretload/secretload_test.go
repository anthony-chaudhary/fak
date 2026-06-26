package secretload

import (
	"errors"
	"strings"
	"testing"
)

// mapSource is a test SecretSource backed by an in-memory map.
type mapSource struct {
	name string
	m    map[string]string
}

func (s mapSource) Name() string { return s.name }
func (s mapSource) Lookup(k string) (string, bool) {
	v, ok := s.m[k]
	if !ok || v == "" {
		return "", false
	}
	return v, true
}

func TestLoaderPriorityOrderFirstSourceWins(t *testing.T) {
	hi := mapSource{name: "hi", m: map[string]string{"K": "from-hi"}}
	lo := mapSource{name: "lo", m: map[string]string{"K": "from-lo", "ONLY_LO": "lo-only"}}
	l := New(hi, lo)

	if v, src, ok := l.LookupSource("K"); !ok || v != "from-hi" || src != "hi" {
		t.Fatalf("priority: got (%q,%q,%v), want (from-hi,hi,true)", v, src, ok)
	}
	if v, src, ok := l.LookupSource("ONLY_LO"); !ok || v != "lo-only" || src != "lo" {
		t.Fatalf("fallthrough: got (%q,%q,%v), want (lo-only,lo,true)", v, src, ok)
	}
	if _, ok := l.Lookup("ABSENT"); ok {
		t.Fatal("absent key reported present")
	}
	if got := l.Sources(); len(got) != 2 || got[0] != "hi" || got[1] != "lo" {
		t.Fatalf("Sources order = %v", got)
	}
}

func TestAddSourceIsLowestPriority(t *testing.T) {
	l := New(mapSource{name: "hi", m: map[string]string{"K": "hi"}})
	l.AddSource(mapSource{name: "lo", m: map[string]string{"K": "lo", "X": "x"}})
	if v, _ := l.Lookup("K"); v != "hi" {
		t.Fatalf("added source must not preempt existing: got %q", v)
	}
	if v, _ := l.Lookup("X"); v != "x" {
		t.Fatalf("added source not consulted: got %q", v)
	}
	l.AddSource(nil) // must be a no-op
	if got := l.Sources(); len(got) != 2 {
		t.Fatalf("nil source was added: %v", got)
	}
}

func TestRequireChecklistMissingPresentInvalid(t *testing.T) {
	src := mapSource{name: "env", m: map[string]string{
		"PRESENT": "ok-value-here",
		"BADFMT":  "short",
	}}
	l := New(src)
	l.Require("PRESENT", "a present secret", nil)
	l.Require("MISSING", "a secret nobody supplies", nil)
	l.Require("BADFMT", "must be >=10 chars", func(v string) error {
		if len(v) < 10 {
			return errors.New("too short")
		}
		return nil
	})

	got := map[string]Status{}
	src2 := map[string]string{}
	for _, r := range l.CheckRequired() {
		got[r.Key] = r.Status
		src2[r.Key] = r.Source
	}
	if got["PRESENT"] != StatusOK {
		t.Errorf("PRESENT = %s, want ok", got["PRESENT"])
	}
	if src2["PRESENT"] != "env" {
		t.Errorf("PRESENT source = %q, want env", src2["PRESENT"])
	}
	if got["MISSING"] != StatusMissing {
		t.Errorf("MISSING = %s, want missing", got["MISSING"])
	}
	if got["BADFMT"] != StatusInvalid {
		t.Errorf("BADFMT = %s, want invalid", got["BADFMT"])
	}

	miss := l.Missing()
	if len(miss) != 2 {
		t.Fatalf("Missing() = %d rows, want 2", len(miss))
	}

	ok, report := l.StartupReport()
	if ok {
		t.Error("StartupReport ok=true with a missing+invalid secret")
	}
	if !strings.Contains(report, "MISSING") || !strings.Contains(report, "INVALID") {
		t.Errorf("report omits a failure class:\n%s", report)
	}
	// The report must name keys but never leak the actual secret value.
	if strings.Contains(report, "ok-value-here") {
		t.Errorf("StartupReport leaked a secret value:\n%s", report)
	}
}

func TestRequireReplacesEarlierDeclaration(t *testing.T) {
	l := New(mapSource{name: "env", m: map[string]string{"K": "v"}})
	l.Require("K", "first", func(string) error { return errors.New("always") })
	l.Require("K", "second", nil) // replaces — now no validator
	got := l.CheckRequired()
	if len(got) != 1 {
		t.Fatalf("duplicate Require created %d rows, want 1", len(got))
	}
	if got[0].Status != StatusOK || got[0].Description != "second" {
		t.Fatalf("replacement not applied: %+v", got[0])
	}
}

func TestRedactMasksKnownValueAndCanonShape(t *testing.T) {
	l := New()
	l.MarkSecret("hunter2-supersecret-value")
	in := "logged token=hunter2-supersecret-value and key sk-abcdef0123456789ABCDEF here"
	out := l.Redact(in)
	if strings.Contains(out, "hunter2-supersecret-value") {
		t.Errorf("known value not masked: %q", out)
	}
	if strings.Contains(out, "sk-abcdef0123456789ABCDEF") {
		t.Errorf("canon shape not masked: %q", out)
	}
	if !strings.Contains(out, RedactPlaceholder) {
		t.Errorf("no placeholder in output: %q", out)
	}
}

func TestRedactShortValueIgnored(t *testing.T) {
	l := New()
	l.MarkSecret("ab") // below minMaskLen
	if l.Redact("value ab stays") != "value ab stays" {
		t.Error("short value should not be masked")
	}
}

func TestRedactAutoMasksResolvedRequiredSecret(t *testing.T) {
	src := mapSource{name: "env", m: map[string]string{"API_KEY": "live-credential-abcdef"}}
	l := New(src)
	l.Require("API_KEY", "the api key", nil)
	// Resolving a required key registers its value for masking automatically.
	if _, ok := l.Lookup("API_KEY"); !ok {
		t.Fatal("API_KEY should resolve")
	}
	out := l.Redact("trace: API_KEY=live-credential-abcdef end")
	if strings.Contains(out, "live-credential-abcdef") {
		t.Errorf("resolved required secret not auto-masked: %q", out)
	}
}

func TestRedactShapesNoLoader(t *testing.T) {
	in := "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 leaked"
	out := RedactShapes(in)
	if strings.Contains(out, "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") {
		t.Errorf("github token shape not masked: %q", out)
	}
}

func TestRedactEmptyAndNoSecretsUnchanged(t *testing.T) {
	l := New()
	if l.Redact("") != "" {
		t.Error("empty in -> empty out")
	}
	clean := "a perfectly ordinary log line with no credentials"
	if l.Redact(clean) != clean {
		t.Error("clean line should pass through unchanged")
	}
}

func TestDefaultLoaderHasOSEnv(t *testing.T) {
	if got := Default().Sources(); len(got) != 1 || got[0] != "os-env" {
		t.Fatalf("Default sources = %v, want [os-env]", got)
	}
}
