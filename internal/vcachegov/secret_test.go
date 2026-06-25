package vcachegov

import "testing"

// secret_test.go covers the Law-D4 content classifier — the fail-closed gate that
// keeps secrets/PII/regulated content out of the provider prefix cache.

func TestWarmableOnlyForCacheable(t *testing.T) {
	if !Warmable(Cacheable) {
		t.Error("Cacheable must be warmable")
	}
	for _, c := range []SecretClassification{Secret, SecretRegulated} {
		if Warmable(c) {
			t.Errorf("%q must NOT be warmable", c)
		}
	}
}

func TestClassifyPrefixFailClosed(t *testing.T) {
	// The zero/unknown label is fail-CLOSED to Secret: a prefix whose content class
	// the canonicalizer could not establish is not trusted into a shared, retained
	// provider cache (Law D4: never warm what you haven't classified safe).
	cases := []struct {
		label string
		want  SecretClassification
	}{
		{"", Secret}, // unknown → fail closed
		{"unknown", Secret},
		{"public", Cacheable},
		{"cacheable", Cacheable},
		{"system", Cacheable},
		{"retrieved", Cacheable},
		{"secret", Secret},
		{"credential", Secret},
		{"api_key", Secret},
		{"token", Secret},
		{"private", Secret},
		{"regulated", SecretRegulated},
		{"pii", SecretRegulated},
		{"customer", SecretRegulated},
		{"totally-new-label", Secret}, // unrecognized → fail closed
	}
	for _, c := range cases {
		if got := ClassifyPrefix(c.label); got != c.want {
			t.Errorf("ClassifyPrefix(%q) = %q, want %q", c.label, got, c.want)
		}
	}
}

func TestCacheableIsZeroValue(t *testing.T) {
	// Cacheable is the zero value so a default-constructed PrefixStats starts
	// cacheable and must OPT INTO a restriction, rather than starting forbidden.
	var s PrefixStats
	if s.Secret != Cacheable {
		t.Fatalf("zero-value Secret = %q, want Cacheable (\"\")", s.Secret)
	}
}
