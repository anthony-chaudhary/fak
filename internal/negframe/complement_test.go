package negframe

import (
	"reflect"
	"strings"
	"testing"
)

func TestResolveExact(t *testing.T) {
	cases := []struct {
		negated, domain string
		wantPositive    string
	}{
		{"shared", "lock-mode", "exclusive"},
		{"exclusive", "lock-mode", "shared"},
		{"true", "boolean", "false"},
		{"false", "boolean", "true"},
		{"refused", "lease-outcome", "granted"},
		{"negative", "polarity", "positive"},
		// case-insensitive membership, canonicalized to the registry spelling
		{"SHARED", "lock-mode", "exclusive"},
		{"Shared", "LOCK-MODE", "exclusive"},
	}
	for _, c := range cases {
		got := Resolve(c.negated, c.domain)
		if got.Kind != Exact {
			t.Errorf("Resolve(%q,%q) kind = %q, want exact (%+v)", c.negated, c.domain, got.Kind, got)
			continue
		}
		if got.Positive != c.wantPositive {
			t.Errorf("Resolve(%q,%q) positive = %q, want %q", c.negated, c.domain, got.Positive, c.wantPositive)
		}
		if !got.Resolved() {
			t.Errorf("Resolve(%q,%q).Resolved() = false, want true", c.negated, c.domain)
		}
		if len(got.Members) != 1 || got.Members[0] != c.wantPositive {
			t.Errorf("Resolve(%q,%q) members = %v, want [%q]", c.negated, c.domain, got.Members, c.wantPositive)
		}
	}
}

func TestResolveCandidates(t *testing.T) {
	got := Resolve("global", "lane-kind")
	if got.Kind != Candidates {
		t.Fatalf("Resolve(global,lane-kind) kind = %q, want candidates (%+v)", got.Kind, got)
	}
	want := []string{"cluster", "keyword"}
	if !reflect.DeepEqual(got.Members, want) {
		t.Errorf("members = %v, want %v", got.Members, want)
	}
	if got.Positive != "" {
		t.Errorf("candidate resolution should carry no single Positive, got %q", got.Positive)
	}
	if !got.Resolved() {
		t.Error("Resolved() = false for a candidate set, want true")
	}
}

func TestResolveInfersDomain(t *testing.T) {
	// A blank domain infers it from the term via DomainOf.
	got := Resolve("shared", "")
	if got.Kind != Exact || got.Positive != "exclusive" {
		t.Fatalf("Resolve(shared,\"\") = %+v, want exact exclusive", got)
	}
	if got.Domain != "lock-mode" {
		t.Errorf("inferred domain = %q, want lock-mode", got.Domain)
	}
}

func TestResolveUnknownFailsClosed(t *testing.T) {
	cases := []struct {
		name, negated, domain string
	}{
		{"unknown domain", "shared", "no-such-domain"},
		{"member not in named domain", "banana", "lock-mode"},
		{"unresolvable inference", "banana", ""},
		{"empty term", "", "lock-mode"},
	}
	for _, c := range cases {
		got := Resolve(c.negated, c.domain)
		if got.Kind != Unknown {
			t.Errorf("%s: Resolve(%q,%q) kind = %q, want unknown", c.name, c.negated, c.domain, got.Kind)
		}
		if got.Resolved() {
			t.Errorf("%s: Resolved() = true, want false (fail-closed)", c.name)
		}
		if got.Reason == "" {
			t.Errorf("%s: Unknown resolution carries no reason", c.name)
		}
		// Fail-closed means NO fabricated positive.
		if got.Positive != "" || len(got.Members) != 0 {
			t.Errorf("%s: Unknown resolution fabricated a positive: %+v", c.name, got)
		}
	}
}

func TestStripNegation(t *testing.T) {
	cases := []struct {
		in       string
		wantTerm string
		wantOK   bool
	}{
		{"not shared", "shared", true},
		{"not-shared", "shared", true},
		{"non-shared", "shared", true},
		{"NOT Shared", "Shared", true},
		{"¬shared", "shared", true},
		{"shared", "shared", false}, // no marker
		{"  not   global ", "global", true},
	}
	for _, c := range cases {
		term, ok := StripNegation(c.in)
		if term != c.wantTerm || ok != c.wantOK {
			t.Errorf("StripNegation(%q) = (%q,%v), want (%q,%v)", c.in, term, ok, c.wantTerm, c.wantOK)
		}
	}
}

func TestDomainOf(t *testing.T) {
	d, ok := DomainOf("shared")
	if !ok || d.Name != "lock-mode" {
		t.Errorf("DomainOf(shared) = (%q,%v), want (lock-mode,true)", d.Name, ok)
	}
	if _, ok := DomainOf("banana"); ok {
		t.Error("DomainOf(banana) resolved a domain, want not-found")
	}
}

// TestRegistryMembersDisjoint enforces the invariant DomainOf relies on: no member appears in two
// domains, or inference would be ambiguous. If a future domain reuses a member, this fails loudly.
func TestRegistryMembersDisjoint(t *testing.T) {
	seen := map[string]string{} // member -> first domain
	for _, d := range Domains() {
		for _, m := range d.Members {
			key := m
			if prev, dup := seen[key]; dup {
				t.Errorf("member %q appears in both %q and %q; DomainOf inference would be ambiguous", m, prev, d.Name)
			}
			seen[key] = d.Name
		}
	}
}

// TestResolveIdempotentOnExact checks the round-trip: resolving the negation of a resolved
// positive returns to the original member (both directions of a 2-member domain).
func TestResolveIdempotentOnExact(t *testing.T) {
	first := Resolve("shared", "lock-mode")
	if first.Kind != Exact {
		t.Fatalf("setup: want exact, got %+v", first)
	}
	back := Resolve(first.Positive, "lock-mode")
	if back.Kind != Exact || back.Positive != "shared" {
		t.Errorf("round-trip: Resolve(%q) = %+v, want exact shared", first.Positive, back)
	}
}

func TestComplementRoutesExactBoundedOpen(t *testing.T) {
	tests := []struct {
		name, original, term, domain string
		class                        ComplementClass
		want                         string
	}{
		{"exact", "not shared", "shared", "lock-mode", ComplementExact, "exclusive"},
		{"bounded", "not global", "global", "lane-kind", ComplementBounded, "one of cluster, keyword (constraint: not global)"},
		{"open", "not blue", "blue", "color-space", ComplementOpen, "not blue"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Complement(tc.original, tc.term, tc.domain)
			if got.Class != tc.class || got.Text != tc.want {
				t.Fatalf("Complement() = %+v, want class=%s text=%q", got, tc.class, tc.want)
			}
		})
	}
}

func TestComplementBoundedPreservesMustKeepToken(t *testing.T) {
	got := Complement("not global under `OFF_TRUNK`", "global", "lane-kind")
	if got.Class != ComplementBounded || !strings.Contains(got.Text, "`OFF_TRUNK`") {
		t.Fatalf("bounded route dropped token: %+v", got)
	}
}

func TestReframeResultCarriesComplementTelemetry(t *testing.T) {
	r := ReframePass("Do not forget to test.")
	r.ComplementClasses = append(r.ComplementClasses, Complement("not shared", "shared", "lock-mode").Class)
	if !reflect.DeepEqual(r.ComplementClasses, []ComplementClass{ComplementExact}) {
		t.Fatalf("complement telemetry = %v", r.ComplementClasses)
	}
}
