package alphafence

import (
	"fmt"
	"strings"
	"testing"
)

const (
	secretA = "Always rank candidate trades by realized slippage before sizing."
	secretB = "Escalate to the overnight desk only when the spread inverts twice."
	shared  = "Licensed under the Apache License, Version 2.0 (the License)."
)

func mustMatcher(t *testing.T, f Fence) *Matcher {
	t.Helper()
	raw, err := f.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMatcher(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestFenceMatchesReflowedPrivateLinesAndNothingElse(t *testing.T) {
	f, err := Build(Options{}, []Source{{Label: "alpha-1", Body: []byte(secretA + "\n" + secretB + "\n")}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := mustMatcher(t, f)
	if label, ok := m.Match("   " + strings.ToUpper(secretA) + "  "); !ok || label != "alpha-1" {
		t.Fatalf("re-cased, re-indented private line not matched: %q %v", label, ok)
	}
	if _, ok := m.Match(strings.ReplaceAll(secretB, " ", "\t  ")); !ok {
		t.Fatal("whitespace-reflowed private line not matched")
	}
	if _, ok := m.Match("An unrelated sentence of comparable length about nothing."); ok {
		t.Fatal("unrelated line matched")
	}
}

func TestFenceFileCarriesNoPlaintext(t *testing.T) {
	f, err := Build(Options{}, []Source{{Label: "alpha-1", Body: []byte(secretA + "\n")}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := f.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	for _, word := range strings.Fields(strings.ToLower(secretA)) {
		if len(word) > 5 && strings.Contains(strings.ToLower(string(raw)), word) {
			t.Fatalf("fence file leaks private word %q:\n%s", word, raw)
		}
	}
}

func TestBuildDropsShortAndAlreadyPublicLines(t *testing.T) {
	body := []byte("}\nshort line\n" + strings.Repeat("-", 60) + "\n" + shared + "\n" + secretA + "\n" + secretA + "\n")
	f, err := Build(Options{}, []Source{{Label: "alpha-1", Body: body}}, [][]byte{[]byte("# header\n  " + shared + "\n")})
	if err != nil {
		t.Fatal(err)
	}
	if f.Lines() != 1 {
		t.Fatalf("fenced %d lines, want only the one distinctive private line", f.Lines())
	}
	m := mustMatcher(t, f)
	if _, ok := m.Match(shared); ok {
		t.Fatal("line already present in public content was fenced")
	}
}

func TestSaltChangesHashes(t *testing.T) {
	src := []Source{{Label: "a", Body: []byte(secretA + "\n")}}
	one, _ := Build(Options{Salt: []byte("salt-one")}, src, nil)
	two, _ := Build(Options{Salt: []byte("salt-two")}, src, nil)
	again, _ := Build(Options{Salt: []byte("salt-one")}, src, nil)
	if one.Entries[0].Hashes[0] == two.Entries[0].Hashes[0] {
		t.Fatal("different salts produced the same hash")
	}
	if one.Entries[0].Hashes[0] != again.Entries[0].Hashes[0] {
		t.Fatal("same salt did not reproduce the hash")
	}
}

func TestCollectorAppliesMinHitsPerUnit(t *testing.T) {
	f, err := Build(Options{MinHits: 2}, []Source{{Label: "a", Body: []byte(secretA + "\n" + secretB + "\n")}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := mustMatcher(t, f)
	c := m.NewCollector()
	c.Observe(3, secretA)
	c.Observe(9, secretA)
	if hits := c.Hits(); hits != nil {
		t.Fatalf("one distinct fenced line repeated met min_hits=2: %+v", hits)
	}
	c.Observe(12, secretB)
	hits := c.Hits()
	if len(hits) != 3 || hits[2].Line != 12 || hits[0].Label != "a" {
		t.Fatalf("unexpected hits: %+v", hits)
	}
	var none *Matcher
	idle := none.NewCollector()
	idle.Observe(1, secretA)
	if idle.Hits() != nil {
		t.Fatal("nil matcher reported a hit")
	}
}

func TestParseRejectsDefectiveFences(t *testing.T) {
	for name, raw := range map[string]string{
		"not json":     "{",
		"wrong schema": `{"schema":"other/1","salt":"aa","min_line_len":24,"min_hits":1}`,
		"empty salt":   `{"schema":"` + Schema + `","salt":"","min_line_len":24,"min_hits":1}`,
		"zero bounds":  `{"schema":"` + Schema + `","salt":"aa","min_line_len":0,"min_hits":1}`,
		"bad hash":     `{"schema":"` + Schema + `","salt":"aa","min_line_len":24,"min_hits":1,"entries":[{"label":"a","hashes":["zz"]}]}`,
	} {
		if _, err := Parse([]byte(raw)); err == nil {
			t.Errorf("%s: defective fence accepted", name)
		}
	}
}

func BenchmarkScan(b *testing.B) {
	var body strings.Builder
	for i := range 20000 {
		fmt.Fprintf(&body, "private workflow instruction number %d stays on this machine\n", i)
	}
	f, err := Build(Options{}, []Source{{Label: "a", Body: []byte(body.String())}}, nil)
	if err != nil {
		b.Fatal(err)
	}
	m, err := NewMatcher(f)
	if err != nil {
		b.Fatal(err)
	}
	line := "an ordinary outbound source line that is not fenced at all"
	b.ReportAllocs()
	for b.Loop() {
		m.Match(line)
	}
}
