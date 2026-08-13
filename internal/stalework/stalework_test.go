package stalework

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPacketDeterministicBoundedAndIntegrated(t *testing.T) {
	d := t.TempDir()
	write := func(p, s string) {
		t.Helper()
		q := filepath.Join(d, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(q), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte(s), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/a.md", "# Operate\nUse `fak serve`; currently Go 1.26 is required.\n"+strings.Repeat("x", 500))
	write("docs/b.md", "# Operate too\nUse internal/gateway for requests.")
	write("docs/notes/old.md", "# Historical study\nOld but valid.")
	write("docs/generated.md", "<!-- Code generated; DO NOT EDIT. -->")
	write("docs/quiet.md", "No dependency and no version assertion.")
	r := fixtureRunner()
	o := Options{Root: d, Limit: 10, Now: time.Unix(1800000000, 0), Run: r, OpenDedupe: map[string]bool{"stale-work:docs/b.md": true}}
	p1, e := Scan(context.Background(), o)
	if e != nil {
		t.Fatal(e)
	}
	p2, e := Scan(context.Background(), o)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(p1, p2) {
		t.Fatal("repeated scan differed")
	}
	if len(p1.Candidates) != 1 || p1.Candidates[0].Path != "docs/a.md" {
		t.Fatalf("candidates: %#v", p1.Candidates)
	}
	if p1.Candidates[0].Score < 70 || len(p1.Candidates[0].Excerpt) > maxExcerpt {
		t.Fatalf("unbounded/weak candidate: %#v", p1.Candidates[0])
	}
	if p1.Candidates[0].Batch != "cmd/fak" {
		t.Fatalf("batch=%q", p1.Candidates[0].Batch)
	}
	if !hasExemption(p1, "historical note") || !hasExemption(p1, "generated or third-party artifact") || !hasExemption(p1, "already-open candidate") {
		t.Fatalf("exemptions: %#v", p1.Exemptions)
	}
	if len(p1.Abstentions) != 1 || p1.Abstentions[0].Path != "docs/quiet.md" {
		t.Fatalf("abstentions: %#v", p1.Abstentions)
	}
}
func TestAgeAloneAbstains(t *testing.T) {
	d := t.TempDir()
	_ = os.WriteFile(filepath.Join(d, "old.md"), []byte("ordinary old prose"), 0644)
	p, e := Scan(context.Background(), Options{Root: d, Paths: []string{"old.md"}, Now: time.Unix(1900000000, 0), Run: fixtureRunner()})
	if e != nil {
		t.Fatal(e)
	}
	if len(p.Candidates) != 0 || len(p.Abstentions) != 1 {
		t.Fatalf("age became verdict: %#v", p)
	}
}
func fixtureRunner() Runner {
	return func(_ context.Context, _ string, a ...string) ([]byte, error) {
		if a[0] == "rev-parse" {
			return []byte("head\n"), nil
		}
		if a[0] == "ls-files" {
			return []byte("docs/quiet.md\ndocs/generated.md\ndocs/b.md\ndocs/a.md\ndocs/notes/old.md\n"), nil
		}
		if a[0] == "log" && a[1] == "-1" {
			return []byte("base|1609459200\n"), nil
		}
		if a[0] == "log" && a[1] == "--format=%H|%s" {
			return []byte("new|semantic dependency change\n"), nil
		}
		return nil, nil
	}
}
func hasExemption(p Packet, s string) bool {
	for _, x := range p.Exemptions {
		if x.Exemption == s {
			return true
		}
	}
	return false
}
