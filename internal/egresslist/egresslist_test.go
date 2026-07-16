package egresslist

import "testing"

// TestDecideBlockAndSubdomain locks the core block behavior: a bare-domain block rule
// refuses the domain itself AND every subdomain (the subdomain-suffix convention the
// research allowlist already uses), while an unrelated host is silent (None).
func TestDecideBlockAndSubdomain(t *testing.T) {
	l := NewBuilder().AddRules("op", []string{"malware.example"}, Block).Build()

	for _, host := range []string{"malware.example", "a.malware.example", "x.y.malware.example"} {
		d := l.Decide(host)
		if d.Kind != Block {
			t.Errorf("Decide(%q) = %v, want Block", host, d.Kind)
		}
		if d.Rule != "malware.example" || d.Source != "op" {
			t.Errorf("Decide(%q) rule/source = %q/%q, want malware.example/op", host, d.Rule, d.Source)
		}
	}
	// A neighbor that merely ends in the same labels but is NOT a subdomain must not match.
	if d := l.Decide("notmalware.example"); d.Kind != None {
		t.Errorf("Decide(notmalware.example) = %v, want None (suffix must be label-aligned)", d.Kind)
	}
	if d := l.Decide("example"); d.Kind != None {
		t.Errorf("Decide(example) = %v, want None (a parent of the rule is not blocked)", d.Kind)
	}
}

// TestAllowWinsOverBlock is the adblock `@@` precedence pin: a host on BOTH a block list
// and the allow list is Allowed. This is what lets an operator subscribe to a broad
// community block list yet keep a sanctioned docs host reachable.
func TestAllowWinsOverBlock(t *testing.T) {
	l := NewBuilder().
		AddRules("community", []string{"example.com"}, Block).
		AddRules("operator-allow", []string{"docs.example.com"}, Allow).
		Build()

	if d := l.Decide("docs.example.com"); d.Kind != Allow || d.Source != "operator-allow" {
		t.Fatalf("Decide(docs.example.com) = %v/%s, want Allow/operator-allow", d.Kind, d.Source)
	}
	// A sibling still under the block rule (no allow carve-out) stays blocked.
	if d := l.Decide("ads.example.com"); d.Kind != Block {
		t.Fatalf("Decide(ads.example.com) = %v, want Block", d.Kind)
	}
}

// TestDecideNormalizesHost proves a raw authority (mixed case, port, brackets, trailing
// root dot) resolves the same as its bare host, so callers can pass what they extracted
// from a URL without pre-cleaning it.
func TestDecideNormalizesHost(t *testing.T) {
	l := NewBuilder().AddRules("op", []string{"blocked.example"}, Block).Build()
	for _, host := range []string{"BLOCKED.example", "blocked.example:443", "blocked.example.", " blocked.example "} {
		if d := l.Decide(host); d.Kind != Block {
			t.Errorf("Decide(%q) = %v, want Block after normalization", host, d.Kind)
		}
	}
}

// TestNilAndEmptyListDecideNone pins the zero-cost absent case: a nil *List and an empty
// list Decide None for every host, so a policy with no egress list never blocks and never
// panics.
func TestNilAndEmptyListDecideNone(t *testing.T) {
	var nilList *List
	if d := nilList.Decide("anything.example"); d.Kind != None {
		t.Fatalf("nil List Decide = %v, want None", d.Kind)
	}
	if b, a := nilList.Counts(); b != 0 || a != 0 {
		t.Fatalf("nil List Counts = %d/%d, want 0/0", b, a)
	}
	empty := NewBuilder().Build()
	if !empty.Empty() {
		t.Fatalf("fresh builder List should be Empty()")
	}
	if d := empty.Decide("anything.example"); d.Kind != None {
		t.Fatalf("empty List Decide = %v, want None", d.Kind)
	}
}

// TestAddFilterTextHostsFile pins the hosts-file grammar: the sink IP is dropped, each
// host becomes a block rule, and localhost/broadcast bookkeeping is ignored.
func TestAddFilterTextHostsFile(t *testing.T) {
	const hosts = `
# a hosts-style block list
127.0.0.1 localhost
0.0.0.0 ads.tracker.example
0.0.0.0 telemetry.example.net # inline comment
bare.example.org
255.255.255.255 broadcasthost
`
	l := NewBuilder().AddFilterText("hostsfile", hosts).Build()

	for _, host := range []string{"ads.tracker.example", "telemetry.example.net", "bare.example.org"} {
		if d := l.Decide(host); d.Kind != Block || d.Source != "hostsfile" {
			t.Errorf("Decide(%q) = %v/%s, want Block/hostsfile", host, d.Kind, d.Source)
		}
	}
	// localhost and broadcasthost must NOT have become block rules.
	if d := l.Decide("localhost"); d.Kind != None {
		t.Errorf("Decide(localhost) = %v, want None (bookkeeping host must not block)", d.Kind)
	}
	if b, _ := l.Counts(); b != 3 {
		t.Errorf("compiled %d block rules, want 3 (sink IP + noise hosts dropped)", b)
	}
}

// TestAddFilterTextAdblock pins the adblock domain-anchor grammar: "||host^" blocks,
// "@@||host^" is an allow exception, and rules this leaf does not model (options,
// element-hiding, wildcards, paths, comments) are skipped rather than approximated.
func TestAddFilterTextAdblock(t *testing.T) {
	const adblock = `
! adblock-style list
||ads.example.com^
||track.example.com^$third-party
@@||docs.example.com^
example.com##.banner
||*.wildcard.example^
||host.example.com/path
/regex-rule/
`
	l := NewBuilder().AddFilterText("adblock", adblock).Build()

	if d := l.Decide("ads.example.com"); d.Kind != Block {
		t.Errorf("Decide(ads.example.com) = %v, want Block", d.Kind)
	}
	if d := l.Decide("docs.example.com"); d.Kind != Allow {
		t.Errorf("Decide(docs.example.com) = %v, want Allow (@@ exception)", d.Kind)
	}
	// The option-bearing, wildcard, path, element-hiding, and regex lines are NOT rules.
	block, allow := l.Counts()
	if block != 1 || allow != 1 {
		t.Fatalf("compiled %d block / %d allow rules, want 1/1 (unmodeled lines skipped)", block, allow)
	}
	// Specifically: the "$third-party" option line must not have blocked track.example.com.
	if d := l.Decide("track.example.com"); d.Kind != None {
		t.Errorf("Decide(track.example.com) = %v, want None (option-bearing rule not modeled)", d.Kind)
	}
}

// TestBundledListRegistry proves the block_lists reference resolves to a shipped list and
// that an unknown / path-shaped name fails closed (ok == false) so the policy loader can
// reject it loudly.
func TestBundledListRegistry(t *testing.T) {
	names := BundledListNames()
	found := false
	for _, n := range names {
		if n == "sample-malware" {
			found = true
		}
	}
	if !found {
		t.Fatalf("BundledListNames() = %v, want it to include sample-malware", names)
	}
	text, ok := BundledList("sample-malware")
	if !ok || text == "" {
		t.Fatalf("BundledList(sample-malware) ok=%v len=%d, want ok/non-empty", ok, len(text))
	}
	// The sample list must actually compile to real block rules.
	l := NewBuilder().AddFilterText("sample-malware", text).Build()
	if d := l.Decide("malware.example"); d.Kind != Block {
		t.Errorf("sample list Decide(malware.example) = %v, want Block", d.Kind)
	}
	if d := l.Decide("ads.example.net"); d.Kind != Block {
		t.Errorf("sample list Decide(ads.example.net) = %v, want Block", d.Kind)
	}
	for _, bad := range []string{"unknown-list", "../etc/passwd", "lists/sample-malware", "sample.malware"} {
		if _, ok := BundledList(bad); ok {
			t.Errorf("BundledList(%q) ok=true, want false (unknown/path-shaped must fail closed)", bad)
		}
	}
}
