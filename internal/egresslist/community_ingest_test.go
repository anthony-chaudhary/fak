package egresslist

import "testing"

// community_ingest_test.go — the acceptance witness for issue #4977: a POPULAR community
// filter list is ingested, ships with provenance + a pinned checksum, resolves by the name
// an operator writes in `egress.block_lists`, and actually refuses the hosts it carries.
//
// WHY THE WITNESS STOPS AT List.Decide AND NOT AT THE ADJUDICATOR. #4977's acceptance also
// asks for the refusal to be proven "end-to-end through the adjudicator". That rung is the
// egress LIST layer in internal/adjudicator/egresslist.go, which imports THIS package — so
// the Policy-level assertion cannot live here without an import cycle, and belongs in
// internal/adjudicator/decide_egresslist_test.go.
//
// THAT HALF IS WRITTEN BUT NOT YET COMMITTED. It is
// TestEgressListBundledCommunityListRefusesItsHosts in
// internal/adjudicator/decide_egresslist_test.go, which drives the same derived probe hosts
// through Adjudicate and asserts EGRESS_BLOCK on the WebFetch and the Bash/curl path alike.
// It is named here rather than left implied so a reviewer can check the claim instead of
// taking it. internal/adjudicator is a hard-self CORE-LOCK surface: it is the
// reference monitor, so `fak commit` refuses to stage an agent's edit to it with
// CORE_SELF_MODIFY unless an INDEPENDENT witness resolver clears the maintenance claim
// (AGENTS.md: "do not clear this by self-report"). A self-authored witness resolves abstain,
// which is the guard working as designed — an agent may not certify its own edit to the
// thing that judges it. The end-to-end rung therefore needs an operator/release maintenance
// path; until then the acceptance is witnessed at the layer below.
//
// What this file owns is every link BELOW that rung — registry name -> BundledList ->
// AddFilterText -> Decide -> Block — which is exactly the call sequence compileEgressList
// makes, so the only unwitnessed step is the adjudicator's own already-tested wiring
// (TestEgressListBundledList covers that wiring against the seed list).

// communityListName is the ingested upstream feed under test. The other bundled list
// (sample-malware) is a hand-authored placeholder whose hosts were written to be blocked;
// asserting against IT proves only that a fixture agrees with itself. This one carries
// thousands of rules nobody in this repo chose, which is the only way to witness that
// ingesting a community list changes what an agent may actually reach.
const communityListName = "stevenblack-curated"

// communityProbeHosts draws probe hosts from the bundled artifact ITSELF rather than
// hardcoding a domain someone read off the upstream once.
//
// WHY DERIVED AND NOT LITERAL. A community feed churns: a host on it today may be gone
// after the next `fak egresslist refresh`, and a test pinned to a literal domain would then
// red the trunk for a routine, correct refresh — training the next maintainer to edit the
// assertion instead of read it. Deriving keeps the claim ("a host on the ingested list is
// refused") true across every refresh while staying a real assertion: nothing here knows
// which hosts are on the list, it asks the list. Rules() is domain-sorted, so first/middle/
// last is a deterministic spread rather than a sample.
func communityProbeHosts(t *testing.T) []string {
	t.Helper()
	text, ok := BundledList(communityListName)
	if !ok {
		t.Fatalf("bundled list %q does not ship", communityListName)
	}
	var blocked []string
	for _, r := range NewBuilder().AddFilterText(communityListName, text).Build().Rules() {
		if r.Kind == Block {
			blocked = append(blocked, r.Domain)
		}
	}
	if len(blocked) < 100 {
		t.Fatalf("bundled list %q compiled to %d block rules, want a real feed's worth: a nearly "+
			"empty list would make every assertion below vacuous", communityListName, len(blocked))
	}
	return []string{blocked[0], blocked[len(blocked)/2], blocked[len(blocked)-1]}
}

// TestCommunityListIsSelectableAndRefusesItsHosts is the #4977 acceptance test: the
// ingested list resolves by the name an operator writes in `egress.block_lists`, and hosts
// it carries Decide to Block — including their subdomains, and without swallowing hosts it
// does not carry.
func TestCommunityListIsSelectableAndRefusesItsHosts(t *testing.T) {
	// Selectable by name: this is the exact lookup internal/policy performs for each
	// `egress.block_lists` entry, and the reason an unknown name can fail LOUD.
	names := BundledListNames()
	found := false
	for _, n := range names {
		if n == communityListName {
			found = true
		}
	}
	if !found {
		t.Fatalf("BundledListNames() = %v, want it to include the ingested %q", names, communityListName)
	}

	text, _ := BundledList(communityListName)
	l := NewBuilder().AddFilterText(communityListName, text).Build()

	for _, host := range communityProbeHosts(t) {
		d := l.Decide(host)
		if d.Kind != Block {
			t.Errorf("Decide(%s) = %v, want Block (host is on the ingested list)", host, d.Kind)
		}
		if d.Source != communityListName {
			t.Errorf("Decide(%s) Source = %q, want %q: a refusal must name which list spoke",
				host, d.Source, communityListName)
		}
		// A block rule covers the subtree: a tracker that moves to a new subdomain of a
		// listed host does not escape the list.
		if d := l.Decide("cdn." + host); d.Kind != Block {
			t.Errorf("Decide(cdn.%s) = %v, want Block (subdomain of a listed host)", host, d.Kind)
		}
	}

	// The converse half: compiling a multi-thousand-rule feed must not turn the egress
	// posture into a deny-all. A host on no rule stays silent (None), so the caller falls
	// through to the next layer instead of being refused.
	if d := l.Decide("not-on-any-community-list.example"); d.Kind != None {
		t.Errorf("Decide(unlisted host) = %v, want None: a block list must not become an allowlist", d.Kind)
	}
}

// TestCommunityListIsNormalizedOnIngest proves the checked-in artifact is the NORMALIZED
// form the issue asks for, not a copy of the messy upstream: every rule is lower-case, is a
// plausible host, and carries none of the hosts-file bookkeeping (localhost/broadcast) or
// unmodeled adblock syntax that the raw upstream is full of. This is what makes the
// artifact reviewable in a diff and deterministic across refreshes.
func TestCommunityListIsNormalizedOnIngest(t *testing.T) {
	text, ok := BundledList(communityListName)
	if !ok {
		t.Fatalf("bundled list %q does not ship", communityListName)
	}
	rules := NewBuilder().AddFilterText(communityListName, text).Build().Rules()
	if len(rules) == 0 {
		t.Fatal("ingested list compiled to zero rules")
	}
	for _, r := range rules {
		if r.Domain != normalizeHost(r.Domain) {
			t.Errorf("rule %q is not normalized (want %q)", r.Domain, normalizeHost(r.Domain))
		}
		if !isPlausibleHost(r.Domain) {
			t.Errorf("rule %q is not a plausible host: non-host junk survived ingest", r.Domain)
		}
		if isNoiseHost(r.Domain) {
			t.Errorf("rule %q is hosts-file bookkeeping and must never become a block rule", r.Domain)
		}
	}
}
