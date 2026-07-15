package reachdelta

import (
	"reflect"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

func TestDeltaClassifiesPolicyWidening(t *testing.T) {
	rule := adjudicator.ArgPredicate{Tool: "shell", Arg: "command", Kind: adjudicator.ArgDenyRegex, Re: regexp.MustCompile(`rm`), Reason: abi.ReasonPolicyBlock}
	base := adjudicator.Policy{
		Deny:            map[string]abi.ReasonCode{"delete": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{".git/"}, ArgPredicates: []adjudicator.ArgPredicate{rule},
		RedactFields: []string{"token"}, EgressExtraDenyHosts: []string{"blocked.example"},
		AdvisoryReasons: map[abi.ReasonCode]bool{}, SecretPosture: adjudicator.SecretFailClosed,
		SecretPatterns: []*regexp.Regexp{regexp.MustCompile(`SECRET`)}, LintWrites: true,
	}
	proposed := adjudicator.Policy{
		Allow: map[string]bool{"write_file": true}, AllowPrefix: []string{"unsafe_"},
		ResearchEgressAllowHosts: []string{"new.example"}, Posture: adjudicator.PostureAdmitAndLog,
		Complain: map[string]bool{"trial_tool": true}, AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonPolicyBlock: true},
		SecretPosture: adjudicator.SecretAdmitAndLog,
	}
	want := []Finding{
		{AdvisoryModeAdded, "POLICY_BLOCK"},
		{ArgumentProtectionLost, argRuleIDs(base)[0]},
		{ComplainModeAdded, "trial_tool"},
		{DefaultPostureWidened, "default"},
		{EgressDenyRemoved, "blocked.example"},
		{ExplicitDenyRemoved, "delete"},
		{NewEgressHostPermitted, "new.example"},
		{NewToolPermitted, "write_file"},
		{NewToolPrefixPermitted, "unsafe_"},
		{RedactionProtectionLost, "token"},
		{SecretPostureWidened, "admit_and_log"},
		{SecretProtectionLost, "SECRET"},
		{SelfModifyProtectionLost, ".git/"},
		{WriteLintDisabled, "lint_writes"},
	}
	if got := Delta(base, proposed); !reflect.DeepEqual(got, want) {
		t.Fatalf("Delta() = %#v\nwant %#v", got, want)
	}
}

func TestDeltaTreatsChangedExplicitDenyAsRemovedProtection(t *testing.T) {
	base := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"delete": abi.ReasonPolicyBlock}}
	proposed := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"delete": abi.ReasonDefaultDeny}}
	want := []Finding{{ExplicitDenyRemoved, "delete"}}
	if got := Delta(base, proposed); !reflect.DeepEqual(got, want) {
		t.Fatalf("Delta() = %#v, want %#v", got, want)
	}
}

func TestDeltaIgnoresRestrictiveAndAlreadyCoveredChanges(t *testing.T) {
	rule := adjudicator.ArgPredicate{Tool: "shell", Arg: "command", Kind: adjudicator.ArgDenyRegex, Re: regexp.MustCompile(`rm`), Reason: abi.ReasonPolicyBlock}
	base := adjudicator.Policy{
		Allow: map[string]bool{"old": true}, AllowPrefix: []string{"read_"},
		Complain: map[string]bool{"search_docs": true}, ResearchEgressAllowHosts: []string{"example.com"},
	}
	proposed := adjudicator.Policy{
		Allow: map[string]bool{"search_docs": true, "read_file": true}, AllowPrefix: []string{"read_safe_"},
		Deny:            map[string]abi.ReasonCode{"delete": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{".git/"}, ArgPredicates: []adjudicator.ArgPredicate{rule},
		RedactFields: []string{"token"}, EgressExtraDenyHosts: []string{"blocked.example"},
		ResearchEgressAllowHosts: []string{"api.example.com"},
	}
	if got := Delta(base, proposed); len(got) != 0 {
		t.Fatalf("restrictive/covered delta = %#v, want empty", got)
	}
}

func TestDeltaSortsMapFindingsDeterministically(t *testing.T) {
	base := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"z": abi.ReasonPolicyBlock, "a": abi.ReasonPolicyBlock}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"z": true, "a": true}}
	want := []Finding{{ExplicitDenyRemoved, "a"}, {ExplicitDenyRemoved, "z"}, {NewToolPermitted, "a"}, {NewToolPermitted, "z"}}
	if got := Delta(base, proposed); !reflect.DeepEqual(got, want) {
		t.Fatalf("Delta() = %#v, want %#v", got, want)
	}
}

func TestSuppressAcceptedUsesLatestLiveExactAcceptedRisk(t *testing.T) {
	f := Finding{NewToolPermitted, "danger"}
	open := knownbad.Record{Signature: f.Signature(), ReasonClass: "accepted-risk", Status: knownbad.StatusOpen, DiscoveredAtUnix: 100, TTLSeconds: 100}
	cases := []struct {
		name    string
		records []knownbad.Record
		now     int64
		want    int
	}{
		{"live", []knownbad.Record{open}, 150, 0},
		{"wrong class", []knownbad.Record{{Signature: f.Signature(), ReasonClass: "bug", Status: knownbad.StatusOpen, DiscoveredAtUnix: 100, TTLSeconds: 100}}, 150, 1},
		{"expired", []knownbad.Record{open}, 201, 1},
		{"resolved latest", []knownbad.Record{open, {Signature: f.Signature(), ReasonClass: "accepted-risk", Status: knownbad.StatusResolved, DiscoveredAtUnix: 160, TTLSeconds: 100}}, 170, 1},
		{"revoked latest", []knownbad.Record{open, {Signature: f.Signature(), ReasonClass: "accepted-risk", Status: knownbad.StatusRevoked, DiscoveredAtUnix: 160, TTLSeconds: 100}}, 170, 1},
		{"unrelated signature", []knownbad.Record{{Signature: "other", ReasonClass: "accepted-risk", Status: knownbad.StatusOpen, DiscoveredAtUnix: 100, TTLSeconds: 100}}, 150, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := len(SuppressAccepted([]Finding{f}, tc.records, tc.now)); got != tc.want {
				t.Fatalf("len = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestRatifyPromotionIsDefaultOffAndDoesNotMutateBase(t *testing.T) {
	base := adjudicator.Policy{Complain: map[string]bool{"search_docs": true}}
	offer := adjudicator.PromotionOffer{Tool: "search_docs", CleanEvents: 10}
	proposed, disabled := RatifyPromotion(false, base, offer, nil, 0)
	if disabled.Approved || disabled.Reason != "disabled" {
		t.Fatalf("disabled verdict = %#v", disabled)
	}
	proposed, enabled := RatifyPromotion(true, base, offer, nil, 0)
	if !enabled.Approved || enabled.Reason != "empty_delta" || len(enabled.Review) != 0 {
		t.Fatalf("enabled verdict = %#v", enabled)
	}
	if base.Allow != nil || !base.Complain["search_docs"] {
		t.Fatal("RatifyPromotion mutated base")
	}
	if !proposed.Allow["search_docs"] || proposed.Complain["search_docs"] {
		t.Fatalf("proposed policy = %#v", proposed)
	}
}

func TestRatifyRequiresReviewForNewReachUnlessAccepted(t *testing.T) {
	proposed := adjudicator.Policy{Allow: map[string]bool{"new_tool": true}}
	verdict := Ratify(true, adjudicator.Policy{}, proposed, nil, 150)
	if verdict.Approved || verdict.Reason != "reach_expansion" || len(verdict.Review) != 1 {
		t.Fatalf("verdict = %#v", verdict)
	}
	accepted := []knownbad.Record{{Signature: verdict.Review[0].Signature(), ReasonClass: "reach-delta-accepted-risk", Status: knownbad.StatusOpen, DiscoveredAtUnix: 100, TTLSeconds: 100}}
	verdict = Ratify(true, adjudicator.Policy{}, proposed, accepted, 150)
	if !verdict.Approved || verdict.Reason != "empty_delta" {
		t.Fatalf("accepted verdict = %#v", verdict)
	}
}
