package witness

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// baseSpec is a well-formed RSL spec: ref main advances to target sha t1,
// authorized by key k1 under the current pinned policy (seq 7).
func baseSpec() RSLSpec {
	return RSLSpec{
		Ref:    "refs/heads/main",
		Target: "t1",
		Policy: RSLPolicy{Seq: 7, AuthorizedKey: []string{"k1", "k2"}},
		Log: []RSLEntry{
			{Ref: "refs/heads/main", Target: "t1", SignerKeyID: "k1", PolicySeq: 7, Signature: "sig"},
		},
	}
}

func TestRSLAuthorized(t *testing.T) {
	res := NewRSLVerifier().Verify(context.Background(), baseSpec())
	if res.Verdict != RSLAuthorized {
		t.Fatalf("matching, admitted, fresh entry => AUTHORIZED, got %v (%s)", res.Verdict, res.Reason)
	}
	if res.MatchedSigner != "k1" {
		t.Fatalf("AUTHORIZED must name the signer, got %q", res.MatchedSigner)
	}
	if res.WitnessOutcome() != abi.WitnessConfirmed {
		t.Fatalf("AUTHORIZED maps to Confirmed, got %v", res.WitnessOutcome())
	}
}

func TestRSLNoMatchingEntryRefutes(t *testing.T) {
	spec := baseSpec()
	// The log authorizes a DIFFERENT target — the claimed move is unattested.
	spec.Log[0].Target = "other"
	res := NewRSLVerifier().Verify(context.Background(), spec)
	if res.Verdict != RSLUnauthorized || res.Reason != "no_authorized_entry" {
		t.Fatalf("no matching entry => UNAUTHORIZED/no_authorized_entry, got %v/%s", res.Verdict, res.Reason)
	}
	if res.WitnessOutcome() != abi.WitnessRefuted {
		t.Fatalf("UNAUTHORIZED maps to Refuted, got %v", res.WitnessOutcome())
	}
}

func TestRSLEmptyLogRefutes(t *testing.T) {
	spec := baseSpec()
	spec.Log = nil
	// An empty/unreadable log WITH a pinned policy is a positive refute: the
	// cross-clone oracle has no record of this authorized advance.
	res := NewRSLVerifier().Verify(context.Background(), spec)
	if res.Verdict != RSLUnauthorized {
		t.Fatalf("empty log under a pinned policy => UNAUTHORIZED, got %v", res.Verdict)
	}
}

func TestRSLUnadmittedKeyRefutes(t *testing.T) {
	spec := baseSpec()
	// The entry is signed by a key the pinned policy does NOT admit.
	spec.Log[0].SignerKeyID = "rogue"
	res := NewRSLVerifier().Verify(context.Background(), spec)
	if res.Verdict != RSLUnauthorized || res.Reason != "no_authorized_entry" {
		t.Fatalf("unadmitted signer => UNAUTHORIZED/no_authorized_entry, got %v/%s", res.Verdict, res.Reason)
	}
}

// TestRSLPolicyRollbackRejected is the CVE-2026-44544 guard: an entry that is
// VALIDLY signed by an admitted key, but authorizes against an OLDER policy
// sequence than the pinned current one, is a rollback and must be refused — not
// silently accepted on the strength of its signature alone.
func TestRSLPolicyRollbackRejected(t *testing.T) {
	spec := baseSpec()
	spec.Policy.Seq = 9       // current pinned policy advanced to 9
	spec.Log[0].PolicySeq = 7 // entry authorizes against the stale, rolled-back policy
	res := NewRSLVerifier().Verify(context.Background(), spec)
	if res.Verdict != RSLUnauthorized {
		t.Fatalf("validly-signed but stale policy => UNAUTHORIZED (rollback), got %v", res.Verdict)
	}
	if res.Reason != "policy_rollback_rejected" {
		t.Fatalf("rollback reason must be precise, got %q", res.Reason)
	}
	if res.WitnessOutcome() != abi.WitnessRefuted {
		t.Fatalf("rollback maps to Refuted, got %v", res.WitnessOutcome())
	}
}

// TestRSLFreshPolicyAccepted: an entry against a policy sequence EQUAL to or
// AHEAD of the pin is fresh and authorizes. (Ahead can happen mid-rotation; only
// strictly-behind is a rollback.)
func TestRSLFreshPolicyAccepted(t *testing.T) {
	spec := baseSpec()
	spec.Policy.Seq = 7
	spec.Log[0].PolicySeq = 8 // ahead of the pin — not a rollback
	if got := NewRSLVerifier().Verify(context.Background(), spec).Verdict; got != RSLAuthorized {
		t.Fatalf("entry ahead of the policy pin authorizes, got %v", got)
	}
}

func TestRSLNoPinnedPolicyAbstains(t *testing.T) {
	for _, spec := range []RSLSpec{
		{Ref: "r", Target: "t", Policy: RSLPolicy{Seq: 5}},                                // no authorized keys
		{Ref: "r", Target: "t", Policy: RSLPolicy{Seq: -1, AuthorizedKey: []string{"k"}}}, // negative seq = unpinned
	} {
		res := NewRSLVerifier().Verify(context.Background(), spec)
		if res.Verdict != RSLNotApplicable || res.Reason != "no_pinned_policy" {
			t.Fatalf("no pinned policy => NOT_APPLICABLE/no_pinned_policy, got %v/%s", res.Verdict, res.Reason)
		}
		if res.WitnessOutcome() != abi.WitnessAbstain {
			t.Fatalf("NOT_APPLICABLE maps to Abstain, got %v", res.WitnessOutcome())
		}
	}
}

func TestRSLMissingRefOrTargetAbstains(t *testing.T) {
	spec := baseSpec()
	spec.Target = ""
	res := NewRSLVerifier().Verify(context.Background(), spec)
	if res.Verdict != RSLNotApplicable || res.Reason != "missing_ref_or_target" {
		t.Fatalf("missing target => NOT_APPLICABLE/missing_ref_or_target, got %v/%s", res.Verdict, res.Reason)
	}
}

// TestRSLInjectedVerifier proves the external-gittuf seam: a custom
// SignatureVerifier (standing in for a real DSSE/SSH verifier) decides admission
// instead of the default key-set membership. Here it rejects the otherwise-valid
// entry, flipping AUTHORIZED to UNAUTHORIZED, and the rest of the adjudication is
// untouched.
func TestRSLInjectedVerifier(t *testing.T) {
	reject := func(RSLEntry, RSLPolicy) bool { return false }
	res := NewRSLVerifierWith(reject).Verify(context.Background(), baseSpec())
	if res.Verdict != RSLUnauthorized {
		t.Fatalf("injected verifier rejecting all => UNAUTHORIZED, got %v", res.Verdict)
	}
	// A nil verifier must fall back to the default key-set seam (still authorizes).
	if got := NewRSLVerifierWith(nil).Verify(context.Background(), baseSpec()).Verdict; got != RSLAuthorized {
		t.Fatalf("nil verifier falls back to default key-set seam, got %v", got)
	}
}

func TestRSLEnabledFlag(t *testing.T) {
	t.Setenv(RSLFlagEnv, "")
	if RSLEnabled() {
		t.Fatal("empty flag => disabled (default off)")
	}
	for _, on := range []string{"1", "true", "TRUE", "yes", "on"} {
		t.Setenv(RSLFlagEnv, on)
		if !RSLEnabled() {
			t.Fatalf("flag %q => enabled", on)
		}
	}
	for _, off := range []string{"0", "false", "no", "off", "garbage"} {
		t.Setenv(RSLFlagEnv, off)
		if RSLEnabled() {
			t.Fatalf("flag %q => disabled", off)
		}
	}
}

// TestRSLResolverFlagOffAbstains: the rsl:<json> claim is a FLAGGED SPIKE. With
// the flag off (default), the resolver abstains WITHOUT consulting the payload —
// the rung is inert for the fleet until opted in.
func TestRSLResolverFlagOffAbstains(t *testing.T) {
	t.Setenv(RSLFlagEnv, "") // default off
	raw, _ := json.Marshal(baseSpec())
	r := NewWithRunner((&fakeGit{code: 0}).run, "")
	if got := r.Resolve(context.Background(), nil, "rsl:"+string(raw)); got != abi.WitnessAbstain {
		t.Fatalf("flag off => Abstain regardless of payload, got %v", got)
	}
}

// TestRSLResolverFlagOn drives the rung end to end through the resolver's claim
// grammar with the flag on: an authorizing log Confirms, a contradicting one
// Refutes, and a malformed payload Abstains.
func TestRSLResolverFlagOn(t *testing.T) {
	t.Setenv(RSLFlagEnv, "1")
	ctx := context.Background()
	r := NewWithRunner((&fakeGit{code: 0}).run, "")

	ok, _ := json.Marshal(baseSpec())
	if got := r.Resolve(ctx, nil, "rsl:"+string(ok)); got != abi.WitnessConfirmed {
		t.Fatalf("flag on + authorized => Confirmed, got %v", got)
	}

	bad := baseSpec()
	bad.Log[0].SignerKeyID = "rogue"
	badRaw, _ := json.Marshal(bad)
	if got := r.Resolve(ctx, nil, "rsl:"+string(badRaw)); got != abi.WitnessRefuted {
		t.Fatalf("flag on + unauthorized => Refuted, got %v", got)
	}

	if got := r.Resolve(ctx, nil, "rsl:{not json"); got != abi.WitnessAbstain {
		t.Fatalf("flag on + malformed payload => Abstain, got %v", got)
	}
}
