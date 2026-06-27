package witness

// The gittuf RSL rung (#826) — a FLAGGED SPIKE, default off.
//
// THE GAP IT CLOSES. Every other rung in this package decides "did this ref
// actually move to where the worker says it did?" against LOCAL git only:
// ancestor-of-HEAD, commit-exists, path-tracked, tree-clean. That is the right
// check against the wrong oracle. A worker's local repository — or a
// compromised forge serving a doctored ref to exactly the clone that is asking —
// can misrepresent ref history to a single clone, and none of the local rungs
// can tell. Authorization for a ref ADVANCE lives across clones and over time,
// not in one working tree.
//
// THE ADDITIVE VALUE, AND ONLY IT. gittuf's Reference State Log (the ref
// `refs/gittuf/reference-state-log`) is a signed, append-only record: every
// authorized ref advance is a signed entry every clone can fetch and verify, so
// "this ref moved here" stops being a single-clone assertion and becomes a
// cross-clone, cryptographically attributable fact. A git SHA is already
// content-addressed, so this does NOT make commit CONTENTS more tamper-evident
// than git's Merkle structure already does — the additive value is purely the
// SIGNATURE ENVELOPE around ref MOVEMENT: who authorized this ref to advance,
// attested across clones. This rung claims no more than that.
//
// TRUST BOUNDARY vs gitgate / the local witness rungs. This is after-the-fact,
// distributed verification — the SAME category as the local witness, made
// cross-clone. It does NOT replace gitgate's pre-call refusal and does NOT
// synchronously decide a live OFF_TRUNK or foreign MERGE_HEAD at the moment a
// worker is about to act. By the time an RSL entry exists, the ref has already
// moved; this catches an UNAUTHORIZED advance, it does not prevent the call. The
// shell-laundering backstop is the bonus, not the headline: because an RSL entry
// is created at ref-advance time regardless of HOW the push was invoked, an
// aliased / eval'd / $()-laundered push that slips past the gitgate argv
// tokenizer still leaves an entry every peer verifies — but only at ref-advance
// GRANULARITY (after the ref moved), never as a pre-call refusal, and never for
// a tool call that advances no ref.
//
// POLICY-ROLLBACK (CVE-2026-44544). A stale-but-validly-signed policy must NOT
// be silently accepted: an attacker who can replay an OLD, correctly-signed
// policy could re-admit a key that the current policy has since revoked. So this
// rung pins policy freshness on EVERY check — the verified entry must reference
// the pinned current policy id, and a strictly-lower policy sequence is REFUTED
// as a rollback even when its signature checks out. Signature validity is
// necessary, not sufficient.
//
// WHAT IS IN-LANE HERE, AND THE EXTERNAL-GITTUF SEAM. This file lands the
// VERIFICATION PATH against an RSL-style signed-log structure in-process: parse
// the log, find the entry that authorizes the claimed (ref, target) advance,
// check it is signed by a key the pinned policy admits, and enforce freshness.
// The structure mirrors gittuf's RSL/policy semantics so the rung's logic is
// exercised and tested without a network fetch or a real signing toolchain. Two
// pieces are a DEFERRED external seam, called out so a follow-on can wire them:
//
//   - The fetch of `refs/gittuf/reference-state-log` (a `git fetch` of the
//     gittuf namespace) and real DSSE/SSH/GPG signature cryptography belong to
//     the gittuf toolchain (github.com/gittuf/gittuf) or `git verify-commit`.
//     This rung takes a SignatureVerifier seam (default: a key-set membership
//     check over an already-extracted signer id) so the cross-clone logic is
//     testable now and a real verifier drops in without touching the rung.
//   - Turning RSL MAINTENANCE on for the fleet (per-writer signing keys + a
//     signed log entry per authorized ref advance, fetched and verified by every
//     peer) is a real, heavy tax on a fast-moving shared trunk. That is exactly
//     why this is a flagged spike, not a default: the spike measures that tax on
//     this repo's churn before anyone argues for default-on. RSLEnabled() is the
//     flag; it is false unless explicitly opted in.

import (
	"context"
	"encoding/json"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// RSLVerdict is the RSL rung's label space, deliberately separate from
// abi.WitnessOutcome (as ExecutionVerdict is) so a caller can distinguish a
// cross-clone ref-move verdict from the local git-evidence verdicts.
type RSLVerdict string

const (
	// RSLAuthorized: the claimed ref advance has a matching, policy-admitted,
	// fresh RSL entry — a cross-clone, attributable authorization.
	RSLAuthorized RSLVerdict = "RSL_AUTHORIZED"
	// RSLUnauthorized: the log was readable and the claim is contradicted — no
	// matching entry, an entry signed by a key the policy does not admit, or a
	// rolled-back (stale) policy. This is a positive REFUTE, not an abstain.
	RSLUnauthorized RSLVerdict = "RSL_UNAUTHORIZED"
	// RSLNotApplicable: the rung could form no evidence either way — the flag is
	// off, the payload is unparseable, or no policy is pinned. Abstains.
	RSLNotApplicable RSLVerdict = "RSL_NOT_APPLICABLE"
)

// RSLEntry is one signed Reference State Log entry: an authorized advance of Ref
// to Target, signed by SignerKeyID, under policy sequence PolicySeq. It mirrors
// the fields of a gittuf RSL reference entry that this rung needs to adjudicate a
// ref move; the real on-the-wire form is a signed git commit in the gittuf
// namespace (see the external seam in the package doc above).
type RSLEntry struct {
	Ref         string `json:"ref"`
	Target      string `json:"target"`
	SignerKeyID string `json:"signer_key_id"`
	PolicySeq   int    `json:"policy_seq"`
	// Signature is the opaque signature blob over the entry. The default
	// verifier does not inspect it (it checks signer-key membership); a real
	// DSSE/SSH verifier dropped in via the seam would.
	Signature string `json:"signature,omitempty"`
}

// RSLPolicy is the pinned current gittuf policy: the set of key ids authorized to
// sign ref advances, plus the policy SEQUENCE used to detect rollback. Seq is the
// freshness pin: an entry that authorizes against a strictly-lower policy
// sequence is a rollback (CVE-2026-44544) and is refused even if its signature is
// valid.
type RSLPolicy struct {
	Seq           int      `json:"seq"`
	AuthorizedKey []string `json:"authorized_keys"`
}

// RSLSpec is the JSON payload accepted by the rsl:<json> witness claim. Log is
// the cross-clone signed log (in production: fetched from
// refs/gittuf/reference-state-log — see the external seam). Policy is the pinned
// current policy. Claim names the ref move being adjudicated.
type RSLSpec struct {
	Ref    string     `json:"ref"`
	Target string     `json:"target"`
	Policy RSLPolicy  `json:"policy"`
	Log    []RSLEntry `json:"log"`
}

// RSLResult is the portable read-back from the RSL rung.
type RSLResult struct {
	Verdict   RSLVerdict `json:"verdict"`
	Reason    string     `json:"reason,omitempty"`
	Ref       string     `json:"ref,omitempty"`
	Target    string     `json:"target,omitempty"`
	PolicySeq int        `json:"policy_seq,omitempty"`
	// MatchedSigner is the signer key id of the authorizing entry, on AUTHORIZED.
	MatchedSigner string `json:"matched_signer,omitempty"`
}

// WitnessOutcome maps the RSL verdict back to the kernel's three-way contract.
// AUTHORIZED confirms, UNAUTHORIZED refutes, NOT_APPLICABLE abstains.
func (r RSLResult) WitnessOutcome() abi.WitnessOutcome {
	switch r.Verdict {
	case RSLAuthorized:
		return abi.WitnessConfirmed
	case RSLUnauthorized:
		return abi.WitnessRefuted
	default:
		return abi.WitnessAbstain
	}
}

// SignatureVerifier is the external-gittuf seam. It reports whether an RSL
// entry's signature is valid under the pinned policy. The default
// (keySetVerifier) checks that the entry's already-extracted SignerKeyID is a
// member of the policy's authorized-key set — enough to exercise the cross-clone
// AUTHORIZE/REFUTE/freshness logic end to end. A real implementation backed by
// the gittuf toolchain (DSSE/SSH/GPG over the signed git object) drops in here
// without touching the rung's adjudication.
type SignatureVerifier func(entry RSLEntry, policy RSLPolicy) bool

// keySetVerifier is the default seam: membership of the entry's signer key in the
// pinned policy's authorized-key set. It is intentionally a stand-in for real
// signature cryptography — the seam exists so that cryptography is the ONLY thing
// a follow-on must add, never the adjudication around it.
func keySetVerifier(entry RSLEntry, policy RSLPolicy) bool {
	if strings.TrimSpace(entry.SignerKeyID) == "" {
		return false
	}
	for _, k := range policy.AuthorizedKey {
		if k == entry.SignerKeyID {
			return true
		}
	}
	return false
}

// RSLFlagEnv is the opt-in flag for the spike. The rung is OFF (abstains
// immediately) unless this env var is set to a truthy value — the "flagged spike,
// default off" contract. It is read per-check so a test or operator can toggle it
// without restarting.
const RSLFlagEnv = "FAK_WITNESS_RSL"

// RSLEnabled reports whether the RSL spike is opted in. Default off.
func RSLEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(RSLFlagEnv))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// RSLVerifier adjudicates a claimed ref move against an RSL-style signed log under
// a pinned policy. Construct with NewRSLVerifier (default key-set seam) or
// NewRSLVerifierWith (inject a real SignatureVerifier). The verifier holds no git
// state: it adjudicates the structure it is handed, which is what makes it the
// pure, testable core of the rung. The network fetch of the gittuf namespace is
// the external seam (see the package doc).
type RSLVerifier struct {
	verify SignatureVerifier
}

// NewRSLVerifier builds the verifier with the default key-set membership seam.
func NewRSLVerifier() *RSLVerifier { return &RSLVerifier{verify: keySetVerifier} }

// NewRSLVerifierWith injects a SignatureVerifier (a real gittuf/DSSE verifier, or
// a test double). A nil verifier falls back to the default key-set seam.
func NewRSLVerifierWith(v SignatureVerifier) *RSLVerifier {
	if v == nil {
		v = keySetVerifier
	}
	return &RSLVerifier{verify: v}
}

// Verify adjudicates the claimed (Ref, Target) advance against the signed log
// under the pinned policy. The order is deliberate and fail-closed:
//
//  1. A pinned policy is required (Seq >= 0 and at least one authorized key);
//     without it there is no oracle, so the rung is NOT_APPLICABLE (abstain).
//  2. Among entries for the same (Ref, Target), the AUTHORIZING entry is the one
//     whose signature the policy admits AND whose policy sequence is NOT a
//     rollback (PolicySeq >= policy.Seq). A strictly-lower sequence is the
//     CVE-2026-44544 rollback and is refused even with a valid signature.
//  3. With no admitted, fresh entry for the claim, the result is UNAUTHORIZED —
//     a positive refute, because the log WAS readable and did not authorize the
//     move. (An unreadable/empty log with a pinned policy is still a refute: the
//     cross-clone oracle says "no record of this authorized advance".)
func (v *RSLVerifier) Verify(ctx context.Context, spec RSLSpec) RSLResult {
	_ = ctx
	ref := strings.TrimSpace(spec.Ref)
	target := strings.TrimSpace(spec.Target)
	res := RSLResult{Ref: ref, Target: target, PolicySeq: spec.Policy.Seq}

	if ref == "" || target == "" {
		res.Verdict = RSLNotApplicable
		res.Reason = "missing_ref_or_target"
		return res
	}
	if spec.Policy.Seq < 0 || len(spec.Policy.AuthorizedKey) == 0 {
		// No pinned policy => no cross-clone oracle => abstain, never a false
		// confirm and never a refute we cannot justify.
		res.Verdict = RSLNotApplicable
		res.Reason = "no_pinned_policy"
		return res
	}

	verify := v.verify
	if verify == nil {
		verify = keySetVerifier
	}

	sawStaleSigned := false
	for _, e := range spec.Log {
		if strings.TrimSpace(e.Ref) != ref || strings.TrimSpace(e.Target) != target {
			continue
		}
		if !verify(e, spec.Policy) {
			continue // signed by a key the pinned policy does not admit
		}
		if e.PolicySeq < spec.Policy.Seq {
			// Validly signed but against a ROLLED-BACK policy (CVE-2026-44544).
			// Record it so the refute reason is precise, but do not authorize.
			sawStaleSigned = true
			continue
		}
		// Admitted signature AND fresh policy: a cross-clone authorization.
		res.Verdict = RSLAuthorized
		res.Reason = ""
		res.MatchedSigner = e.SignerKeyID
		return res
	}

	res.Verdict = RSLUnauthorized
	if sawStaleSigned {
		res.Reason = "policy_rollback_rejected" // CVE-2026-44544 guard fired
	} else {
		res.Reason = "no_authorized_entry"
	}
	return res
}

// resolveRSL backs the rsl:<json> claim on Resolver. It is a FLAGGED SPIKE: off
// by default. When the flag is off it abstains (NOT_APPLICABLE) without parsing,
// so the rung is inert for the fleet until explicitly opted in. When on, it
// adjudicates the payload via RSLVerifier and maps to the kernel's three-way
// witness contract.
func (r *Resolver) resolveRSL(ctx context.Context, raw string) abi.WitnessOutcome {
	if !RSLEnabled() {
		return abi.WitnessAbstain
	}
	var spec RSLSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return abi.WitnessAbstain
	}
	return NewRSLVerifier().Verify(ctx, spec).WitnessOutcome()
}
