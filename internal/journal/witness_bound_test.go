package journal

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// offTrunkProse is the LONGEST witness value in this host's real corpus: 447
// bytes, 12 rows across .dispatch-runs/guard-audit/*.jsonl (verbatim). It is the
// gitgate OFF_TRUNK refusal — the single most recovery-critical string the guard
// emits, because the remedy it names ("use the collision-safe sanctioned route
// instead: fak worktree worker prepare …") lives entirely in its second half. A
// bound that truncates this value converts a recoverable refusal into a dead end,
// which is the dominant agent-mortality lever on this fleet. Every bound chosen
// here is chosen to keep it INTACT.
const offTrunkProse = "off-trunk refused: `git checkout -b` / `git switch -c` / raw `git worktree add` " +
	"opens an unmanaged branch or worktree. Work directly on the configured development branch. " +
	"For an explicitly requested detached worker, use the collision-safe sanctioned route instead: " +
	"`fak worktree worker prepare --id <worker-id> --scope <path>`, then " +
	"`fak worktree worker land --id <worker-id>` and `fak worktree worker reap --id <worker-id>` " +
	"(AGENTS.md OFF_TRUNK)."

// secretExfilProse is a real ctxmmu witness (2 rows in the corpus, verbatim). It
// is the reason the witness scrub is NOT ArgsLabel's secretish() whole-string
// drop: that rule keys on the SUBSTRING "secret", so reusing it here would blank
// the witness on precisely the SECRET_EXFIL refusals an operator most needs to
// read. Measured cost of the naive reuse: 2 distinct values / 4 rows destroyed.
const secretExfilProse = "ctxmmu SECRET_EXFIL secret_pattern quarantine_id=q1"

func witnessEvent(claim string) abi.Event {
	return abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"cd fak"}`)},
		},
		Verdict: &abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "gitgate",
			Payload: abi.WitnessPayload{Claim: claim},
		},
	}
}

func witnessRow(t *testing.T, claim string) Row {
	t.Helper()
	j := OpenMemory()
	j.Emit(witnessEvent(claim))
	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("wrote %d rows, want 1", len(rows))
	}
	return rows[0]
}

// TestWitnessIsBoundedInTheJournal is the gap this file closes. witnessOf
// returned WitnessPayload.Claim VERBATIM — no bound, no scrub — while
// internal/guardcorpus copies row.Witness into the EXPORTED dataset under a
// package comment asserting the field "the journal producer already bounded and
// scrubbed". That assertion was false for Witness (it was only ever true for
// ArgsLabel), so this is a demonstrated documentation/behaviour divergence, not
// merely a regression guard.
//
// The claim is not rung-authored by construction: live producers concatenate
// call-derived bytes into it — internal/adjudicator/egresslist.go builds
// "egress restricted, host not allowlisted: "+dests[0] from a host parsed out of
// the CALL ARGS, and internal/adjudicator/lintwrites.go builds a witness from the
// target PATH plus a Go/JSON parser message that embeds the offending source
// token. An unbounded field fed from call args is a disclosure surface no matter
// how well-behaved today's rungs are.
func TestWitnessIsBoundedInTheJournal(t *testing.T) {
	long := "egress restricted, host not allowlisted: " + strings.Repeat("a", 4096)
	row := witnessRow(t, long)
	if len(row.Witness) > maxWitnessLen+len(boundEllipsis) {
		t.Errorf("Witness len = %d, want <= %d (unbounded call-derived bytes reach the exported dataset)",
			len(row.Witness), maxWitnessLen+len(boundEllipsis))
	}
	if !strings.HasPrefix(row.Witness, "egress restricted, host not allowlisted: ") {
		t.Errorf("bound must keep the rung's leading prose, got %q", row.Witness)
	}
}

// TestWitnessBoundKeepsRefusalProseIntact pins the design constraint that decides
// the bound's VALUE. Reusing ArgsLabel's 96-byte bound would truncate 57 of 630
// witness rows (9.05%) and discard 26.22% of all witness prose bytes in the
// corpus — the recovery instructions specifically, since the long values are
// exactly the gate's own "here is how to comply" text. The bound is therefore set
// above the observed maximum (447), so measured truncation cost is ZERO.
func TestWitnessBoundKeepsRefusalProseIntact(t *testing.T) {
	if len(offTrunkProse) != 447 {
		t.Fatalf("fixture drift: offTrunkProse is %d bytes, want the corpus max of 447", len(offTrunkProse))
	}
	row := witnessRow(t, offTrunkProse)
	if row.Witness != offTrunkProse {
		t.Errorf("the 447-byte OFF_TRUNK remedy must survive byte-for-byte;\n got %q\nwant %q", row.Witness, offTrunkProse)
	}
	if maxWitnessLen < 447 {
		t.Errorf("maxWitnessLen = %d truncates the observed corpus maximum (447)", maxWitnessLen)
	}
}

// TestWitnessScrubKeepsSecretExfilProse pins that the witness scrub is
// value-targeted, never ArgsLabel's whole-string secretish() drop. Verified
// against the whole corpus: of ~156 distinct witness values, the only ones
// carrying a secretish needle are these SECRET_EXFIL rows, and NO corpus value
// has a secretish assignment key — so the rule chosen here alters ZERO real rows
// (verified by replaying boundWitness over every live corpus row) while still
// redacting an assigned secret value.
func TestWitnessScrubKeepsSecretExfilProse(t *testing.T) {
	if !secretish(secretExfilProse) {
		t.Fatalf("fixture drift: %q no longer trips secretish(), so this test no longer guards the reuse hazard", secretExfilProse)
	}
	row := witnessRow(t, secretExfilProse)
	if row.Witness != secretExfilProse {
		t.Errorf("SECRET_EXFIL witness must survive whole;\n got %q\nwant %q", row.Witness, secretExfilProse)
	}
}

// TestWitnessRedactsAssignedSecretValue is a REGRESSION GUARD, not a demonstrated
// old bug: no row in this host's corpus carries a secretish assignment today. It
// pins the half of the guardcorpus contract that says "scrubbed" — a rung that
// concatenates an arg value behind a secret-shaped key must not put that value on
// the exported wire. The payloads below are obviously-synthetic placeholders.
func TestWitnessRedactsAssignedSecretValue(t *testing.T) {
	for _, tc := range []struct {
		name, claim, wantGone string
	}{
		{"equals", "egress blocked by denylist: api_key=SYNTHETIC_NOT_A_SECRET", "SYNTHETIC_NOT_A_SECRET"},
		{"header", "egress refused: Authorization: SYNTHETIC_NOT_A_SECRET", "SYNTHETIC_NOT_A_SECRET"},
		{"token", "witness refused: token=SYNTHETIC_NOT_A_SECRET rejected", "SYNTHETIC_NOT_A_SECRET"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row := witnessRow(t, tc.claim)
			if strings.Contains(row.Witness, tc.wantGone) {
				t.Errorf("assigned secret value reached the journal: %q", row.Witness)
			}
			if !strings.Contains(row.Witness, redactedValue) {
				t.Errorf("expected the redaction marker %q in %q", redactedValue, row.Witness)
			}
		})
	}
}

// TestWitnessBoundDoesNotAffectChainVerification pins the property that makes
// this change safe to land against journals already on disk: chainHash's
// pre-image is (Seq..ResultDigest) and does NOT include Witness, so changing the
// value this field records can never invalidate a historical row's hash. Same
// reasoning #5863 used when it appended deny_rule outside the pre-image.
func TestWitnessBoundDoesNotAffectChainVerification(t *testing.T) {
	j := OpenMemory()
	j.Emit(witnessEvent(offTrunkProse))
	j.Emit(witnessEvent(strings.Repeat("z", 9000)))
	j.Emit(witnessEvent(secretExfilProse))
	rows := j.Recent(0)
	if n, err := VerifyRows(rows); err != nil || n != 3 {
		t.Fatalf("witness bound must not affect hash-chain verification: n=%d err=%v", n, err)
	}
	// The pre-image is field-listed, so an identical row with a DIFFERENT witness
	// hashes identically — the direct proof that historical journals still verify.
	a := rows[0]
	b := a
	b.Witness = "anything else entirely"
	if chainHash(a.PrevHash, a) != chainHash(b.PrevHash, b) {
		t.Error("Witness leaked into chainHash's pre-image: historical journals would stop verifying")
	}
}
