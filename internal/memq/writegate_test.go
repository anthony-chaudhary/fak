package memq

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// #2431 acceptance: the write arm probes every concrete claim BEFORE a memory
// note lands. A note citing a claim that does not verify persists with an
// unverified-at-write stamp naming the failed claim; a note whose claims all
// verify carries verified-at-write with the probe witness. The split verifier
// keeps the test hermetic (no git) — value "deadbeef" is stale, everything else
// fresh, exactly as the read-gate tests use it.
func TestWriteTimeClaimProbe(t *testing.T) {
	ctx := context.Background()

	// A commit SHA that does not resolve -> unverified-at-write, claim named.
	bad := "---\nname: guess\ndescription: a self-report\nmetadata:\n  type: project\n---\n\nLanded in commit deadbeef.\n"
	stamped, findings := ProbeAtWrite(ctx, []byte(bad), splitVerifier)
	if len(findings) == 0 {
		t.Fatal("a note naming a commit SHA must produce a probe finding")
	}
	status, detail := writeVerifyStamp(string(stamped))
	if status != WriteUnverified {
		t.Fatalf("note with a failing claim must stamp %s, got %q (findings=%+v)", WriteUnverified, status, findings)
	}
	if !strings.Contains(detail, "deadbeef") {
		t.Fatalf("unverified stamp must name the failed claim, got detail %q", detail)
	}
	// The write is NOT blocked — the body survives untouched, only frontmatter grows.
	if !strings.Contains(string(stamped), "Landed in commit deadbeef.") {
		t.Fatal("a failing probe must still persist the note body (write is not blocked)")
	}

	// A note whose only concrete claim (a live repo path) verifies -> verified.
	good := "---\nname: real\ndescription: a checked note\nmetadata:\n  type: feedback\n---\n\nUse the helper in internal/memq/exec.go.\n"
	gstamped, gfindings := ProbeAtWrite(ctx, []byte(good), splitVerifier)
	if len(gfindings) == 0 {
		t.Fatal("a note naming a repo path must produce a probe finding")
	}
	gstatus, gdetail := writeVerifyStamp(string(gstamped))
	if gstatus != WriteVerified {
		t.Fatalf("note whose claims all verify must stamp %s, got %q (findings=%+v)", WriteVerified, gstatus, gfindings)
	}
	if !strings.Contains(gdetail, "internal/memq/exec.go") {
		t.Fatalf("verified stamp witness must name the probed claim, got %q", gdetail)
	}

	// A prose-only note has nothing checkable -> no verdict is invented.
	prose := "---\nname: pref\ndescription: terse\nmetadata:\n  type: user\n---\n\nThe user prefers terse answers.\n"
	pstamped, pfindings := ProbeAtWrite(ctx, []byte(prose), splitVerifier)
	if len(pfindings) != 0 {
		t.Fatalf("a claim-free note must probe nothing, got %+v", pfindings)
	}
	if s, _ := writeVerifyStamp(string(pstamped)); s != "" {
		t.Fatalf("a claim-free note must not carry a write verdict, got %q", s)
	}

	// Re-probing is idempotent: a second pass replaces, not stacks, the stamp.
	restamped, _ := ProbeAtWrite(ctx, stamped, splitVerifier)
	if n := strings.Count(string(restamped), "write_verify:"); n != 1 {
		t.Fatalf("re-probe must replace the stamp, found %d write_verify lines", n)
	}
}

// #2431 acceptance: the hedged render path fires for a note flagged
// unverified-at-write — the doubt recorded at the door survives to the table even
// when the live re-check cannot confirm or refute the claim.
func TestRecallHedgesUnverifiedAtWrite(t *testing.T) {
	ctx := context.Background()

	// Probe a note whose SHA claim fails at write, then persist the stamped note.
	raw := "---\nname: doorstamp\ndescription: unproven at write\nmetadata:\n  type: project\n---\n\nLanded in commit deadbeef.\n"
	stamped, _ := ProbeAtWrite(ctx, []byte(raw), splitVerifier)
	if s, _ := writeVerifyStamp(string(stamped)); s != WriteUnverified {
		t.Fatalf("precondition: note must be stamped %s, got %q", WriteUnverified, s)
	}
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Door stamp](door.md) — unproven at write\n",
		map[string]string{"door.md": string(stamped)})

	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	// A verifier that can neither confirm nor refute: only the write-time stamp
	// carries the doubt now.
	b.WithVerifier(func(_ context.Context, claims []recall.ArtifactClaim) []recall.ArtifactFinding {
		out := make([]recall.ArtifactFinding, 0, len(claims))
		for _, c := range claims {
			out = append(out, recall.ArtifactFinding{Claim: c, Status: recall.ArtifactUnverifiable, Detail: "offline"})
		}
		return out
	})

	reason, err := b.HedgeReason(ctx, "door.md")
	if err != nil {
		t.Fatal(err)
	}
	if reason == "" {
		t.Fatal("an unverified-at-write note must render hedged even when the live re-check is inconclusive")
	}
	if !strings.Contains(reason, "unverified at write") || !strings.Contains(reason, "deadbeef") {
		t.Fatalf("hedge reason must cite the write-time verdict and the failing claim, got %q", reason)
	}

	// A note that verified at write AND still verifies live renders plainly.
	good := "---\nname: solid\ndescription: checked and still true\nmetadata:\n  type: feedback\n---\n\nUse the helper in internal/memq/exec.go.\n"
	gstamped, _ := ProbeAtWrite(ctx, []byte(good), splitVerifier)
	gdir := fixtureNotesStore(t,
		"# Memory index\n\n- [Solid](solid.md) — checked\n",
		map[string]string{"solid.md": string(gstamped)})
	gb, err := NewNotesBackend(gdir)
	if err != nil {
		t.Fatal(err)
	}
	gb.WithVerifier(splitVerifier)
	greason, err := gb.HedgeReason(ctx, "solid.md")
	if err != nil {
		t.Fatal(err)
	}
	if greason != "" {
		t.Fatalf("a verified-at-write note whose claims still verify must render plainly, got hedge %q", greason)
	}
}
