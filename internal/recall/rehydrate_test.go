package recall

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// A recalled memory naming a 40-hex commit SHA — concrete enough for ExtractArtifactClaims
// to lift it as an ArtifactGitSHA without needing a "commit" cue.
const recalledSHAText = "fixed it in deadbeefdeadbeefdeadbeefdeadbeefdeadbeef which shipped the gate"

// stubVerifier returns a fixed status for every claim — the injected DOS/test seam.
func stubVerifier(status ArtifactStatus, detail string) ArtifactVerifier {
	return func(_ context.Context, claims []ArtifactClaim) []ArtifactFinding {
		out := make([]ArtifactFinding, 0, len(claims))
		for _, c := range claims {
			out = append(out, ArtifactFinding{Claim: c, Status: status, Detail: detail})
		}
		return out
	}
}

// #1184 acceptance: a memory naming a SHA orphaned by history is marked STALE_RECALL and
// withheld at a cold-or-longer wake.
func TestRehydrateRecallRung_staleSHAWithheld(t *testing.T) {
	rung := NewRehydrateRecallRung(recalledSHAText, stubVerifier(ArtifactStale, "commit resolves but is not reachable from HEAD"))
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted {
		t.Fatalf("a stale recalled SHA must refuse admission; got admitted=%v", adm)
	}
	if adm.RefusedBy != rehydrate.StaleRecall {
		t.Fatalf("refusal reason = %q, want STALE_RECALL", adm.RefusedBy)
	}
	if !strings.Contains(adm.Detail, "deadbeef") {
		t.Fatalf("refusal detail should name the stale artifact; got %q", adm.Detail)
	}
}

// #1184 acceptance: a still-true memory passes.
func TestRehydrateRecallRung_freshSHAAdmitted(t *testing.T) {
	rung := NewRehydrateRecallRung(recalledSHAText, stubVerifier(ArtifactFresh, "commit resolves and is reachable from HEAD"))
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if !adm.Admitted {
		t.Fatalf("a fresh recalled SHA must clear; got refusedBy=%q detail=%q", adm.RefusedBy, adm.Detail)
	}
}

// An artifact the gate merely CANNOT verify (no git checkout) must not wedge the wake — the
// gate refuses only on proven staleness.
func TestRehydrateRecallRung_unverifiableClears(t *testing.T) {
	rung := NewRehydrateRecallRung(recalledSHAText, stubVerifier(ArtifactUnverifiable, "git root unavailable"))
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if !adm.Admitted {
		t.Fatalf("an unverifiable artifact must clear (refuse only on proven staleness); got refusedBy=%q", adm.RefusedBy)
	}
}

// Recalled text naming no concrete artifact has nothing that can have gone stale.
func TestRehydrateRecallRung_noArtifactsClears(t *testing.T) {
	rung := NewRehydrateRecallRung("a vague prose memory with no SHA, path, or flag", stubVerifier(ArtifactStale, "should not be called"))
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if !adm.Admitted {
		t.Fatalf("text with no concrete artifact must clear; got refusedBy=%q", adm.RefusedBy)
	}
}

// The rung fires at Cold (its canonical band), not at a warm wake: a 5-minute resume runs
// zero rungs and is admitted verbatim even if the memory would be stale.
func TestRehydrateRecallRung_doesNotFireWhenWarm(t *testing.T) {
	rung := NewRehydrateRecallRung(recalledSHAText, stubVerifier(ArtifactStale, "stale"))
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Warm)
	if !adm.Admitted {
		t.Fatalf("a warm wake runs no rungs and must admit verbatim; got refusedBy=%q", adm.RefusedBy)
	}
	if len(adm.Ran) != 0 {
		t.Fatalf("a warm wake must run zero rungs; ran %v", adm.RanReasons())
	}
}

// The default (git-backed) verifier is wired: a bogus, non-resolving SHA is proven stale and
// withheld end-to-end. Skips where git/the repo is unavailable (the verifier returns
// UNVERIFIABLE there, which is a clear, not a refusal — exercised by the stub test above).
func TestRehydrateRecallRung_defaultVerifierStaleSHA(t *testing.T) {
	const bogus = "commit ffffffffffffffffffffffffffffffffffffffff never existed"
	findings := DefaultArtifactVerifier(context.Background(), ExtractArtifactClaims(bogus))
	if len(findings) == 0 || findings[0].Status == ArtifactUnverifiable {
		t.Skip("git/repo unavailable: default verifier cannot adjudicate here")
	}
	rung := NewRehydrateRecallRung(bogus, nil) // nil -> DefaultArtifactVerifier
	adm := rehydrate.NewGate(rung).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted || adm.RefusedBy != rehydrate.StaleRecall {
		t.Fatalf("a bogus non-resolving SHA must refuse with STALE_RECALL via the default verifier; got %+v", adm)
	}
}
