package recall

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// rehydrate.go — the #1184 recall-revalidation RUNG for the staged wake gate
// (internal/rehydrate, #1181). The orchestrator owns WHEN a rung fires; this leaf supplies
// the StaleRecall rung's CHECK, exactly as internal/leaseref supplies the StaleLease fence.
//
// The check reuses the SAME read-time artifact guard the loaded session already runs
// (ExtractArtifactClaims + ArtifactVerifier in reverify.go, #1158): a recalled memory that
// names a concrete artifact — a commit SHA, a repo-relative path, a CLI flag — whose history
// has moved on is no longer fact, so on a Cold+ wake the gate withholds it with STALE_RECALL
// rather than re-injecting a stale memory as confirmed.

// NewRehydrateRecallRung builds the StaleRecall rung that re-verifies the concrete artifacts
// named in `text` (the recalled pages about to be admitted on this wake) before they are
// handed back as fact. It fires at the canonical StaleRecall band (Cold) — the band at which
// a recalled memory's artifacts first decay. `verifier` is the test/DOS seam: nil uses the
// DefaultArtifactVerifier (git-backed: a SHA must resolve and stay reachable, a path must
// exist, a flag must still appear in the checkout); a test injects a stub to drive
// stale/fresh deterministically.
//
// Refusal is on POSITIVE staleness only: a provably-orphaned artifact refuses with
// STALE_RECALL so the wake gate withholds the page. Text that names no concrete artifact
// clears (nothing could have gone stale), and an UNVERIFIABLE artifact also clears — the gate
// never wedges a wake over something it merely cannot check (e.g. off a git checkout), it only
// withholds what it can prove is stale.
func NewRehydrateRecallRung(text string, verifier ArtifactVerifier) rehydrate.Rung {
	return rehydrate.NewRung(rehydrate.StaleRecall, func(ctx context.Context) rehydrate.Verdict {
		claims := ExtractArtifactClaims(text)
		if len(claims) == 0 {
			return rehydrate.Clear() // no concrete artifact to verify -> nothing can be stale
		}
		v := verifier
		if v == nil {
			v = DefaultArtifactVerifier
		}
		var stale []string
		for _, f := range v(ctx, claims) {
			if f.Status != ArtifactStale {
				continue // FRESH and UNVERIFIABLE both clear: refuse only on proven staleness
			}
			label := fmt.Sprintf("%s %q", f.Claim.Kind, f.Claim.Value)
			if d := strings.TrimSpace(f.Detail); d != "" {
				label += ": " + d
			}
			stale = append(stale, label)
		}
		if len(stale) == 0 {
			return rehydrate.Clear()
		}
		sort.Strings(stale)
		return rehydrate.Refuse(rehydrate.StaleRecall,
			"recalled memory names a stale artifact (withhold, do not inject as fact): "+strings.Join(stale, "; "))
	})
}
