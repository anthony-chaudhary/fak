// Issue #4144 (wiring half): the production on-disk wiring for relay's cross-host WARM-resume
// seam. relay owns the PURE port — the SessionImageResolver over an injected load probe, and
// the ResolveResumeMode warm/cold fold (internal/relay/hostmigrate.go). This file is the one
// place a real bundle is touched: it maps the filesystem to relay's fail-closed tri-state and
// pages the verified image in.
//
// The dependency points the RIGHT way. relay is a foundation-tier package that must import no
// integrator, so it cannot import sessionimage. sessionimage is the tier-4 integrator (it
// already composes recall/session/trajectory/ctxplan into a bundle), so it may depend DOWN on
// relay's port — this file is that legal downward edge. The tier DAG (architest
// TestNoUpwardImports) holds: relay never learns sessionimage exists.
package sessionimage

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// RelayResumeProbe is the production image-load probe for relay.SessionImageResolver: it
// reports whether the bundle at handle loads and integrity-verifies via LoadDir, mapping a
// real filesystem to relay's resolver contract, fail-closed:
//
//   - the bundle path cannot be stat'd for a reason OTHER than "does not exist" (a permission
//     error, an I/O error, an unreachable mount) -> (false, err) -> ResolveUnknown. The
//     offload store is unreachable; never mistake that for a missing image.
//   - the bundle path does not exist                              -> (false, nil) -> ResolveDangling.
//   - the bundle exists but LoadDir errors (a missing/absent part, a version mismatch, a
//     sha256 digest mismatch — truncated or tampered)             -> (false, nil) -> ResolveDangling.
//     The store is reachable but there is no whole, verifiable image there, so a warm resume
//     is unsafe; fall back to cold. A corrupt image is NEVER reported verified.
//   - the bundle loads and every part hashes clean                -> (true, nil)  -> ResolveVerified.
func RelayResumeProbe(handle string) (bool, error) {
	h := strings.TrimSpace(handle)
	if h == "" {
		return false, nil
	}
	if _, err := os.Stat(h); err != nil {
		if os.IsNotExist(err) {
			return false, nil // reachable location, image is gone -> dangling
		}
		return false, err // cannot reach the store at all -> unknown (fail closed)
	}
	if _, err := LoadDir(h); err != nil {
		// The bundle is present but is not a whole, integrity-verified image (absent part,
		// wrong version, or a digest mismatch from a truncated/tampered offload). Reachable
		// but not resolvable as a whole -> dangling, never verified.
		return false, nil
	}
	return true, nil
}

// RelayResolver is the production wiring a successor uses: a relay.SessionImageResolver whose
// probe loads and integrity-verifies real bundles on disk (RelayResumeProbe). Pass it to
// relay.ResolveResumeMode to fold a baton's session-image pointer into the warm/cold decision.
func RelayResolver() relay.SessionImageResolver {
	return relay.NewSessionImageResolver(RelayResumeProbe)
}

// LoadWarmImage is the production warm-branch load step: it loads and integrity-verifies the
// image bundle at handle and returns the *Image the successor rehydrates. It reuses LoadDir, so
// the SAME sha256 part-index check RelayResumeProbe made gates the actual load — a warm resume
// never reads bytes the resolver did not already prove whole. Callers reach it only after
// relay.ResolveResumeMode returns relay.ResumeWarm.
func LoadWarmImage(handle string) (*Image, error) {
	return LoadDir(strings.TrimSpace(handle))
}
