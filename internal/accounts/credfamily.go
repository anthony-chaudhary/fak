package accounts

// credfamily.go — the OAuth token-FAMILY hazard an adopted credential creates, and the
// primitives that let the enroll path resolve it deliberately instead of leaving it armed.
//
// WHY THIS EXISTS (witnessed 2026-08-06). `add --adopt` / `enroll-current` COPY a source dir's
// .credentials.json into the new seat. That leaves two config dirs holding a byte-identical
// claudeAiOauth block — one accessToken and, critically, one refreshToken. Claude Code rotates
// the refresh token when it refreshes: the side that refreshes FIRST receives a new
// access+refresh pair, and the provider then rejects the old pair. The other dir is holding
// exactly that old pair, so it is silently dead — its next request 401s and the only recovery
// is a human /login.
//
// The witness that pinned this down: a seat enrolled from ~/.claude refreshed at 11:06 local;
// the credential ~/.claude was still holding (same bytes, its own expiresAt five and a half
// hours in the FUTURE) then returned
//
//	HTTP 401 authentication_error "Invalid authentication credentials"
//
// on api/oauth/profile. The token had not expired — it had been INVALIDATED by the copy's
// rotation. The operator's own interactive session was logged out mid-task with no warning.
//
// THE INESCAPABLE CONSEQUENCE. Two dirs cannot share one login and both keep refreshing. There
// is no ordering that saves both: whoever refreshes first survives and the other is dead. So an
// adopt is not really a COPY of a login, it is a MOVE of one — and the only honest thing a tool
// can do is perform that move at a moment the operator chose (and is watching), rather than let
// it detonate later at whatever random instant the first refresh happens to occur.
//
// These primitives are pure (a fingerprint compare over on-disk bytes, no network, no spawn) so
// the policy is unit-tested here; the enroll path composes them with TriggerRefresh, which
// already knows how to CAUSE a rotation and witness it from the file rather than an exit code.

import (
	"crypto/sha256"
	"encoding/hex"
)

// credentialRefreshToken returns the refreshToken a dir's live session credential holds, or ""
// when the dir carries no parseable credential / no refresh token. It reads through the shared
// credentialTokens decoder (credidentity.go) rather than re-opening the file: the ACCESS token
// identifies the account, the REFRESH token identifies the token FAMILY, but both come from one
// read of one block. The raw token is never a value to log; use RefreshFamilyID for anything
// operator-visible.
func credentialRefreshToken(dir string) string {
	_, refresh := credentialTokens(dir)
	return refresh
}

// RefreshFamilyID is a short, NON-SECRET fingerprint of the OAuth token family a config dir is
// currently on: the first 8 hex of the sha256 of its refresh token. It exists so an operator (and
// a test) can see at a glance whether two dirs are on the SAME family, and whether a divorce
// actually moved one of them off it, without a secret ever reaching a log line or a terminal.
// "" means the dir holds no refresh token (no credential, a torn file, or a hollow credential
// whose tokens are empty strings — the shape a dead seat has).
func RefreshFamilyID(dir string) string {
	tok := credentialRefreshToken(dir)
	if tok == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(tok))
	return hex.EncodeToString(sum[:4])
}

// FamilyShare is the verdict on whether two config dirs are sharing one OAuth token family.
type FamilyShare struct {
	// Shared is true iff BOTH dirs hold a non-empty refresh token and the tokens are identical —
	// the armed state where the next refresh by either side silently kills the other.
	Shared bool
	// FamilyID is the shared family's non-secret fingerprint (RefreshFamilyID) when Shared;
	// otherwise the target's own family, so a caller can always report which family a seat is on.
	FamilyID string
	// SourceID / TargetID are each dir's family fingerprint ("" when the dir holds no refresh
	// token). Reported so an operator can SEE the divergence a divorce is supposed to produce.
	SourceID string
	TargetID string
}

// DetectSharedRefreshFamily reports whether srcDir and dstDir are on the same OAuth token family.
// Only a byte-identical, non-empty refresh token counts as shared: two dirs that merely serve the
// same ACCOUNT (a duplicate login, each with its own family) are not shared — they waste one
// rate-limit bucket, which is DetectEnrollCollision's concern, not this one. A dir with no
// credential, or a hollow credential whose tokens are empty strings, is never "sharing" anything.
func DetectSharedRefreshFamily(srcDir, dstDir string) FamilyShare {
	src, dst := RefreshFamilyID(srcDir), RefreshFamilyID(dstDir)
	out := FamilyShare{SourceID: src, TargetID: dst, FamilyID: dst}
	if src == "" || dst == "" {
		return out
	}
	if credentialRefreshToken(srcDir) != credentialRefreshToken(dstDir) {
		return out
	}
	out.Shared = true
	out.FamilyID = dst
	return out
}

// DivorceOutcome is the closed classification of what a family divorce achieved. It is what the
// enroll path reports, and it is graded on the FILE (did the target's family fingerprint actually
// change?), never on the refresh spawn's exit code — the same "witness the artifact, not the
// self-report" discipline TriggerRefresh applies to the expiry.
type DivorceOutcome string

const (
	// DivorceNotNeeded: the dirs were never on the same family, so nothing had to move.
	DivorceNotNeeded DivorceOutcome = "not_needed"
	// DivorceDone: the target moved onto its OWN family. The seat is now independently
	// refreshable — and, unavoidably, the SOURCE dir's copy of the old family is now dead and
	// needs a human /login. That consequence is stated, never hidden.
	DivorceDone DivorceOutcome = "divorced"
	// DivorceFailed: the refresh did not move the target off the shared family. Nothing is broken
	// YET — both dirs still hold a working credential — but the hazard is still armed, and the
	// most likely cause is that the credential cannot refresh at all (a seat that will need a
	// human login the first time its access token lapses).
	DivorceFailed DivorceOutcome = "failed"
)

// DivorceReport is the operator-facing result of a divorce attempt.
type DivorceReport struct {
	Outcome DivorceOutcome `json:"outcome"`
	// Before/After are the target's non-secret family fingerprints around the refresh. A divorce
	// is proven exactly when they differ.
	Before string `json:"before,omitempty"`
	After  string `json:"after,omitempty"`
	// SourceDir is the dir whose credential is invalidated by a successful divorce — the one that
	// now needs a human relogin.
	SourceDir string `json:"source_dir,omitempty"`
	// Refreshed carries TriggerRefresh's own verdict (did the on-disk EXPIRY advance). A divorce
	// can be Done with Refreshed true; Refreshed false with a changed family would be surprising
	// and is worth surfacing rather than smoothing over.
	Refreshed bool `json:"refreshed"`
	// Err is the refresh spawn's error when it had one (surfaced, never swallowed).
	Err error `json:"-"`
}

// Divorced reports whether the target provably owns its own token family now.
func (r DivorceReport) Divorced() bool {
	return r.Outcome == DivorceNotNeeded || r.Outcome == DivorceDone
}
