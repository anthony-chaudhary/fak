package accounts

// credexpirynudge.go — make a refresh happen NOW on a credential that is not yet near expiry.
//
// WHY THIS EXISTS (dogfooded 2026-08-06). TriggerRefresh causes a rotation by running one
// `claude -p`, but Claude Code only rotates when its access token is actually near expiry. So on a
// credential with hours left, the spawn is a legitimate NO-OP — and a caller that grades "nothing
// moved" as "this credential cannot refresh" reports a healthy seat as dead. That false alarm was
// observed the first time `fak accounts refresh` ran against a seat refreshed 20 minutes earlier.
//
// Two callers genuinely need the rotation to happen regardless of remaining lifetime:
//
//   - the token-family DIVORCE (credfamily_divorce.go), which must move a freshly COPIED
//     credential off the family it shares with its source — a family it will otherwise keep
//     sharing for hours, with either side able to silently invalidate the other at any moment;
//   - an operator/scheduled `fak accounts refresh --force`, proving a seat CAN still refresh
//     rather than waiting for dispatch to find out that it cannot.
//
// The nudge is the narrowest possible intervention: rewrite ONLY claudeAiOauth.expiresAt to the
// recent past so Claude Code sees an expired access token and performs its own refresh. The
// tokens themselves are never touched, fak never mints or posts anything, and the returned restore
// closure puts the ORIGINAL BYTES back — so a nudge whose refresh then fails leaves the credential
// exactly as it was, rather than leaving a still-valid token looking expired.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// NudgeWindow is how far before expiry a credential counts as "due" for a refresh. A spawn inside
// this window rotates on its own, so no nudge is needed; outside it, only a forced caller nudges.
const NudgeWindow = 30 * time.Minute

// CredentialDueForRefresh reports whether dir's credential is missing, already expired, or expires
// within NudgeWindow — i.e. whether a plain TriggerRefresh would rotate it on its own. ok is false
// when there is no readable expiry to judge (treated as due, since an unreadable credential is not
// something to declare healthy).
func CredentialDueForRefresh(dir string, now time.Time) bool {
	exp, ok := credExpiry(filepath.Join(dir, ".credentials.json"))
	if !ok {
		return true
	}
	return !exp.After(now.Add(NudgeWindow))
}

// CredentialExpiry returns dir's recorded credential expiry, or ok=false when there is none to
// read. Exported so a caller can REPORT remaining lifetime rather than only branch on it.
func CredentialExpiry(dir string) (time.Time, bool) {
	return credExpiry(filepath.Join(dir, ".credentials.json"))
}

// NudgeExpiryForRefresh backdates dir's credential expiry so the next `claude -p` must refresh,
// and returns a restore closure that rewrites the original file bytes verbatim.
//
// It refuses to touch a credential it cannot parse or that carries no access token: a torn or
// hollow file is not something to rewrite, and the caller's "did the family move" check will
// report the real problem. The restore closure is safe to call unconditionally (nil when there is
// nothing to restore) and is idempotent.
func NudgeExpiryForRefresh(dir string, now time.Time) (restore func() error, err error) {
	path := filepath.Join(dir, ".credentials.json")
	original, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Round-trip through a generic map so every field we do not understand survives untouched.
	var doc map[string]any
	if err := json.Unmarshal(original, &doc); err != nil {
		return nil, fmt.Errorf("credential at %s is not parseable JSON: %w", path, err)
	}
	block, ok := doc["claudeAiOauth"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("credential at %s carries no claudeAiOauth block", path)
	}
	if tok, _ := block["accessToken"].(string); tok == "" {
		return nil, fmt.Errorf("credential at %s carries no access token (hollow); nothing to refresh", path)
	}

	block["expiresAt"] = now.Add(-time.Minute).UnixMilli()
	doc["claudeAiOauth"] = block
	nudged, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	info, statErr := os.Stat(path)
	mode := os.FileMode(0o600)
	if statErr == nil {
		mode = info.Mode().Perm()
	}
	if err := os.WriteFile(path, nudged, mode); err != nil {
		return nil, err
	}
	return func() error { return os.WriteFile(path, original, mode) }, nil
}
