package accounts

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// credrefresh.go — proactively TRIGGER Claude Code's own OAuth refresh for a headless seat,
// instead of only DETECTING staleness and waiting for something else to rewrite the file.
//
// WHY THIS EXISTS. A seat's <config-dir>/.credentials.json holds a claudeAiOauth block with an
// accessToken that expires ~every 8h AND a refreshToken that can mint a fresh one. Interactive
// Claude Code rotates the token in place before it lapses; a HEADLESS/subprocess `claude` does
// not reliably do so (a known upstream bug), and fak's guard historically only POLLED/PARKED
// waiting for a concurrent `claude` or a human to rewrite the file — so an idle seat's token
// died and a manual /login was the only recovery.
//
// The fix is deliberately NOT to POST the OAuth refresh grant ourselves: Claude Pro/Max OAuth
// credentials are, per Anthropic's policy, for Claude Code and claude.ai — fak minting a token
// with Claude Code's client id would be exactly the restricted third-party use, on a public
// repo, and brittle to a client-id rotation. Instead we CAUSE the refresh the guard already
// waits for: run one minimal `claude -p` against the seat's config dir, which makes Claude Code
// refresh ITS OWN .credentials.json, then re-read the on-disk expiry to witness the rotation.
// This is the exact recovery the code already documents ("run `claude` once … (refreshes the
// token)", cmd/fak/guard.go) — we just trigger it proactively rather than hope it happens.
//
// The ground truth is the file, never the spawn's exit code: refreshed is true only when the
// on-disk expiry actually ADVANCED. A spawn that ran but did not rotate the token (or that
// cleared the file) reports refreshed=false, so the caller falls through to its human-relogin
// backstop rather than trusting a no-op.

// RefreshSpawn runs one credential-refreshing `claude` invocation against cfgDir and returns
// when it has finished (or ctx expired). It is the injected seam so tests never exec a real
// binary. The production implementation is DefaultRefreshSpawn.
type RefreshSpawn func(ctx context.Context, cfgDir string) error

// refreshEnvBlockers are the environment variables that, when set, make `claude -p` bill an API
// key and CLEAR the seat's .credentials.json instead of refreshing the subscription OAuth token
// (observed in the account-lifecycle runbook). The refresh spawn MUST run with all of them unset
// or it can destroy the very credential it means to refresh. CLAUDE_CODE_OAUTH_TOKEN is included
// so the child refreshes the config dir's own on-disk credential, not an ambient pinned token.
var refreshEnvBlockers = []string{
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_BASE_URL",
	"CLAUDE_CODE_OAUTH_TOKEN",
}

// refreshModel is the cheapest model for the throwaway refresh turn — the request's only purpose
// is to make Claude Code rotate its token; the answer is discarded.
const refreshModel = "claude-haiku-4-5-20251001"

// TriggerRefresh causes Claude Code to refresh the OAuth credential in cfgDir and reports whether
// the on-disk token actually rotated. It reads the credential's expiry, runs spawn (default:
// DefaultRefreshSpawn — a minimal `claude -p` with the API-key/base-url env scrubbed), then
// re-reads the expiry: refreshed is true iff the new expiry is strictly later than the old one.
// A nil spawn uses DefaultRefreshSpawn; a nil now uses time.Now. The returned err is the spawn's
// error (surfaced, never swallowed) — but refreshed can still be false with a nil err when the
// spawn ran cleanly yet did not advance the expiry (e.g. the refresh token itself is dead), which
// the caller treats as "route to the human relogin backstop", not a hard failure.
func TriggerRefresh(ctx context.Context, cfgDir string, spawn RefreshSpawn, now func() time.Time) (refreshed bool, err error) {
	if now == nil {
		now = time.Now
	}
	if spawn == nil {
		spawn = DefaultRefreshSpawn
	}
	if strings.TrimSpace(cfgDir) == "" {
		return false, nil
	}
	credPath := filepath.Join(cfgDir, ".credentials.json")
	before, hadBefore := credExpiry(credPath)

	spawnErr := spawn(ctx, cfgDir)

	after, hasAfter := credExpiry(credPath)
	if !hasAfter {
		// The spawn left no parseable credential — including the API-key failure mode that
		// CLEARS the file. Nothing to vouch for; surface any spawn error, refreshed stays false.
		return false, spawnErr
	}
	if !hadBefore {
		// No prior expiry to compare against (missing/torn/no-expiry before): treat a now-present
		// future-dated token as a refresh, so a first-time rotation is not silently missed.
		return after.After(now()), spawnErr
	}
	return after.After(before), spawnErr
}

// credExpiry reads the claudeAiOauth.expiresAt (epoch MILLISECONDS) from a .credentials.json path.
// ok is false when the file is missing, unparseable (a torn read mid-rewrite), carries no token,
// or records no positive expiry — the raw freshness fact, not the send-safe verdict. It mirrors
// cmd/fak/guard_support.go's credExpiresAt; it lives here so this leaf stays free of a cmd import.
func credExpiry(path string) (time.Time, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return time.Time{}, false
	}
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken string `json:"accessToken"`
			ExpiresAt   int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return time.Time{}, false
	}
	if strings.TrimSpace(doc.ClaudeAIOauth.AccessToken) == "" {
		return time.Time{}, false
	}
	if doc.ClaudeAIOauth.ExpiresAt <= 0 {
		return time.Time{}, false
	}
	return time.UnixMilli(doc.ClaudeAIOauth.ExpiresAt), true
}

// DefaultRefreshSpawn runs `claude -p "ok" --model <haiku>` with CLAUDE_CONFIG_DIR pinned to
// cfgDir and refreshEnvBlockers stripped, feeding empty stdin so the CLI does not stall on a
// pipe. Its sole purpose is the token-rotation side effect; stdout/stderr are discarded. The
// caller supplies the timeout via ctx (a short deadline is expected).
func DefaultRefreshSpawn(ctx context.Context, cfgDir string) error {
	cmd := exec.CommandContext(ctx, ClaudeExe(), "-p", "ok", "--model", refreshModel)
	cmd.Env = refreshEnv(os.Environ(), cfgDir)
	cmd.Stdin = strings.NewReader("")
	return cmd.Run()
}

// refreshEnv returns base with CLAUDE_CONFIG_DIR set to cfgDir and every refreshEnvBlockers entry
// removed. Split out (pure, over an explicit base slice) so a test can assert the scrub without
// spawning anything — the API-key/base-url scrub is the highest-risk line in the whole change.
func refreshEnv(base []string, cfgDir string) []string {
	blocked := make(map[string]bool, len(refreshEnvBlockers))
	for _, k := range refreshEnvBlockers {
		blocked[k] = true
	}
	out := make([]string, 0, len(base)+1)
	for _, kv := range base {
		name := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			name = kv[:i]
		}
		if blocked[name] {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "CLAUDE_CONFIG_DIR="+cfgDir)
}

// ClaudeExe resolves the claude binary from the fleet-wide convention: FLEET_CLAUDE_EXE, the
// FAK_CLAUDE_EXE back-compat fallback, PATH, then the conventional install path under $HOME. It
// is the one home for this resolution so the Go call sites (this refresh spawn, the resume
// watchdog's rwClaudeExe) and the Python/PowerShell launchers all agree via FLEET_CLAUDE_EXE.
func ClaudeExe() string {
	if v := strings.TrimSpace(os.Getenv("FLEET_CLAUDE_EXE")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_EXE")); v != "" {
		return v
	}
	for _, name := range []string{"claude", "claude.exe"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return "claude"
	}
	return filepath.Join(home, ".local", "bin", "claude")
}
