package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/term"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/secretload"
)

const guardAnthropicOAuthSecretKey = "CLAUDE_SUBSCRIPTION_OAUTH_TOKEN"

// logDojoEpisodeStart records the start of a live dojo episode when --dojo is
// enabled. Shutdown writes a second completed row if the gateway observed a
// provider-cache turn window.
func logDojoEpisodeStart(mode string) error {
	return logDojoEpisodeFile(mode, nil)
}

// findRepoRoot walks up from start to the nearest dir containing .git; falls back to start.
func findRepoRoot(start string) string {
	cur := filepath.Clean(start)
	for {
		if _, err := os.Stat(filepath.Join(cur, ".git")); err == nil {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return start
		}
		cur = parent
	}
}

type guardOAuthEnvSource struct {
	key string
	env string
}

func (s guardOAuthEnvSource) Name() string { return "$" + s.env }

func (s guardOAuthEnvSource) Lookup(key string) (string, bool) {
	if key != s.key || s.env == "" {
		return "", false
	}
	v := strings.TrimSpace(os.Getenv(s.env))
	return v, v != ""
}

type guardOAuthCredentialsSource struct {
	key  string
	path string
	now  func() time.Time
	warn io.Writer
}

func (s guardOAuthCredentialsSource) Name() string { return s.path }

func (s guardOAuthCredentialsSource) Lookup(key string) (string, bool) {
	if key != s.key {
		return "", false
	}
	now := time.Now
	if s.now != nil {
		now = s.now
	}
	// Claude Code refreshes this file ~hourly by rewriting it, so a read can race the
	// rewrite and catch a torn/empty/partial body that fails to parse. The window closes
	// in microseconds, so when the file EXISTS but the current read does not yield a token,
	// retry a few times over a few ms before giving up — that keeps a transient torn read
	// from being reported as "no active login" and falling through to the sibling
	// .oauth-token (a DIFFERENT, possibly-stale setup token). A genuinely-absent file (the
	// first os.Stat error) still misses immediately.
	const tornReadRetries = 3
	for attempt := 0; ; attempt++ {
		v, ok, transient := s.readOnce(now)
		if ok || !transient || attempt >= tornReadRetries {
			return v, ok
		}
		time.Sleep(15 * time.Millisecond)
	}
}

// readOnce performs a single resolve of the credentials file. It returns the token and
// ok=true on success; on failure, transient reports whether the failure looks like a
// torn/racing read of an EXISTING file (worth a brief retry) versus a definitive miss —
// an absent file, or a present-but-expired token (which must NOT be sent: an expired
// bearer 401s, so it is treated as absent so a fresher source or the per-request 401
// refresh can take over).
func (s guardOAuthCredentialsSource) readOnce(now func() time.Time) (tok string, ok bool, transient bool) {
	b, err := os.ReadFile(s.path)
	if err != nil {
		// A missing file is a definitive miss; any other read error (a momentary
		// permission/lock blip during the rewrite) is worth a short retry.
		return "", false, !os.IsNotExist(err)
	}
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken string `json:"accessToken"`
			ExpiresAt   int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		// File exists but does not parse — a torn read mid-rewrite. Retry.
		return "", false, true
	}
	v := strings.TrimSpace(doc.ClaudeAIOauth.AccessToken)
	if v == "" {
		// Parsed but no token (truncated-but-valid JSON, or a transitional empty write).
		return "", false, true
	}
	// An expired access token is a KNOWN-BAD bearer (the upstream 401s on it). Treat it as
	// absent rather than send it: a higher-priority source already lost, so falling through
	// lets the long-lived .oauth-token setup token answer, and the per-request 401 refresh
	// (agent.HTTPPlanner) re-reads the file once Claude Code rewrites the rotated token in.
	if exp := doc.ClaudeAIOauth.ExpiresAt; exp > 0 && exp < now().UnixMilli() {
		if s.warn != nil {
			fmt.Fprintf(s.warn, "fak guard: WARNING — the OAuth token in %s expired; Claude Code normally refreshes it. Re-run `claude` once, or use `claude setup-token` for a long-lived token.\n", s.path)
		}
		return "", false, false
	}
	return v, true, false
}

// credExpiresAt reads the .credentials.json accessToken's expiresAt WITHOUT collapsing an
// expired token to "absent" (unlike readOnce, which the Lookup contract requires to do): the
// #1183 StaleCred rehydrate rung (accounts.CredFreshness / CredCheck) needs the raw freshness
// fact — is the token live right now — not the "safe to send" verdict. ok is false when the
// file is missing/unparseable/carries no token (nothing to judge); a token with expiresAt<=0
// (no expiry recorded) is treated as always-fresh, matching Claude Code's own convention of
// omitting expiresAt for a token that does not rotate.
func credExpiresAt(path string) (expiresAt time.Time, ok bool) {
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
		return time.Time{}, true // no expiry recorded: treat as never-expiring
	}
	return time.UnixMilli(doc.ClaudeAIOauth.ExpiresAt), true
}

type guardOAuthFileSource struct {
	key  string
	path string
}

func (s guardOAuthFileSource) Name() string { return s.path }

func (s guardOAuthFileSource) Lookup(key string) (string, bool) {
	if key != s.key {
		return "", false
	}
	b, err := os.ReadFile(s.path)
	if err != nil {
		return "", false
	}
	v := strings.TrimSpace(string(b))
	return v, v != ""
}

func guardAnthropicOAuthLoader(tokenEnv, cfgDir string, now func() time.Time, warn io.Writer) (*secretload.Loader, []string) {
	tried := make([]string, 0, 3)
	sources := make([]secretload.SecretSource, 0, 3)

	if tokenEnv != "" {
		tried = append(tried, "$"+tokenEnv)
		sources = append(sources, guardOAuthEnvSource{key: guardAnthropicOAuthSecretKey, env: tokenEnv})
	}

	credPath := filepath.Join(cfgDir, ".credentials.json")
	tried = append(tried, credPath)
	sources = append(sources, guardOAuthCredentialsSource{
		key:  guardAnthropicOAuthSecretKey,
		path: credPath,
		now:  now,
		warn: warn,
	})

	setupPath := filepath.Join(cfgDir, ".oauth-token")
	tried = append(tried, setupPath)
	sources = append(sources, guardOAuthFileSource{key: guardAnthropicOAuthSecretKey, path: setupPath})

	l := secretload.New(sources...)
	l.Require(guardAnthropicOAuthSecretKey, "Claude Pro/Max subscription OAuth token", nil)
	return l, tried
}

// resolveAnthropicOAuthToken finds a Claude Pro/Max SUBSCRIPTION OAuth token to
// authenticate the upstream with, in priority order:
//  1. the named env var (default CLAUDE_CODE_OAUTH_TOKEN) — the explicit
//     headless/automation override;
//  2. <claude-config>/.credentials.json -> claudeAiOauth.accessToken — the active
//     Claude Code login token. This mirrors the credential direct Claude Code is
//     using right now, which matters because a stale or org-disallowed setup token
//     can exist beside a working interactive login;
//  3. <claude-config>/.oauth-token — a long-lived setup-token file. This remains
//     the fallback for headless homes with no interactive login, and callers that
//     want to force a specific setup token can still put it in tokenEnv.
//
// <claude-config> is $CLAUDE_CONFIG_DIR (first entry if it is a list) when set,
// else ~/.claude. Returns the token and a human source label, or an error that
// names every place it looked so the operator can fix the setup.
func resolveAnthropicOAuthToken(tokenEnv string) (token, source string, err error) {
	// Boot-time/diagnostic callers want the expired-token WARNING on stderr (it fires at most
	// once per resolve here). The hot per-request refresh path must pass io.Discard via
	// resolveAnthropicOAuthTokenWarn so a genuinely-expired credential does not reprint the
	// multi-line warning every turn and bury the agent's output.
	return resolveAnthropicOAuthTokenWarn(tokenEnv, os.Stderr)
}

// resolveAnthropicOAuthTokenWarn is resolveAnthropicOAuthToken with the expired-token warning
// sink made explicit. The credential file's expiry warning (guardOAuthCredentialsSource.warn)
// is invaluable ONCE at startup but becomes stderr spam when re-emitted on the per-request
// rotation re-read (apiKeyFunc in cmdGuard), so that path passes io.Discard while the boot
// path passes os.Stderr. Routing is unchanged — same loader, same source precedence.
func resolveAnthropicOAuthTokenWarn(tokenEnv string, warn io.Writer) (token, source string, err error) {
	loader, tried := guardAnthropicOAuthLoader(tokenEnv, guardClaudeConfigDir(), time.Now, warn)
	if v, src, ok := loader.LookupSource(guardAnthropicOAuthSecretKey); ok {
		return v, src, nil
	}

	return "", "", fmt.Errorf("no Claude subscription OAuth token found (looked in: %s). Log into Claude Code (`claude`), or create a long-lived one with `claude setup-token` and export it as %s", strings.Join(tried, ", "), tokenEnv)
}

// guardSubscriptionLoginPresent reports whether a Claude subscription login EXISTS on disk,
// independent of whether its token is readable RIGHT NOW. Claude Code rewrites
// <claude-config>/.credentials.json roughly hourly and the OAuth access token it holds is
// short-lived, so a single boot-time read can legitimately catch the file mid-rotation (or
// holding a just-expired token) and miss — even though a live login is there and the token
// will rotate back in within seconds. resolveAnthropicOAuthToken correctly returns "absent"
// in that window (a torn/expired read must NOT be sent as a bearer), but the guard's
// pin-vs-passthrough boot decision must NOT read that transient miss as "no subscription":
// demoting to passthrough strips the placeholder that keeps the wrapped agent out of its own
// /login, so the agent hangs on a login prompt for a session that would have recovered on the
// first per-request token re-resolve (the 'stuck on login sometimes' race). This is the cheap
// disk witness that separates "a login is present, the token is just briefly unreadable" (pin
// on intent; the per-request APIKeyFunc recovers the fresh token) from "no subscription at
// all" (genuinely fall back to passthrough). The named env override (CLAUDE_CODE_OAUTH_TOKEN)
// counts as present when set. Existence only — it never reads or validates the token.
func guardSubscriptionLoginPresent(tokenEnv string) bool {
	if tokenEnv != "" && strings.TrimSpace(os.Getenv(tokenEnv)) != "" {
		return true
	}
	cfgDir := guardClaudeConfigDir()
	if accounts.DeriveIdentity(cfgDir).HasCreds {
		return true
	}
	if fi, err := os.Stat(filepath.Join(cfgDir, ".oauth-token")); err == nil && !fi.IsDir() {
		return true
	}
	return false
}

// cmdGuardStdinInteractive reports whether the guard process's stdin is a real interactive
// terminal where a user could complete an OAuth /login. The headless no-token fail-loud gate
// uses this so an attended user is never blocked, while an automated/headless run — where a
// blocked login is unrecoverable — fails loud with guidance instead of hanging.
//
// It uses term.IsTerminal, NOT the stdlib os.ModeCharDevice test: on Windows a redirected
// stdin (`NUL` / `< /dev/null`) reports AS a character device, so a FileMode check treats the
// exact headless-automation case as interactive and the gate never fires (caught by field
// test, not the unit test). term.IsTerminal calls GetConsoleMode on Windows / isatty on Unix,
// which distinguishes a console from a redirected handle.
func cmdGuardStdinInteractive() bool {
	return guardFdIsTerminal(int(os.Stdin.Fd()))
}

// guardFdIsTerminal is the seam over term.IsTerminal, named so the gate can be reasoned about
// (and the real os.Stdin fd swapped in tests) without depending on the test's own stdin.
func guardFdIsTerminal(fd int) bool {
	return term.IsTerminal(fd)
}

// guardDebugStatsToSharedStderr decides whether the per-turn `fak-turn …` economy line
// streams to the SHARED terminal stderr. The line is invaluable on a headless / piped /
// scripted run, but on an ATTENDED interactive launch the wrapped agent (Claude Code)
// paints a full-screen alternate-screen TUI over THIS same terminal — so a per-turn stderr
// write lands on top of the agent's UI and corrupts the session view. There the economy
// belongs in the `fak info` split pane (the dedicated fak section), the exit summary, and
// /metrics, not bleeding into the agent pane.
//
// Precedence, highest first: --debug-stats=false / --quiet silence everything (false). An
// EXPLICIT --debug-stats (userSet) is a knowing opt-in and always streams here (true).
// Otherwise an attended interactive child sharing this terminal auto-suppresses the
// shared-stderr stream (false — the pane + exit summary still carry it); a headless / piped
// launch keeps it (true — there is no full-screen UI to corrupt and a captured log helps).
func guardDebugStatsToSharedStderr(debugStats, quiet, userSet, stdinInteractive, childInteractive bool) bool {
	if !debugStats || quiet {
		return false
	}
	if userSet {
		return true
	}
	return !(stdinInteractive && childInteractive)
}

// guardClaudeConfigDir resolves the directory that holds Claude Code's per-account
// credentials: $CLAUDE_CONFIG_DIR (first path if it is an OS-list) when set, else
// ~/.claude. A home that cannot be resolved degrades to the literal ".claude".
func guardClaudeConfigDir() string {
	if v := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); v != "" {
		if i := strings.IndexByte(v, os.PathListSeparator); i >= 0 {
			v = v[:i]
		}
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".claude"
	}
	return filepath.Join(home, ".claude")
}

// printGuardBanner explains exactly what is now in front of the agent: where the
// gateway is, what it proxies to, which floor is loaded, the single env var injected
// into the child, and WHERE TO WATCH IT — the live metrics/debug endpoints, the durable
// audit journal, and the structured log stream. It goes to stderr so it never pollutes a
// `-p` JSON run the child writes to stdout.
func printGuardBanner(w io.Writer, version, buildStamp, gwURL, provider, baseURL, floorSource, injectVar, injectVal, logLabel, auditLabel string, refusalCarryForward []guardRefusalCarry, remoteServe, local bool, localLabel string, command []string) {
	fmt.Fprintf(w, "fak guard %s — kernel-adjudicated: %s\n", version, strings.Join(command, " "))
	// The embedded build stamp, not the version, is the reliable "is THIS guard binary current?"
	// signal: the version reads the tree's VERSION file, so a stale binary still looks current
	// (the screenshot confusion). A +uncommitted marker means the running guard was built from a
	// dirty tree. See guardBannerBuildStamp.
	fmt.Fprintf(w, "  build      : %s\n", buildStamp)
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	switch {
	case local:
		// The model runs IN-KERNEL on this box; there is no upstream API call at all. Say so
		// plainly — it is the headline of `fak guard --gguf`: a local model + your harness.
		fmt.Fprintf(w, "  upstream   : in-kernel %s   (LOCAL — fak runs the model itself; no API key, no network) via the %s wire\n", localLabel, provider)
	case remoteServe:
		// Tell the operator the dev turn's INFERENCE is on the lab box they chose, not a
		// public API — the whole point of --remote-serve.
		fmt.Fprintf(w, "  upstream   : %s   (remote fak serve on a lab box, %s wire)\n", baseURL, provider)
	default:
		fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
	}
	fmt.Fprintf(w, "  floor      : %s\n", floorSource)
	fmt.Fprintf(w, "  wired via  : %s=%s   (child only — your shell is untouched)\n", injectVar, injectVal)
	// Observability: the live scrape surfaces are on the gateway URL above (unauth on
	// loopback); the audit journal is ON by default (auditLabel says where), the log
	// stream survives the session only if asked for.
	fmt.Fprintf(w, "  metrics    : %s/metrics  ·  %s/debug/vars  ·  %s/v1/fak/events\n", gwURL, gwURL, gwURL)
	// Point operators at the cache-value metric family by name — it lives on /metrics
	// above, but nothing told them to scrape for it (#1077, epic #1072). These are the
	// numbers that answer "what did fak's owned KV cache actually save this session?".
	fmt.Fprintf(w, "  cache value: scrape %s/metrics for the fak_vcache_* family (saved_token_equiv, hit_rate, multiplier, proven)\n", gwURL)
	fmt.Fprintf(w, "  audit log  : %s\n", auditLabel)
	fmt.Fprint(w, formatGuardRefusalCarryForward(refusalCarryForward))
	fmt.Fprintf(w, "  gateway log: %s\n", logLabel)
	fmt.Fprintln(w, "  every tool call the agent proposes crosses the capability floor before it runs.")
}

// guardLogSink builds the gateway's structured-log destination from the --log value.
// "" (default) mutes it (a no-op) to keep the wrapped agent's terminal clean; "-" or
// "stderr" streams it to stderr; any other value appends to that file. It returns the
// log function, an optional closer (the opened file), and a human label for the banner.
// A file that cannot be opened is fatal — an operator who asked for a log and silently
// got none is worse than a loud failure.
func guardLogSink(logPath string, stderr io.Writer) (logf func(string, ...any), closer io.Closer, label string) {
	switch strings.TrimSpace(logPath) {
	case "":
		return func(string, ...any) {}, nil, "off (--log FILE or --log - to enable)"
	case "-", "stderr":
		lg := log.New(stderr, "fak-gateway ", log.LstdFlags|log.Lmsgprefix)
		return lg.Printf, nil, "stderr"
	default:
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		must(err)
		lg := log.New(f, "", log.LstdFlags)
		return lg.Printf, f, logPath
	}
}

// guardAuditPlan is the PURE decision behind guard's default-on audit journal: it
// returns the path to enable (""=> do not enable) and whether the off was an
// explicit opt-out, given the flags and whether a journal is already active at
// boot (FAK_AUDIT_JOURNAL). Kept side-effect-free so the precedence — boot env
// wins, then --no-audit / --audit off, then --audit PATH, then the repo-local
// default — is unit-tested without touching the process-global journal.
func guardAuditPlan(auditPath string, noAudit, bootActive bool) (enablePath string, optedOut bool) {
	if bootActive {
		return "", false // FAK_AUDIT_JOURNAL already registered an emitter; nothing to enable
	}
	if noAudit || strings.EqualFold(strings.TrimSpace(auditPath), "off") {
		return "", true
	}
	p := strings.TrimSpace(auditPath)
	if p == "" {
		p = guardDefaultAuditPath()
	}
	return p, false
}

// guardDefaultAuditPath is where fak guard writes its durable decision journal
// when the operator names none: .dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl.
// That is the same repo-local discovery root guardrsi folds, so an attended
// `fak guard -- claude` refusal has a per-session witness row beside fleet runs.
func guardDefaultAuditPath() string {
	root := findRepoRoot(".")
	return filepath.Join(guardAuditDir(root),
		fmt.Sprintf("interactive-%d-%s.jsonl", os.Getpid(), guardAuditPathHash(root)))
}

func guardAuditDir(root string) string {
	return filepath.Join(root, ".dispatch-runs", "guard-audit")
}

func guardPolicyDigest(policyBytes []byte) string {
	sum := sha256.Sum256(policyBytes)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func guardAuditPathHash(root string) string {
	abs, err := filepath.Abs(root)
	if err == nil {
		root = abs
	}
	sum := sha256.Sum256([]byte(filepath.Clean(root)))
	return hex.EncodeToString(sum[:])[:12]
}

// guardReadableAuditPath is the reader-side default used by diagnostics and the
// guard TUI. A separate process cannot know another guard session's PID-derived
// writer path, so readers prefer the newest repo-local journal when the operator
// did not name FAK_AUDIT_JOURNAL.
func guardReadableAuditPath() string {
	if p := strings.TrimSpace(os.Getenv("FAK_AUDIT_JOURNAL")); p != "" {
		return p
	}
	if p := latestGuardAuditJournalPath(); p != "" {
		return p
	}
	return guardDefaultAuditPath()
}

func latestGuardAuditJournalPath() string {
	dir := guardAuditDir(findRepoRoot("."))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var bestPath, bestName string
	var bestMod time.Time
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".jsonl") {
			continue
		}
		info, err := ent.Info()
		if err != nil {
			continue
		}
		mod := info.ModTime()
		if bestPath == "" || mod.After(bestMod) || (mod.Equal(bestMod) && ent.Name() > bestName) {
			bestName = ent.Name()
			bestMod = mod
			bestPath = filepath.Join(dir, ent.Name())
		}
	}
	return bestPath
}

// guardEnableAudit turns the durable, hash-chained decision journal ON for the
// session per guardAuditPlan and returns a human label for the banner plus the
// active journal (nil when disabled). A failure to open a REQUESTED path is fatal
// (must) — an operator who asked for an audit trail and silently got none is worse
// than a loud failure, mirroring guardLogSink's file-sink contract.
func guardEnableAudit(auditPath string, noAudit bool) (label string, active *journal.Journal) {
	// A boot-time FAK_AUDIT_JOURNAL already registered an emitter we cannot
	// unregister; respect it (and note --no-audit cannot turn it off).
	if j := journal.Active(); j != nil {
		if p := j.Path(); p != "" {
			return p + "  (durable, hash-chained; from FAK_AUDIT_JOURNAL)", j
		}
		return "active  (durable, hash-chained; from FAK_AUDIT_JOURNAL)", j
	}
	path, optedOut := guardAuditPlan(auditPath, noAudit, false)
	if path == "" {
		if optedOut {
			return "off  (default-on; disabled by --no-audit / --audit off)", nil
		}
		return "off", nil
	}
	j, err := journal.Enable(path)
	must(err)
	return path + "  (durable, hash-chained — verify with: fak audit verify <path>)", j
}
