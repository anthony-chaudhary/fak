package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// saveAccountsRegistry persists reg to path, printing the shared "fak accounts: %v"
// error on failure. It returns false (the caller should then `return 1`) when the save
// failed - the save-and-report step repeated by every accounts mutation verb.
func saveAccountsRegistry(stderr io.Writer, path string, reg accounts.Registry) bool {
	if err := accounts.SaveRegistry(path, reg); err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return false
	}
	return true
}

func requireAccountsHome(stderr io.Writer, homeDir string) bool {
	if homeDir != "" {
		return true
	}
	fmt.Fprintln(stderr, "fak accounts: cannot resolve home dir")
	return false
}

// addParams carries the resolved flags for `fak accounts add` from the dispatcher.
type addParams struct {
	name     string
	reserved bool
	chrome   string
	noLogin  bool
	token    string
	suffix   string
	noSync   bool

	// baseURL and extraEnv enroll a seat that fronts a THIRD-PARTY Anthropic-compatible
	// endpoint. baseURL is that endpoint; extraEnv is the raw repeatable --env KEY=VALUE list,
	// parsed and validated (non-secret names only) by parseSeatExtraEnv before it reaches the
	// registry. Both empty is an ordinary first-party seat, unchanged.
	baseURL  string
	extraEnv []string

	// adopt enrolls by copying an EXISTING login bundle from `from` (default ~/.claude)
	// instead of running interactive `claude setup-token`; force reconciles an existing
	// target dir/registry row in place rather than refusing.
	adopt bool
	from  string
	force bool

	// apiKeyEnv, when non-empty, enrolls an API-KEY seat (CredKindAPIKey, #5331) instead of a
	// subscription-OAuth seat: it is the NAME of the environment variable holding the account's
	// Anthropic API key (e.g. "ANTHROPIC_API_KEY"). The registry stores ONLY this reference,
	// never the secret. This path runs no setup-token and copies no credential bundle - the
	// credential is the env var - so it is mutually exclusive with --adopt/--no-login/--token.
	apiKeyEnv string

	// probeIdentity (adopt only) reconciles the adopted seat's identity against the account its
	// live credential ACTUALLY serves - a network probe of the OAuth profile endpoint - and
	// prefers the credential over stale on-disk .claude.json metadata, overwriting the seeded
	// oauthAccount when they disagree. This is the fix for a seat whose .claude.json names one
	// account while its .credentials.json (a later /login into a shared dir) serves another.
	// The probe is now the DEFAULT for every adopt (the credential is live, so the ground truth
	// is always available and the same-network cost buys a deterministically-correct identity);
	// `enroll-current` forces it on, and a plain `add --adopt` gets it unless --no-probe-identity
	// is passed. probeIdentity only records that a caller EXPLICITLY forced it (enroll-current),
	// which matters for the "warn loudly on probe failure" policy below; the default-on decision
	// is `!noProbeIdentity`.
	probeIdentity bool
	// noProbeIdentity (adopt only) opts OUT of the default credential-identity probe, restoring the
	// historical disk-only derivation. Use it for a deliberately offline enrollment where no OAuth
	// endpoint is reachable and the copied .claude.json identity is trusted as-is. Ignored when
	// probeIdentity is set (enroll-current always probes).
	noProbeIdentity bool
	// probeURL overrides the OAuth profile endpoint (accounts.DefaultProfileURL when empty).
	// It exists as a test/advanced seam, sourced from $FAK_OAUTH_PROFILE_URL at the CLI layer.
	probeURL string

	// noDivorce (adopt only) opts OUT of the default post-copy token-family divorce. An adopt
	// COPIES the source's .credentials.json, so both dirs end up holding ONE OAuth refresh token -
	// and the first side to refresh rotates the family and silently 401s the other (witnessed
	// 2026-08-06: an enrolled seat's refresh logged the operator's own interactive session out,
	// its access token still hours from expiry). By default the enroll therefore refreshes the new
	// seat immediately, so the seat provably owns its own family and the source's now-dead
	// credential is reported at enroll time instead of detonating later. Pass this to keep the
	// copy byte-identical and control the timing yourself - the hazard then stays armed, which is
	// the whole reason it is opt-out rather than opt-in.
	noDivorce bool

	// divorceSpawn overrides the refresh spawn used by the family divorce (nil = the real
	// `claude -p`). Test seam only, mirroring accounts.RefreshSpawn.
	divorceSpawn accounts.RefreshSpawn

	// dryRun prints the enrollment plan and returns without any mutation - no dir created, no
	// credential copied, no OAuth probe, no registry write, no view sync (#3954). It short-circuits
	// after the read-only refusals (bad target, existing dir, duplicate name, missing source) so a
	// dry run still surfaces those, but performs zero writes and zero network calls.
	dryRun bool

	homeDir      string
	registryPath string
	dosView      string
	jobView      string
}

// runAccountsAdd is the end-to-end "enroll an account" flow. It is deliberately the
// ONLY place the multi-file account-enrollment runbook lives, so adding an account is one
// command instead of: hand-edit three rosters, hand-derive the uuid, work around the
// out-of-tree guard, remember the projects/ marker. The steps, in order:
//
//  1. resolve an ISOLATED config dir (~/.claude-<name>[-suffix]); refuse to clobber ~/.claude
//     or an existing dir, so a stray login never lands on the live session. With
//     --adopt --force an existing target dir is allowed (reconcile in place).
//  2. obtain the credential. Default: run `CLAUDE_CONFIG_DIR=<dir> claude setup-token`
//     (inheriting the TTY for the browser+paste), or read --token/stdin with --no-login, then
//     twin-check (GateTokenWrite) and write <dir>/.oauth-token. With --adopt: copy the EXISTING
//     login bundle (.credentials.json and/or .oauth-token) from the source seat (--from, else
//     ~/.claude) - the account you are already logged into becomes a rotation seat with no
//     setup-token. A copied .oauth-token is still twin-checked.
//  3. record identity. Default: probe the OAuth profile endpoint for the email + account UUID -
//     ground truth that also proves the credential works. With --adopt: derive identity from
//     the copied disk state (DeriveIdentity) - the credential is already a proven live login;
//     fall back to a token probe only if disk identity is empty and a token was copied.
//  4. seed the dir's markers so every consumer recognizes it: .claude.json (identity, so the
//     roster shows WHO it is, not "-"), projects/ (the fleet discovery gate), and settings.json
//     (the registry's defaults.settings, so the seat launches WITH the bypass/permission
//     defaults instead of "losing" them until a later sync).
//  5. upsert the canonical registry record (identity + policy) and SaveRegistry. Under --force
//     an existing row for the name is replaced in place, not duplicated.
//  6. regenerate the roster views (sync) so the dos + job rosters reflect the new account (this
//     also re-projects settings.json across the whole roster).
func runAccountsAdd(stdout, stderr io.Writer, p addParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts add --name <name> [--reserved] [--chrome-profile P] [--no-login [--token -]] [--adopt [--from <seat|dir>] [--force]]")
		return 2
	}
	if !requireAccountsHome(stderr, p.homeDir) {
		return 1
	}

	// API-KEY seat (#5331): --api-key-env names the env var holding the account's Anthropic API
	// key (a REFERENCE, never the secret). It is a distinct credential KIND with no setup-token
	// and no bundle to copy, so it is mutually exclusive with the OAuth acquisition flags, and
	// the value must look like an env-var NAME so a pasted key is refused before anything is written.
	apiKeyEnv := strings.TrimSpace(p.apiKeyEnv)
	if apiKeyEnv != "" {
		if p.adopt || p.noLogin || p.token != "" {
			fmt.Fprintln(stderr, "fak accounts: --api-key-env cannot be combined with --adopt/--no-login/--token (an api-key seat carries no setup-token or copied bundle)")
			return 1
		}
		if !accounts.ValidAPIKeyEnvName(apiKeyEnv) {
			fmt.Fprintf(stderr, "fak accounts: --api-key-env %q is not a valid env-var NAME (want e.g. ANTHROPIC_API_KEY; never paste the secret)\n", apiKeyEnv)
			return 2
		}
	}

	// Canonicalize the roster name to carry the suffix (the host convention, e.g. day26 ->
	// day26-netra), so the registry name matches the dir basename and `remove --name <name>`
	// uses the same handle the rosters show.
	rosterName := rosterAccountName(p.name, p.suffix)
	dir := accountDir(p.homeDir, p.name, p.suffix)
	// The TARGET must never be the live default seat - a new account gets a fresh, isolated
	// home so no login can clobber ~/.claude. This holds even under --force.
	if filepath.Clean(dir) == filepath.Clean(filepath.Join(p.homeDir, ".claude")) {
		fmt.Fprintln(stderr, "fak accounts: refusing to add into the default ~/.claude seat")
		return 1
	}
	// A never-clobber-an-existing-dir guard, so a stray add can't overwrite a live seat.
	// `--adopt --force` is the deliberate exception: it reconciles an already-seeded dir in
	// place (refresh creds, re-derive identity, upsert the row). A tombstoned (.DELETED) dir
	// is never a reconcile target.
	reconcile := (p.adopt || apiKeyEnv != "") && p.force
	if _, err := os.Stat(dir); err == nil {
		if !reconcile {
			fmt.Fprintf(stderr, "fak accounts: config dir already exists: %s (pick another --name, or pass --adopt --force to reconcile it)\n", dir)
			return 1
		}
		if strings.Contains(strings.ToLower(filepath.Base(dir)), ".deleted") {
			fmt.Fprintf(stderr, "fak accounts: refusing to reconcile a tombstoned (.DELETED) dir: %s\n", dir)
			return 1
		}
	}

	// Load the canonical registry up front so a duplicate name fails before we log in.
	reg := accounts.Registry{}
	if _, err := os.Stat(p.registryPath); err == nil {
		loaded, err := accounts.LoadRegistry(p.registryPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg = loaded
	}
	for _, h := range reg.Homes {
		if h.Name == rosterName && !reconcile {
			fmt.Fprintf(stderr, "fak accounts: %q is already in the registry (pass --adopt --force to reconcile it)\n", rosterName)
			return 1
		}
	}

	// Dry run (#3954): every durable mutation and network probe lives BELOW this point (MkdirAll,
	// SnapshotBeforeOverwrite, copyLoginBundle, the OAuth identity probe, the registry write, the
	// view sync). Short-circuit here - after the read-only refusals above have had their say - so a
	// dry run reports exactly what WOULD happen and touches nothing. The source seat is still
	// resolved (read-only) so an adopt from a missing source fails the same way it would for real.
	if p.dryRun {
		return dryRunAddPlan(stdout, stderr, p, reg, rosterName, dir, reconcile)
	}

	state := accountsAddState{
		params: p, registry: reg, rosterName: rosterName, dir: dir,
		apiKeyEnv: apiKeyEnv, reconcile: reconcile,
	}
	if code := acquireAccountsAddCredential(stdout, stderr, &state); code != 0 {
		return code
	}
	return commitAccountsAdd(stdout, stderr, &state)
}

// dryRunAddPlan prints the enrollment plan for `--dry-run` and returns without mutating anything.
// It performs only read-only work: resolving the adopt source (so a missing source still fails as it
// would for real) and describing the mutations the real run would perform. Kept in lockstep with the
// mutation sequence in runAccountsAdd - every "would" line names a step that lives below the dry-run
// short-circuit there.
func dryRunAddPlan(stdout, stderr io.Writer, p addParams, reg accounts.Registry, rosterName, dir string, reconcile bool) int {
	regVerb := "add"
	if reconcile {
		regVerb = "update (reconcile in place)"
	}
	fmt.Fprintf(stdout, "DRY RUN: no dir created, no credential copied, no probe, no registry write, no view sync\n")
	fmt.Fprintf(stdout, "  target dir:    %s\n", dir)
	fmt.Fprintf(stdout, "  roster name:   %s (reserved=%v)\n", rosterName, p.reserved)
	// A third-party-endpoint seat: show the endpoint and the overlay's variable NAMES, and
	// validate the overlay HERE too, so `--dry-run` catches a credential-shaped name before the
	// operator re-runs for real. Names only, never values - a dry-run plan is meant to be
	// pasteable, and ANTHROPIC_CUSTOM_HEADERS carries arbitrary header text.
	if base := strings.TrimSpace(p.baseURL); base != "" {
		fmt.Fprintf(stdout, "  endpoint:      %s (third-party Anthropic-compatible; launch requires --guard=false)\n", base)
	}
	if extra, err := parseSeatExtraEnv(p.extraEnv); err != nil {
		fmt.Fprintf(stderr, "fak accounts add: %v\n", err)
		return 2
	} else if len(extra) > 0 {
		fmt.Fprintf(stdout, "  seat env:      would set %s (values not shown)\n", strings.Join(sortedMapKeys(extra), ", "))
	}
	if env := strings.TrimSpace(p.apiKeyEnv); env != "" {
		// API-key seat (#5331): the credential is the env-var REFERENCE; nothing is minted or copied.
		fmt.Fprintf(stdout, "  credential:    would record the env-var REFERENCE $%s (kind=api_key; the key itself is never stored)\n", env)
		fmt.Fprintf(stdout, "  identity:      would derive from the key reference (offline; no OAuth profile probe)\n")
	} else if p.adopt {
		// A missing/invalid source is a read-only refusal that must fire under --dry-run too.
		src, srcOK := resolveAdoptSource(stderr, p, reg)
		if !srcOK {
			return 1
		}
		fmt.Fprintf(stdout, "  credential:    would ADOPT the login bundle from %s\n", src)
		if p.noProbeIdentity {
			fmt.Fprintf(stdout, "  identity:      would derive from copied disk state (probe disabled)\n")
		} else {
			fmt.Fprintf(stdout, "  identity:      would PROBE the live credential's OAuth profile to record the true account\n")
		}
	} else {
		fmt.Fprintf(stdout, "  credential:    would MINT a new setup-token via `claude setup-token`\n")
		fmt.Fprintf(stdout, "  identity:      would PROBE the new token's OAuth profile\n")
	}
	fmt.Fprintf(stdout, "  registry:      would %s %q -> %s\n", regVerb, rosterName, dir)
	if p.noSync {
		fmt.Fprintf(stdout, "  views:         sync SKIPPED (--no-sync)\n")
	} else {
		fmt.Fprintf(stdout, "  views:         would regenerate the dos + job roster views and re-verify the seat is servable\n")
	}
	fmt.Fprintf(stdout, "re-run without --dry-run to apply - ~/.claude untouched\n")
	return 0
}

// verifyServableAfterSync re-reads the registry (Refresh()ed, so LoginStatus reflects the just-copied
// dir) and the rendered view files, then asserts the new seat is serveable and present in each
// CONFIGURED view. It only warns - never changes the exit code - because it runs after the enroll has
// already committed; its job is to turn a silent "enrolled but nothing will call it" into a visible
// one. A disabled view (empty path) is not counted against servability.
func verifyServableAfterSync(stdout, stderr io.Writer, p addParams, rosterName string) {
	reg, err := accounts.LoadRegistry(p.registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: warning: could not re-verify servability: %v\n", err)
		return
	}
	reg = reg.Refresh()
	dosText := readViewFileForVerify(p.dosView)
	jobText := readViewFileForVerify(p.jobView)
	rep := accounts.VerifySeatServable(reg, rosterName, dosText, jobText)
	var problems []string
	if !rep.DosServable {
		problems = append(problems, fmt.Sprintf("not serveable (login status %q)", rep.LoginStatus))
	}
	if p.dosView != "" && !rep.InDosView {
		problems = append(problems, "missing from the dos roster view")
	}
	if p.jobView != "" && !rep.InJobView {
		problems = append(problems, "missing from the job roster view")
	}
	if len(problems) == 0 {
		switch {
		case p.dosView != "" && p.jobView != "":
			fmt.Fprintf(stdout, "servable: seat %q is ready in both roster views\n", rosterName)
		case p.dosView != "" || p.jobView != "":
			fmt.Fprintf(stdout, "servable: seat %q is ready in the configured roster view\n", rosterName)
		default:
			fmt.Fprintf(stdout, "servable: seat %q is serveable (no roster views configured to check)\n", rosterName)
		}
		return
	}
	fmt.Fprintf(stderr, "fak accounts: warning: seat %q enrolled but NOT yet servable - %s\n", rosterName, strings.Join(problems, "; "))
	fmt.Fprintln(stderr, "  run `fak accounts status --probe` to inspect, or `fak accounts sync` to regenerate the views")
}

// readViewFileForVerify reads a rendered view file for the servability check, returning "" (treated
// as "not checked") for an unconfigured or unreadable view rather than failing.
func readViewFileForVerify(path string) string {
	if path == "" {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

// resolveSourceSeat resolves the --from source for an adopt: an empty value is the default
// ~/.claude seat; a value that names an existing directory (or is path-shaped) is used as-is;
// otherwise it is treated as a seat NAME and resolved through the registry (its Dir), falling
// back to the ~/.claude-<name>[-suffix] convention. It never returns a nonexistent dir without
// saying so, so an adopt fails loudly rather than copying from thin air.
func resolveSourceSeat(homeDir, from string, reg accounts.Registry) (string, error) {
	if strings.TrimSpace(from) == "" {
		return filepath.Join(homeDir, ".claude"), nil
	}
	from = strings.TrimSpace(from)
	// A path-shaped value (separator, ~, or an existing dir) is taken literally.
	if strings.ContainsAny(from, `/\`) || strings.HasPrefix(from, "~") {
		clean := filepath.Clean(pathutil.ExpandTilde(from))
		if fi, err := os.Stat(clean); err != nil || !fi.IsDir() {
			return "", fmt.Errorf("--from source dir not found: %s", clean)
		}
		return clean, nil
	}
	// A bare name: prefer the registry's recorded Dir for that seat, else the dir convention.
	for _, h := range reg.Homes {
		if h.Name == from && h.Dir != "" {
			if fi, err := os.Stat(h.Dir); err == nil && fi.IsDir() {
				return h.Dir, nil
			}
		}
	}
	// "default" is the well-known handle for ~/.claude.
	if from == "default" {
		return filepath.Join(homeDir, ".claude"), nil
	}
	cand := accountDir(homeDir, from, firstNonEmpty(os.Getenv("FAK_ACCOUNT_SUFFIX"), "-netra"))
	if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
		return cand, nil
	}
	return "", fmt.Errorf("--from %q: no such seat (not a registry name, and %s does not exist)", from, cand)
}

// copyLoginBundle copies an EXISTING login's credential bundle from src into dst for an adopt.
// It copies whichever of .credentials.json (the auto-refreshing session) and .oauth-token (the
// static setup-token) the source carries, at 0600, and seeds dst/.claude.json's oauthAccount
// from the source's identity so the seat shows WHO it is. At least one credential file must
// exist, else it errors (an adopt from a not-logged-in dir is refused rather than half-done).
// It returns the human labels of what it copied for the summary line.
//
// homeRoot is the dir under which sibling seats are discovered; when set, a source .oauth-token
// that is a KNOWN cross-account twin (its fingerprint already belongs to a DIFFERENT account's
// live login among the siblings) is deliberately SKIPPED as long as the source also carries a
// live .credentials.json - that session is the real credential, and carrying the twin would only
// trip the GateTokenWrite smear guard and burn the other account's rate-limit bucket. The
// skipped token's short fingerprint is returned so the caller can explain the omission. Passing
// "" for homeRoot disables the sibling lookup and copies both files (the prior behavior).
func copyLoginBundle(src, dst, homeRoot string) (copied []string, skippedTwin string, err error) {
	skipToken := twinTokenToSkip(src, homeRoot)
	for _, name := range []string{".credentials.json", ".oauth-token"} {
		if name == ".oauth-token" && skipToken != "" {
			skippedTwin = skipToken
			continue
		}
		sp := filepath.Join(src, name)
		b, rerr := os.ReadFile(sp)
		if rerr != nil {
			continue // absent is fine; we require only that at least one exists
		}
		if fi, serr := os.Stat(sp); serr == nil && fi.IsDir() {
			continue
		}
		if werr := os.WriteFile(filepath.Join(dst, name), b, 0o600); werr != nil {
			return copied, skippedTwin, fmt.Errorf("copy %s: %w", name, werr)
		}
		copied = append(copied, name)
	}
	if len(copied) == 0 {
		return nil, skippedTwin, fmt.Errorf("source %s carries no login (.credentials.json / .oauth-token) to adopt", src)
	}
	// Seed identity from the source's oauthAccount so the roster shows WHO, not "-". The source
	// dir's fresher .claude.json (root vs in-dir) is resolved by DeriveIdentity's stateIdentity.
	sid := accounts.DeriveIdentity(src)
	if err := seedClaudeJSON(dst, accounts.ProbedIdentity{Email: sid.Email, AccountUUID: sid.AccountUUID}); err != nil {
		return copied, skippedTwin, fmt.Errorf("seed identity: %w", err)
	}
	return copied, skippedTwin, nil
}

// twinTokenToSkip decides whether the source's .oauth-token is a cross-account twin that must NOT
// ride along into the adopted seat. It returns the token's short fingerprint (to explain the
// skip) only when ALL of these hold: the source carries a usable live .credentials.json (so the
// seat still has a credential without the token), the source carries a .oauth-token, and that
// token's fingerprint matches a SIBLING home under homeRoot whose live login names a DIFFERENT
// account than the source's own login. That is exactly the GateTokenWrite refusal condition -
// caught here so the adopt copies the clean session instead of copying-then-refusing. It returns
// "" (copy the token) whenever homeRoot is unset, the token is the only credential, or the token
// legitimately belongs to the source's own account.
func twinTokenToSkip(src, homeRoot string) string {
	if homeRoot == "" {
		return ""
	}
	fp := tokenFingerprintFor(filepath.Join(src, ".oauth-token"))
	if fp == "" {
		return "" // no token to skip
	}
	// The token may only be dropped when a real session credential remains. hasLiveCredentials
	// mirrors the launcher's "can this dir actually serve" check on .credentials.json alone.
	if !hasLiveSessionCred(src) {
		return "" // token is the only credential - keep it
	}
	srcAcct := accounts.DeriveIdentity(src).AccountKey()
	homes, derr := accounts.Discover(homeRoot)
	if derr != nil {
		return ""
	}
	for _, h := range homes {
		if h.Identity.TokenFP != fp {
			continue
		}
		other := h.Identity.AccountKey()
		if other == "" {
			continue
		}
		// A sibling carries this exact token AND is logged into a different account than the
		// source: the token is that other account's. Skipping it keeps july8's seat off july7's
		// bucket. When srcAcct is "" (source login unknown) any identified other-owner is enough
		// to treat the token as foreign.
		if srcAcct == "" || other != srcAcct {
			return fp
		}
	}
	return ""
}

// hasLiveSessionCred reports whether dir carries a usable auto-refreshing session credential
// (.credentials.json with a non-empty claudeAiOauth access/refresh token) - the credential that
// lets a seat serve WITHOUT a .oauth-token. It is deliberately narrow: a bare .oauth-token does
// NOT count here, since the whole point is deciding whether the token is redundant.
func hasLiveSessionCred(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, ".credentials.json"))
	if err != nil {
		return false
	}
	var doc struct {
		ClaudeAiOauth struct {
			AccessToken  string `json:"accessToken"`
			RefreshToken string `json:"refreshToken"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return false
	}
	return strings.TrimSpace(doc.ClaudeAiOauth.AccessToken) != "" ||
		strings.TrimSpace(doc.ClaudeAiOauth.RefreshToken) != ""
}

// tokenFingerprintFor returns the same short fingerprint accounts.DeriveIdentity records for a
// dir's .oauth-token, computed for an explicit token-file path. "" means absent/empty.
func tokenFingerprintFor(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	tok := strings.TrimSpace(string(b))
	if tok == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(tok))
	return hex.EncodeToString(sum[:6])
}

// upsertHome replaces an existing registry row for home.Name in place (so --adopt --force
// refreshes rather than duplicates) or appends it when the name is new. It returns "added" or
// "updated" for the caller's summary line.
func upsertHome(reg *accounts.Registry, home accounts.Home) string {
	for i := range reg.Homes {
		if reg.Homes[i].Name == home.Name {
			// Preserve authored policy the row already carries; refresh dir + identity + the
			// add-time flags. Keep any role/note/rehome the seat had.
			existing := reg.Homes[i]
			existing.Dir = home.Dir
			existing.Identity = home.Identity
			existing.Reserved = home.Reserved
			if home.ChromeProfile != "" {
				existing.ChromeProfile = home.ChromeProfile
			}
			// Carry the credential KIND + api-key reference from the incoming row (#5331): an
			// api-key add lands them; a plain OAuth reconcile of the same name clears them, which
			// is the intended "convert this seat's kind" semantics of an explicit re-enroll.
			existing.CredKind = home.CredKind
			existing.APIKeyEnv = home.APIKeyEnv
			// A reconcile re-activates a seat that had been tombstoned.
			existing.Status = ""
			existing.Enabled = nil
			reg.Homes[i] = existing
			return "updated"
		}
	}
	reg.Homes = append(reg.Homes, home)
	return "added"
}

// rosterAccountName canonicalizes a --name to the suffixed roster handle (e.g. day26 ->
// day26-netra), matching the dir basename so the registry name, the dir, and the rosters all
// use one handle and `remove --name <name>` works with the name the rosters show.
func rosterAccountName(name, suffix string) string {
	if suffix != "" && !strings.HasSuffix(name, suffix) {
		return name + suffix
	}
	return name
}

// removeParams carries the resolved flags for `fak accounts remove`.
type removeParams struct {
	name string
	// byAccount, when set, selects the account-scoped retirement (#4669): retire EVERY active
	// seat resolving to this account bucket (an email, account UUID, raw bucket key, or seat
	// name) instead of the single --name seat.
	byAccount    string
	rehomeTo     string
	reason       string
	archive      bool
	terminal     bool
	registryPath string
	dosView      string
	jobView      string
	noSync       bool
}

// sameDir reports whether two paths name the same directory, tolerant of separators and
// (on Windows) case - used to refuse archiving the live CLAUDE_CONFIG_DIR out from under the
// running session.
func sameDir(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }

// loadRegistryOrErr loads the canonical accounts registry, printing the house error
// and returning ok=false on failure - the LoadRegistry+error-print prelude the
// registry-mutating subcommands repeat.
func loadRegistryOrErr(stderr io.Writer, registryPath string) (accounts.Registry, bool) {
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return reg, false
	}
	return reg, true
}

// resolveAdoptSource resolves the seat an `--adopt` enroll copies its login bundle FROM, and
// refuses a missing or invalid source. The dry run and the real run have to agree about what
// "the source" is and refuse the same source for the same reason - a --dry-run that happily
// resolved a source the real run then rejected would print a plan nobody can execute - so both
// ask here. ok=false means the refusal has already been printed and the caller must exit 1.
func resolveAdoptSource(stderr io.Writer, p addParams, reg accounts.Registry) (string, bool) {
	src, err := resolveSourceSeat(p.homeDir, p.from, reg)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return "", false
	}
	return src, true
}

// resolveRehomeTarget picks the seat a removal falls FORWARD to and resolves it. Both removal
// forms choose it the same way - the explicit --rehome-to, else the registry's anchor seat -
// and refuse the same two ways: no target at all, and a target the registry cannot resolve.
// Sharing that is what stops one form from tombstoning against a rehome the other form would
// have rejected. self names the seat being retired when the caller has exactly one (the
// --name form), which is excluded from the anchor fallback and refused as an explicit target;
// the account form passes "" because it retires a whole bucket and catches rehome-into-retired
// by identity, which no name comparison could see. ok=false means the refusal has already been
// printed and the caller must exit 1.
func resolveRehomeTarget(stderr io.Writer, reg accounts.Registry, rehomeTo, self string) (string, accounts.Home, bool) {
	rehome := rehomeTo
	if rehome == "" {
		if def, ok := reg.Default(); ok && (self == "" || def.Name != self) {
			rehome = def.Name
		}
	}
	if rehome == "" {
		fmt.Fprintln(stderr, "fak accounts: no --rehome-to and no default seat to fall forward to; pass --rehome-to <seat>")
		return "", accounts.Home{}, false
	}
	if self != "" && rehome == self {
		fmt.Fprintf(stderr, "fak accounts: cannot rehome %q to itself\n", self)
		return "", accounts.Home{}, false
	}
	live, _, err := reg.Resolve(rehome)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: invalid rehome target %q: %v\n", rehome, err)
		return "", accounts.Home{}, false
	}
	return rehome, live, true
}

// syncViewsUnlessNoSync re-syncs the dos/job views unless noSync is set, returning the
// nonzero exit code to propagate on a sync failure (0 otherwise). It folds the
// `if !noSync { syncViews … }` tail the registry-mutating subcommands share.
func syncViewsUnlessNoSync(stdout, stderr io.Writer, registryPath, dosView, jobView string, noSync bool) int {
	if noSync {
		return 0
	}
	if _, code := syncViews(stdout, stderr, registryPath, dosView, jobView); code != 0 {
		return code
	}
	return 0
}

// runAccountsRemove tombstones an account in the canonical registry and regenerates the
// views - the single-source inverse of add. It sets the home to status=tombstoned with a
// rehome target (so anything pinned to it falls forward) and records the audit fields
// (tombstoned_at, tombstone_reason), then re-syncs so the account drops from the dos view's
// active rows while remaining available only in the canonical registry. It does NOT delete
// the config dir - that is a separate, destructive operator step.
func runAccountsRemove(stdout, stderr io.Writer, p removeParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts remove --name <name> [--rehome-to <seat>] [--reason <text>] [--archive]")
		// Name the account-scoped form here too: a bare `remove` is exactly where an operator who
		// means "retire this account" lands, and offering only --name is what let a duplicate seat
		// keep the account live after its canonical seat was tombstoned (#4669).
		fmt.Fprintln(stderr, "   or: fak accounts remove --by-account <email|uuid|seat> [--rehome-to <seat>] [--reason <text>] [--archive]")
		fmt.Fprintln(stderr, "       --name retires ONE seat; --by-account retires the WHOLE account (every active seat resolving to it)")
		return 2
	}
	reg, ok := loadRegistryOrErr(stderr, p.registryPath)
	if !ok {
		return 1
	}
	idx := -1
	for i := range reg.Homes {
		if reg.Homes[i].Name == p.name {
			idx = i
			break
		}
	}
	if idx < 0 {
		fmt.Fprintf(stderr, "fak accounts: %q not in registry\n", p.name)
		return 1
	}
	if !reg.Homes[idx].Active() {
		fmt.Fprintf(stderr, "fak accounts: %q is already tombstoned\n", p.name)
		return 1
	}
	// Resolve the rehome target unless this is an explicit terminal tombstone. Terminal
	// retirement is reserved for the final account in one harness; cross-provider routing
	// (for example Claude -> Codex) belongs to fleet-accounts, not this registry.
	rehome, liveRehome := "", accounts.Home{}
	if !p.terminal {
		var ok bool
		rehome, liveRehome, ok = resolveRehomeTarget(stderr, reg, p.rehomeTo, p.name)
		if !ok {
			return 1
		}
		if liveRehome.Name == p.name {
			fmt.Fprintf(stderr, "fak accounts: cannot rehome %q through %q because it resolves back to itself\n", p.name, rehome)
			return 1
		}
	}
	reason := p.reason
	if reason == "" {
		reason = "removed via `fak accounts remove`"
	}
	// Peers on the SAME account bucket as this seat, computed BEFORE the tombstone over a
	// Refreshed COPY (so disk-derived identities drive the grouping without persisting them).
	// After this removal these seats still resolve to the account, so a note points the operator
	// at the account-scoped retirement instead of relying on catching the `dup ->` line by eye
	// (#4669) - the exact july6-netra papercut the issue reports.
	peers := reconcileRefreshed(reg)[p.name].Peers

	date := time.Now().UTC().Format("2006-01-02")
	liveName := liveRehome.Name
	if p.terminal {
		liveName = ""
	}
	movedRoles, code := applyTombstone(stdout, stderr, &reg, idx, rehome, liveName, reason, p.archive, p.terminal, date)
	if code != 0 {
		return code
	}

	if !saveAccountsRegistry(stderr, p.registryPath, reg) {
		return 1
	}
	if p.terminal {
		fmt.Fprintf(stdout, "registry: terminal-tombstoned %s (no same-harness rehome)\n", p.name)
	} else {
		fmt.Fprintf(stdout, "registry: tombstoned %s -> rehome %s\n", p.name, rehome)
	}
	for _, role := range movedRoles {
		fmt.Fprintf(stdout, "registry: role %s -> %s (was %s)\n", role, liveRehome.Name, p.name)
	}
	if len(peers) > 0 {
		fmt.Fprintf(stdout, "note: also reachable via %s - pass --by-account to retire the whole account\n", strings.Join(peers, ", "))
	}
	if code := syncViewsUnlessNoSync(stdout, stderr, p.registryPath, p.dosView, p.jobView, p.noSync); code != 0 {
		return code
	}
	if p.archive {
		fmt.Fprintf(stdout, "removed + archived account %q (now %q; dir renamed, tombstoned in registry + views)\n", p.name, reg.Homes[idx].Name)
	} else {
		fmt.Fprintf(stdout, "removed account %q (config dir left in place; tombstoned in registry + views)\n", p.name)
	}
	return 0
}

// applyTombstone performs the registry-side retirement of the ACTIVE seat at reg.Homes[idx]:
// it sets status=tombstoned + rehome + audit fields, disables the seat, and moves any roles it
// held onto liveRehomeName (which the caller has already resolved to an active seat). WITHOUT
// archive the seat is tombstoned in place under its own name, so it FLATTENS the pool: every OTHER
// seat whose rehome edge named it is repointed forward to liveRehomeName, compressing `C -> S -> L`
// to `C -> L` instead of letting the registry accrete `tombstoned -> … -> live` chains as
// intermediate hops retire (#4672). With archive it instead renames the seat's config dir to
// <dir>.DELETED-<date> and repoints the registry handle (name + dir) plus any inbound rehome edge
// that named the old handle onto the renamed one - a repoint `restore` reverses - the manual
// dir-rename + hand-edit-registry dance, done for you. It refuses the live CLAUDE_CONFIG_DIR, since
// you cannot move the dir the current session runs from.
//
// It mutates reg only; the caller saves + syncs + prints the summary. It returns the roles it
// moved (for the summary) and a nonzero code (with a printed error) on an archive filesystem
// refusal/failure. rehome is the handle recorded on the seat (may itself rehome forward);
// liveRehomeName is its resolved live seat, used for the role move. date is passed in so an
// account-scoped retirement stamps every seat with ONE archive date.
func applyTombstone(stdout, stderr io.Writer, reg *accounts.Registry, idx int, rehome, liveRehomeName, reason string, archive, terminal bool, date string) (movedRoles []string, code int) {
	fromName := reg.Homes[idx].Name
	reg.Homes[idx].Status = accounts.StatusTombstoned
	reg.Homes[idx].RehomeTo = rehome
	reg.Homes[idx].Terminal = terminal
	reg.Homes[idx].TombstonedAt = time.Now().UTC().Format(time.RFC3339)
	reg.Homes[idx].TombstoneReason = reason
	disabled := false
	reg.Homes[idx].Enabled = &disabled
	movedRoles = moveRolesOffHome(reg, fromName, liveRehomeName)

	if archive {
		oldName, oldDir := reg.Homes[idx].Name, reg.Homes[idx].Dir
		if oldDir != "" {
			if live := os.Getenv("CLAUDE_CONFIG_DIR"); live != "" && sameDir(live, oldDir) {
				fmt.Fprintf(stderr, "fak accounts: refusing to archive %q - it is the live CLAUDE_CONFIG_DIR; archive it from another session\n", fromName)
				return movedRoles, 1
			}
			newDir := oldDir + ".DELETED-" + date
			if _, err := os.Stat(newDir); err == nil {
				fmt.Fprintf(stderr, "fak accounts: archive target already exists: %s\n", newDir)
				return movedRoles, 1
			}
			if _, err := os.Stat(oldDir); err == nil {
				if err := os.Rename(oldDir, newDir); err != nil {
					fmt.Fprintf(stderr, "fak accounts: archive rename %s -> %s: %v\n", oldDir, newDir, err)
					return movedRoles, 1
				}
				fmt.Fprintf(stdout, "archived dir: %s -> %s\n", oldDir, newDir)
			} else {
				fmt.Fprintf(stdout, "archive: dir %s absent - repointing the registry only\n", oldDir)
			}
			reg.Homes[idx].Dir = newDir
		}
		newName := oldName + ".DELETED-" + date
		reg.Homes[idx].Name = newName
		for i := range reg.Homes {
			if reg.Homes[i].RehomeTo == oldName {
				reg.Homes[i].RehomeTo = newName
			}
		}
	} else {
		// Flatten inbound rehome edges past a seat tombstoned IN PLACE (#4672). The seat keeps its
		// name here (no --archive rename), so every OTHER seat that rehomed to it would keep naming
		// a now-dead seat - and as intermediate hops retire the pool accretes tombstoned->…->live
		// chains that `list --all` renders as rehomes pointing at dead seats. Repoint each such edge
		// forward to the live seat this removal falls to. (The archive branch above already repoints
		// inbound edges onto the renamed handle, which `restore` reverses; a plain tombstone has no
		// rename and nothing to reverse, so flattening forward is the correct hygiene.) idx is
		// skipped naturally - its own RehomeTo is the rehome handle, never fromName (a self-rehome is
		// refused upstream) - and fromName != liveRehomeName for the same reason, so no self-loop.
		for i := range reg.Homes {
			if i != idx && reg.Homes[i].RehomeTo == fromName {
				reg.Homes[i].RehomeTo = liveRehomeName
			}
		}
	}
	return movedRoles, 0
}

// reconcileRefreshed returns the reconcile verdicts computed over a Refreshed CLONE of reg, so
// disk-derived identities drive the account grouping WITHOUT mutating (or later persisting) the
// caller's registry.
func reconcileRefreshed(reg accounts.Registry) map[string]accounts.SeatIdentity {
	return reconcileRefreshedRegistry(reg).Reconcile()
}

// resolveAccountBucket maps a --by-account selector to the account-bucket key Reconcile groups
// on, plus the ACTIVE seats resolving to it. The selector may be an account email, an account
// UUID, the raw bucket key (uuid:… / tok:… / apikey:…), or a seat NAME - whichever handle an
// operator has to hand. reg must already be Refreshed (identities derived). It refuses
// ambiguity: a selector that names seats on more than one distinct bucket returns ok=false with
// the matched buckets, so the caller can list them rather than guess which account to retire.
func resolveAccountBucket(reg accounts.Registry, selector string) (bucket string, seats []accounts.Home, buckets []string) {
	sel := strings.TrimSpace(selector)
	matched := map[string]bool{}
	for _, h := range reg.Homes {
		if !h.Active() {
			continue
		}
		key := h.Identity.AccountKey()
		if key == "" {
			continue // no derivable identity - nothing to bucket on
		}
		if h.Name == sel ||
			(h.Identity.Email != "" && strings.EqualFold(h.Identity.Email, sel)) ||
			(h.Identity.AccountUUID != "" && h.Identity.AccountUUID == sel) ||
			key == sel {
			matched[key] = true
		}
	}
	for k := range matched {
		buckets = append(buckets, k)
	}
	sort.Strings(buckets)
	if len(buckets) != 1 {
		return "", nil, buckets
	}
	bucket = buckets[0]
	for _, h := range reg.Homes {
		if h.Active() && h.Identity.AccountKey() == bucket {
			seats = append(seats, h)
		}
	}
	sort.Slice(seats, func(i, j int) bool { return seats[i].Name < seats[j].Name })
	return bucket, seats, buckets
}

// runAccountsRemoveByAccount retires a WHOLE account (#4669): it tombstones EVERY active seat
// resolving to the selected account bucket in one audited pass, with a single rehome target and
// reason, so a duplicate seat that was identity_mismatched onto the account (the `dup ->` line
// in `list`) cannot leave the account live after its canonical seat is removed. It prints the
// full set of seats it will touch BEFORE acting, and refuses (structured, non-zero) when the
// rehome target itself resolves back into the account being retired - otherwise the account
// would never actually retire.
func runAccountsRemoveByAccount(stdout, stderr io.Writer, p removeParams) int {
	reg, ok := loadRegistryOrErr(stderr, p.registryPath)
	if !ok {
		return 1
	}
	// Resolve over a Refreshed copy so disk-derived identities drive the bucketing; the SAVED
	// registry stays the loaded one (only tombstone fields change), matching the --name path.
	refreshed := reconcileRefreshedRegistry(reg)
	bucket, seats, buckets := resolveAccountBucket(refreshed, p.byAccount)
	switch {
	case len(buckets) == 0:
		fmt.Fprintf(stderr, "fak accounts: no active seat resolves to account %q (want an email, account UUID, bucket key, or seat name)\n", p.byAccount)
		return 1
	case len(buckets) > 1:
		fmt.Fprintf(stderr, "fak accounts: %q is ambiguous - it names %d accounts (%s); retire one at a time with --name, or pass a UUID/email that selects exactly one\n",
			p.byAccount, len(buckets), strings.Join(buckets, ", "))
		return 1
	}

	// Resolve the rehome target unless this is an explicit final-account retirement.
	// Cross-provider destinations belong to fleet-accounts; a Claude tombstone must not fake
	// a Codex home inside the Claude registry merely to satisfy a same-harness edge.
	rehome, liveRehome := "", accounts.Home{}
	if !p.terminal {
		var ok bool
		rehome, liveRehome, ok = resolveRehomeTarget(stderr, reg, p.rehomeTo, "")
		if !ok {
			return 1
		}
		if liveKey := accountKeyForName(refreshed, liveRehome.Name); liveKey == bucket {
			fmt.Fprintf(stderr, "fak accounts: REFUSED (rehome-into-retired): rehome target %q resolves to the account being retired (%s); pick a live seat on a DIFFERENT account or pass --terminal when no same-harness account remains\n", rehome, bucket)
			return 1
		}
	}

	reason := p.reason
	if reason == "" {
		reason = "removed via `fak accounts remove --by-account`"
	}

	// Print the full set BEFORE acting, so the operator sees every seat one command will touch.
	names := make([]string, len(seats))
	for i, s := range seats {
		names[i] = s.Name
	}
	fmt.Fprintf(stdout, "retiring account %s - %d seat(s): %s\n", bucket, len(seats), strings.Join(names, ", "))
	if p.terminal {
		fmt.Fprintf(stdout, "  terminal retirement (no same-harness rehome); reason: %s\n", reason)
	} else {
		fmt.Fprintf(stdout, "  rehome -> %s; reason: %s\n", rehome, reason)
	}

	date := time.Now().UTC().Format("2006-01-02")
	// Pre-flight the archive so a multi-seat retirement is all-or-nothing at the first rename:
	// refuse up front if any seat is the live CLAUDE_CONFIG_DIR or its .DELETED target exists,
	// rather than renaming some dirs and then bailing with the registry unsaved.
	if p.archive {
		if code := preflightArchive(stderr, seats, date); code != 0 {
			return code
		}
	}

	for _, s := range seats {
		idx := homeIndex(reg, s.Name)
		if idx < 0 {
			continue // resolved from the refreshed clone; the name is always present in reg
		}
		movedRoles, code := applyTombstone(stdout, stderr, &reg, idx, rehome, liveRehome.Name, reason, p.archive, p.terminal, date)
		if code != 0 {
			return code
		}
		fmt.Fprintf(stdout, "registry: tombstoned %s -> rehome %s\n", s.Name, rehome)
		for _, role := range movedRoles {
			fmt.Fprintf(stdout, "registry: role %s -> %s (was %s)\n", role, liveRehome.Name, s.Name)
		}
	}

	if !saveAccountsRegistry(stderr, p.registryPath, reg) {
		return 1
	}
	if code := syncViewsUnlessNoSync(stdout, stderr, p.registryPath, p.dosView, p.jobView, p.noSync); code != 0 {
		return code
	}
	if p.terminal {
		fmt.Fprintf(stdout, "removed account %s � %d seat(s) terminal-tombstoned (config dirs left in place unless --archive)\n", bucket, len(seats))
	} else {
		fmt.Fprintf(stdout, "removed account %s � %d seat(s) tombstoned; rehome -> %s (config dirs left in place unless --archive)\n", bucket, len(seats), rehome)
	}
	return 0
}

// reconcileRefreshedRegistry returns a Refreshed CLONE of reg (identities derived from disk)
// without mutating the caller's registry - the same slice-clone guard reconcileRefreshed uses,
// but returning the registry itself so a caller can resolve buckets over it.
func reconcileRefreshedRegistry(reg accounts.Registry) accounts.Registry {
	homes := make([]accounts.Home, len(reg.Homes))
	copy(homes, reg.Homes)
	cp := reg
	cp.Homes = homes
	return cp.Refresh()
}

// accountKeyForName returns the account-bucket key of the named seat in reg (already Refreshed),
// or "" when the name is absent or has no derivable identity.
func accountKeyForName(reg accounts.Registry, name string) string {
	for _, h := range reg.Homes {
		if h.Name == name {
			return h.Identity.AccountKey()
		}
	}
	return ""
}

// homeIndex returns the index of the seat named name in reg.Homes, or -1 when absent.
func homeIndex(reg accounts.Registry, name string) int {
	for i := range reg.Homes {
		if reg.Homes[i].Name == name {
			return i
		}
	}
	return -1
}

// preflightArchive refuses an account-scoped `--archive` retirement up front when any seat can't
// be archived (it is the live CLAUDE_CONFIG_DIR, or its <dir>.DELETED-<date> target already
// exists), so the multi-seat rename is all-or-nothing rather than renaming some dirs and then
// bailing with the registry unsaved. It mutates nothing.
func preflightArchive(stderr io.Writer, seats []accounts.Home, date string) int {
	live := os.Getenv("CLAUDE_CONFIG_DIR")
	for _, s := range seats {
		if s.Dir == "" {
			continue
		}
		if live != "" && sameDir(live, s.Dir) {
			fmt.Fprintf(stderr, "fak accounts: refusing to archive %q - it is the live CLAUDE_CONFIG_DIR; archive it from another session\n", s.Name)
			return 1
		}
		if _, err := os.Stat(s.Dir + ".DELETED-" + date); err == nil {
			fmt.Fprintf(stderr, "fak accounts: archive target already exists: %s\n", s.Dir+".DELETED-"+date)
			return 1
		}
	}
	return 0
}

// moveRolesOffHome points every role currently filled by from at to, returning the roles
// moved in stable order. A role must name an active seat, so tombstoning a role-holder has
// to move the role before SaveRegistry's validation self-check can accept the registry.
func moveRolesOffHome(reg *accounts.Registry, from, to string) []string {
	if reg.Roles == nil {
		return nil
	}
	var moved []string
	for role, name := range reg.Roles {
		if name != from {
			continue
		}
		if to == "" {
			delete(reg.Roles, role)
		} else {
			reg.Roles[role] = to
		}
		moved = append(moved, role)
	}
	sort.Strings(moved)
	return moved
}

// setRoleParams carries the resolved flags for `fak accounts set-role`.
type setRoleParams struct {
	role         string // "active", "anchor", … (RoleActive is the set-default alias's role)
	name         string
	registryPath string
	dosView      string
	jobView      string
	noSync       bool
}

// runAccountsSetRole points a well-known ROLE at <name>: it sets reg.Roles[role]=name,
// validates (the target must be an active, serveable home), and regenerates the views. This is
// the deterministic one-command inverse of hand-editing the registry's roles - and the reason
// roles exist: the launch seat (RoleActive) and the rehome anchor (RoleAnchor) are SEPARATE,
// so pointing one never disturbs the other. RoleActive is surfaced as active_default in the
// dos view. Refuses a missing or tombstoned target (a tombstone can never serve, so it can
// never fill a role).
func runAccountsSetRole(stdout, stderr io.Writer, p setRoleParams) int {
	if p.role == "" || p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts set-role <role> --name <name>   (role: active|anchor)")
		return 2
	}
	reg, ok := loadRegistryOrErr(stderr, p.registryPath)
	if !ok {
		return 1
	}
	h, ok := homeByName(reg, p.name)
	if !ok {
		fmt.Fprintf(stderr, "fak accounts: %q not in registry\n", p.name)
		return 1
	}
	if !h.Active() {
		fmt.Fprintf(stderr, "fak accounts: %q is tombstoned and cannot fill role %q\n", p.name, p.role)
		return 1
	}
	if cur, ok := reg.Roles[p.role]; ok && cur == p.name {
		fmt.Fprintf(stdout, "%q already fills role %q\n", p.name, p.role)
		return 0
	}
	if reg.Roles == nil {
		reg.Roles = map[string]string{}
	}
	reg.Roles[p.role] = p.name
	if !validateAndSaveAccounts(stderr, p.registryPath, reg, "fak accounts: %v\n") {
		return 1
	}
	fmt.Fprintf(stdout, "registry: role %s -> %s\n", p.role, p.name)
	if code := syncViewsUnlessNoSync(stdout, stderr, p.registryPath, p.dosView, p.jobView, p.noSync); code != 0 {
		return code
	}
	fmt.Fprintf(stdout, "set role %q -> account %q\n", p.role, p.name)
	return 0
}

// homeByName returns the registry home with the given name.
func homeByName(reg accounts.Registry, name string) (accounts.Home, bool) {
	for _, h := range reg.Homes {
		if h.Name == name {
			return h, true
		}
	}
	return accounts.Home{}, false
}

// accountDir resolves the isolated config dir for a new account: ~/.claude-<name> when <name>
// already ends with the suffix, else ~/.claude-<name><suffix>. The suffix matches the host's
// roster convention (default "-netra") so a new seat sits alongside its peers.
func accountDir(home, name, suffix string) string {
	base := name
	if suffix != "" && !strings.HasSuffix(name, suffix) {
		base = name + suffix
	}
	return filepath.Join(home, ".claude-"+base)
}
