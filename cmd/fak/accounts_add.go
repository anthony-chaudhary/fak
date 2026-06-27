package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// addParams carries the resolved flags for `fak accounts add` from the dispatcher.
type addParams struct {
	name     string
	reserved bool
	chrome   string
	noLogin  bool
	token    string
	suffix   string
	noSync   bool

	homeDir      string
	registryPath string
	dosView      string
	jobView      string
}

// runAccountsAdd is the end-to-end "enroll a brand-new account" flow. It is deliberately the
// ONLY place the multi-file account-enrollment runbook lives, so adding an account is one
// command instead of: hand-edit three rosters, hand-derive the uuid, work around the
// out-of-tree guard, remember the projects/ marker. The steps, in order:
//
//  1. resolve an ISOLATED config dir (~/.claude-<name>[-suffix]); refuse to clobber ~/.claude
//     or an existing dir, so a stray login never lands on the live session.
//  2. obtain the setup-token — either by running `CLAUDE_CONFIG_DIR=<dir> claude setup-token`
//     (inheriting the TTY for the browser+paste), or from --token/stdin with --no-login.
//  3. write <dir>/.oauth-token, but twin-check FIRST (GateTokenWrite) so we never enroll a
//     token that belongs to a DIFFERENT account already on disk (the cross-account smear).
//  4. probe the OAuth profile endpoint for the email + account UUID — ground truth that also
//     proves the credential works.
//  5. seed the dir's markers so every consumer recognizes it: .claude.json (identity, so the
//     roster shows WHO it is, not "-") and projects/ (the fleet discovery gate).
//  6. upsert the canonical registry record (identity + policy) and SaveRegistry.
//  7. regenerate the roster views (sync) so the dos + job rosters reflect the new account.
func runAccountsAdd(stdout, stderr io.Writer, p addParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts add --name <name> [--reserved] [--chrome-profile P] [--no-login [--token -]]")
		return 2
	}
	if p.homeDir == "" {
		fmt.Fprintln(stderr, "fak accounts: cannot resolve home dir")
		return 1
	}

	// Canonicalize the roster name to carry the suffix (the host convention, e.g. day26 ->
	// day26-netra), so the registry name matches the dir basename and `remove --name <name>`
	// uses the same handle the rosters show.
	rosterName := rosterAccountName(p.name, p.suffix)
	dir := accountDir(p.homeDir, p.name, p.suffix)
	// Refuse to ever target the live default seat or an existing dir — a new account gets a
	// fresh, isolated home so no login can clobber ~/.claude.
	if filepath.Clean(dir) == filepath.Clean(filepath.Join(p.homeDir, ".claude")) {
		fmt.Fprintln(stderr, "fak accounts: refusing to add into the default ~/.claude seat")
		return 1
	}
	if _, err := os.Stat(dir); err == nil {
		fmt.Fprintf(stderr, "fak accounts: config dir already exists: %s (pick another --name)\n", dir)
		return 1
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
		if h.Name == rosterName {
			fmt.Fprintf(stderr, "fak accounts: %q is already in the registry\n", rosterName)
			return 1
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak accounts: mkdir %s: %v\n", dir, err)
		return 1
	}

	// Step 2: obtain the token.
	token, err := obtainToken(stdout, stderr, dir, p)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	token = strings.TrimSpace(token)
	if !strings.HasPrefix(token, "sk-ant-oat") {
		fmt.Fprintf(stderr, "fak accounts: not a setup-token (want sk-ant-oat…), got %d chars\n", len(token))
		return 1
	}

	// Step 3: twin-check BEFORE persisting, then write the token.
	verdict := accounts.GateTokenWrite(dir, token, p.homeDir)
	if !verdict.Allow {
		fmt.Fprintf(stderr, "fak accounts: REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
		return 1
	}
	if err := os.WriteFile(filepath.Join(dir, ".oauth-token"), []byte(token+"\n"), 0o600); err != nil {
		fmt.Fprintf(stderr, "fak accounts: write token: %v\n", err)
		return 1
	}

	// Step 4: probe identity (ground truth + proves the credential works).
	id, err := accounts.ProbeToken(nil, "", token)
	if err != nil {
		// A probe failure is not fatal to enrollment (the dir + token are written), but it
		// means we cannot record identity and the credential may be bad — surface it loudly.
		fmt.Fprintf(stderr, "fak accounts: warning: identity probe failed: %v\n", err)
		fmt.Fprintln(stderr, "  the seat is created with a token but no recorded identity; run `fak accounts discover --write` after first login")
	} else {
		fmt.Fprintf(stdout, "probed identity: %s (%s)\n", id.Email, id.AccountUUID)
	}

	// Step 5: seed markers so every consumer recognizes the seat.
	if err := seedClaudeJSON(dir, id); err != nil {
		fmt.Fprintf(stderr, "fak accounts: warning: seed .claude.json: %v\n", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "projects"), 0o755); err != nil {
		fmt.Fprintf(stderr, "fak accounts: warning: create projects/ marker: %v\n", err)
	}

	// Step 6: upsert the canonical registry record.
	home := accounts.Home{
		Name:          rosterName,
		Dir:           dir,
		Reserved:      p.reserved,
		ChromeProfile: p.chrome,
		Identity:      accounts.DeriveIdentity(dir),
	}
	reg.Homes = append(reg.Homes, home)
	if err := accounts.SaveRegistry(p.registryPath, reg); err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "registry: added %s -> %s\n", rosterName, dir)

	// Step 7: regenerate the roster views.
	if !p.noSync {
		synced, serr := syncViews(stdout, stderr, p.registryPath, p.dosView, p.jobView)
		if serr != 0 {
			return serr
		}
		fmt.Fprintf(stdout, "synced %d roster view(s)\n", synced)
	}

	fmt.Fprintf(stdout, "added account %q (dir=%s, reserved=%v) — ~/.claude untouched\n", rosterName, dir, p.reserved)
	return 0
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
	name         string
	rehomeTo     string
	reason       string
	registryPath string
	dosView      string
	jobView      string
	noSync       bool
}

// runAccountsRemove tombstones an account in the canonical registry and regenerates the
// views — the single-source inverse of add. It sets the home to status=tombstoned with a
// rehome target (so anything pinned to it falls forward) and records the audit fields
// (tombstoned_at, tombstone_reason), then re-syncs so the account drops from the dos view's
// active rows and appears under the job view's tombstoned_accounts block. It does NOT delete
// the config dir — that is a separate, destructive operator step.
func runAccountsRemove(stdout, stderr io.Writer, p removeParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts remove --name <name> [--rehome-to <seat>] [--reason <text>]")
		return 2
	}
	reg, err := accounts.LoadRegistry(p.registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
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
	// Resolve the rehome target: the flag, else the registry's default seat.
	rehome := p.rehomeTo
	if rehome == "" {
		if def, ok := reg.Default(); ok && def.Name != p.name {
			rehome = def.Name
		}
	}
	if rehome == "" {
		fmt.Fprintln(stderr, "fak accounts: no --rehome-to and no default seat to fall forward to; pass --rehome-to <seat>")
		return 1
	}
	if rehome == p.name {
		fmt.Fprintf(stderr, "fak accounts: cannot rehome %q to itself\n", p.name)
		return 1
	}
	reason := p.reason
	if reason == "" {
		reason = "removed via `fak accounts remove`"
	}
	reg.Homes[idx].Status = accounts.StatusTombstoned
	reg.Homes[idx].RehomeTo = rehome
	reg.Homes[idx].TombstonedAt = time.Now().UTC().Format(time.RFC3339)
	reg.Homes[idx].TombstoneReason = reason
	disabled := false
	reg.Homes[idx].Enabled = &disabled
	if err := accounts.SaveRegistry(p.registryPath, reg); err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "registry: tombstoned %s -> rehome %s\n", p.name, rehome)
	if !p.noSync {
		if _, code := syncViews(stdout, stderr, p.registryPath, p.dosView, p.jobView); code != 0 {
			return code
		}
	}
	fmt.Fprintf(stdout, "removed account %q (config dir left in place; tombstoned in registry + views)\n", p.name)
	return 0
}

// setDefaultParams carries the resolved flags for `fak accounts set-default`.
type setDefaultParams struct {
	name         string
	registryPath string
	dosView      string
	jobView      string
	noSync       bool
}

// runAccountsSetDefault makes <name> the single default (active) seat: it clears any existing
// default and marks the named active home default, then validates and regenerates the views.
// This is the deterministic one-command inverse of hand-editing `default: true` in the
// registry — the default is what a bare launch / the watchdog picks, surfaced as
// `active_default` in the dos view. Refuses a missing or tombstoned target (a tombstone can
// never serve, so it can never be the default).
func runAccountsSetDefault(stdout, stderr io.Writer, p setDefaultParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts set-default --name <name>")
		return 2
	}
	reg, err := accounts.LoadRegistry(p.registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
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
		fmt.Fprintf(stderr, "fak accounts: %q is tombstoned and cannot be the default\n", p.name)
		return 1
	}
	if reg.Homes[idx].Default {
		fmt.Fprintf(stdout, "%q is already the default seat\n", p.name)
		return 0
	}
	// Clear any prior default (Validate allows at most one), then set the new one.
	for i := range reg.Homes {
		reg.Homes[i].Default = false
	}
	reg.Homes[idx].Default = true
	if err := reg.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	if err := accounts.SaveRegistry(p.registryPath, reg); err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "registry: default seat -> %s\n", p.name)
	if !p.noSync {
		if _, code := syncViews(stdout, stderr, p.registryPath, p.dosView, p.jobView); code != 0 {
			return code
		}
	}
	fmt.Fprintf(stdout, "set default account %q (active_default in the dos view)\n", p.name)
	return 0
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

// obtainToken returns the setup-token, either by running `claude setup-token` in the isolated
// dir or by reading --token / stdin under --no-login.
func obtainToken(stdout, stderr io.Writer, dir string, p addParams) (string, error) {
	if p.noLogin || p.token != "" {
		if p.token != "" && p.token != "-" {
			return p.token, nil
		}
		fmt.Fprintln(stderr, "reading setup-token from stdin…")
		b, err := io.ReadAll(bufio.NewReader(os.Stdin))
		if err != nil {
			return "", fmt.Errorf("read token from stdin: %w", err)
		}
		return string(b), nil
	}
	// Interactive: run `claude setup-token` with CLAUDE_CONFIG_DIR pointed at the isolated dir
	// so the login lands there, NOT in ~/.claude. Inherit the TTY for the browser + paste.
	fmt.Fprintf(stdout, "running `claude setup-token` for %s (CLAUDE_CONFIG_DIR=%s)…\n", p.name, dir)
	cmd := exec.Command("claude", "setup-token")
	cmd.Env = append(os.Environ(), "CLAUDE_CONFIG_DIR="+dir)
	cmd.Stdin, cmd.Stderr = os.Stdin, os.Stderr
	// Capture stdout so we can recover the printed token, while still echoing it for the user.
	var buf strings.Builder
	cmd.Stdout = io.MultiWriter(os.Stdout, &buf)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("claude setup-token: %w", err)
	}
	return extractToken(buf.String()), nil
}

// extractToken pulls the sk-ant-oat… token out of `claude setup-token` output (which prints
// some preamble around it).
func extractToken(out string) string {
	for _, f := range strings.Fields(out) {
		if strings.HasPrefix(f, "sk-ant-oat") {
			return f
		}
	}
	return strings.TrimSpace(out)
}

// seedClaudeJSON writes a minimal .claude.json carrying the probed identity, so a fresh seat
// shows WHO it is in the roster (not "-") before its first interactive `claude` run. It does
// nothing when the identity is empty, and never overwrites an existing .claude.json.
func seedClaudeJSON(dir string, id accounts.ProbedIdentity) error {
	if id.Email == "" && id.AccountUUID == "" {
		return nil
	}
	path := filepath.Join(dir, ".claude.json")
	if _, err := os.Stat(path); err == nil {
		return nil // never clobber claude's own file
	}
	doc := map[string]any{
		"oauthAccount": map[string]any{
			"emailAddress": id.Email,
			"accountUuid":  id.AccountUUID,
		},
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
