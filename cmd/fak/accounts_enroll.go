package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// `fak accounts enroll-current` — turn the login you are CURRENTLY signed into (the session's
// CLAUDE_CONFIG_DIR, or ~/.claude) into a first-class, identity-true rotation seat in one command.
//
// It is `add --adopt` aimed at the current session, with one critical difference: it ALWAYS
// probes the OAuth profile endpoint for the account the live credential ACTUALLY serves and
// records THAT, rather than trusting the source dir's .claude.json metadata. That metadata lies
// exactly when it matters — after a `/login` rewrote .credentials.json (the real credential) but
// left oauthAccount naming the previous account — so a metadata-trusting adopt would enroll the
// wrong account, and the twin `.oauth-token` (if it belongs to a different account among the
// siblings) is deliberately not copied. This closes the "enrolled seat silently bills/represents
// the wrong account" gap for the common case of promoting the seat you are already using.
type enrollParams struct {
	name     string
	from     string // override the source: a seat name / dir; empty = current session dir
	reserved bool
	force    bool
	suffix   string
	noSync   bool
	probeURL string // OAuth profile endpoint override ($FAK_OAUTH_PROFILE_URL); "" = default
	dryRun   bool   // print the enrollment plan without mutating anything (#3954)
	// noDivorce opts OUT of the default post-copy OAuth token-family divorce. See addParams.
	// enroll-current is the path that hits this hazard hardest: its source is by definition a dir
	// still in live use, so leaving the copied family shared logs that session out later, without
	// warning (witnessed 2026-08-06).
	noDivorce bool

	homeDir      string
	registryPath string
	dosView      string
	jobView      string
}

// currentSessionDir resolves the config dir the CURRENT session runs from: an explicit --from
// wins; else $CLAUDE_CONFIG_DIR (what a `fak guard` / launched seat exports); else ~/.claude.
func currentSessionDir(homeDir, from string) string {
	if from != "" {
		return from // resolveSourceSeat interprets a name / path / "default"
	}
	if cc := os.Getenv("CLAUDE_CONFIG_DIR"); cc != "" {
		return cc
	}
	return filepath.Join(homeDir, ".claude")
}

// runAccountsEnrollCurrent enrolls the current session's login as a new rotation seat. It is a
// thin, honest wrapper over the shared adopt core (runAccountsAdd with adopt+probeIdentity), so
// the twin-skip, isolated-dir refusal, marker seeding, registry upsert, and view sync are exactly
// the audited add path — only the source (current session) and the always-on credential probe
// differ.
func runAccountsEnrollCurrent(stdout, stderr io.Writer, p enrollParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts enroll-current --name <seat> [--from <seat|dir>] [--reserved] [--force]")
		return 2
	}
	if !requireAccountsHome(stderr, p.homeDir) {
		return 1
	}
	src := currentSessionDir(p.homeDir, p.from)
	verb := "enrolling"
	if p.dryRun {
		verb = "would enroll"
	}
	fmt.Fprintf(stdout, "%s the current login (%s) as seat %q with a live credential-identity probe…\n", verb, src, p.name)
	return runAccountsAdd(stdout, stderr, addParams{
		name:          p.name,
		reserved:      p.reserved,
		suffix:        p.suffix,
		noSync:        p.noSync,
		adopt:         true,
		from:          src,
		force:         p.force,
		probeIdentity: true,
		probeURL:      p.probeURL,
		noDivorce:     p.noDivorce,
		dryRun:        p.dryRun,
		homeDir:       p.homeDir,
		registryPath:  p.registryPath,
		dosView:       p.dosView,
		jobView:       p.jobView,
	})
}

// enrollProfileURL returns the OAuth profile endpoint override for enrollment probes: the
// $FAK_OAUTH_PROFILE_URL test/advanced seam, else "" (accounts.DefaultProfileURL). Named so the
// dependency on the accounts default is explicit at the call site.
func enrollProfileURL() string {
	if u := os.Getenv("FAK_OAUTH_PROFILE_URL"); u != "" {
		return u
	}
	return "" // accounts.ProbeToken falls back to accounts.DefaultProfileURL
}

var _ = accounts.DefaultProfileURL // documents the fallback endpoint enrollProfileURL defers to
