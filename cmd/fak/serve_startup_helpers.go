package main

import (
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// serveExit is os.Exit, injectable so tests can observe a nonzero serve exit.
var serveExit = os.Exit

// warnIfNotFakWorkspace emits a loud stderr advisory when the serve cwd is not inside a
// fak workspace (no dos.toml found walking upward). It never blocks startup — the caller
// serves regardless. This is the #3094 mis-binding guard: fak serve's dojo corpus,
// devindex, and session-state planes all resolve against the workspace root, which
// defaults to cwd; launched from a sibling repo they silently bind the wrong tree.
func warnIfNotFakWorkspace(stderr io.Writer) {
	if _, err := branchrole.FindRoot(""); err == nil {
		return // inside a fak workspace (dos.toml found) — nothing to warn about.
	}
	wd, err := os.Getwd()
	if err != nil {
		wd = "(unknown cwd)"
	}
	// A warning rather than a refusal, which is exactly why it needs the same shape
	// as a bail: nothing stops, so an operator who skims this line serves a
	// silently mis-bound tree for the rest of the session.
	writeConfigBail(stderr, configBail{
		Verb:    "fak serve",
		Reason:  bailNotAWorkspace,
		Summary: "WARNING — this is not a fak workspace; the dojo corpus, devindex, and session-state planes will bind THIS cwd, not a fak repo",
		Knobs: []bailKnob{
			bailCWD(wd, "no dos.toml found walking upward").want("a directory inside your fak checkout"),
		},
		// Nothing stops here, so the fix has to stay INLINE — a warning that makes
		// you run a second command to learn the remedy is one you scroll past.
		Check: "git rev-parse --show-toplevel   # the repo this cwd actually resolves to\n          launch fak serve from a fak checkout, or set the MCP server entry's \"cwd\" to your fak workspace root",
	})
}

const deferToolsDisabledWarning = "fak serve: WARNING: --defer-tools=false is set; advertising the full tool registry carries significant token overhead (~5,500+ tokens / ~22KB per turn) and increases per-turn latency. Consider keeping --defer-tools=true (the default) with progressive disclosure."

func warnIfDeferToolsDisabled(stderr io.Writer, deferTools bool) {
	if !deferTools {
		fmt.Fprintln(stderr, deferToolsDisabledWarning)
	}
}

// serveKeyPrincipals resolves the repeated `--key-principal PRINCIPAL=ENV_VAR` specs into
// the gateway.Config.KeyPrincipals map (#5332), reporting a refusal on stderr rather than
// returning it — the same (value, ok) shape resolveRequiredKey uses, so both auth flags
// fail closed identically. ok=false means the caller MUST NOT boot: a gateway serving with
// a keyset the operator believes is armed, but which never resolved, does not merely lose
// attribution — with no keyset matched, principalFor falls through to the CALLER-supplied
// X-Fak-Principal header, and that value is what the account allowlist adjudicates. So a
// half-resolved keyset is a tenant-isolation hole, not a cosmetic one.
//
// No specs is not a failure: it returns a nil map, which gateway.New turns into a nil
// keyset, leaving the --require-key-env single-bearer path byte-for-byte unchanged.
func serveKeyPrincipals(specs []string, lookupEnv func(string) string, stderr io.Writer) (map[string]string, bool) {
	keyPrincipals, err := gateway.ParseKeyPrincipals(specs, lookupEnv)
	if err != nil {
		// The summary keeps the original sentence intact, X-Fak-Principal clause and
		// all: it is what says WHY this fails closed rather than warning, and the
		// block is meant to add a next step, never to cost the reason.
		writeConfigBail(stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailKeyPrincipalUnresolved,
			Summary: fmt.Sprintf("--key-principal %v — refusing to start a gateway whose tenant keyset did not fully resolve (an unresolved binding leaves that tenant attributed by the caller-supplied X-Fak-Principal header instead of by its key)", err),
			Knobs: []bailKnob{
				bailFlag("key-principal", strings.Join(specs, " ")).want("PRINCIPAL=ENV_VAR per tenant, each naming a set, distinct env var"),
			},
		})
		return nil, false
	}
	return keyPrincipals, true
}

func loadServeRouteFile[T any](flagName, path, want string, load func(string) (T, error)) *T {
	loaded, err := load(path)
	if err != nil {
		writeConfigBail(os.Stderr, configBail{
			Verb: "fak serve", Reason: bailRouteManifestInvalid,
			Summary: fmt.Sprintf("--%s did not load: %v", flagName, err),
			Knobs:   []bailKnob{bailFlag(flagName, path), bailFile(path, "did not load").want(want)},
			Bind:    []string{"path=" + path},
		})
		os.Exit(1)
	}
	return &loaded
}

// hostPortOf extracts the host:port from a base URL, dropping any path suffix
// (a --base-url carries a /v1 path that must not participate in the comparison).
// An empty or unparseable URL yields "".
func hostPortOf(raw string) string {
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	host := u.Hostname()
	if host == "" {
		return ""
	}
	port := u.Port()
	if port == "" {
		return host
	}
	return host + ":" + port
}

// checkReplicaBindingConsistency refuses a reused node-global roster whose loopback
// account binding disagrees with this process's --base-url. The H200 half-node witness:
// a replica gateway launched with the CORRECT --base-url (this replica's engine port)
// but a REUSED --route-accounts roster whose local-engine binding pointed at the OTHER
// replica's port. The roster, not the --base-url flag, is authoritative for the request
// plan, so every request returned 200 from the other replica while this replica's GPUs
// served nothing — and the facade stayed healthy, so the defect was invisible.
//
// SCOPING IS LOAD-BEARING: the check applies only to LOOPBACK accounts. A literal
// "every account base_url equals --base-url" would refuse legitimate mixed rosters
// (one local engine + remote openai/bedrock accounts); the witness defect is
// specifically a loopback account pointing at the wrong local engine. The comparison
// is host:port only, so a /v1 path suffix on either side does not false-positive.
//
// Returns nil when there is no proxy target (baseURL == "") or no roster (roster == nil)
// — the pre-#2528 behavior, byte-identical.
func checkReplicaBindingConsistency(baseURL string, roster *modelroute.Roster) error {
	if baseURL == "" || roster == nil {
		return nil
	}
	want := hostPortOf(baseURL)
	if want == "" {
		return nil
	}
	for _, acctURL := range roster.LoopbackBaseURLs() {
		got := hostPortOf(acctURL)
		if got == "" {
			continue
		}
		if got != want {
			return fmt.Errorf("%s: loopback account binding %s (host:port %s) disagrees with --base-url %s (host:port %s) — the roster, not the flag, is authoritative for the request plan, so every request would forward to the other replica while this one's GPUs serve nothing",
				bailReplicaBindingMismatch, acctURL, got, baseURL, want)
		}
	}
	return nil
}
