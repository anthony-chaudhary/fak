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

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accountsadd_identity.go — the credential-and-identity seam of `fak accounts add`, lifted out
// of accounts_add.go under the god-file growth gate (>1500 lines): obtaining the setup-token
// (interactive `claude setup-token`, or --token/stdin under --no-login), resolving WHICH account a
// seat's live credential actually serves (the OAuth profile probe and its disk-only fallback), and
// writing that identity into the seat's .claude.json. Pure relocation — the enrollment flow that
// calls these stays in accounts_add.go.

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

// adoptedIdentityWithProbe resolves an adopted seat's identity by reconciling its on-disk
// metadata against the account its live credential ACTUALLY serves (an OAuth profile probe), and
// prefers the credential. When the two disagree it OVERWRITES the seat's .claude.json oauthAccount
// with the credential identity — the durable fix for the mislabel, so `list`/`status`/`discover`
// (all disk-only) report the true account forever after, not just this run. A probe that cannot
// run — no live session credential to probe, or an endpoint/transport error — is non-fatal: it
// degrades to the SAME disk+token derivation a --no-probe-identity adopt uses, so the probe-on
// default is never worse than disk-only. profileURL is "" for the real endpoint. `explicit` is
// true only when a caller FORCED the probe (enroll-current); it controls how loud a probe failure
// is: an explicitly-requested probe that fails warns (the operator asked for the guarantee and did
// not get it), while the default-on probe falling back is quiet (it is the expected offline path).
func adoptedIdentityWithProbe(stdout, stderr io.Writer, dir, profileURL string, explicit bool) accounts.ProbedIdentity {
	probe := func(tok string) (accounts.ProbedIdentity, error) {
		return accounts.ProbeToken(nil, profileURL, tok)
	}
	res := accounts.ResolveCredentialIdentity(dir, probe)
	switch {
	case res.Stale:
		fmt.Fprintf(stdout, "identity reconcile: on-disk .claude.json names %s but the live credential serves %s — using the credential identity\n",
			identityLabel(res.Disk.Email, res.Disk.AccountUUID), identityLabel(res.Credential.Email, res.Credential.AccountUUID))
		if err := writeClaudeJSONIdentity(dir, res.Credential); err != nil {
			fmt.Fprintf(stderr, "fak accounts: warning: rewrite .claude.json to credential identity: %v\n", err)
		}
		return res.Credential
	case res.Probed:
		// Probed and agreed (or filled an empty disk identity): the credential is ground truth.
		return res.Credential
	case res.ProbeErr != nil && explicit:
		// The operator explicitly asked for the probe guarantee and the endpoint failed — warn,
		// then fall through to the disk+token derivation so enrollment still completes.
		fmt.Fprintf(stderr, "fak accounts: warning: could not probe the credential identity (%v); trusting the copied on-disk identity\n", res.ProbeErr)
	}
	// No live credential to probe, or a quiet default-on fallback: use the full disk+token
	// derivation (identical to a --no-probe-identity adopt), never a possibly-empty disk read.
	return adoptedIdentityDiskOnly(dir, profileURL)
}

// adoptedIdentityDiskOnly derives an adopted seat's identity primarily from disk: the copied
// .claude.json metadata via DeriveIdentity, falling back to a token probe ONLY when disk identity
// is empty and a bare .oauth-token was copied (the historical --no-probe-identity behavior). It is
// the shared fallback for both the opt-out path and a probe that could not run. profileURL is the
// OAuth endpoint override ("" = the real DefaultProfileURL) — threaded through so the token-only
// fallback probe honors the same $FAK_OAUTH_PROFILE_URL test/advanced seam as the primary probe,
// rather than silently hitting the real endpoint.
func adoptedIdentityDiskOnly(dir, profileURL string) accounts.ProbedIdentity {
	derived := accounts.DeriveIdentity(dir)
	id := accounts.ProbedIdentity{Email: derived.Email, AccountUUID: derived.AccountUUID}
	if id.Email == "" && id.AccountUUID == "" {
		if tok, terr := os.ReadFile(filepath.Join(dir, ".oauth-token")); terr == nil {
			if probed, perr := accounts.ProbeToken(nil, profileURL, string(tok)); perr == nil {
				id = probed
			}
		}
	}
	return id
}

// identityLabel renders an identity as "email (uuid)", "email", "(uuid)", or "unknown" for a
// human-readable reconcile line.
func identityLabel(email, uuid string) string {
	switch {
	case email != "" && uuid != "":
		return fmt.Sprintf("%s (%s)", email, uuid)
	case email != "":
		return email
	case uuid != "":
		return "(" + uuid + ")"
	default:
		return "unknown"
	}
}

// writeClaudeJSONIdentity writes (OVERWRITING) a dir's .claude.json oauthAccount block to the
// given identity, preserving any other top-level keys the file already carries. Unlike
// seedClaudeJSON it deliberately replaces a stale oauthAccount — the reconcile's whole point.
func writeClaudeJSONIdentity(dir string, id accounts.ProbedIdentity) error {
	if id.Email == "" && id.AccountUUID == "" {
		return nil
	}
	path := filepath.Join(dir, ".claude.json")
	doc := map[string]any{}
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, &doc) // best-effort: a corrupt file is replaced wholesale
	}
	if doc == nil {
		doc = make(map[string]any)
	}
	doc["oauthAccount"] = claudeOAuthAccountBlock(id)
	return writeClaudeJSONDoc(path, doc)
}

// claudeOAuthAccountBlock renders the .claude.json "oauthAccount" object for a probed
// identity. Both writers in this file emit the same two keys, so the roster reads the same
// shape whether a seat was seeded fresh or reconciled in place.
func claudeOAuthAccountBlock(id accounts.ProbedIdentity) map[string]any {
	return map[string]any{
		"emailAddress": id.Email,
		"accountUuid":  id.AccountUUID,
	}
}

// writeClaudeJSONDoc writes doc to path in claude's own 2-space-indented layout with a
// trailing newline -- the exact byte form both writers here have always produced.
func writeClaudeJSONDoc(path string, doc map[string]any) error {
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
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
	return writeClaudeJSONDoc(path, map[string]any{"oauthAccount": claudeOAuthAccountBlock(id)})
}
