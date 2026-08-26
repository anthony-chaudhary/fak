package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

type accountsAddState struct {
	params          addParams
	registry        accounts.Registry
	rosterName, dir string
	apiKeyEnv       string
	reconcile       bool
	identity        accounts.ProbedIdentity
}

func acquireAccountsAddCredential(stdout, stderr io.Writer, state *accountsAddState) int {
	p, reg := state.params, state.registry
	rosterName, dir, apiKeyEnv := state.rosterName, state.dir, state.apiKeyEnv
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak accounts: mkdir %s: %v\n", dir, err)
		return 1
	}

	// Step 2 (identity holder): the enrollment identity. adopt derives it from the copied
	// disk state; setup-token derives it from the profile probe. Set in the branch below.
	var id accounts.ProbedIdentity

	if apiKeyEnv != "" {
		// API-KEY seat (#5331): there is NO credential to obtain or copy - the credential is the
		// Anthropic API key held in the $apiKeyEnv env var, and the registry keeps only the NAME.
		// Identity is derived from the KEY's org/workspace; offline that probe cannot run, so id
		// stays empty here (no OAuth email/uuid) and the seat's stable identity is its env-var
		// reference (DeriveAPIKeyIdentity at the upsert below). TODO(#5331): a live Console/profile
		// probe of the key would fill the org email/uuid.
		fmt.Fprintf(stdout, "api-key seat: credential is $%s (reference only; the key is never stored in the registry)\n", apiKeyEnv)
		if _, present := os.LookupEnv(apiKeyEnv); !present {
			fmt.Fprintf(stdout, "note: $%s is not set in this environment yet; the seat enrolls now and reads it at launch\n", apiKeyEnv)
		}
	} else if p.adopt {
		// ADOPT: copy an EXISTING login's bundle from the source seat into the isolated dir,
		// instead of minting a fresh setup-token. The source is already an enrolled, twin-clean
		// login, so the credential is proven by being live; we still twin-check a copied
		// .oauth-token, since that is the exact surface GateTokenWrite guards.
		src, srcOK := resolveAdoptSource(stderr, p, reg)
		if !srcOK {
			return 1
		}
		if sameDir(src, dir) {
			fmt.Fprintf(stderr, "fak accounts: --from source and target are the same dir (%s)\n", dir)
			return 1
		}
		// Backup-on-write (#3987): before copyLoginBundle overwrites this dir's credential blobs,
		// snapshot whatever is already there so the prior account is always recoverable. On a fresh
		// add the dir is empty and this is a no-op; on a --force reconcile it captures the live
		// credential being replaced, so even an in-place refresh can be undone with
		// `restore-credential`. A backup miss is a non-fatal warning - it never blocks the enroll.
		if snaps, berr := accounts.SnapshotBeforeOverwrite(accounts.BackupRoot(p.homeDir), rosterName, dir, time.Now()); berr != nil {
			fmt.Fprintf(stderr, "fak accounts: warning: pre-overwrite credential backup failed: %v\n", berr)
		} else if len(snaps) > 0 {
			fmt.Fprintf(stdout, "backed up %d prior credential blob(s) before overwrite (restore with `fak accounts restore-credential --name %s`)\n", len(snaps), rosterName)
		}
		copied, skipped, err := copyLoginBundle(src, dir, p.homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: adopt from %s: %v\n", src, err)
			return 1
		}
		if skipped != "" {
			// A cross-account twin setup-token was deliberately left behind (the source's live
			// .credentials.json is the real credential; carrying the twin would trip the
			// GateTokenWrite smear guard and burn the OTHER account's bucket). Surface it so the
			// operator knows WHY the seat has no .oauth-token.
			fmt.Fprintf(stdout, "skipped cross-account twin .oauth-token from %s (%s); seat runs on its own .credentials.json\n", src, skipped)
		}
		// If a token came across, twin-check it against the rest of the tree (the smear guard).
		if tok, terr := os.ReadFile(filepath.Join(dir, ".oauth-token")); terr == nil {
			if verdict := accounts.GateTokenWrite(dir, string(tok), p.homeDir); !verdict.Allow {
				fmt.Fprintf(stderr, "fak accounts: REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
				return 1
			}
		}
		fmt.Fprintf(stdout, "adopted login from %s (%s)\n", src, strings.Join(copied, ", "))
		// Record identity. The copied disk metadata (.claude.json oauthAccount) is only a CLAIM -
		// it lies exactly when the source is a shared dir a later /login rewrote .credentials.json
		// on without touching oauthAccount. So by DEFAULT reconcile that claim against the account
		// the copied LIVE credential actually serves (an OAuth profile probe) and let the credential
		// win, writing a corrected .claude.json so every later disk read is right. --no-probe-identity
		// opts back into the historical disk-only derivation for a deliberately offline enrollment.
		// A probe that can't run (no live session credential, offline, endpoint error) degrades to
		// the same disk+token derivation, so a plain adopt is never WORSE than before - only better
		// when the network is there.
		if p.noProbeIdentity {
			id = adoptedIdentityDiskOnly(dir, p.probeURL)
		} else {
			id = adoptedIdentityWithProbe(stdout, stderr, dir, p.probeURL, p.probeIdentity)
		}
		if id.Email != "" || id.AccountUUID != "" {
			fmt.Fprintf(stdout, "adopted identity: %s (%s)\n", id.Email, id.AccountUUID)
		} else {
			fmt.Fprintln(stderr, "fak accounts: warning: adopted a credential but could not derive identity; run `fak accounts discover --write` after first login")
		}
		// Step 3b: DIVORCE the copied credential's OAuth token family. The copy above left this
		// seat and `src` holding one refresh token; the first of them to refresh rotates the family
		// and the other is instantly dead (see internal/accounts/credfamily.go for the witness).
		// Resolving it here - while the operator is watching the enroll - is the only way the cost
		// lands at a chosen moment instead of mid-task hours later. It doubles as the seat's
		// refresh-capability proof: a seat that cannot rotate its own token is a seat that will
		// demand a human /login, and we would rather learn that now than at dispatch.
		divorceAdoptedFamily(stdout, stderr, src, dir, p)
	} else {
		// SETUP-TOKEN: mint (or read) a brand-new setup-token, twin-check, write, then probe.
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
		verdict := accounts.GateTokenWrite(dir, token, p.homeDir)
		if !verdict.Allow {
			fmt.Fprintf(stderr, "fak accounts: REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
			return 1
		}
		if err := os.WriteFile(filepath.Join(dir, ".oauth-token"), []byte(token+"\n"), 0o600); err != nil {
			fmt.Fprintf(stderr, "fak accounts: write token: %v\n", err)
			return 1
		}
		probed, err := accounts.ProbeToken(nil, "", token)
		if err != nil {
			// A probe failure is not fatal to enrollment (the dir + token are written), but it
			// means we cannot record identity and the credential may be bad - surface it loudly.
			fmt.Fprintf(stderr, "fak accounts: warning: identity probe failed: %v\n", err)
			fmt.Fprintln(stderr, "  the seat is created with a token but no recorded identity; run `fak accounts discover --write` after first login")
		} else {
			id = probed
			fmt.Fprintf(stdout, "probed identity: %s (%s)\n", id.Email, id.AccountUUID)
		}
	}

	state.identity = id
	return 0
}

func commitAccountsAdd(stdout, stderr io.Writer, state *accountsAddState) int {
	p, reg := state.params, state.registry
	rosterName, dir, apiKeyEnv := state.rosterName, state.dir, state.apiKeyEnv
	reconcile, id := state.reconcile, state.identity
	// Step 4b (#3954): refuse a login-identity hijack BEFORE writing the seat. GateTokenWrite above
	// already guards the token-fingerprint smear; this guards the orthogonal axis - the account the
	// just-probed credential ACTUALLY serves colliding with an EXISTING seat. reg is the pre-add
	// registry (recorded identities, deliberately un-Refreshed) so the check keys on the registry's
	// binding, not current disk. A duplicate (a different active seat already owns this account) is
	// refused unless --force; a rebind of this seat onto its own new account is enroll-current's job,
	// so it is surfaced, not blocked; an unprobed identity warns rather than guesses.
	// An api-key seat has no probed OAuth account to collide on (its identity is the key's
	// org, which we don't probe offline), so the login-hijack check does not apply - skip it
	// rather than warn "could not verify identity collision" on every api-key enroll (#5331).
	if apiKeyEnv == "" {
		switch col := accounts.DetectEnrollCollision(reg, rosterName, id); col.Kind {
		case accounts.EnrollDuplicate:
			if !p.force {
				// The dir was created THIS run (a duplicate-refuse implies !force, and reconcile needs
				// force, so an existing dir would have been refused earlier). Remove it so a refused
				// hijack leaves no half-seat behind - the registry (the authority) was never touched.
				if !reconcile {
					_ = os.RemoveAll(dir)
				}
				fmt.Fprintf(stderr, "fak accounts: REFUSED (identity-hijack): %s\n", col.Detail)
				fmt.Fprintf(stderr, "  enrolling it as %q would collapse two seats onto one rate-limit bucket. Pick a remedy:\n", rosterName)
				fmt.Fprintf(stderr, "    fresh dir     - meant to add a DIFFERENT account? log in again under a FRESH config dir\n")
				fmt.Fprintf(stderr, "                    (CLAUDE_CONFIG_DIR=<new dir> claude /login), then re-run. Never log into another seat's dir.\n")
				fmt.Fprintf(stderr, "    canonicalize  - this login should BE seat %q? rebind that seat onto it in place:\n", col.ConflictSeat)
				fmt.Fprintf(stderr, "                    fak accounts enroll-current --name %s --force\n", col.ConflictSeat)
				fmt.Fprintf(stderr, "    tombstone     - retiring seat %q instead? tombstone it with fall-forward, then re-run:\n", col.ConflictSeat)
				fmt.Fprintf(stderr, "                    fak accounts remove --name %s --rehome-to <seat>\n", col.ConflictSeat)
				fmt.Fprintf(stderr, "  (--force enrolls the duplicate anyway: two seats on one bucket, one of which the rotation drops)\n")
				return 1
			}
			fmt.Fprintf(stderr, "fak accounts: warning: --force enrolling a duplicate of seat %q (%s)\n", col.ConflictSeat, col.Account)
		case accounts.EnrollRebind:
			fmt.Fprintf(stdout, "note: %s\n", col.Detail)
		case accounts.EnrollUnknown:
			fmt.Fprintf(stderr, "fak accounts: warning: could not verify identity collision (%s)\n", col.Detail)
		}
	}

	// Step 5: seed markers so every consumer recognizes the seat. An api-key seat has no
	// OAuth oauthAccount to seed (id is empty and its identity is the key), so skip the
	// .claude.json identity seed for it; projects/ + settings.json still land below.
	if apiKeyEnv == "" {
		if err := seedClaudeJSON(dir, id); err != nil {
			fmt.Fprintf(stderr, "fak accounts: warning: seed .claude.json: %v\n", err)
		}
	}
	if err := os.MkdirAll(filepath.Join(dir, "projects"), 0o755); err != nil {
		fmt.Fprintf(stderr, "fak accounts: warning: create projects/ marker: %v\n", err)
	}
	// Seed the seat's settings.json from the registry's per-account defaults (defaults.settings)
	// so the new account launches WITH the bypass/permission defaults, not without them until a
	// later `sync`. Claude reads only its own settings.json, so this is what stops the bypass
	// default "getting lost" for a brand-new seat. Scoped to the just-added home, and - like the
	// .claude.json/projects markers above - done regardless of --no-sync, since it is part of
	// making the seat usable, not a roster-view refresh. A registry with no defaults.settings
	// block is a clean no-op.
	newHome := accounts.Home{Name: rosterName, Dir: dir}
	if code := projectSettingsForHomes(stdout, stderr, reg, []accounts.Home{newHome}); code != 0 {
		fmt.Fprintln(stderr, "fak accounts: warning: could not seed settings.json for the new seat")
	}

	// Step 6: upsert the canonical registry record. A reconcile (--adopt --force) REPLACES an
	// existing row for the name in place (refreshing its dir + identity + these flags) rather
	// than appending a duplicate; a fresh add appends.
	// A third-party-endpoint seat's overlay is validated BEFORE the registry is written, so a
	// credential-shaped variable never reaches the plaintext file at all. (The launch path
	// re-checks, since a registry can also be hand-edited.)
	extraEnv, eerr := parseSeatExtraEnv(p.extraEnv)
	if eerr != nil {
		fmt.Fprintf(stderr, "fak accounts add: %v\n", eerr)
		return 2
	}
	home := accounts.Home{
		Name:          rosterName,
		Dir:           dir,
		Reserved:      p.reserved,
		ChromeProfile: p.chrome,
		BaseURL:       strings.TrimSpace(p.baseURL),
		ExtraEnv:      extraEnv,
		Identity:      accounts.DeriveIdentity(dir),
	}
	if apiKeyEnv != "" {
		// API-KEY seat (#5331): mark the credential kind + env-var reference, and derive identity
		// from the KEY reference (never a disk OAuth probe, which would read it as credential-less).
		home.CredKind = accounts.CredKindAPIKey
		home.APIKeyEnv = apiKeyEnv
		home.Identity = accounts.DeriveAPIKeyIdentity(dir, apiKeyEnv, os.LookupEnv)
	}
	verb := upsertHome(&reg, home)
	if !saveAccountsRegistry(stderr, p.registryPath, reg) {
		return 1
	}
	fmt.Fprintf(stdout, "registry: %s %s -> %s\n", verb, rosterName, dir)

	// Step 7: regenerate the roster views.
	if !p.noSync {
		synced, serr := syncViews(stdout, stderr, p.registryPath, p.dosView, p.jobView)
		if serr != 0 {
			return serr
		}
		fmt.Fprintf(stdout, "synced %d roster view(s)\n", synced)
		// Step 7b (#3954): auto-verify the seat we just enrolled is actually usable - serveable in
		// the registry projection AND present in each rendered roster view. Advisory only: the dir +
		// registry + views already landed, so a miss is a loud warning pointing at the gap, not a
		// failure of the enroll that already committed.
		verifyServableAfterSync(stdout, stderr, p, rosterName)
	}

	action := "added"
	if reconcile {
		action = "reconciled"
	}
	fmt.Fprintf(stdout, "%s account %q (dir=%s, reserved=%v) - ~/.claude untouched\n", action, rosterName, dir, p.reserved)
	return 0

}
