package accounts

// twinguard — the PRE-WRITE gate that stops Regression A: a login/switch flow writing
// one account's setup token into a config home whose interactive login is a DIFFERENT
// account. That smear is how a single account's token landed byte-identical in three
// homes (one rate-limit bucket presented as three seats), which then walls and surfaces
// to Claude Code as "Your organization has disabled Claude subscription access".
//
// Reconcile (accounts.go) DETECTS the smear after the fact (the token-twin flag). This
// file REFUSES it before the bytes hit disk, and audits the live tree for an existing
// smear. The asymmetry that makes the gate correct: two homes sharing a token is only a
// bug when their interactive logins name DIFFERENT accounts. The default home (~/.claude)
// and a named home (~/.claude-gem8-netra) that are BOTH logged into gem8@ legitimately
// share gem8's token — same account, two dir names. The gate keys on the LOGIN identity,
// not the dir name, so it permits the legitimate case and refuses only the cross-account
// write.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
)

// fingerprintToken returns the same one-way fingerprint tokenFingerprint computes for an
// on-disk .oauth-token, but for a token VALUE held in memory (the bytes a write is about
// to persist). It lets the gate compare a to-be-written token against homes' identities
// without the secret entering the registry. Whitespace is trimmed so it matches the
// on-disk fingerprint of the same token written with a trailing newline.
func fingerprintToken(token string) string {
	tok := trimToken(token)
	if tok == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(tok))
	return hex.EncodeToString(sum[:6])
}

// trimToken strips ASCII leading/trailing whitespace from a token string so an in-memory
// value and its newline-terminated on-disk form fingerprint identically.
func trimToken(s string) string {
	start, end := 0, len(s)
	for start < end && isSpace(s[start]) {
		start++
	}
	for end > start && isSpace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f'
}

// TokenWriteVerdict is the result of gating a setup-token write against a target home.
type TokenWriteVerdict struct {
	// Allow is the one-bit decision: true => the write is safe to perform.
	Allow bool `json:"allow"`
	// Reason is a short machine-stable token: "ok", "cross-account", or "empty-token".
	Reason string `json:"reason"`
	// Detail is a human-readable explanation for the operator.
	Detail string `json:"detail,omitempty"`
	// DirAccount is the account the TARGET dir is logged into (email or uuid), "" if none.
	DirAccount string `json:"dir_account,omitempty"`
	// TokenTwins names the OTHER config homes (under the same home root) already logged
	// into a different account than this token's — i.e. the homes a cross-account write
	// would make this dir collide with. Advisory; populated only for a refusal.
	TokenTwins []string `json:"token_twins,omitempty"`
}

// GateTokenWrite decides whether writing setupToken into dir is safe — the chokepoint
// every setup-token write (save-token, enroll, a switch flow) MUST consult first. It is
// pure over disk-derived identity and never writes anything.
//
// The rule: a setup token belongs to exactly one account. If the TARGET dir is logged
// into a known account (its .claude.json oauthAccount), and that account is NOT the one
// the token authenticates as, the write is REFUSED — it would make the dir a token-twin,
// burning another account's rate-limit bucket on every headless launch. The write is
// ALLOWED when (a) the dir has no interactive login to contradict (a fresh/half-set-up
// home — there is nothing to lie about), or (b) the token's account matches the dir's.
//
// We cannot read the account FROM a raw setup token (it is opaque), so "the token's
// account" is established by matching its fingerprint to a config home already logged in
// and carrying that same token. homeRoot is the dir under which sibling homes are
// discovered (typically ~); pass "" to skip the cross-home lookup and gate on the target
// dir's own login alone (which already catches the common case: writing token X into a
// dir logged into account Y when some sibling home is logged into X).
func GateTokenWrite(dir, setupToken, homeRoot string) TokenWriteVerdict {
	if trimToken(setupToken) == "" {
		return TokenWriteVerdict{Allow: false, Reason: "empty-token",
			Detail: "refusing to write an empty setup token"}
	}
	fp := fingerprintToken(setupToken)
	target := DeriveIdentity(dir)
	dirAcct := target.AccountKey()

	// Find which account, if any, this token already authenticates as: a sibling home
	// that is BOTH logged in AND carries this exact token fingerprint tells us the
	// token's true owner. Without a home root we can't look, so tokenAcct stays "".
	var tokenAcct string
	var twins []string
	if homeRoot != "" {
		if homes, err := Discover(homeRoot); err == nil {
			for _, h := range homes {
				id := h.Identity
				if id.TokenFP == fp && id.AccountKey() != "" {
					if tokenAcct == "" {
						tokenAcct = id.AccountKey()
					}
				}
			}
			// Collect the homes whose login differs from the TARGET dir's login — the
			// seats a wrong write would collide this dir with. Only meaningful when the
			// target has its own login to differ from.
			if dirAcct != "" {
				for _, h := range homes {
					if h.Identity.AccountKey() != "" && h.Identity.AccountKey() != dirAcct {
						twins = append(twins, h.Name)
					}
				}
				sort.Strings(twins)
			}
		}
	}

	// The refusal: the dir is logged into a known account, and we KNOW the token belongs
	// to a different one. This is the cross-account smear, refused before it lands.
	if dirAcct != "" && tokenAcct != "" && dirAcct != tokenAcct {
		return TokenWriteVerdict{
			Allow:      false,
			Reason:     "cross-account",
			DirAccount: identityLabel(target),
			TokenTwins: twins,
			Detail: fmt.Sprintf("refusing to write a setup token owned by %s into a dir logged in as %s "+
				"(this is the token-twin smear: headless launches here would burn the OTHER account's bucket). "+
				"Write the token into ITS OWN dir, or re-login this dir's intended account.",
				labelKey(tokenAcct), identityLabel(target)),
		}
	}

	return TokenWriteVerdict{Allow: true, Reason: "ok", DirAccount: identityLabel(target)}
}

// identityLabel is the friendliest human label for a home's identity: its login email if
// known, else its account key, else "(no login)".
func identityLabel(id Identity) string {
	if id.Email != "" {
		return id.Email
	}
	if k := id.AccountKey(); k != "" {
		return k
	}
	return "(no login)"
}

// labelKey turns an AccountKey ("uuid:…"/"tok:…") into a shorter operator-facing label.
func labelKey(key string) string {
	return key
}

// TwinFinding is one cross-account token collision: a set of config homes that share one
// setup-token fingerprint but whose interactive logins name MORE THAN ONE account. That
// is the smear — distinct from the benign case of two dir-names for one account.
type TwinFinding struct {
	// Fingerprint is the shared one-way token fingerprint (not the secret).
	Fingerprint string `json:"fingerprint"`
	// Homes are the config-home names sharing this token (sorted).
	Homes []string `json:"homes"`
	// Accounts are the DISTINCT logins among those homes (sorted) — len>1 is the bug.
	Accounts []string `json:"accounts"`
}

// AuditTokenTwins scans every config home under homeRoot and returns the cross-account
// token collisions: groups that share one setup-token fingerprint across two or more
// DISTINCT logins. A group of homes that share a token but all resolve to ONE account
// (the default + named-dir case) is NOT reported — that sharing is legitimate. An empty
// result means the tree is clean. This is the read-only audit a guard/CI gate reds on.
func AuditTokenTwins(homeRoot string) ([]TwinFinding, error) {
	homes, err := Discover(homeRoot)
	if err != nil {
		return nil, err
	}
	byFP := map[string][]Home{}
	for _, h := range homes {
		if fp := h.Identity.TokenFP; fp != "" {
			byFP[fp] = append(byFP[fp], h)
		}
	}
	var findings []TwinFinding
	for fp, group := range byFP {
		if len(group) < 2 {
			continue
		}
		acctSet := map[string]bool{}
		var names []string
		for _, h := range group {
			names = append(names, h.Name)
			if k := h.Identity.AccountKey(); k != "" {
				acctSet[identityLabel(h.Identity)] = true
			}
		}
		if len(acctSet) < 2 {
			continue // shared token, but all one account — legitimate
		}
		var accts []string
		for a := range acctSet {
			accts = append(accts, a)
		}
		sort.Strings(names)
		sort.Strings(accts)
		findings = append(findings, TwinFinding{Fingerprint: fp, Homes: names, Accounts: accts})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].Fingerprint < findings[j].Fingerprint })
	return findings, nil
}
