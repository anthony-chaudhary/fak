package fleetaccounts

import "testing"

// TestAccountIdentityFromAliasesAndNesting pins the alias/nesting extraction the #2253 fold
// leans on: camelCase oauthAccount spellings map onto the same canonical keys as snake_case
// registry rows, a nested identity blob is folded in, values are trimmed+lower-cased so a
// case/whitespace skew never reads as a mismatch, and token_fp is extracted but is NOT a key
// a match verdict is decided on (identityKeys excludes it).
func TestAccountIdentityFromAliasesAndNesting(t *testing.T) {
	if got := accountIdentityFrom(nil); got != nil {
		t.Fatalf("nil info -> %v, want nil", got)
	}
	if got := accountIdentityFrom(map[string]any{"unrelated": "x"}); got != nil {
		t.Fatalf("no identity field -> %v, want nil", got)
	}

	id := accountIdentityFrom(map[string]any{
		"accountUuid": "  AAAA  ",
		"tokenFP":     "FP1",
		"oauthAccount": map[string]any{
			"emailAddress":     "User@Example.COM",
			"organizationUuid": "Org-9",
		},
	})
	if id["account_uuid"] != "aaaa" {
		t.Fatalf("account_uuid = %q, want aaaa (trimmed+lowered from camelCase alias)", id["account_uuid"])
	}
	if id["login_email"] != "user@example.com" {
		t.Fatalf("login_email = %q, want user@example.com (nested camelCase alias)", id["login_email"])
	}
	if id["org_uuid"] != "org-9" {
		t.Fatalf("org_uuid = %q, want org-9 (nested camelCase alias)", id["org_uuid"])
	}
	if id["token_fp"] != "fp1" {
		t.Fatalf("token_fp = %q, want fp1 (extracted)", id["token_fp"])
	}
	// token_fp is extracted but must never decide a verdict.
	for _, k := range identityKeys {
		if k == "token_fp" {
			t.Fatalf("token_fp must not be an identity match key")
		}
	}

	// Top-level snake_case wins over a nested spelling of the same key (first source checked).
	top := accountIdentityFrom(map[string]any{
		"account_uuid": "top",
		"identity":     map[string]any{"account_uuid": "nested"},
	})
	if top["account_uuid"] != "top" {
		t.Fatalf("account_uuid = %q, want top (top-level source precedes nested)", top["account_uuid"])
	}
}

// TestIdentityMatchDecidability pins _identity_match's two-bit contract: it decides on the
// first key BOTH sides carry (account_uuid leads), and returns decided=false when the two
// share no key — the undecidable case the caller must fall through on rather than treat as a
// mismatch.
func TestIdentityMatchDecidability(t *testing.T) {
	eq, decided := identityMatch(
		map[string]string{"account_uuid": "a", "login_email": "x"},
		map[string]string{"account_uuid": "a", "login_email": "y"},
	)
	if !eq || !decided {
		t.Fatalf("shared account_uuid equal -> (%v,%v), want (true,true); account_uuid outranks a differing email", eq, decided)
	}

	eq, decided = identityMatch(
		map[string]string{"account_uuid": "a"},
		map[string]string{"account_uuid": "b"},
	)
	if eq || !decided {
		t.Fatalf("shared account_uuid differ -> (%v,%v), want (false,true)", eq, decided)
	}

	eq, decided = identityMatch(
		map[string]string{"account_uuid": "a"},
		map[string]string{"org_uuid": "o"},
	)
	if eq || decided {
		t.Fatalf("no shared key -> (%v,%v), want (false,false) undecidable", eq, decided)
	}
}

// idSession builds a session row carrying an identity under raw with a set age, so the
// verdict-order test can control which session is freshest.
func idSession(ageMin float64, raw map[string]any) Session {
	return Session{AgeMin: ageMin, hasAge: true, raw: raw}
}

// TestThrottleMatchesCurrentIdentityVerdictOrder walks the candidate precedence
// (probe -> config -> registry row -> freshest session) and the fail-closed defaults of the
// #2253 fold, driving each verdict through an explicit candidate so no filesystem is touched
// (dir="" skips the current-config read).
func TestThrottleMatchesCurrentIdentityVerdictOrder(t *testing.T) {
	// A throttle with no stamped identity always holds (fail-closed) regardless of candidates.
	if !throttleMatchesCurrentIdentity(".claude-a", "", map[string]any{"weekly": "x"}, Registry{}, nil, nil) {
		t.Fatalf("throttle with no identity must hold (return true)")
	}

	thr := map[string]any{"account_uuid": "AAAA"}

	// Probe identity is the first candidate: a probe mismatch clears the hold even when a later
	// registry row would match.
	reg := Registry{Accounts: []map[string]any{{"account": ".claude-a", "account_uuid": "AAAA"}}}
	if throttleMatchesCurrentIdentity(".claude-a", "", thr, reg, nil, map[string]string{"account_uuid": "BBBB"}) {
		t.Fatalf("probe-identity mismatch must clear the hold (return false) before the registry row is consulted")
	}

	// Registry accounts[] row decides when probe/config are absent: a foreign uuid clears.
	regMismatch := Registry{Accounts: []map[string]any{{"account": ".claude-a", "account_uuid": "BBBB"}}}
	if throttleMatchesCurrentIdentity(".claude-a", "", thr, regMismatch, nil, nil) {
		t.Fatalf("registry-row mismatch must clear the hold (return false)")
	}
	if !throttleMatchesCurrentIdentity(".claude-a", "", thr, reg, nil, nil) {
		t.Fatalf("registry-row match must keep the hold (return true)")
	}

	// Freshest session decides over an older one: the low-age row carrying a foreign uuid wins
	// even though the older row matches.
	sessions := []Session{
		idSession(90, map[string]any{"account_uuid": "AAAA"}), // older, matches
		idSession(3, map[string]any{"account_uuid": "BBBB"}),  // freshest, mismatch
	}
	if throttleMatchesCurrentIdentity(".claude-a", "", thr, Registry{}, sessions, nil) {
		t.Fatalf("freshest session mismatch must decide and clear the hold (return false)")
	}

	// Undecidable everywhere -> fail-closed hold: throttle stamps only org_uuid, every candidate
	// carries only account_uuid, so no pair shares a key and the seat stays held.
	orgThr := map[string]any{"org_uuid": "O1"}
	uuidOnly := []Session{idSession(1, map[string]any{"account_uuid": "AAAA"})}
	if !throttleMatchesCurrentIdentity(".claude-a", "", orgThr, Registry{}, uuidOnly, map[string]string{"account_uuid": "AAAA"}) {
		t.Fatalf("no shared identity key anywhere must fail closed (return true)")
	}
}
