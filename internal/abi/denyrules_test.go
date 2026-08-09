package abi

import (
	"strings"
	"testing"
)

// TestDenyRuleIDAdmitsTheAuthoredSpellings pins the two spellings the shipped
// rungs actually author — the arg-predicate family's snake_case atom and
// gitgate's hyphenated law prefix with its "refused:" punctuation — onto one
// canonical on-the-wire id.
func TestDenyRuleIDAdmitsTheAuthoredSpellings(t *testing.T) {
	cases := []struct{ raw, want string }{
		// arg-predicate rung: the WHOLE bounded detail string it already builds.
		{"rm_rf recursive/forced delete", DenyRuleRmRf},
		{"rce_pipe download|interpreter", DenyRuleRCEPipe},
		{"out_of_tree_write", DenyRuleOutOfTreeWrite},
		{"shell_dialect Get-Content", DenyRuleShellDialect},
		{"sudo_local doas", DenyRuleSudoLocal},
		{`deny_regex /\brm\s+-[A-Za-z]*[rRfF]/`, DenyRuleDenyRegex},
		{"allow_glob ./out/**", DenyRuleAllowGlob},
		{"max_bytes 4096", DenyRuleMaxBytes},
		// gitgate: the WHOLE law, whose leading atom is the law's authored id.
		{"skip-hooks refused: `git config core.hooksPath ...` persistently redirects", DenyRuleSkipHooks},
		{"off-trunk refused: `git checkout -b` opens an unmanaged branch", DenyRuleOffTrunk},
		{"NEVER_AMEND_SHARED: shared-history rewrite refused", DenyRuleNeverAmendShared},
		{"commit-by-explicit-path: `git add -A/--all` stages everything", DenyRuleCommitByPath},
		{"reset-hard refused: `git reset --hard` discards", DenyRuleResetHard},
		{"clean-force refused: `git clean -f` deletes untracked files", DenyRuleCleanForce},
		// bare canonical ids round-trip.
		{DenyRuleSelfModifyCommand, DenyRuleSelfModifyCommand},
	}
	for _, c := range cases {
		got, ok := DenyRuleID(c.raw)
		if !ok || got != c.want {
			t.Errorf("DenyRuleID(%q) = %q,%v; want %q,true", c.raw, got, ok, c.want)
		}
	}
}

// TestDenyRuleIDRejectsEverythingUnregistered is the disclosure guard. The field
// backed by this vocabulary is copied VERBATIM into the exported guard corpus, so
// its safety rests entirely on the value space being a compile-time closed set.
// A non-member must be rejected WHOLE — never trimmed, filtered, or truncated
// into the set, which is exactly how a scrub-and-filter label fused an
// env-assignment value into args_label before #5863.
func TestDenyRuleIDRejectsEverythingUnregistered(t *testing.T) {
	hostile := []string{
		"",
		"   ",
		"AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIhunter2EXAMPLEKEY",
		"sk-ant-api03-REDACTEDLOOKINGVALUE",
		"Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",
		"MYVAR=hunter2",
		"password",
		"/home/user/.ssh/id_rsa",
		"C:\\work\\fak\\.credentials.json",
		// near-misses: a prefix or suffix of a real id must NOT be admitted.
		"rm_r",
		"rm_rfx",
		"xrm_rf",
		"_rm_rf",
		"skip_hook",
		// a hostile value that CONTAINS a real id but is not one.
		"rm_rf=hunter2",
		"rm_rf/../../etc/passwd",
		// over the length pre-filter.
		strings.Repeat("a", maxDenyRuleIDLen+1),
	}
	for _, raw := range hostile {
		got, ok := DenyRuleID(raw)
		if ok || got != "" {
			t.Errorf("DenyRuleID(%q) = %q,%v; want \"\",false — the vocabulary must be closed", raw, got, ok)
		}
	}
}

// TestDenyRuleIDNeverReturnsInputBytes is the structural statement of the
// property above: the returned value is always one of the declared literals, so
// no byte of the caller's input can survive into the field. Anything else would
// mean the function is filtering rather than testing membership.
func TestDenyRuleIDNeverReturnsInputBytes(t *testing.T) {
	declared := map[string]bool{}
	for _, id := range DenyRuleIDs() {
		declared[id] = true
	}
	probes := []string{
		"rm_rf recursive/forced delete",
		"skip-hooks refused: x",
		"NEVER_AMEND_SHARED: y",
		"sk-ant-api03-SECRET",
		"totally-unknown-rung",
		"rm_rf\x00hunter2",
	}
	for _, p := range probes {
		got, ok := DenyRuleID(p)
		if !ok {
			if got != "" {
				t.Errorf("DenyRuleID(%q) rejected but returned %q, want \"\"", p, got)
			}
			continue
		}
		if !declared[got] {
			t.Errorf("DenyRuleID(%q) returned %q, which is NOT a declared literal", p, got)
		}
	}
}

// TestDenyRuleIDsIsSortedAndCanonical keeps the vocabulary itself well-formed:
// every declared id is already in canonical form (so a consumer switching on the
// exported list matches what lands on the wire), and the list is sorted so a
// diff of the value space is reviewable.
func TestDenyRuleIDsIsSortedAndCanonical(t *testing.T) {
	ids := DenyRuleIDs()
	if len(ids) == 0 {
		t.Fatal("DenyRuleIDs is empty")
	}
	for i, id := range ids {
		if i > 0 && ids[i-1] >= id {
			t.Fatalf("DenyRuleIDs not sorted/unique at %d: %q then %q", i, ids[i-1], id)
		}
		if got, ok := DenyRuleID(id); !ok || got != id {
			t.Errorf("declared id %q is not canonical: DenyRuleID(%q) = %q,%v", id, id, got, ok)
		}
		for _, r := range id {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			default:
				t.Errorf("declared id %q holds %q; ids are [a-z0-9_] only", id, r)
			}
		}
	}
}
