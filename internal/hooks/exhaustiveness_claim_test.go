package hooks

// exhaustiveness_claim_test.go — bind this package's prose gate COUNTS to the registry they
// quantify over (#5605, epic #5601).
//
// A count a human types into prose is correct on the day it is written and unowned every day
// after. This package carried a live instance: the package doc described the registry as holding
// a number that was right for the Python-era set and had drifted by nine gates, because ten gates
// were added since and nothing told any of those authors they had a debt. Every one of those
// additions was correct; the header was the only thing that was wrong, and it was wrong silently.
//
// fak already solves this shape twice, mechanically — architest's TestEveryPackageDeclaresTier
// (every internal package has a tier row) and failclosed_ledger_test.go's bidirectional
// registry/ledger membership. Both bind a claim to the registry it quantifies over. This points
// the same idea at a third kind of claim: a NUMBER in a comment.
//
// The rule: a comment that makes an EXHAUSTIVENESS claim about a gate registry — "all N gates",
// "every N gates", "the N gates" — must state that registry's live length. A bare number next to
// the word "gate" is NOT such a claim ("a tier-1 gate", "one gate per interpreter"), and is
// deliberately ignored: the quantifier is what turns a mention into a claim, and flagging
// mentions would make the check noise nobody keeps.

import (
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// gateCountClaimRE matches an exhaustiveness claim about a gate registry. The leading quantifier
// is REQUIRED — that is the whole discrimination between a claim and a mention.
var gateCountClaimRE = regexp.MustCompile(`(?i)\b(all|every|the)\s+(\d+)\s+((?:pre-commit|commit-msg|hygiene)\s+)?gates?\b`)

// gateCountClaim is one such claim, located well enough to fix.
type gateCountClaim struct {
	File     string
	Line     int
	Text     string // the matched phrase, for the failure message
	Claimed  int
	Registry string // "hygiene" or "pre-commit"
}

// findGateCountClaims scans Go COMMENTS (never string literals — the checker's own fixtures would
// otherwise trip it) for exhaustiveness claims.
func findGateCountClaims(t *testing.T, dir string) []gateCountClaim {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	var out []gateCountClaim
	for _, pkg := range pkgs {
		for name, f := range pkg.Files {
			for _, cg := range f.Comments {
				for _, c := range cg.List {
					out = append(out, claimsInComment(name, fset.Position(c.Pos()).Line, c.Text)...)
				}
			}
		}
	}
	return out
}

func claimsInComment(file string, line int, text string) []gateCountClaim {
	var out []gateCountClaim
	for _, m := range gateCountClaimRE.FindAllStringSubmatch(text, -1) {
		n, err := strconv.Atoi(m[2])
		if err != nil {
			continue
		}
		registry := "pre-commit"
		if strings.EqualFold(strings.TrimSpace(m[3]), "hygiene") {
			registry = "hygiene"
		}
		out = append(out, gateCountClaim{
			File: file, Line: line, Text: strings.TrimSpace(m[0]), Claimed: n, Registry: registry,
		})
	}
	return out
}

// The claim under test. Adding a gate without updating the package doc fails HERE, at the moment
// the debt is created, instead of leaving a reader to trust a number nobody owns.
func TestGateCountClaimsMatchTheRegistry(t *testing.T) {
	claims := findGateCountClaims(t, ".")

	// Non-vacuity: this package's doc is expected to state its own size. A scanner that silently
	// found nothing would pass forever while the prose rotted — the exact failure being fixed.
	if len(claims) == 0 {
		t.Fatal("no gate-count claim found anywhere in package hooks: either the package doc stopped stating the registry's size (state it, so a reader need not count the registry by hand), or this scanner has stopped matching it")
	}

	for _, msg := range gateCountClaimMismatches(claims, liveGateCounts()) {
		t.Error(msg)
	}
}

// liveGateCounts is the denominator side of the binding: the registries, read at test time.
func liveGateCounts() map[string]int {
	return map[string]int{
		"pre-commit": len(PreCommitGates()),
		"hygiene":    len(HygieneGates()),
	}
}

// gateCountClaimMismatches is the comparison itself, kept separate from the tree walk so the
// ENFORCEMENT can be tested on a stale claim without staling the real package doc to do it.
func gateCountClaimMismatches(claims []gateCountClaim, live map[string]int) []string {
	var out []string
	for _, c := range claims {
		want := live[c.Registry]
		if c.Claimed != want {
			out = append(out, strings.Join([]string{
				c.File + ":" + strconv.Itoa(c.Line) + " claims " + strconv.Quote(c.Text) +
					", but the " + c.Registry + " gate registry holds " + strconv.Itoa(want) + ".",
				"  A count in prose is unowned the day after it is written — this is the mechanism that owns it.",
				"  Fix the comment to say " + strconv.Itoa(want) + " (or drop the number if the sentence does not need one).",
			}, "\n"))
		}
	}
	return out
}

// The enforcement path, proven on a stale claim. Without this, a green run only shows that the
// tree happens to agree right now — not that disagreement would be caught.
func TestGateCountClaimMismatchIsReportedWithBothNumbers(t *testing.T) {
	live := liveGateCounts()
	stale := claimsInComment("hooks.go", 8, "// This package collapses all 8 gates into one Go process")
	if len(stale) != 1 {
		t.Fatalf("fixture did not parse as a claim: %+v", stale)
	}
	if stale[0].Claimed == live["pre-commit"] {
		t.Skipf("the registry now holds %d, so this fixture is no longer stale", live["pre-commit"])
	}

	msgs := gateCountClaimMismatches(stale, live)
	if len(msgs) != 1 {
		t.Fatalf("a stale claim produced %d mismatches, want exactly 1", len(msgs))
	}
	// The message has to carry BOTH numbers, or the reader cannot tell what to change it to.
	for _, want := range []string{"hooks.go:8", "all 8 gates", strconv.Itoa(live["pre-commit"])} {
		if !strings.Contains(msgs[0], want) {
			t.Errorf("mismatch message does not mention %q:\n%s", want, msgs[0])
		}
	}

	// And a claim that AGREES must produce nothing — the check must not fire on every count.
	agreeing := claimsInComment("hooks.go", 8, "// registers all "+strconv.Itoa(live["pre-commit"])+" gates today")
	if msgs := gateCountClaimMismatches(agreeing, live); len(msgs) != 0 {
		t.Fatalf("an accurate claim was reported as a mismatch: %v", msgs)
	}
}

// The scanner's own contract, on fixtures rather than on the tree — so "the real package happens to
// be consistent right now" can never be mistaken for "the check works".
func TestGateCountClaimScannerMatchesClaimsAndIgnoresMentions(t *testing.T) {
	for _, tc := range []struct {
		name     string
		comment  string
		want     int // -1 = expect no claim
		registry string
	}{
		{name: "all N gates is a claim", comment: "// collapses all 17 gates into one process", want: 17, registry: "pre-commit"},
		{name: "the N gates is a claim", comment: "// the 17 gates run in order", want: 17, registry: "pre-commit"},
		{name: "every N gates is a claim", comment: "// every 17 gates share one diff read", want: 17, registry: "pre-commit"},
		{name: "singular is a claim too", comment: "// all 1 gate", want: 1, registry: "pre-commit"},
		{name: "hygiene is routed to its own registry", comment: "// all 9 hygiene gates run whole-tree", want: 9, registry: "hygiene"},
		{name: "pre-commit qualifier is explicit", comment: "// all 17 pre-commit gates", want: 17, registry: "pre-commit"},
		{name: "case is ignored", comment: "// ALL 17 GATES", want: 17, registry: "pre-commit"},

		// Mentions, NOT claims — no exhaustiveness quantifier. The tier-1 case is real: it appears
		// in gate_pythongate.go and a quantifier-free scanner flags it as a bogus count of one.
		{name: "a hyphenated tier is not a claim", comment: "// so the tier-1 gate stays import-clean", want: -1},
		{name: "a bare count is not a claim", comment: "// 7 gates were ported from Python", want: -1},
		{name: "one spelled out is not a claim", comment: "// one Python interpreter per gate", want: -1},
		{name: "an unrelated noun is not a claim", comment: "// all 8 checkers", want: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := claimsInComment("fixture.go", 1, tc.comment)
			if tc.want < 0 {
				if len(got) != 0 {
					t.Fatalf("comment %q was read as a claim (%+v); a mention without a quantifier must be ignored", tc.comment, got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("comment %q yielded %d claims, want exactly 1", tc.comment, len(got))
			}
			if got[0].Claimed != tc.want {
				t.Errorf("comment %q claimed %d, want %d", tc.comment, got[0].Claimed, tc.want)
			}
			if got[0].Registry != tc.registry {
				t.Errorf("comment %q routed to registry %q, want %q", tc.comment, got[0].Registry, tc.registry)
			}
		})
	}
}

// The drift this was written for, restated as a test: had the mechanism existed, a stale count
// would have failed rather than shipped. Kept as a direct assertion so the regression is legible
// without re-reading git history.
func TestGateCountClaimDetectsAStaleCount(t *testing.T) {
	stale := claimsInComment("hooks.go", 8, "// This package collapses all 8 gates into one Go process")
	if len(stale) != 1 {
		t.Fatalf("the historical stale header did not parse as a claim: %+v", stale)
	}
	if stale[0].Claimed == len(PreCommitGates()) {
		t.Fatalf("fixture is no longer stale: the registry now holds %d, so this test proves nothing — pick a different number", len(PreCommitGates()))
	}
}
