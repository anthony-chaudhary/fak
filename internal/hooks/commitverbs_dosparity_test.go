package hooks

import (
	"sort"
	"strings"
	"testing"
)

// commitverbs_dosparity — the #2089 alignment invariants between fak's commit-subject accept-set
// (commitVerbs, gate_commitmsg.go) and the DOS commit-audit referee's code-effect verbs
// (dosCodeEffectVerbs, dos_witness_verbs.go). The two sets are DELIBERATELY not equal: fak's is
// wider to keep its own false-flag rate ~1%, and the abstainHazard predictor bridges the "fak
// accepts / referee abstains" direction. These tests pin BOTH directions so a preview-clean
// feat/perf subject and a referee-witnessed subject stay in agreement:
//
//   (a) no SILENT abstain: every verb fak accepts on a feat/perf subject is either witnessed by
//       the referee or flagged by abstainHazard — a green preview never hides an ABSTAIN.
//   (b) SUPERSET: every imperative base verb the referee witnesses is accepted by fak — a subject
//       that BINDS at the referee is never red-flagged by `fak commit --preview` as ungradeable.

// refereeVerbBases maps the referee's non-regular inflected code verbs to the imperative base fak
// carries, so the superset check compares base-to-base (fak enforces imperative mood, so it need
// not carry the referee's inflected forms — only a base that covers each).
var refereeVerbBases = map[string]string{
	"drove":  "drive",
	"driven": "drive",
	"fed":    "feed",
	"showed": "show",
	"bound":  "bind",
}

// regularBases returns the candidate imperative base forms of a possibly-inflected verb by
// stripping -s/-es/-ies/-ed/-ied. It is intentionally over-generative; membership in commitVerbs
// decides.
func regularBases(v string) []string {
	out := []string{v}
	switch {
	case strings.HasSuffix(v, "ies"):
		out = append(out, strings.TrimSuffix(v, "ies")+"y")
	case strings.HasSuffix(v, "es"):
		out = append(out, strings.TrimSuffix(v, "es"), strings.TrimSuffix(v, "s"))
	case strings.HasSuffix(v, "s"):
		out = append(out, strings.TrimSuffix(v, "s"))
	}
	switch {
	case strings.HasSuffix(v, "ied"):
		out = append(out, strings.TrimSuffix(v, "ied")+"y")
	case strings.HasSuffix(v, "ed"):
		out = append(out, strings.TrimSuffix(v, "ed"), strings.TrimSuffix(v, "d"))
	}
	if b, ok := refereeVerbBases[v]; ok {
		out = append(out, b)
	}
	return out
}

// TestCommitVerbsSupersetOfRefereeCodeVerbs asserts direction (b): every code-effect verb the DOS
// referee witnesses is accepted by fak's gate (directly or via an imperative base form fak
// carries). A failure means fak red-flags a subject the referee would happily bind — the exact
// misalignment #2089 names.
func TestCommitVerbsSupersetOfRefereeCodeVerbs(t *testing.T) {
	var missing []string
	for v := range dosCodeEffectVerbs {
		accepted := false
		for _, base := range regularBases(v) {
			if commitVerbs[base] {
				accepted = true
				break
			}
		}
		if !accepted {
			missing = append(missing, v)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("fak commitVerbs rejects %d referee-witnessed code verb(s) — a subject that BINDS "+
			"at the DOS referee would be red-flagged by `fak commit --preview`: %v", len(missing), missing)
	}
}

// TestCommitVerbsNoSilentRefereeAbstain asserts direction (a): every verb fak accepts, on a
// feat/perf subject, either binds at the referee (dosWouldAbstainOnCodeEffect=false) or is flagged
// by abstainHazard. A failure means a green `fak commit --preview` hides a referee ABSTAIN — a
// change that would land unwitnessed with no warning.
func TestCommitVerbsNoSilentRefereeAbstain(t *testing.T) {
	var silent []string
	for v := range commitVerbs {
		subject := "feat(pkg): " + v + " the widget (fak pkg)"
		if ok, why := CommitMsgVerdict(subject); !ok {
			t.Errorf("commitVerbs contains %q but CommitMsgVerdict rejects %q: %s", v, subject, why)
			continue
		}
		if _, abstains := dosWouldAbstainOnCodeEffect(v+" the widget", "(pkg)"); abstains && abstainHazard(subject) == "" {
			silent = append(silent, v)
		}
	}
	sort.Strings(silent)
	if len(silent) > 0 {
		t.Errorf("%d fak-accepted verb(s) SILENTLY abstain at the referee (feat/perf, no abstainHazard "+
			"warning) — a green preview hides an unwitnessed change: %v", len(silent), silent)
	}
}
