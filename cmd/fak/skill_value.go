package main

// fak skill value — the per-skill outcome-VALUE ledger (issue #2873, epic
// #2871). Where `fak skill footprint` measures a skill's resident COST and
// Hermes' curator counts a skill's USAGE, this verb measures a skill's VALUE:
// the witnessed pass / cost / latency lift of sessions that LOADED the skill
// against matched same-task-class sessions that did NOT — the ablation-harness
// "loaded vs not-loaded" arm keyed by skill id. Skills whose measured pass-lift
// is <= 0 (with a comparable arm to say so) are surfaced for auto-revert; the
// valuation-basis gate (--gate) flags any active skill promoted with no
// measurement basis, the #2796 discipline applied to skills.
//
// Deterministic and offline: it folds a pre-written JSONL ledger of session
// outcomes (fak-skill-value-ledger/1). No ledger yet ⇒ it reports not-yet, never
// a fabricated lift.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/skillvalue"
)

// basisRow is one skill's promotion valuation basis, read from the --basis file
// (JSONL). A skill absent here — or present with an empty basis — is ungrounded.
type basisRow struct {
	SkillID        string `json:"skill_id"`
	ValuationBasis string `json:"valuation_basis"`
}

func runSkillValue(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill value", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledgerPath := fs.String("ledger", "", "session-outcome ledger (default: <root>/"+skillvalue.DefaultLedgerRel+")")
	basisPath := fs.String("basis", "", "valuation-basis JSONL ({skill_id,valuation_basis} per line); default: none (every active skill is ungrounded)")
	asJSON := fs.Bool("json", false, "emit machine-readable rollup JSON")
	gate := fs.Bool("gate", false, "exit non-zero if any active skill is promoted with no valuation basis (#2796 mirror)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak skill value: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := repoRoot()
	lpath := *ledgerPath
	if lpath == "" {
		lpath = root + "/" + skillvalue.DefaultLedgerRel
	}
	var sessions []skillvalue.SessionRow
	if b, err := os.ReadFile(lpath); err == nil {
		sessions = skillvalue.ParseLedger(string(b))
	} else if !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak skill value: read ledger %s: %v\n", lpath, err)
		return 1
	}

	basis := map[string]string{}
	if *basisPath != "" {
		b, err := os.ReadFile(*basisPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak skill value: read basis %s: %v\n", *basisPath, err)
			return 1
		}
		for _, r := range jsonlledger.Parse(string(b), func(r basisRow) bool { return r.SkillID != "" }) {
			basis[r.SkillID] = r.ValuationBasis
		}
	}

	r := skillvalue.Compute(sessions, basis)

	if *asJSON {
		if err := writeIndentedJSON(stdout, r); err != nil {
			fmt.Fprintf(stderr, "fak skill value: %v\n", err)
			return 1
		}
	} else {
		renderSkillValue(stdout, r, lpath)
	}

	if *gate {
		if g := r.Gate(); !g.OK {
			fmt.Fprintf(stderr, "fak skill value: valuation-basis gate FAILED — %d active skill(s) promoted with no basis: %s\n",
				len(g.Ungrounded), strings.Join(g.Ungrounded, ", "))
			return 1
		}
	}
	return 0
}

func renderSkillValue(w io.Writer, r skillvalue.Rollup, ledgerPath string) {
	if r.Sessions == 0 {
		fmt.Fprintf(w, "skill value: not yet — no %s rows at %s\n", skillvalue.LedgerSchema, ledgerPath)
		fmt.Fprintln(w, "  (write session-outcome rows to the ledger, then re-run to measure per-skill lift)")
		return
	}
	revert := r.AutoRevert()
	gate := r.Gate()
	fmt.Fprintf(w, "skill value: %d session(s), %d skill(s) — %d auto-revert, %d ungrounded\n",
		r.Sessions, len(r.Skills), len(revert), len(gate.Ungrounded))
	for _, s := range r.Skills {
		fmt.Fprintf(w, "  %-24s lift %+.3f  (loaded %.2f vs base %.2f)  cost %+.4f  lat %+.1fms  n=%d/%d",
			s.SkillID, s.PassLift, s.LoadedPass, s.BaselinePass, s.CostDelta, s.LatencyDelta, s.ComparableN, s.BaselineN)
		if len(s.Flags) > 0 {
			flags := append([]string(nil), s.Flags...)
			sort.Strings(flags)
			fmt.Fprintf(w, "  [%s]", strings.Join(flags, ","))
		}
		if s.ValuationBasis != "" {
			fmt.Fprintf(w, "  basis=%s", s.ValuationBasis)
		}
		fmt.Fprintln(w)
	}
	if len(revert) > 0 {
		fmt.Fprintf(w, "  → auto-revert (measured lift <= 0): %s\n", strings.Join(revert, ", "))
	}
	if !gate.OK {
		fmt.Fprintf(w, "  → valuation-basis gate: %d ungrounded skill(s): %s\n", len(gate.Ungrounded), strings.Join(gate.Ungrounded, ", "))
	}
}
