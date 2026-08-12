package main

// trajctl_depth.go — the impure shell over internal/depthadmit: the shipped
// consumer that makes the depth fold load-bearing rather than a library nobody
// calls.
//
// It does two things, and both are the same read:
//
//	fak trajctl depth --id ID   prints how far down its declared plan an objective
//	                            actually got, plus the handoff line naming the next
//	                            phase — the successor's resume point.
//	fak trajctl close --status met   is now GATED on that read. Before this, `close`
//	                            wrote `met` on a six-phase plan with one phase
//	                            witnessed and nothing objected; a shallow close was
//	                            indistinguishable from a deep one.
//
// WHERE THE WITNESS COMES FROM. Not a self-report and not a fresh git shell: the
// W3 `witnessed-commit-progress` score rows already in the ledger. That scorer
// (internal/trajctl commitscorer.go) appends an EvidenceRef per phase whose
// candidate commit resolved to EvidenceVerified, carrying the phase id in Detail.
// Reading those Details back is therefore a re-read of an already-verified
// resolution, which is why this shell needs no resolver of its own.
//
// LATEST ROW, NOT THE UNION. Only the most recent W3 row counts. Unioning every
// row a scoring pass ever wrote would keep crediting a phase whose commit later
// went dangling — the ledger would remember a witness that no longer exists.
// Taking the latest snapshot means credit can be LOST, which is exactly what
// depthadmit.PersistenceRegressed is for.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/depthadmit"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// trajctlDepthInput projects an objective and the folded ledger into the pure
// fold's input: the declared plan, plus the phase ids the latest W3 commit-progress
// row verified.
func trajctlDepthInput(obj trajctl.Objective, st trajctl.State) depthadmit.Input {
	in := depthadmit.Input{Plan: make([]depthadmit.Phase, 0, len(obj.Plan))}
	for _, p := range obj.Plan {
		in.Plan = append(in.Plan, depthadmit.Phase{ID: p.ID, Title: p.Title})
	}
	in.Witnessed = trajctlWitnessedPhases(obj.ID, st)
	return in
}

// trajctlWitnessedPhases returns the phase ids the LATEST W3 witnessed-commit-progress
// row verified for objID. A missing row (scoring never ran, or ran before any phase
// resolved) yields nothing, so an unscored objective reads as shallow rather than as
// silently complete — the fail-closed direction.
func trajctlWitnessedPhases(objID string, st trajctl.State) []string {
	latest := -1
	for i, row := range st.Scores {
		if row.ObjectiveID != objID {
			continue
		}
		if row.Method != trajctl.CommitScorerMethod || row.Witness != trajctl.W3 {
			continue
		}
		latest = i
	}
	if latest < 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, ev := range st.Scores[latest].Evidence {
		if ev.Kind != "commit" {
			continue
		}
		phase := strings.TrimSpace(ev.Detail)
		if phase == "" || seen[phase] {
			continue
		}
		seen[phase] = true
		out = append(out, phase)
	}
	sort.Strings(out)
	return out
}

func runTrajctlDepth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl depth", flag.ContinueOnError)
	fs.SetOutput(stderr)
	id := fs.String("id", "", "objective id to read (default: every open objective)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the depth reports as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	st := trajctl.Fold(trajctl.ReadLedgerFile(trajctlLedgerPath(*ledger)))

	var ids []string
	if *id != "" {
		if _, ok := st.Objectives[*id]; !ok {
			fmt.Fprintf(stderr, "fak trajctl depth: unknown objective %q\n", *id)
			return 1
		}
		ids = []string{*id}
	} else {
		for oid, obj := range st.Objectives {
			if obj.Status == trajctl.StatusActive || obj.Status == trajctl.StatusPaused {
				ids = append(ids, oid)
			}
		}
		sort.Strings(ids)
	}

	type entry struct {
		ObjectiveID string            `json:"objective_id"`
		Status      string            `json:"status"`
		Report      depthadmit.Report `json:"report"`
		Handoff     string            `json:"handoff"`
	}
	out := make([]entry, 0, len(ids))
	for _, oid := range ids {
		obj := st.Objectives[oid]
		rep := depthadmit.Fold(trajctlDepthInput(obj, st))
		out = append(out, entry{
			ObjectiveID: oid,
			Status:      string(obj.Status),
			Report:      rep,
			Handoff:     depthadmit.HandoffLine(oid, rep),
		})
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(stderr, "fak trajctl depth: %v\n", err)
			return 1
		}
		return 0
	}
	if len(out) == 0 {
		fmt.Fprintln(stdout, "no open objectives")
		return 0
	}
	for _, e := range out {
		fmt.Fprintf(stdout, "%s [%s] %s\n", e.ObjectiveID, e.Status, e.Report.Verdict)
		fmt.Fprintf(stdout, "  %s\n", e.Handoff)
		if len(e.Report.Coverage.Foreign) > 0 {
			fmt.Fprintf(stdout, "  off-plan witnesses (not credited): %s\n", strings.Join(e.Report.Coverage.Foreign, ", "))
		}
	}
	return 0
}
