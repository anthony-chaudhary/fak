package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const codexWorkflowDefaultReportSchema = "fak.codex_workflow_default_report.v1"

type codexWorkflowDefaultReport struct {
	Schema          string         `json:"schema"`
	CodexHome       string         `json:"codex_home"`
	WitnessFiles    int            `json:"witness_files"`
	ValidDecisions  int            `json:"valid_decisions"`
	Malformed       int            `json:"malformed"`
	GuardJoined     int            `json:"guard_joined"`
	Classifications map[string]int `json:"classifications"`
	Decisions       map[string]int `json:"decisions"`
	ObservedUse     int            `json:"observed_workflow_use"`
	UnknownOutcome  int            `json:"unknown_outcome"`
	EvidenceNote    string         `json:"evidence_note"`
}

func runSessionsWorkflowDefaultReport(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("sessions workflow-default-report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if code, done := parseFlagsRejectArgs(fs, args, stderr); done {
		return code
	}
	report, err := collectCodexWorkflowDefaultReport(*codexHome)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions workflow-default-report: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak sessions workflow-default-report")
	}
	fmt.Fprintf(stdout, "guarded Codex workflow default: %d valid / %d files; malformed=%d guard-joined=%d\n", report.ValidDecisions, report.WitnessFiles, report.Malformed, report.GuardJoined)
	fmt.Fprintf(stdout, "classifications=%v decisions=%v observed-use=%d unknown-outcome=%d\n", report.Classifications, report.Decisions, report.ObservedUse, report.UnknownOutcome)
	fmt.Fprintln(stdout, report.EvidenceNote)
	return 0
}

func collectCodexWorkflowDefaultReport(codexHome string) (codexWorkflowDefaultReport, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return codexWorkflowDefaultReport{}, err
	}
	report := codexWorkflowDefaultReport{
		Schema: codexWorkflowDefaultReportSchema, CodexHome: home,
		Classifications: map[string]int{}, Decisions: map[string]int{},
		EvidenceNote: "injection is FAK-authored evidence; observed workflow use requires a joined downstream receipt and is not inferred from injection",
	}
	entries, err := os.ReadDir(filepath.Join(home, "fak-workflow-defaults"))
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil
		}
		return report, err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		report.WitnessFiles++
		raw, err := os.ReadFile(filepath.Join(home, "fak-workflow-defaults", entry.Name()))
		if err != nil {
			report.Malformed++
			continue
		}
		var witness codexWorkflowDefaultWitness
		if json.Unmarshal(raw, &witness) != nil || witness.Schema != "fak.codex_workflow_default.v1" || witness.SessionID == "" || witness.Classification == "" || witness.Decision == "" {
			report.Malformed++
			continue
		}
		report.ValidDecisions++
		report.Classifications[witness.Classification]++
		report.Decisions[witness.Decision]++
		if codexGuardWitnessExists(home, witness.SessionID) {
			report.GuardJoined++
		}
		report.UnknownOutcome++
	}
	return report, nil
}
