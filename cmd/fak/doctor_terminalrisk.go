package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/terminalrisk"
)

func runDoctorTerminalRisk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor terminal-risk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "back up settings and set global graphicsAPI=direct2d")
	settings := fs.String("settings", defaultWTSettingsPath(), "Windows Terminal settings.json path")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	facts, err := gatherTerminalRiskFactsFn(*settings)
	if err != nil {
		fmt.Fprintf(stderr, "fak doctor terminal-risk: %v\n", err)
		return 1
	}
	report, err := terminalrisk.Assess(facts)
	if err != nil {
		fmt.Fprintf(stderr, "fak doctor terminal-risk: %v\n", err)
		return 1
	}
	backup := ""
	if *apply && report.Risk {
		backup, err = terminalrisk.ApplyDirect2D(*settings, facts.Settings, time.Now())
		if err != nil {
			fmt.Fprintf(stderr, "fak doctor terminal-risk: %v\n", err)
			return 1
		}
		facts.Settings, err = osReadFile(*settings)
		if err != nil {
			return 1
		}
		report, err = terminalrisk.Assess(facts)
		if err != nil {
			return 1
		}
	}
	result := struct {
		terminalrisk.Report
		Backup          string `json:"backup,omitempty"`
		RestartRequired bool   `json:"restart_required"`
		RDPAdvice       string `json:"rdp_advice"`
	}{report, backup, backup != "", terminalRiskRDPAdvice()}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak doctor terminal-risk")
	}
	verdict := "CLEAN"
	if report.Risk {
		verdict = "RISK"
	}
	fmt.Fprintf(stdout, "terminal render risk: %s — %s\n", verdict, report.Reason)
	if backup != "" {
		fmt.Fprintf(stdout, "backup: %s\nWindows Terminal must be fully restarted.\n", backup)
	}
	if result.RDPAdvice != "" {
		fmt.Fprintln(stdout, result.RDPAdvice)
	}
	if report.Risk {
		return 3
	}
	return 0
}

var osReadFile = os.ReadFile

var gatherTerminalRiskFactsFn = gatherTerminalRiskFacts
