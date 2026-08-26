package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
)

var sessionJournalAuditNow = time.Now

// runSessionJournalAudit implements the read-only launch-to-provider-journal proof.
// It always emits a report, including when an authority is unreadable; every RED
// verdict exits one so automation cannot mistake an unproven cohort for healthy.
func runSessionJournalAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session journal-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.Duration("since", 24*time.Hour, "recent SessionStart identity window")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding the live resume_identity.jsonl authority")
	homeFlag := fs.String("home", "", "user home under which all .claude* and .codex* roots are discovered")
	asJSON := fs.Bool("json", false, "emit the versioned audit report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 || *since <= 0 {
		fmt.Fprintln(stderr, "usage: fak session journal-audit [--since 24h] [--reg-dir DIR] [--home DIR] [--json]")
		return 2
	}

	regDir := resolveSweepRegDir(*regDirFlag)
	identityPath := resume.IdentityLedgerPath(regDir)
	identityRows, invalidLines, readErr := resume.LoadIdentityRowsStrict(regDir)
	authorityErrors := []sessiondiag.JournalAuthorityError{}
	if readErr != nil {
		authorityErrors = append(authorityErrors, sessiondiag.JournalAuthorityError{
			Path: identityPath, Code: "IDENTITY_JOURNAL_READ_FAILED",
			Detail: "cannot read the live resume identity authority",
		})
	}
	if invalidLines > 0 {
		authorityErrors = append(authorityErrors, sessiondiag.JournalAuthorityError{
			Path: identityPath, Code: "IDENTITY_JOURNAL_INVALID_JSON",
			Detail: fmt.Sprintf("%d non-empty row(s) are invalid JSON", invalidLines),
		})
	}
	launches := make([]sessiondiag.JournalLaunchIdentity, 0, len(identityRows))
	for _, row := range identityRows {
		launchAt, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(row.TS))
		if err != nil {
			authorityErrors = append(authorityErrors, sessiondiag.JournalAuthorityError{
				Path: identityPath, Code: "IDENTITY_JOURNAL_INVALID_TIMESTAMP",
				Detail: "an identity row has no parseable launch timestamp",
			})
			continue
		}
		launches = append(launches, sessiondiag.JournalLaunchIdentity{
			Identity: row.UUID, Trace: row.Trace, LaunchAt: launchAt,
			Provider: row.Provider, Account: row.Account, Via: row.Via, Source: row.Source,
		})
	}
	home := resolveFleetUserHome(*homeFlag, "FLEET_USER_HOME")
	report := sessiondiag.AuditRecentLaunches(sessiondiag.JournalAuditOptions{
		Now: sessionJournalAuditNow(), Window: *since, IdentityPath: identityPath,
		Identities: launches, UserHome: home,
		CodexHome:       strings.TrimSpace(os.Getenv("CODEX_HOME")),
		ClaudeConfigDir: strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")),
		AuthorityErrors: authorityErrors,
	})
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(stderr, "fak session journal-audit: encode report:", err)
			return 1
		}
	} else {
		renderSessionJournalAudit(stdout, report)
	}
	if report.Verdict != sessiondiag.JournalVerdictGreen {
		return 1
	}
	return 0
}

func renderSessionJournalAudit(w io.Writer, report sessiondiag.JournalAuditReport) {
	fmt.Fprintf(w, "SESSION JOURNAL AUDIT %s — %s\n", strings.ToUpper(report.Verdict), report.Summary)
	fmt.Fprintf(w, "window=%s..%s identity_journal=%s roots=%d\n", report.Window.Start, report.Window.End, report.IdentityJournal, len(report.Roots))
	if len(report.AuthorityErrors) > 0 {
		for _, item := range report.AuthorityErrors {
			fmt.Fprintf(w, "authority_error provider=%s code=%s path=%s\n", fallbackJournalValue(item.Provider, "all"), item.Code, item.Path)
		}
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	printed := false
	for _, row := range report.Rows {
		if row.Status == sessiondiag.JournalStatusAdvanced {
			continue
		}
		if !printed {
			fmt.Fprintln(tw, "IDENTITY\tPROVIDER\tSTATUS\tLAUNCH\tCURSOR\tPATH")
			printed = true
		}
		cursor := "-"
		if row.BaselineCursor != nil {
			cursor = row.BaselineCursor.ID + "@" + row.BaselineCursor.At
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", row.Identity, row.Provider, row.Status, row.LaunchAt, cursor, fallbackJournalValue(row.TranscriptPath, "-"))
	}
	_ = tw.Flush()
	if report.Verdict != sessiondiag.JournalVerdictGreen {
		fmt.Fprintln(w, "next: repair unreadable authority or the listed exact identity joins, then rerun this read-only audit")
	}
}

func fallbackJournalValue(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
