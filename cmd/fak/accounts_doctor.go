package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// `fak accounts doctor` — the one-command recover/clean fold over the seat registry.
// `status` reports each seat's login state; doctor goes one rung further: it folds the
// config plane (registry + disk truth) and, when FLEET_REG_DIR is wired, the active
// probe ledger into ONE closed per-seat recovery action, and — with --write — applies
// the deterministic, non-destructive repairs itself (tombstone+rehome a seat whose
// config dir vanished, through the exact same audited path as `remove`). Everything
// judgment-shaped (re-login, credits, duplicate collapse) stays a reported action with
// the exact command, never an auto-mutation.
const doctorSchema = "fak.accounts.doctor.v1"

// doctorAction is the closed per-seat recovery vocabulary. Exactly one action is
// assigned per seat; "none" means the seat needs nothing.
type doctorAction string

const (
	doctorNone           doctorAction = "none"
	doctorRelogin        doctorAction = "relogin"          // needs_login, or a fresh auth/access wall
	doctorWaitReset      doctorAction = "wait_reset"       // fresh usage limit; recovers by itself
	doctorTopUp          doctorAction = "top_up"           // fresh credit wall; needs billing, not code
	doctorPrune          doctorAction = "prune"            // config dir vanished; tombstone+rehome (auto with --write)
	doctorHydrate        doctorAction = "hydrate"          // canonical same-account home is missing creds/sessions; copy from ready peer
	doctorEnableOrRemove doctorAction = "enable_or_remove" // explicitly disabled; operator judgment
	doctorDedupe         doctorAction = "dedupe"           // duplicate identity bucket; retire the extra seat
)

// doctorSeat is one seat's folded verdict.
type doctorSeat struct {
	Name      string       `json:"name"`
	Status    string       `json:"status"`
	Action    doctorAction `json:"action"`
	AutoFix   bool         `json:"auto_fix"`
	Reason    string       `json:"reason,omitempty"`
	Command   string       `json:"command,omitempty"`
	Reset     string       `json:"reset,omitempty"`
	Source    string       `json:"source,omitempty"`
	Applied   bool         `json:"applied,omitempty"`
	ApplyNote string       `json:"apply_note,omitempty"`
}

// acctDoctorReport is the machine-readable doctor surface.
type acctDoctorReport struct {
	Schema      string       `json:"schema"`
	Registry    string       `json:"registry"`
	ProbeLedger bool         `json:"probe_ledger_consulted"`
	Seats       []doctorSeat `json:"seats"`
	Actionable  int          `json:"actionable"`
	AutoFixable int          `json:"auto_fixable"`
	Applied     int          `json:"applied"`
}

type acctFixSummary struct {
	Actionable  int            `json:"actionable"`
	AutoFixable int            `json:"auto_fixable"`
	ByAction    map[string]int `json:"by_action,omitempty"`
	Seats       []acctFixSeat  `json:"seats,omitempty"`
}

type acctFixSeat struct {
	Name    string `json:"name"`
	Status  string `json:"status"`
	Action  string `json:"action"`
	Command string `json:"command,omitempty"`
	Reason  string `json:"reason,omitempty"`
	Reset   string `json:"reset,omitempty"`
}

// accountsDoctor folds every seat into a recovery action and (with write) applies the
// auto-fixable ones. Exit 0 when nothing is left to do, 1 while actions remain — so a
// watchdog can run `fak accounts doctor --write` and alert on nonzero.
func accountsDoctor(stdout, stderr io.Writer, registryPath, dosView, jobView string, asJSON, write bool) int {
	reg, ok := loadRegistryOrErr(stderr, registryPath)
	if !ok {
		return 1
	}
	reg = reg.Refresh()
	report := buildAccountsDoctorReport(registryPath, reg)

	if write {
		for i := range report.Seats {
			s := &report.Seats[i]
			if s.Action != doctorPrune && s.Action != doctorHydrate {
				continue
			}
			switch s.Action {
			case doctorPrune:
				// The exact audited remove path: tombstone + rehome to the anchor, move any
				// roles off the seat, defer the view re-sync to one pass below. A seat the
				// remove path refuses (e.g. no anchor to rehome to) stays reported, not fixed.
				var out, errBuf bytes.Buffer
				code := runAccountsRemove(&out, &errBuf, removeParams{
					name:         s.Name,
					reason:       "fak accounts doctor: config directory missing",
					registryPath: registryPath,
					dosView:      dosView,
					jobView:      jobView,
					noSync:       true,
				})
				if code == 0 {
					s.Applied = true
					s.ApplyNote = strings.TrimSpace(out.String())
				} else {
					s.ApplyNote = "skipped: " + strings.TrimSpace(errBuf.String())
				}
			case doctorHydrate:
				note, err := applyAccountHydrate(reg, s.Name, s.Source)
				if err != nil {
					s.ApplyNote = "skipped: " + err.Error()
				} else {
					s.Applied = true
					s.ApplyNote = note
				}
			}
		}
		applied := 0
		for _, s := range report.Seats {
			if s.Applied {
				applied++
			}
		}
		if applied > 0 {
			if code := syncViewsUnlessNoSync(stdout, stderr, registryPath, dosView, jobView, false); code != 0 {
				return code
			}
		}
	}

	foldDoctorReportCounts(&report)

	if asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		stdout.Write(append(b, '\n'))
	} else {
		printDoctorTable(stdout, report, write)
	}
	if report.Actionable > 0 {
		return 1
	}
	return 0
}

func buildAccountsDoctorReport(registryPath string, reg accounts.Registry) acctDoctorReport {
	report := acctDoctorReport{
		Schema:      doctorSchema,
		Registry:    registryPath,
		ProbeLedger: strings.TrimSpace(os.Getenv("FLEET_REG_DIR")) != "",
	}
	login := reg.LoginReport()
	for _, obs := range login.Seats {
		report.Seats = append(report.Seats, foldDoctorSeat(obs, report.ProbeLedger))
	}
	markHydrateRepairs(&report, login.Seats)
	foldDoctorReportCounts(&report)
	return report
}

func foldDoctorReportCounts(report *acctDoctorReport) {
	report.Actionable = 0
	report.AutoFixable = 0
	report.Applied = 0
	for _, s := range report.Seats {
		if s.Applied {
			report.Applied++
			continue
		}
		if s.Action != doctorNone {
			report.Actionable++
			if s.AutoFix {
				report.AutoFixable++
			}
		}
	}
}

func summarizeAccountFixes(report acctDoctorReport) acctFixSummary {
	sum := acctFixSummary{Actionable: report.Actionable, AutoFixable: report.AutoFixable}
	for _, s := range report.Seats {
		if s.Applied || s.Action == doctorNone {
			continue
		}
		if sum.ByAction == nil {
			sum.ByAction = map[string]int{}
		}
		action := string(s.Action)
		sum.ByAction[action]++
		sum.Seats = append(sum.Seats, acctFixSeat{
			Name:    s.Name,
			Status:  s.Status,
			Action:  action,
			Command: firstNonEmpty(s.Command, s.Source),
			Reason:  s.Reason,
			Reset:   s.Reset,
		})
	}
	return sum
}

func markHydrateRepairs(report *acctDoctorReport, seats []accounts.LoginObservation) {
	byName := map[string]accounts.LoginObservation{}
	for _, obs := range seats {
		byName[obs.Name] = obs
	}
	for i := range report.Seats {
		s := &report.Seats[i]
		if s.Action != doctorRelogin || accounts.LoginStatus(s.Status) != accounts.LoginNeedsLogin {
			continue
		}
		target, ok := byName[s.Name]
		if !ok || target.Account == "" || target.IdentityRole != accounts.RoleCanonical {
			continue
		}
		var src accounts.LoginObservation
		for _, peerName := range target.Peers {
			peer, ok := byName[peerName]
			if !ok || peer.Account != target.Account || peer.Status != accounts.LoginReady || !peer.HasCreds || peer.Dir == "" {
				continue
			}
			if src.Name == "" || hydrateSourceRank(peer) > hydrateSourceRank(src) || (hydrateSourceRank(peer) == hydrateSourceRank(src) && peer.Name < src.Name) {
				src = peer
			}
		}
		if src.Name == "" {
			continue
		}
		s.Action = doctorHydrate
		s.AutoFix = true
		s.Source = src.Name
		s.Reason = "canonical home missing live credentials/sessions; same account is ready in " + src.Name
		s.Command = "fak accounts doctor --write"
	}
}

func hydrateSourceRank(obs accounts.LoginObservation) int {
	if strings.EqualFold(obs.Name, "default") {
		return 1
	}
	return 2
}

func applyAccountHydrate(reg accounts.Registry, targetName, sourceName string) (string, error) {
	target, ok := homeByName(reg, targetName)
	if !ok {
		return "", fmt.Errorf("target %q not in registry", targetName)
	}
	source, ok := homeByName(reg, sourceName)
	if !ok {
		return "", fmt.Errorf("source %q not in registry", sourceName)
	}
	if target.Identity.AccountKey() == "" || target.Identity.AccountKey() != source.Identity.AccountKey() {
		return "", fmt.Errorf("source %q and target %q are not the same account bucket", sourceName, targetName)
	}
	if target.Dir == "" || source.Dir == "" {
		return "", fmt.Errorf("source/target config dir missing")
	}
	if err := os.MkdirAll(target.Dir, 0o755); err != nil {
		return "", err
	}
	backupDir := filepath.Join(target.Dir, "backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return "", err
	}
	stamp := time.Now().UTC().Format("20060102T150405Z")
	copiedCred := ""
	if fileExists(filepath.Join(source.Dir, ".credentials.json")) {
		if err := backupIfExists(filepath.Join(target.Dir, ".credentials.json"), backupDir, ".credentials.json.before-hydrate-"+stamp+".bak"); err != nil {
			return "", err
		}
		if err := copyFile(filepath.Join(source.Dir, ".credentials.json"), filepath.Join(target.Dir, ".credentials.json")); err != nil {
			return "", err
		}
		if err := backupIfExists(filepath.Join(target.Dir, ".oauth-token"), backupDir, ".oauth-token.before-hydrate-"+stamp+".bak"); err != nil {
			return "", err
		}
		_ = os.Remove(filepath.Join(target.Dir, ".oauth-token"))
		copiedCred = ".credentials.json"
	} else if fileExists(filepath.Join(source.Dir, ".oauth-token")) {
		if err := backupIfExists(filepath.Join(target.Dir, ".oauth-token"), backupDir, ".oauth-token.before-hydrate-"+stamp+".bak"); err != nil {
			return "", err
		}
		if err := copyFile(filepath.Join(source.Dir, ".oauth-token"), filepath.Join(target.Dir, ".oauth-token")); err != nil {
			return "", err
		}
		copiedCred = ".oauth-token"
	} else {
		return "", fmt.Errorf("source %q has no credential file to copy", sourceName)
	}
	copiedSessions, err := copyMissingProjectFiles(filepath.Join(source.Dir, "projects"), filepath.Join(target.Dir, "projects"))
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("hydrated %s from %s: copied %s and %d missing project file(s)", targetName, sourceName, copiedCred, copiedSessions), nil
}

func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}

func backupIfExists(path, backupDir, name string) error {
	if !fileExists(path) {
		return nil
	}
	return copyFile(path, filepath.Join(backupDir, name))
}

func copyMissingProjectFiles(srcRoot, dstRoot string) (int, error) {
	if fi, err := os.Stat(srcRoot); err != nil || !fi.IsDir() {
		return 0, nil
	}
	copied := 0
	err := filepath.WalkDir(srcRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(srcRoot, path)
		if err != nil || rel == "." {
			return err
		}
		dst := filepath.Join(dstRoot, rel)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if fileExists(dst) {
			return nil
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := copyFile(path, dst); err != nil {
			return err
		}
		copied++
		return nil
	})
	return copied, err
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	_ = os.Remove(dst)
	return os.Rename(tmp, dst)
}

func accountFixSummary(registryPath string, reg accounts.Registry) acctFixSummary {
	return summarizeAccountFixes(buildAccountsDoctorReport(registryPath, reg))
}

// foldDoctorSeat maps one login observation (plus the optional fresh probe-ledger
// verdict) onto the closed action vocabulary. Hard config states win over runtime
// walls; the duplicate warning surfaces only on a seat that is otherwise healthy.
func foldDoctorSeat(obs accounts.LoginObservation, consultLedger bool) doctorSeat {
	seat := doctorSeat{Name: obs.Name, Status: string(obs.Status), Action: doctorNone, Reason: obs.Reason}
	switch obs.Status {
	case accounts.LoginMissingDir:
		seat.Action = doctorPrune
		seat.AutoFix = true
		seat.Command = "fak accounts remove --name " + obs.Name
		return seat
	case accounts.LoginNeedsLogin, accounts.LoginIdentityMismatch:
		seat.Action = doctorRelogin
		seat.Command = loginCommandFor(obs.Dir)
		return seat
	case accounts.LoginDisabled:
		seat.Action = doctorEnableOrRemove
		seat.Command = "fak accounts remove --name " + obs.Name + "  (or re-enable it in the registry)"
		return seat
	case accounts.LoginTombstoned:
		return seat // already retired; Resolve/Serve fall forward past it
	}
	// Ready seat: overlay the freshest active-probe verdict, when the prober is wired.
	if consultLedger && obs.Dir != "" {
		if fp := fleetaccounts.FreshProbeFromLedger(filepath.Base(obs.Dir), "", time.Now().UTC(), 0); fp != nil && !fp.Available {
			seat.Reason = fp.BlockReason
			seat.Reset = fp.Reset
			switch fp.BlockKind {
			case "usage":
				seat.Action = doctorWaitReset
			case "credit":
				seat.Action = doctorTopUp
			default: // auth / access
				seat.Action = doctorRelogin
				seat.Command = loginCommandFor(obs.Dir)
			}
			return seat
		}
	}
	for _, w := range obs.Warnings {
		if w == accounts.LoginWarningDuplicateBucket {
			seat.Action = doctorDedupe
			seat.Reason = "duplicate of " + obs.Canonical
			seat.Command = "fak accounts remove --name " + obs.Name + " --reason duplicate-of-" + obs.Canonical
			return seat
		}
	}
	return seat
}

// loginCommandFor renders the exact re-login command for a seat's config dir.
func loginCommandFor(dir string) string {
	return "CLAUDE_CONFIG_DIR=" + dir + " claude /login"
}

func printDoctorTable(w io.Writer, report acctDoctorReport, write bool) {
	fmt.Fprintf(w, "schema %s\n", report.Schema)
	if !report.ProbeLedger {
		fmt.Fprintln(w, "note: FLEET_REG_DIR unset — probe ledger not consulted (config-plane verdicts only)")
	}
	tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SEAT\tSTATUS\tACTION\tAUTO\tDETAIL")
	for _, s := range report.Seats {
		detail := s.Reason
		if s.Applied {
			detail = "APPLIED: " + s.ApplyNote
		} else if s.ApplyNote != "" {
			detail = s.ApplyNote
		} else if s.Command != "" {
			detail = s.Command
		}
		auto := ""
		if s.AutoFix {
			auto = "yes"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", s.Name, s.Status, s.Action, auto, detail)
	}
	tw.Flush()
	fmt.Fprintf(w, "actionable: %d  auto-fixable: %d  applied: %d\n", report.Actionable, report.AutoFixable, report.Applied)
	if !write && report.AutoFixable > 0 {
		fmt.Fprintln(w, "run `fak accounts doctor --write` to apply the auto-fixable repairs")
	}
}
