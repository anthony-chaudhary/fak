package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
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
//
// The `recovery_worklist` key folds the walled seats by recoverability (#3580): the
// `recoverable` list names the seats one operator action away from serving (a `login` or a
// credential `re-read` per #3216) with the servable-seat gain if reclaimed, and
// `hard_walled` counts the seats a usage/credit/access wall leaves only time can clear —
// so an operator can grow effective supply from seats already owned instead of provisioning
// new accounts. A fully-offerable roster yields an empty worklist.
const doctorSchema = "fak.accounts.doctor.v1"

// doctorAction is the closed per-seat recovery vocabulary. Exactly one action is
// assigned per seat; "none" means the seat needs nothing.
type doctorAction string

const (
	doctorNone           doctorAction = "none"
	doctorRelogin        doctorAction = "relogin"          // needs_login, or a fresh auth wall
	doctorWaitReset      doctorAction = "wait_reset"       // fresh usage limit; recovers by itself
	doctorTopUp          doctorAction = "top_up"           // fresh credit wall; needs billing, not code
	doctorAccessBlocked  doctorAction = "access_blocked"   // subscription access disabled upstream; re-login can't restore it
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
	// Recovery splits a walled seat by whether an operator action reclaims it NOW
	// (recoverable) vs a hard wall that only time/billing/admin clears (hard). Empty
	// on a healthy seat or a cleanup-only action (prune/dedupe/enable-or-remove),
	// which grow no servable supply.
	Recovery string `json:"recovery,omitempty"`
}

// The recovery split: a walled seat is either one operator action away from serving
// (recoverable — a `claude /login` or a credential re-read per #3216) or hard-walled
// (only a usage-cap reset, a credit top-up, or an upstream access-restore clears it).
const (
	recoveryRecoverable = "recoverable"
	recoveryHard        = "hard"
)

// recoverySeat is one entry on the doctor recovery worklist: a walled seat an operator
// can reclaim now, the action word (`login` / `re-read credential`), the exact command,
// and the servable-seat gain (+1 seat) if reclaimed.
type recoverySeat struct {
	Name     string `json:"name"`
	Status   string `json:"status"`
	Action   string `json:"action"`
	Command  string `json:"command,omitempty"`
	SeatGain int    `json:"seat_gain"`
}

// recoveryWorklist is the actionable "grow servable supply now" fold surfaced under the
// documented `recovery_worklist` key: the seats one operator action away from serving,
// the count of hard-walled seats that can only wait, and the total servable-seat gain if
// every recoverable seat is reclaimed. Empty (recoverable:[], gains 0) when the roster is
// fully offerable — the cheapest new seat is often a walled one already owned.
type recoveryWorklist struct {
	Recoverable      []recoverySeat `json:"recoverable"`
	HardWalled       int            `json:"hard_walled"`
	ServableSeatGain int            `json:"servable_seat_gain"`
}

// acctDoctorReport is the machine-readable doctor surface.
type acctDoctorReport struct {
	Schema      string           `json:"schema"`
	Registry    string           `json:"registry"`
	ProbeLedger bool             `json:"probe_ledger_consulted"`
	Seats       []doctorSeat     `json:"seats"`
	Actionable  int              `json:"actionable"`
	AutoFixable int              `json:"auto_fixable"`
	Applied     int              `json:"applied"`
	Recovery    recoveryWorklist `json:"recovery_worklist"`
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
	foldRecoveryWorklist(report)
}

// classifyRecovery maps a folded per-seat action onto the recoverable/hard split and the
// operator action word. Recoverable = one login or credential re-read away from serving;
// hard = only a cap reset, a credit top-up, or an upstream access-restore clears it. An
// identity_mismatch (creds present but disk disagrees, the #3216 mislabel) is reclaimed by
// re-reading the credential, not a fresh login. Cleanup-only actions and healthy seats grow
// no servable supply, so they carry no class.
func classifyRecovery(action doctorAction, status string) (class, actionWord string) {
	switch action {
	case doctorRelogin:
		if accounts.LoginStatus(status) == accounts.LoginIdentityMismatch {
			return recoveryRecoverable, "re-read credential"
		}
		return recoveryRecoverable, "login"
	case doctorHydrate:
		return recoveryRecoverable, "re-read credential"
	case doctorWaitReset:
		return recoveryHard, "wait for reset"
	case doctorTopUp:
		return recoveryHard, "top up credit"
	case doctorAccessBlocked:
		return recoveryHard, "restore access"
	default:
		return "", ""
	}
}

// foldRecoveryWorklist classifies each not-yet-applied seat as recoverable/hard and folds
// the actionable recovery worklist: which walled seats an operator can reclaim now, by what
// action, and the servable-seat gain (one seat per reclaimed seat). Pure post-pass over the
// folded seats, so a fully-offerable roster yields an empty worklist.
func foldRecoveryWorklist(report *acctDoctorReport) {
	work := recoveryWorklist{Recoverable: []recoverySeat{}}
	for i := range report.Seats {
		s := &report.Seats[i]
		if s.Applied {
			s.Recovery = ""
			continue
		}
		class, action := classifyRecovery(s.Action, s.Status)
		s.Recovery = class
		switch class {
		case recoveryRecoverable:
			work.Recoverable = append(work.Recoverable, recoverySeat{
				Name: s.Name, Status: s.Status, Action: action,
				Command: firstNonEmpty(s.Command, s.Source), SeatGain: 1,
			})
			work.ServableSeatGain++
		case recoveryHard:
			work.HardWalled++
		}
	}
	report.Recovery = work
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

// backupAndCopyCredential snapshots whatever the target already holds under name into
// backupDir and only then copies the source's file over it, so a hydrate is always
// reversible. The source keeps its own copy. Both credential shapes a seat can carry
// (.credentials.json and .oauth-token) are installed by this one back-up-then-overwrite
// rule; which files a hydrate touches still differs by shape (a .credentials.json hydrate
// additionally retires the target's stale .oauth-token), but no shape can ever be
// overwritten without being snapshotted first.
func backupAndCopyCredential(sourceDir, targetDir, backupDir, name, stamp string) error {
	if err := backupIfExists(targetDir, backupDir, name, stamp); err != nil {
		return err
	}
	return copyFile(filepath.Join(sourceDir, name), filepath.Join(targetDir, name))
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
		if err := backupAndCopyCredential(source.Dir, target.Dir, backupDir, ".credentials.json", stamp); err != nil {
			return "", err
		}
		if err := backupIfExists(target.Dir, backupDir, ".oauth-token", stamp); err != nil {
			return "", err
		}
		_ = os.Remove(filepath.Join(target.Dir, ".oauth-token"))
		copiedCred = ".credentials.json"
	} else if fileExists(filepath.Join(source.Dir, ".oauth-token")) {
		if err := backupAndCopyCredential(source.Dir, target.Dir, backupDir, ".oauth-token", stamp); err != nil {
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
	note := fmt.Sprintf("hydrated %s from %s: copied %s and %d missing project file(s)", targetName, sourceName, copiedCred, copiedSessions)
	// A COPIED .credentials.json leaves both dirs on ONE OAuth token family, and Claude Code rotates
	// the refresh token when it refreshes — so the first of the pair to refresh silently invalidates
	// the other (internal/accounts/credfamily.go carries the 401 witness). A hydrate therefore cannot
	// produce two independently-refreshable seats, and this repair is reachable unattended from
	// `accounts doctor --write`, where the loser would be the HEALTHY source seat that was working
	// fine. The enroll path resolves its own copy immediately; here the choice of which seat to keep
	// is the operator's, so surface the hazard on the repair note rather than silently picking.
	if share := accounts.DetectSharedRefreshFamily(source.Dir, target.Dir); share.Shared {
		note += fmt.Sprintf("; WARNING: %s and %s now share OAuth token family %s — the first to refresh will silently 401 the other; split them with `fak accounts refresh --name %s --force`",
			sourceName, targetName, share.FamilyID, targetName)
	}
	return note, nil
}

func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}

// hydrateBackupKeep bounds how many timestamped before-hydrate backups of ONE credential file
// a seat retains. Every hydrate writes a fresh "<file>.before-hydrate-<stamp>.bak" under
// <seat>/backups/, so with no cap the dir grows strictly monotonically for the seat's whole
// lifetime AND keeps every superseded plaintext credential/OAuth token forever (#3505). Five is
// deep enough to walk back through a bad hydrate chain while keeping the retained-secret window
// short. The cap is per file, so a .credentials.json chain never evicts an .oauth-token pre-image.
const hydrateBackupKeep = 5

// hydrateBackupPrefix is the name stem shared by every before-hydrate backup of `file`. It is
// BOTH what the writer stamps and what the prune matches on, so the writer and the reaper can
// never disagree about which entries in backups/ are before-hydrate backups of this file. The
// two credential names are prefix-disjoint (neither stem prefixes the other), so a prune of one
// file's chain can never see the other's.
func hydrateBackupPrefix(file string) string { return file + ".before-hydrate-" }

// backupIfExists copies dir/<file> into backupDir as its stamped before-hydrate backup, then
// prunes that file's chain back to hydrateBackupKeep. An absent source is a no-op, not an error —
// a seat with no prior credential simply has no pre-image to keep. The cap lives HERE, on the
// write path, rather than in a separate reaper pass: one write, one prune, same function, so no
// call site can add an unbounded backup by forgetting to reap (#3505).
func backupIfExists(dir, backupDir, file, stamp string) error {
	path := filepath.Join(dir, file)
	if !fileExists(path) {
		return nil
	}
	if err := copyFile(path, filepath.Join(backupDir, hydrateBackupPrefix(file)+stamp+".bak")); err != nil {
		return err
	}
	return pruneHydrateBackups(backupDir, file, hydrateBackupKeep)
}

// pruneHydrateBackups deletes the OLDEST "<file>.before-hydrate-<stamp>.bak" entries in backupDir
// until at most `keep` survive. The stamp is a sortable UTC token, so ordering the names
// lexicographically is ordering them by age, and no stat call is needed to find the oldest.
// keep <= 0 is a no-op: a cap must never prune to empty and discard the pre-image the operator
// just took. A missing backupDir is likewise not an error — nothing backed up yet is a valid
// state. Only names ending .bak are considered, so a torn copy's .bak.tmp is never mistaken for
// a completed backup and counted against the cap.
func pruneHydrateBackups(backupDir, file string, keep int) error {
	if keep <= 0 {
		return nil
	}
	ents, err := os.ReadDir(backupDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	prefix := hydrateBackupPrefix(file)
	var names []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if n := e.Name(); strings.HasPrefix(n, prefix) && strings.HasSuffix(n, ".bak") {
			names = append(names, n)
		}
	}
	if len(names) <= keep {
		return nil
	}
	sort.Strings(names) // oldest stamp first
	for _, n := range names[:len(names)-keep] {
		if err := os.Remove(filepath.Join(backupDir, n)); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
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
			case "access":
				// Subscription access was disabled upstream ("use an API key instead" /
				// "ask your admin to enable access"). Re-login re-auths the SAME disabled
				// account and hits the SAME wall — it cannot restore serving, so this is
				// operator judgment, not a relogin fix.
				seat.Action = doctorAccessBlocked
				seat.Command = "fak accounts remove --name " + obs.Name + "  (subscription access disabled upstream; re-login can't restore it — use an API key or ask your org admin to re-enable)"
			default: // auth
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
	rw := report.Recovery
	fmt.Fprintf(w, "recovery worklist: %d recoverable seat(s), servable-seat gain +%d (%d hard-walled)\n",
		len(rw.Recoverable), rw.ServableSeatGain, rw.HardWalled)
	for _, s := range rw.Recoverable {
		fmt.Fprintf(w, "  reclaim %s via %s: %s\n", s.Name, s.Action, firstNonEmpty(s.Command, "(no command)"))
	}
	if !write && report.AutoFixable > 0 {
		fmt.Fprintln(w, "run `fak accounts doctor --write` to apply the auto-fixable repairs")
	}
}
