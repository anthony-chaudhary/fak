package main

// assume.go — `fak assume`, the thin impure shell over the pure
// internal/assumecheck kernel (#3819 C1 spine; #3820 C2 registry wiring; #3821 C3
// witness-driver plurality, epic #3818). The kernel judges; this file only
// GATHERS the evidence, prints the verdict, and maps it to an exit code — the
// same split stallscan.go keeps with internal/stallscan.
//
//	fak assume list                             # every registered assumption + its wiring
//	fak assume list --json                      # the same registry as JSON
//	fak assume check seat-launchable            # the spine's one wired assumption (human)
//	fak assume check seat-launchable --json     # the same verdict as JSON
//	fak assume check seat-launchable --seat X   # check an explicitly named seat
//
// `check` routes through the declarative registry (assumecheck.Lookup): an unknown
// id is a usage error naming the known ids, never a guessed check. Witness DISPATCH
// is name-resolved (#3821 C3): each wired row's gatherer builds the probe TARGET
// for its assumption and resolves the DECLARED WitnessKind's driver from the
// assumecheck driver registry (assumecheck.ResolveDriver) — the driver gathers,
// the kernel judges. seat-launchable keeps its bespoke ledger-read gatherer
// (gatherSeatLaunchableEvidence; WitnessLedgerRead has no generic driver). A
// registered-but-unwired (declared-only) row hands the kernel unwitnessed evidence
// of its declared kind, so it verdicts UNVERIFIABLE with the wiring gap as the
// explanation — never a fabricated HOLDS.
//
// THE ONE WIRED ASSUMPTION: "a seat named launch-clean by `fak accounts
// doctor`/preflight is actually launchable" — witnessed against the REAL
// launchability authority behind `fak accounts next`:
// Registry.RotationPlanWithHeadroom over the refreshed registry, with the live
// runtime headroom + usage-cooldown overlay folded in (accounts_headroom.go),
// exactly what `next` decides launches from. The failure class this catches is
// authority drift: the doctor/preflight config plane calling a seat clean while the
// rotation authority would refuse to hand it out.
//
// Exit codes: 0 the assumption HOLDS (or `list` printed); 1 runtime error; 2 usage;
// 3 VIOLATED (the gate a script watches); 4 UNVERIFIABLE or STALE (cannot witness —
// still nonzero, the kernel is fail-closed).

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/assumecheck"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdAssume(argv []string) { os.Exit(runAssume(os.Stdout, os.Stderr, argv)) }

const assumeUsage = `usage: fak assume <check|list>
  fak assume check <assumption-id> [--seat <name>] [--registry <path>] [--home <dir>] [--no-headroom] [--json]
  fak assume list [--json]`

// runAssume is the testable core (stdout/stderr injected, exit code returned).
func runAssume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, assumeUsage)
		return 2
	}
	switch argv[0] {
	case "check":
		return runAssumeCheck(stdout, stderr, argv[1:])
	case "list":
		return runAssumeList(stdout, stderr, argv[1:])
	default:
		fmt.Fprintln(stderr, assumeUsage)
		return 2
	}
}

// runAssumeCheck witnesses ONE registered assumption and maps its closed verdict to
// an exit code.
func runAssumeCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("assume check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defHome, _ := os.UserHomeDir()
	regDefault := os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if regDefault == "" && defHome != "" {
		regDefault = filepath.Join(defHome, ".claude-accounts", "registry.json")
	}
	registryPath := fs.String("registry", regDefault, "path to the config-home registry.json (same default as `fak accounts`)")
	homeDir := fs.String("home", defHome, "home dir to discover ~/.claude* under when no registry exists")
	seat := fs.String("seat", "", "seat to check (default: the doctor-named launch seat — the active role if doctor-clean, else the first clean rotation candidate)")
	asJSON := fs.Bool("json", false, "emit the verdict + evidence as JSON")
	noHeadroom := fs.Bool("no-headroom", false, "witness against the registry-only rotation plan, without the live runtime headroom/cooldown overlay (mirrors `fak accounts next --no-headroom`)")
	// Tolerate the assumption id as a leading positional before flags, the same
	// leading-positional split parseAccountsCmd does.
	rest := argv
	lead := 0
	for lead < len(rest) && !strings.HasPrefix(rest[lead], "-") {
		lead++
	}
	if !parseFlags(fs, rest[lead:]) {
		return 2
	}
	id := ""
	if lead > 0 {
		id = strings.TrimSpace(rest[0])
	}
	if id == "" {
		id = assumecheck.SeatLaunchable.ID
	}
	a, ok := assumecheck.Lookup(id)
	if !ok {
		fmt.Fprintf(stderr, "fak assume: unknown assumption %q — known ids: %s\n", id, strings.Join(assumeKnownIDs(), ", "))
		return 2
	}

	ev, seatName := gatherAssumptionEvidence(a, assumeGatherParams{
		registryPath: pathutil.ExpandTilde(*registryPath),
		homeDir:      pathutil.ExpandTilde(*homeDir),
		seat:         strings.TrimSpace(*seat),
		useHeadroom:  !*noHeadroom,
	})
	v, gerr := assumecheck.GuardAssumption(a, ev)
	refused := gerr != nil && errors.Is(gerr, assumecheck.ErrAssumptionViolated)

	if *asJSON {
		rec := map[string]any{
			"schema":     "fak.assume.check.v1",
			"assumption": a,
			"seat":       seatName,
			"evidence":   ev,
			"verdict":    v,
			"refused":    refused,
		}
		if code := encodeJSONOrFail(stdout, stderr, rec, "fak assume"); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "assumption : %s (%s/%s, owner %s)\n", a.ID, a.Level, a.WitnessKind, a.Owner)
		fmt.Fprintf(stdout, "wiring     : %s\n", a.WitnessStatus)
		fmt.Fprintf(stdout, "statement  : %s\n", a.Statement)
		fmt.Fprintf(stdout, "seat       : %s\n", dash(seatName))
		fmt.Fprintf(stdout, "evidence   : witnessed=%t holds=%t%s\n", ev.Witnessed, ev.Holds, detailTail(ev.Detail))
		fmt.Fprintf(stdout, "outcome    : %s\n", v.Outcome)
		fmt.Fprintf(stdout, "reason     : %s\n", v.Reason)
		if refused {
			// The typed-guard seam end-to-end: the same errors.Is branch a programmatic
			// caller takes, rendered with the structured refusal token it would emit.
			fmt.Fprintf(stdout, "refusal    : %s — %v\n", a.RefusalReason, gerr)
		}
	}

	switch v.Outcome {
	case assumecheck.OutcomeHolds:
		return 0
	case assumecheck.OutcomeViolated:
		return 3
	default: // UNVERIFIABLE | STALE — cannot witness; fail closed, distinct from VIOLATED
		return 4
	}
}

// runAssumeList enumerates the declarative registry (#3820): one row per registered
// assumption with its scope, declared witness kind, owner, and whether this shell
// actually has a witness gatherer wired for it — so a declared-only row's
// UNVERIFIABLE is legible from the menu, not a surprise at check time.
func runAssumeList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("assume list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the registry (with per-row wiring) as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	rows := assumecheck.Registry()
	if *asJSON {
		type listRow struct {
			Assumption assumecheck.Assumption `json:"assumption"`
			Wired      bool                   `json:"wired"`
		}
		out := make([]listRow, 0, len(rows))
		for _, a := range rows {
			_, wired := assumeWitnessGatherers[a.ID]
			out = append(out, listRow{Assumption: a, Wired: wired})
		}
		rec := map[string]any{
			"schema":      "fak.assume.list.v1",
			"assumptions": out,
		}
		return encodeJSONOrFail(stdout, stderr, rec, "fak assume")
	}
	fmt.Fprintf(stdout, "%-26s %-10s %-14s %-9s %s\n", "ID", "LEVEL", "WITNESS", "OWNER", "WIRING")
	for _, a := range rows {
		wiring := string(assumecheck.WitnessDeclaredOnly) + " (gatherer pending)"
		if _, wired := assumeWitnessGatherers[a.ID]; wired {
			wiring = string(assumecheck.WitnessWired)
		}
		fmt.Fprintf(stdout, "%-26s %-10s %-14s %-9s %s\n", a.ID, a.Level, a.WitnessKind, a.Owner, wiring)
	}
	return 0
}

// assumeKnownIDs is the registry's id menu in registry order, for the unknown-id
// usage error.
func assumeKnownIDs() []string {
	rows := assumecheck.Registry()
	ids := make([]string, 0, len(rows))
	for _, a := range rows {
		ids = append(ids, a.ID)
	}
	return ids
}

// assumeGatherParams carries the shell flags a witness gatherer may read.
type assumeGatherParams struct {
	registryPath string
	homeDir      string
	seat         string
	useHeadroom  bool
}

// assumeWitnessGatherers is this shell's witness-dispatch table: the assumptions
// whose evidence it can GATHER today. Each entry builds the probe TARGET for its
// assumption; the driver-backed ones then resolve the row's DECLARED WitnessKind
// from the assumecheck driver registry (#3821 C3) and let that driver produce the
// evidence. seat-launchable keeps its bespoke ledger-read gatherer —
// WitnessLedgerRead is a per-assumption authority read, not one of the four
// generic probe kinds. The registry's per-row WitnessStatus declares the same
// fact as data — TestAssumeWiringMatchesDeclaredStatus binds the two so the
// marker can never drift from the dispatch table.
var assumeWitnessGatherers = map[string]func(assumeGatherParams) (assumecheck.Evidence, string){
	assumecheck.SeatLaunchable.ID: func(p assumeGatherParams) (assumecheck.Evidence, string) {
		return gatherSeatLaunchableEvidence(p.registryPath, p.homeDir, p.seat, p.useHeadroom)
	},
	"seat-config-dir-present": gatherSeatConfigDirEvidence,
	"seat-pool-not-depleted":  gatherSeatPoolEvidence,
	"kernel-loop-alive":       gatherKernelLoopEvidence,
}

// assumeProbeTimeout bounds one witness probe. Generous: the slowest wired probe
// (`dos loop --json`, a pip console-script shim spawning a python grandchild)
// takes tens of seconds cold on this fleet's Windows hosts.
const assumeProbeTimeout = 90 * time.Second

// assumeGatherViaDriver is the name-resolved dispatch step (#3821 C3): resolve
// the DECLARED witness kind's driver from the assumecheck driver registry and
// gather through it. No driver registered for the kind is the fail-closed
// branch — unwitnessed evidence of the declared kind, so the kernel verdicts
// UNVERIFIABLE naming the gap, never a cross-kind substitution.
func assumeGatherViaDriver(kind assumecheck.WitnessKind, target assumecheck.Target) assumecheck.Evidence {
	d, ok := assumecheck.ResolveDriver(kind)
	if !ok {
		return assumecheck.Evidence{
			Kind:   kind,
			Detail: fmt.Sprintf("no witness driver registered for kind %s", kind),
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), assumeProbeTimeout)
	defer cancel()
	return d.Gather(ctx, target)
}

// assumeSeatInRotation is the in-rotation predicate the seat-config-dir witness
// scopes to: active status, enabled, and not policy-reserved — the seats a
// launcher can actually be handed, the ones whose vanished config dir is the
// doctor's prune class rather than expected retirement.
func assumeSeatInRotation(h accounts.Home) bool {
	return h.Active() && h.EnabledOrDefault() && !h.Reserved
}

// gatherSeatConfigDirEvidence witnesses seat-config-dir-present through the
// config-flag driver (path presence IS the flag's source of truth): every
// in-rotation registry seat's config dir must still exist on disk. One missing
// dir is a witnessed violation naming the seat (the doctor prune class); a dir
// that cannot be checked at all makes the whole claim unverifiable — "every
// seat" cannot be confirmed past an unreadable one. With --seat the probe
// narrows to that seat; a seat the registry does not know, holds out of
// rotation, or records no dir for is an absent premise, not a violation.
func gatherSeatConfigDirEvidence(p assumeGatherParams) (assumecheck.Evidence, string) {
	ev := assumecheck.Evidence{Kind: assumecheck.WitnessConfigFlag}
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		ev.Detail = fmt.Sprintf("registry unreadable: %v", err)
		return ev, ""
	}
	reg = reg.Refresh()

	type dirProbe struct{ seat, dir string }
	var probes []dirProbe
	if p.seat != "" {
		var home accounts.Home
		known := false
		for _, h := range reg.Homes {
			if h.Name == p.seat {
				home, known = h, true
				break
			}
		}
		switch {
		case !known:
			ev.Detail = fmt.Sprintf("premise absent: registry has no seat %q", p.seat)
			return ev, p.seat
		case !assumeSeatInRotation(home):
			ev.Detail = fmt.Sprintf("premise absent: seat %q is not in rotation (status/enabled/reserved hold it out), so the prune-class claim does not apply", p.seat)
			return ev, p.seat
		case strings.TrimSpace(home.Dir) == "":
			ev.Detail = fmt.Sprintf("premise absent: seat %q records no config dir to witness", p.seat)
			return ev, p.seat
		}
		probes = append(probes, dirProbe{home.Name, home.Dir})
	} else {
		for _, h := range reg.Homes {
			if assumeSeatInRotation(h) && strings.TrimSpace(h.Dir) != "" {
				probes = append(probes, dirProbe{h.Name, h.Dir})
			}
		}
		if len(probes) == 0 {
			ev.Detail = "no in-rotation seat records a config dir to witness"
			return ev, ""
		}
	}

	for _, pr := range probes {
		got := assumeGatherViaDriver(assumecheck.WitnessConfigFlag, assumecheck.Target{Path: pr.dir})
		switch {
		case !got.Witnessed:
			ev.Detail = fmt.Sprintf("seat %q config dir could not be checked — %s", pr.seat, got.Detail)
			return ev, pr.seat
		case !got.Holds:
			ev.Witnessed = true
			ev.Detail = fmt.Sprintf("seat %q config dir is missing from disk (%s) — the doctor prune (tombstone+rehome) class", pr.seat, pr.dir)
			return ev, pr.seat
		}
	}
	ev.Witnessed = true
	ev.Holds = true
	ev.Detail = fmt.Sprintf("%d in-rotation seat config dir(s) present on disk", len(probes))
	return ev, p.seat
}

// gatherSeatPoolEvidence witnesses seat-pool-not-depleted through the
// command-probe driver with an IN-PROCESS probe: the depletion signal is the
// same dispatchtick.BuildSeatPool fold dispatch preflight's SeatCheck reads
// (dispatchPreflightSeat), already reachable in this process — spawning a
// subprocess to re-ask ourselves would add noise, not evidence. The probe
// returns the exit-like tri-state the driver contract maps.
func gatherSeatPoolEvidence(_ assumeGatherParams) (assumecheck.Evidence, string) {
	root, err := os.Getwd()
	if err != nil {
		root = "."
	}
	return assumeGatherViaDriver(assumecheck.WitnessCommandProbe, assumecheck.Target{
		Probe: func(context.Context) (string, int, error) {
			return assumeSeatPoolTriState(dispatchPreflightSeat(root, io.Discard, ""))
		},
	}), ""
}

// assumeSeatPoolTriState maps the dispatch seat gate's folded SeatCheck onto the
// command-probe exit tri-state: free seats hold (0), a depleted pool is a
// witnessed refute (1), an unreadable roster cannot witness (err). Pure — split
// out so the mapping is table-testable without a roster on disk.
func assumeSeatPoolTriState(seat dispatchtick.SeatCheck) (string, int, error) {
	if seat.Error != "" {
		return "", -1, errors.New("seat pool unreadable: " + seat.Error)
	}
	detail := fmt.Sprintf("seat pool total=%s free=%s leased=%s (the dispatchtick.BuildSeatPool fold behind preflight's SeatCheck)",
		assumeIntPtrLabel(seat.Total), assumeIntPtrLabel(seat.Free), assumeIntPtrLabel(seat.Leased))
	if seat.Depleted {
		return detail + ": depleted", 1, nil
	}
	return detail, 0, nil
}

// gatherKernelLoopEvidence witnesses kernel-loop-alive through the command-probe
// driver with an in-process probe around the SAME `dos loop --json` command
// dispatchPreflightKernel folds into preflight's KernelCheck. The dos CLI exits
// 0 whenever it can ANSWER, so the raw exit code cannot carry liveness — the
// probe reads the answered verdict and re-expresses it as the exit-like
// tri-state the driver contract maps.
func gatherKernelLoopEvidence(_ assumeGatherParams) (assumecheck.Evidence, string) {
	root, err := os.Getwd()
	if err != nil {
		root = "."
	}
	return assumeGatherViaDriver(assumecheck.WitnessCommandProbe, assumecheck.Target{
		Probe: func(context.Context) (string, int, error) {
			doc, err := dispatchRunExternalJSON(root, 60*time.Second, "dos", "loop", "--workspace", root, "--json")
			return assumeKernelLoopTriState(doc, err)
		},
	}), ""
}

// assumeKernelLoopTriState maps the `dos loop --json` answer onto the
// command-probe exit tri-state: a non-refusing verdict is a live, admitting
// loop (0); a HALT/REFUSE verdict is a witnessed refusal (1); a probe that
// could not run, or answered without a verdict, cannot witness either way
// (err). Pure — split out so the mapping is table-testable without the dos CLI.
func assumeKernelLoopTriState(doc map[string]any, err error) (string, int, error) {
	if err != nil {
		return "", -1, fmt.Errorf("`dos loop --json` probe could not run: %w", err)
	}
	verdict := strings.TrimSpace(dispatchMapString(doc, "verdict"))
	if verdict == "" {
		return "", -1, errors.New("`dos loop --json` answered without a verdict — nothing to witness")
	}
	detail := fmt.Sprintf("`dos loop --json` verdict=%s alive=%s target=%s",
		verdict, assumeIntPtrLabel(intPtrFromAny(doc["alive"])), assumeIntPtrLabel(intPtrFromAny(doc["target"])))
	up := strings.ToUpper(verdict)
	if strings.Contains(up, "HALT") || strings.HasPrefix(up, "REFUSE") {
		return detail + ": the loop is refusing/halted", 1, nil
	}
	return detail, 0, nil
}

// assumeIntPtrLabel renders an optional count for a detail line ("?" = the
// probe carried no value).
func assumeIntPtrLabel(p *int) string {
	if p == nil {
		return "?"
	}
	return fmt.Sprintf("%d", *p)
}

// gatherAssumptionEvidence routes a registered assumption to its wired gatherer.
// For a declared-only row it hands the kernel UNWITNESSED evidence of the DECLARED
// kind — so Check verdicts UNVERIFIABLE with the wiring gap as the explanation,
// never a fabricated HOLDS and never the cross-witness UNVERIFIABLE that would
// misname the problem as a kind mismatch.
func gatherAssumptionEvidence(a assumecheck.Assumption, p assumeGatherParams) (assumecheck.Evidence, string) {
	if gather, wired := assumeWitnessGatherers[a.ID]; wired {
		return gather(p)
	}
	return assumecheck.Evidence{
		Kind:   a.WitnessKind,
		Detail: fmt.Sprintf("assumption %q is declared-only: no witness gatherer is wired for it in this shell", a.ID),
	}, ""
}

// detailTail renders an evidence detail as a one-line tail for the human report.
func detailTail(detail string) string {
	if detail == "" {
		return ""
	}
	return " — " + detail
}

// gatherSeatLaunchableEvidence is the INLINE witness for the spine's one assumption
// (the C2 registry will own witness dispatch): it reads the same sources the two
// authorities read — the doctor fold for the CLAIM (which seat is named
// launch-clean) and the `accounts next` rotation plan for the WITNESS (is that seat
// actually launchable) — and folds them into one assumecheck.Evidence. It never
// judges; the kernel does. Returned with the seat name the claim resolved to ("",
// when no seat could be named).
//
// Witnessed=false (-> UNVERIFIABLE) when the registry is unreadable, when the doctor
// names no launch-clean seat, when the premise is absent for an explicit --seat
// (doctor marks it as needing action, or does not know it), or when registry POLICY
// (reserved/disabled/tombstoned) holds the seat out of rotation — the authority
// makes no launchability claim about a seat it never adjudicates. Witnessed=true
// with Holds=false (-> VIOLATED) is the real drift: a doctor-clean seat the rotation
// authority cannot or will not launch (unservable, walled/cooled bucket, or not
// modeled at all).
func gatherSeatLaunchableEvidence(registryPath, homeDir, seat string, useHeadroom bool) (assumecheck.Evidence, string) {
	ev := assumecheck.Evidence{Kind: assumecheck.WitnessLedgerRead}

	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		ev.Detail = fmt.Sprintf("registry unreadable: %v", err)
		return ev, ""
	}
	reg = reg.Refresh()
	report := buildAccountsDoctorReport(registryPath, reg)
	rowByName := make(map[string]doctorSeat, len(report.Seats))
	for _, s := range report.Seats {
		rowByName[s.Name] = s
	}
	homeByName := make(map[string]accounts.Home, len(reg.Homes))
	for _, h := range reg.Homes {
		homeByName[h.Name] = h
	}

	// THE CLAIM: resolve the seat the doctor/preflight plane names launch-clean.
	name := seat
	if name == "" {
		name = defaultDoctorSeat(reg, report, homeByName)
		if name == "" {
			ev.Detail = "accounts doctor names no launch-clean seat (action=none, non-tombstoned, in-rotation) to check"
			return ev, ""
		}
	}
	row, known := rowByName[name]
	if !known {
		ev.Detail = fmt.Sprintf("premise absent: accounts doctor does not know seat %q", name)
		return ev, name
	}
	if row.Status == string(accounts.LoginTombstoned) {
		ev.Detail = fmt.Sprintf("premise absent: seat %q is tombstoned — doctor names no launch claim about a retired seat", name)
		return ev, name
	}
	if row.Action != doctorNone {
		ev.Detail = fmt.Sprintf("premise absent: doctor marks %q action=%s (%s), not launch-clean", name, row.Action, firstNonEmpty(row.Reason, row.Status))
		return ev, name
	}

	// THE WITNESS: the rotation plan behind `fak accounts next` — the refreshed
	// registry plus (by default) the live runtime headroom/cooldown overlay.
	var hr accounts.RotationHeadroom
	if useHeadroom {
		hr = rotationHeadroom(homeDir)
	}
	plan := reg.RotationPlanWithHeadroom(hr)

	poolSeat, via, inPool := rotationPoolSeatFor(plan, name)
	if !inPool {
		if st, excluded := rotationExclusionFor(plan, name); excluded {
			switch st {
			case accounts.RotationReserved, accounts.RotationDisabled, accounts.RotationTombstoned:
				// Policy holds the seat out: the authority never adjudicates its
				// launchability, so there is nothing to witness either way.
				ev.Detail = fmt.Sprintf("registry policy holds %q out of rotation (%s): the `accounts next` authority makes no launchability claim about it", name, st)
				return ev, name
			default:
				ev.Witnessed = true
				ev.Detail = fmt.Sprintf("the `accounts next` authority excludes %q as %s: doctor-clean but not launchable", name, st)
				return ev, name
			}
		}
		ev.Witnessed = true
		ev.Detail = fmt.Sprintf("the `accounts next` authority does not model seat %q at all (absent from pool and exclusions)", name)
		return ev, name
	}

	ev.Witnessed = true
	label := name
	if via != "" {
		label = fmt.Sprintf("%s (bucket served via canonical seat %s)", name, via)
	}
	switch {
	case !poolSeat.CanServe:
		ev.Detail = fmt.Sprintf("%s is in the rotation pool but can_serve=false", label)
	case poolSeat.Headroom != nil && accounts.Classify(*poolSeat.Headroom) == accounts.TierWalled:
		ev.Detail = fmt.Sprintf("%s is in the rotation pool but its account is runtime-walled (headroom=%s): `accounts next` would not hand it out", label, accounts.HeadroomLabel(*poolSeat.Headroom))
	default:
		ev.Holds = true
		hrLabel := "no signal"
		if poolSeat.Headroom != nil {
			hrLabel = accounts.HeadroomLabel(*poolSeat.Headroom)
		}
		ev.Detail = fmt.Sprintf("%s is in the `accounts next` rotation pool, can_serve=true, headroom=%s", label, hrLabel)
	}
	return ev, name
}

// defaultDoctorSeat picks the seat the doctor/preflight plane would name for a
// launch when the caller names none: the active-role seat when doctor marks it clean
// and policy keeps it in rotation, else the first such seat in report order. Seats
// held out of rotation by POLICY (reserved) are skipped — a launcher is never handed
// one by default, so checking one by default would manufacture a false premise.
func defaultDoctorSeat(reg accounts.Registry, report acctDoctorReport, homeByName map[string]accounts.Home) string {
	clean := func(name string) bool {
		row, ok := rowByNameLookup(report, name)
		if !ok || row.Action != doctorNone || row.Status == string(accounts.LoginTombstoned) {
			return false
		}
		h, ok := homeByName[name]
		return ok && h.Active() && !h.Reserved
	}
	if h, ok := reg.Role(accounts.RoleActive); ok && clean(h.Name) {
		return h.Name
	}
	for _, s := range report.Seats {
		if clean(s.Name) {
			return s.Name
		}
	}
	return ""
}

// rowByNameLookup finds one doctor row by seat name.
func rowByNameLookup(report acctDoctorReport, name string) (doctorSeat, bool) {
	for _, s := range report.Seats {
		if s.Name == name {
			return s, true
		}
	}
	return doctorSeat{}, false
}

// rotationPoolSeatFor resolves the pool seat that serves `name`'s account: the seat
// itself when it is in the pool, or — when the plan collapsed it as a duplicate —
// the canonical seat its bucket is served by (via names that canonical; "" when the
// seat is in the pool directly).
func rotationPoolSeatFor(plan accounts.RotationResult, name string) (accounts.RotationSeat, string, bool) {
	for _, s := range plan.Pool {
		if s.Name == name {
			return s, "", true
		}
	}
	for _, s := range plan.Excluded {
		if s.Name == name && s.Status == accounts.RotationDuplicate && s.Canonical != "" {
			for _, c := range plan.Pool {
				if c.Name == s.Canonical {
					return c, s.Canonical, true
				}
			}
		}
	}
	return accounts.RotationSeat{}, "", false
}

// rotationExclusionFor returns the closed exclusion status the plan recorded for
// `name`, when it recorded one.
func rotationExclusionFor(plan accounts.RotationResult, name string) (accounts.RotationStatus, bool) {
	for _, s := range plan.Excluded {
		if s.Name == name {
			return s.Status, true
		}
	}
	return "", false
}
