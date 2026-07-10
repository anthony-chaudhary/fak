package main

// assume.go — `fak assume check`, the thin impure shell over the pure
// internal/assumecheck kernel (#3819, epic #3818 C1). The kernel judges; this file
// only GATHERS the evidence, prints the verdict, and maps it to an exit code — the
// same split stallscan.go keeps with internal/stallscan.
//
//	fak assume check seat-launchable            # the spine's one wired assumption (human)
//	fak assume check seat-launchable --json     # the same verdict as JSON
//	fak assume check seat-launchable --seat X   # check an explicitly named seat
//
// THE ONE WIRED ASSUMPTION (hardcoded inline for the spine; the registry is C2):
// "a seat named launch-clean by `fak accounts doctor`/preflight is actually
// launchable" — witnessed against the REAL launchability authority behind
// `fak accounts next`: Registry.RotationPlanWithHeadroom over the refreshed
// registry, with the live runtime headroom + usage-cooldown overlay folded in
// (accounts_headroom.go), exactly what `next` decides launches from. The failure
// class this catches is authority drift: the doctor/preflight config plane calling a
// seat clean while the rotation authority would refuse to hand it out.
//
// Exit codes: 0 the assumption HOLDS; 1 runtime error; 2 usage; 3 VIOLATED (the
// gate a script watches); 4 UNVERIFIABLE or STALE (cannot witness — still nonzero,
// the kernel is fail-closed).

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/assumecheck"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdAssume(argv []string) { os.Exit(runAssume(os.Stdout, os.Stderr, argv)) }

// runAssume is the testable core (stdout/stderr injected, exit code returned).
func runAssume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "check" {
		fmt.Fprintln(stderr, "usage: fak assume check <assumption-id> [--seat <name>] [--registry <path>] [--home <dir>] [--no-headroom] [--json]")
		return 2
	}
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
	rest := argv[1:]
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
	if id != assumecheck.SeatLaunchable.ID {
		fmt.Fprintf(stderr, "fak assume: unknown assumption %q — the C1 spine carries exactly one (%s); the assumption registry is #3818 C2\n", id, assumecheck.SeatLaunchable.ID)
		return 2
	}

	a := assumecheck.SeatLaunchable
	ev, seatName := gatherSeatLaunchableEvidence(pathutil.ExpandTilde(*registryPath), pathutil.ExpandTilde(*homeDir), strings.TrimSpace(*seat), !*noHeadroom)
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
