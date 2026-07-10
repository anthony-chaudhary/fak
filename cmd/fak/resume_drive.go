package main

// resume_drive.go — `fak resume drive <uuid>`, the read-only operator lens onto the drive-CARRY
// channel (#4135). It answers the one question an operator cannot ask today: "if the watchdog
// relaunches this transcript UUID now, what drive-state comes back?" — BEFORE it happens. The
// carry channel's whole promise is that an 80%-spent budget is not thrown away on relaunch; this
// verb makes that promise inspectable and is the cheapest human-facing regression tripwire.
//
// It folds the SAME durable resume_drivestate.jsonl store the watchdog reads: resume.FoldDriveStates
// for the operator hold token (running/paused/draining/stopped) and resume.FoldDriveCarry for the
// carried budget/objective the sibling cluster's producer records. It renders whatever the fold
// exposes, so the richer carry fields render with no re-wiring here.
//
// STRICTLY read-only: it never launches, never admits, and never appends a row to any ledger — the
// test pins the launch ledger byte-length identical before/after. It is a per-uuid dry-run
// inspector, distinct from `fak resume identity` (a uuid<->trace resolver) and `fak resume
// watchdog` (the launcher).

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// driveReadout is the machine shape `fak resume drive <uuid> --json` emits: the operator hold
// token (if any) and the carried record (if any) a relaunch would restore for this uuid. Carry is
// a pointer so a uuid with no carry marshals it away (a fresh uuid is just its id + held:false).
type driveReadout struct {
	UUID     string             `json:"uuid"`
	Held     bool               `json:"held"`
	State    string             `json:"state,omitempty"`
	HasCarry bool               `json:"has_carry"`
	Carry    *resume.DriveCarry `json:"carry,omitempty"`
}

// runResumeDrive folds the durable drive-state store and prints the drive-state a watchdog
// relaunch would restore for one transcript UUID. Read-only: it appends nothing. Exit 0 on any
// clean read (a uuid with no record is a valid "would come up fresh", not an error), 2 on usage.
// Streams are explicit so a test drives it without a process.
func runResumeDrive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_drivestate.jsonl (default: the same regDir the watchdog resolves — $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	asJSON := fs.Bool("json", false, "emit the folded drive-state record as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	uuid := strings.TrimSpace(fs.Arg(0))
	if uuid == "" {
		fmt.Fprintln(stderr, "fak resume drive: want a transcript UUID to inspect, e.g. `fak resume drive <uuid>`")
		return 2
	}
	regDir := resolveSweepRegDir(*regDirFlag)

	rows := loadDriveStateRows(regDir)
	state := resume.FoldDriveStates(rows)[uuid]
	carry, hasCarry := resume.FoldDriveCarry(rows)[uuid]

	out := driveReadout{
		UUID:     uuid,
		Held:     state.HeldByOperator(),
		State:    string(state),
		HasCarry: hasCarry,
	}
	if hasCarry {
		c := carry
		out.Carry = &c
	}

	if *asJSON {
		data, _ := json.Marshal(out)
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	// Human, strictly read-only — this never launches, so it never prints "launched".
	if !hasCarry && state == "" {
		fmt.Fprintf(stdout, "%s: no recorded drive-state — relaunch would come up fresh\n", uuid)
		return 0
	}
	fmt.Fprintf(stdout, "%s: relaunch would restore drive-state\n", uuid)
	if state != "" {
		hold := ""
		if out.Held {
			hold = " (operator hold)"
		}
		fmt.Fprintf(stdout, "  state: %s%s\n", state, hold)
	}
	if hasCarry {
		if fields := driveCarryFields(carry); len(fields) > 0 {
			fmt.Fprintf(stdout, "  carry: %s\n", strings.Join(fields, " "))
		}
		if txt := strings.TrimSpace(carry.ObjectiveText); txt != "" {
			fmt.Fprintf(stdout, "  objective: %q\n", txt)
		}
	}
	return 0
}

// loadDriveStateRows reads the append-only drive-state store and parses it into rows — the same
// read rwLoadDriveStates does, but returning the rows so this verb can fold BOTH the hold token
// (FoldDriveStates) and the carried record (FoldDriveCarry) from one read. A missing / unreadable
// store yields nil rows, which fold to no hold and no carry — the honest "come up fresh" floor.
func loadDriveStateRows(regDir string) []resume.DriveStateRow {
	raw, err := os.ReadFile(rwDriveStateLedger(regDir))
	if err != nil {
		return nil
	}
	return jsonlledger.Parse[resume.DriveStateRow](string(raw), nil)
}

// driveCarryFields renders the non-zero carry axes as "key=value" tokens (empty ones dropped, the
// same clean-when-absent discipline identityProvenance keeps). ObjectiveText is rendered on its
// own quoted line by the caller, so it is not folded in here.
func driveCarryFields(c resume.DriveCarry) []string {
	var f []string
	if c.TurnsLeft != 0 {
		f = append(f, fmt.Sprintf("turns_left=%d", c.TurnsLeft))
	}
	if c.TokensLeft != 0 {
		f = append(f, fmt.Sprintf("tokens_left=%d", c.TokensLeft))
	}
	if c.ContextTokensLeft != 0 {
		f = append(f, fmt.Sprintf("context_tokens_left=%d", c.ContextTokensLeft))
	}
	if c.SpendMicroCentsLeft != 0 {
		f = append(f, fmt.Sprintf("spend_micro_cents_left=%d", c.SpendMicroCentsLeft))
	}
	if c.TimeLeftNanos != 0 {
		f = append(f, fmt.Sprintf("time_left_nanos=%d", c.TimeLeftNanos))
	}
	if c.Generation != 0 {
		f = append(f, fmt.Sprintf("generation=%d", c.Generation))
	}
	if c.ObjectivePinID != "" {
		f = append(f, "objective_pin_id="+c.ObjectivePinID)
	}
	if c.ObjectiveDigest != "" {
		f = append(f, "objective_digest="+c.ObjectiveDigest)
	}
	return f
}
