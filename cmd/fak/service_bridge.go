package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scmbridge"
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// `fak service bridge` (#4756): the one desired-state / read-back seam for
// the Windows supervision plane. It projects a portable fak.service.v1 spec
// into its native form for one launch role (SCM machine service, S4U
// watchdog task, InteractiveToken session broker), reconciles the projection
// against an authoritative native read-back (a document, or --live query-only
// SCM capture on Windows), optionally records the read-back phase in the
// observed-event ledger, and referees the #4698 destructive matrix from a
// captured ledger (--judge).

// bridgeLiveReadBack captures the authoritative local SCM read-back; the
// Windows build assigns the real query-only implementation.
var bridgeLiveReadBack = func(unit string) (scmbridge.Observed, error) {
	return scmbridge.Observed{}, errors.New("live SCM read-back requires windows")
}

var bridgeNowMS = func() int64 { return time.Now().UnixMilli() }

func bridgeRoleFromString(s string) (scmbridge.Role, bool) {
	switch s {
	case "machine", string(scmbridge.RoleMachine):
		return scmbridge.RoleMachine, true
	case "watchdog", string(scmbridge.RoleWatchdog):
		return scmbridge.RoleWatchdog, true
	case "broker", string(scmbridge.RoleBroker):
		return scmbridge.RoleBroker, true
	}
	return "", false
}

type bridgeOutput struct {
	Projection *scmbridge.Projection `json:"projection"`
	Observed   *scmbridge.Observed   `json:"observed,omitempty"`
	Report     *scmbridge.Report     `json:"report,omitempty"`
	Phase      servicespec.Phase     `json:"phase,omitempty"`
}

func runServiceBridge(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("service bridge", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "machine-readable output")
	roleStr := fs.String("role", "", "launch role: machine|watchdog|broker")
	specFile := fs.String("spec", "", "fak.service.v1 desired-state document")
	sha := fs.String("sha256", "", "expected installed-binary SHA-256 (provenance pin)")
	principal := fs.String("principal", "", "task principal for watchdog/broker roles")
	observedFile := fs.String("observed", "", "native read-back document (JSON) to reconcile against")
	live := fs.Bool("live", false, "capture the read-back from the local SCM (query-only; windows)")
	unit := fs.String("unit", "", "SCM service name for --live (default: the projected unit)")
	ledgerDir := fs.String("ledger-dir", "", "record the read-back phase in the observed-event ledger")
	judgeProbe := fs.String("judge", "", "probe verdict from the ledger: terminal-kill|termservice-reset|host-reboot|scm-process-kill")
	service := fs.String("service", "", "identity service filter for --judge")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	if *judgeProbe != "" {
		return runBridgeJudge(stdout, stderr, *judgeProbe, *ledgerDir, *service, *asJSON)
	}
	if *specFile == "" || *roleStr == "" {
		fmt.Fprintln(stderr, "fak service bridge: --spec and --role are required (or --judge)")
		return 2
	}
	role, ok := bridgeRoleFromString(*roleStr)
	if !ok {
		fmt.Fprintf(stderr, "fak service bridge: unknown role %q\n", *roleStr)
		return 2
	}
	data, err := os.ReadFile(*specFile)
	if err != nil {
		fmt.Fprintln(stderr, "fak service bridge:", err)
		return 1
	}
	spec, err := servicespec.ParseSpec(data)
	if err != nil {
		fmt.Fprintln(stderr, "fak service bridge:", err)
		return 1
	}
	proj, err := scmbridge.Project(spec, scmbridge.ProjectInput{Role: role, BinarySHA256: *sha, TaskPrincipal: *principal})
	if err != nil {
		fmt.Fprintln(stderr, "fak service bridge:", err)
		return 1
	}
	out := bridgeOutput{Projection: proj}
	if *observedFile == "" && !*live {
		return writeBridgeOutput(stdout, out, *asJSON, true)
	}
	var got scmbridge.Observed
	if *live {
		u := *unit
		if u == "" {
			u = proj.UnitName
		}
		if got, err = bridgeLiveReadBack(u); err != nil {
			fmt.Fprintln(stderr, "fak service bridge: live read-back:", err)
			return 1
		}
	} else {
		b, err := os.ReadFile(*observedFile)
		if err != nil {
			fmt.Fprintln(stderr, "fak service bridge:", err)
			return 1
		}
		if err := json.Unmarshal(b, &got); err != nil {
			fmt.Fprintln(stderr, "fak service bridge: observed document:", err)
			return 1
		}
	}
	rep := scmbridge.Reconcile(proj, got)
	out.Observed, out.Report = &got, &rep
	if got.Present {
		if proj.Manager == scmbridge.ManagerSCM {
			out.Phase = scmbridge.PhaseFromSCMState(got.Status, got.PID)
		} else {
			out.Phase = scmbridge.PhaseFromTaskState(got.Status)
		}
	}
	if *ledgerDir != "" && got.Present {
		led, rc := openServiceLedger(stderr, *ledgerDir)
		if rc != 0 {
			return rc
		}
		ev := serviceledger.Event{
			Type:        serviceledger.EventReadiness,
			AtUnixMS:    bridgeNowMS(),
			Source:      serviceledger.SourceFak,
			Identity:    proj.Identity,
			Phase:       out.Phase,
			Correlation: serviceledger.Correlation{PID: got.PID},
			Detail:      fmt.Sprintf("bridge read-back role=%s status=%s", proj.Role, got.Status),
		}
		if _, _, err := led.Append(ev); err != nil {
			fmt.Fprintln(stderr, "fak service bridge: ledger:", err)
			return 1
		}
	}
	return writeBridgeOutput(stdout, out, *asJSON, rep.InSync)
}

func writeBridgeOutput(stdout io.Writer, out bridgeOutput, asJSON, inSync bool) int {
	if asJSON {
		_ = json.NewEncoder(stdout).Encode(out)
	} else {
		p := out.Projection
		fmt.Fprintf(stdout, "%s %s role=%s principal=%s\n", p.Manager, p.UnitName, p.Role, p.Principal)
		if out.Report != nil {
			if out.Report.InSync {
				fmt.Fprintf(stdout, "in sync (phase=%s)\n", out.Phase)
			}
			for _, d := range out.Report.Divergences {
				fmt.Fprintf(stdout, "diverged %s: want=%s got=%s\n", d.Axis, d.Want, d.Got)
			}
		}
	}
	if !inSync {
		return 4
	}
	return 0
}

// runBridgeJudge referees one destructive probe of the #4698 matrix against
// the captured observed-event ledger. Exit 0 only when the stop was
// INDEPENDENTLY corroborated and the resume evidence followed; 4 otherwise.
func runBridgeJudge(stdout, stderr io.Writer, probe, ledgerDir, service string, asJSON bool) int {
	var p scmbridge.Probe
	for _, c := range scmbridge.AllProbes {
		if string(c) == probe {
			p = c
		}
	}
	if p == "" {
		fmt.Fprintf(stderr, "fak service bridge: unknown probe %q\n", probe)
		return 2
	}
	led, rc := openServiceLedger(stderr, ledgerDir)
	if rc != 0 {
		return rc
	}
	v := scmbridge.Judge(p, filterServiceEvents(led.Events(), service))
	if asJSON {
		_ = json.NewEncoder(stdout).Encode(v)
	} else {
		fmt.Fprintf(stdout, "%s corroborated=%v resumed=%v", v.Probe, v.Corroborated, v.Resumed)
		for _, m := range v.Missing {
			fmt.Fprintf(stdout, " missing=%s", m)
		}
		fmt.Fprintln(stdout)
	}
	if v.Passed() {
		return 0
	}
	return 4
}
