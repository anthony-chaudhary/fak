package main

// serve_durability.go — durable session state ON BY DEFAULT (#1365, part of #1352).
//
// The cold-resume snapshot of #629 (restoreServeSessions / dumpServeSessions) was gated
// behind an explicit `--session-state FILE`, so out of the box even a graceful Ctrl-C
// persisted nothing and the durability story was invisible until an operator had already
// lost a session. This file is the default-on resolution plus the operator-facing POSTURE
// both `fak serve` (a boot banner) and `fak doctor serve` (a readiness row) render.
//
// The resolution is a PURE fold over the flag value and an injected env reader, so the
// whole default-on/opt-out matrix is unit-witnessable with no serve process and no disk.
// The low-level restore/dump helpers keep their "" == off contract untouched: the default
// is resolved HERE, at the boot seam, and the resolved path is what those helpers see. An
// explicit `--session-state FILE` therefore still behaves byte-for-byte as it did.

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const (
	// serveSessionStateEnv relocates or disables the default cold-resume snapshot without
	// a flag — the documented opt-out for an operator who cannot edit the command line
	// (a unit file, a container entrypoint). A path relocates it; "off" disables it.
	serveSessionStateEnv = "FAK_SESSION_STATE"
	// serveSessionStateOff is the literal that disables persistence, accepted by both
	// --session-state and FAK_SESSION_STATE so the two knobs never disagree on spelling.
	serveSessionStateOff = "off"
	// serveSessionStateFile is the default snapshot filename under the per-user fak state
	// dir, sitting beside the always-on session descriptor registry (#5825).
	serveSessionStateFile = "session-state.snap"
)

// serveDurabilityPosture is what `fak serve` persists across a restart, where it lands,
// which knob decided that, and which signals flush it — the three questions #1365 asks
// `fak doctor` to answer. It is data, not behavior, so both renderers fold the same value.
type serveDurabilityPosture struct {
	Enabled   bool     `json:"enabled"`
	Path      string   `json:"path,omitempty"`     // resolved snapshot path ("" when disabled)
	Source    string   `json:"source"`             // which knob decided: "flag" | "env" | "default"
	Persisted []string `json:"persisted"`          // what the snapshot carries
	Signals   []string `json:"signals"`            // signals that trigger the graceful flush
	Registry  string   `json:"registry,omitempty"` // the always-on descriptor registry (#5825)
}

// servePersistedFields names what a cold-resume snapshot actually carries, so the doctor
// answers "what is persisted" with the real record shape rather than a vague "state".
func servePersistedFields() []string {
	return []string{"run-state + stop reason", "context budget (turns/tokens)", "priority", "pace", "revision"}
}

// serveFlushSignalNames renders the signals whose graceful drain flushes the snapshot
// (#1359). It reads the SAME per-platform terminatingSignals() the serve loop registers,
// so the doctor can never advertise a signal this build does not actually handle — the
// Windows build honestly reports two, a POSIX build three.
func serveFlushSignalNames() []string {
	sigs := terminatingSignals()
	names := make([]string, 0, len(sigs))
	for _, s := range sigs {
		names = append(names, s.String())
	}
	return names
}

// defaultServeSessionStatePath is the no-flag snapshot location: the per-user fak state
// dir, beside the session descriptor registry defaultSessionRegistryPath() already uses,
// so the two durability planes live together. A host with no resolvable user config dir
// falls back to the workspace-local .fak/ — never to "off", because a missing config dir
// must not silently take durability down with it.
func defaultServeSessionStatePath() string {
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", serveSessionStateFile)
	}
	return filepath.Join(".fak", serveSessionStateFile)
}

// resolveServeSessionState folds the --session-state flag value and the environment into
// the posture a boot will run with. Precedence is flag > env > default, and "off" at
// either knob disables persistence:
//
//	--session-state FILE  → that file            (source "flag") — unchanged from #629
//	--session-state off   → disabled             (source "flag")
//	FAK_SESSION_STATE=…   → that file / disabled (source "env")
//	(nothing)             → the default path     (source "default") — the #1365 flip
//
// Pure: env is injected, no disk is touched, so every arm is unit-witnessable.
func resolveServeSessionState(flagValue string, env func(string) string) serveDurabilityPosture {
	p := serveDurabilityPosture{
		Persisted: servePersistedFields(),
		Signals:   serveFlushSignalNames(),
		Registry:  defaultSessionRegistryPath(),
	}
	raw, source := strings.TrimSpace(flagValue), "flag"
	if raw == "" {
		if env != nil {
			raw = strings.TrimSpace(env(serveSessionStateEnv))
		}
		source = "env"
	}
	if raw == "" {
		raw, source = defaultServeSessionStatePath(), "default"
	}
	p.Source = source
	if strings.EqualFold(raw, serveSessionStateOff) {
		return p // Enabled stays false; Path stays empty so the helpers no-op
	}
	p.Enabled = true
	p.Path = pathutil.ExpandTilde(raw)
	return p
}

// serveDurabilityOptOutHint is the one line that tells an operator how to turn the
// default off, reused by the banner and the doctor row so the two never drift.
func serveDurabilityOptOutHint() string {
	return "opt out with --session-state " + serveSessionStateOff + " or " + serveSessionStateEnv + "=" + serveSessionStateOff
}

// writeServeDurabilityBanner prints the boot line that makes the posture visible BEFORE a
// session is lost rather than after — the operability half of #1365. A disabled posture is
// stated just as loudly as an enabled one: silence is what the issue is about.
func writeServeDurabilityBanner(w io.Writer, p serveDurabilityPosture) {
	if w == nil {
		return
	}
	if !p.Enabled {
		fmt.Fprintf(w, "fak serve: session durability OFF (%s) — a restart re-attaches nothing; unset the %s opt-out to restore the default\n",
			p.Source, serveSessionStateEnv)
		return
	}
	fmt.Fprintf(w, "fak serve: session durability ON (%s) — %s → %s; flushed on %s (%s)\n",
		p.Source, strings.Join(p.Persisted, ", "), p.Path, strings.Join(p.Signals, "/"), serveDurabilityOptOutHint())
}

// serveDurabilityRow classifies the posture as one `fak doctor serve` readiness row. An
// enabled posture is green and names the path, the persisted fields, the flush signals and
// the descriptor registry; a disabled one is yellow (an operator may have opted out on
// purpose — that is a posture to surface, not a failure to red) with the re-enable action.
func serveDurabilityRow(p serveDurabilityPosture) serveReadinessRow {
	row := serveReadinessRow{Check: "session-durability"}
	if p.Enabled {
		row.Status = sevOK
		row.Finding = fmt.Sprintf("durable session state ON (%s) at %s — persists %s; flushed on %s; descriptor registry %s",
			p.Source, p.Path, strings.Join(p.Persisted, ", "), strings.Join(p.Signals, "/"), p.Registry)
		row.Remediation = serveDurabilityOptOutHint()
	} else {
		row.Status = sevWarn
		row.Finding = fmt.Sprintf("session persistence is OFF (%s) — a restart starts every session at its defaults, so even a graceful Ctrl-C loses the drive state", p.Source)
		row.Remediation = "drop the --session-state " + serveSessionStateOff + " / " + serveSessionStateEnv + "=" + serveSessionStateOff + " opt-out, or name a path, to restore the default durable posture"
	}
	row.Tier = serveTierLabel(row.Status)
	return row
}

// withServeDurabilityRow appends the durability row to a host-readiness report and
// re-rolls the summary, so the rollup and finding count stay consistent with the rows
// actually present. Kept separate from buildServeReadiness because that fold is over pure
// HOST facts; the posture is a separate injected fact with its own resolution.
func withServeDurabilityRow(rep serveReadinessReport, p serveDurabilityPosture) serveReadinessReport {
	rep.Durability = &p
	rep.Rows = append(rep.Rows, serveDurabilityRow(p))
	worst := sevOK
	rep.Findings = 0
	for _, r := range rep.Rows {
		if serveStatusRank(r.Status) > serveStatusRank(worst) {
			worst = r.Status
		}
		if r.Status != sevOK {
			rep.Findings++
		}
	}
	rep.Rollup = serveTierLabel(worst)
	return rep
}
