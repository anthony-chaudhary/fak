package main

// resume_stopped_liveness.go — the DRIVER-LIVENESS producer for `fak resume stopped`.
//
// #5386 made the mid-tool tail honest: an unmatched trailing tool_use with no evidence
// about the owning process is MIDTOOL_UNKNOWN, and it DEFERS instead of being asserted as
// a crash. It landed the vocabulary, the mapping and the deferral — but nothing that
// PRODUCES the evidence, so `internal/resume/stopped.Classify` (the LivenessUnknown
// wrapper) was the only production caller and every mid-tool row deferred unconditionally
// (#5440). This file is the missing producer: it observes the host process table, binds a
// transcript to a driver process, and hands the stopped leaf a REAL Liveness.
//
// The one rule that governs every line here: a non-empty Liveness is a CLAIM, and a claim
// needs positive evidence in that direction. Nothing below reads a clock, an mtime, an age
// or a file size — those are exactly the guesses #5386 removed, and re-introducing one as
// a "fallback" would hide it better than the original. The two claimable shapes are:
//
//	live — a process that is running RIGHT NOW names this session on its command line
//	       (the `claude --resume <sid>` form the rest of this package already keys on), or
//	       the pid durably recorded for this session is still running.
//	gone — the pid durably recorded for this session is ABSENT from a process table that is
//	       provably complete.
//
// There are TWO producers of that recorded pid, and both are read here (#5542). The resume
// watchdog's launch ledger records one for a session IT resumed — but a FIRST-GENERATION
// worker (`claude -p …`, no session id anywhere on its argv) was never resumed by anything,
// so no launch row exists for it and it resolved to "no recorded driver pid" every time: the
// deferral was correct and permanent for that whole population. The second producer is the
// guard SessionStart hook, which witnesses its own driver process at start and records the pid
// on the durable identity row. The launch ledger keeps precedence where both answer, because a
// launch row is written at the newest spawn; the identity store only fills sessions the launch
// ledger has no pid for, so no verdict this file already reached can change.
//
// Everything else is LivenessUnknown and says so: an unreadable or empty table (#5385
// reports the POSIX census comes back empty on some hosts, which makes `gone` unwitnessable
// there), a session with no recorded driver pid (absence from the table is then not
// evidence of death — a driver launched without the id on its argv would look identical),
// a pid whose command line the platform would not surrender, and a pid that is running but
// is demonstrably some other program (pid reuse witnesses neither life nor death).

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resume/stopped"
)

// stoppedProcRelations enumerates the host's processes (pid + command line) through the
// same audited cross-platform census `fak ps`, `fak resume admit` and the resume watchdog
// already read — one enumeration implementation, not a fork. Production binds
// procguard.CollectRelations; a test injects a fixture, matching the established seam shape
// in this package (fleetCollectRelations, loopReapCollectRelations,
// sessionQueryCollectRelations).
var stoppedProcRelations = procguard.CollectRelations

// stoppedSelfPID reports the pid of the process TAKING the snapshot. It backs the
// completeness self-check below and is injectable so a test can stage a table that cannot
// see its own reader (the shape #5385 describes) without spawning anything.
var stoppedSelfPID = os.Getpid

// stoppedDriverFacts is one folded process-table snapshot plus the durable launch record —
// the whole evidence base a liveness claim may rest on.
type stoppedDriverFacts struct {
	// readable marks a snapshot that was actually obtained: the collector returned no error
	// AND returned at least one row. A table with zero rows is not an empty host, it is a
	// census that did not work (#5385), and it may witness nothing.
	readable bool
	// selfSeen marks a snapshot that contains the pid of the process that took it. A table
	// which cannot see its own reader is provably incomplete, so an ABSENCE in it is not
	// evidence of death. Only a self-seen snapshot may witness LivenessGone; a POSITIVE
	// match needs no such proof (finding a process is finding a process).
	selfSeen bool
	// pids maps every observed pid to its lowercased command line, or "" when the platform
	// did not surrender one.
	pids map[int]string
	// cmdlineUnread counts observed processes whose command line could NOT be read. Those
	// rows are NOT EXAMINED for a session id — they can hide a live driver — so they are
	// counted and reported rather than rendering identically to a row that was examined and
	// did not match. Nothing here is capped or sampled: the count exists to keep the
	// unexaminable remainder visible, not to bound the scan.
	cmdlineUnread int
	// scanned is the number of processes the snapshot returned, and scanErr the collector's
	// own error text — both echoed so an operator can tell "no drivers" from "no census".
	scanned int
	scanErr string
	// launchPIDs maps a session id to the LAST driver pid durably recorded for it, folded
	// across BOTH producers (the launch ledger first, then the session-start identity store
	// for the sessions it does not cover). This is the "recorded driver PID"
	// internal/resume/stopped names as a fact that may assert a liveness value; without one,
	// this triage has no handle on the process at all.
	launchPIDs map[string]int
	// identityPIDs counts how many of those entries came from the session-start identity
	// store rather than the launch ledger, so the operator line can show that the
	// first-generation population is being covered at all rather than silently deferring.
	identityPIDs int
}

// stoppedLivenessReason values are the closed set of explanations livenessFor attaches to
// its verdict, so an operator reading --json sees WHY a row was (or was not) witnessed
// rather than having to re-derive it.
const (
	stoppedWhyTableUnread   = "host process table unreadable or empty — no driver liveness is observable on this host"
	stoppedWhyNamedLive     = "a running process names this session on its command line"
	stoppedWhyNoRecordedPID = "no recorded driver pid for this session — absence from the process table is not evidence of death"
	stoppedWhyIncomplete    = "process table does not contain its own reader — provably incomplete, so an absence proves nothing"
)

// livenessFor resolves the driver-liveness evidence for one session id, with the reason
// that decided it. It returns a non-empty Liveness ONLY on positive evidence; every other
// shape returns LivenessUnknown and means it (#5386).
func (f stoppedDriverFacts) livenessFor(sid string) (stopped.Liveness, string) {
	sid = strings.ToLower(strings.TrimSpace(sid))
	if !f.readable {
		return stopped.LivenessUnknown, fmt.Sprintf("%s (rows=%d err=%q)", stoppedWhyTableUnread, f.scanned, f.scanErr)
	}
	if sid == "" {
		return stopped.LivenessUnknown, "no session id to bind a driver process to"
	}
	// Positive match first: a running process that names this session is live evidence, and
	// it stands whether or not the rest of the table is complete.
	for _, cmd := range f.pids {
		if cmd != "" && strings.Contains(cmd, sid) {
			return stopped.LivenessLive, stoppedWhyNamedLive
		}
	}
	pid, ok := f.launchPIDs[sid]
	if !ok || pid <= 0 {
		return stopped.LivenessUnknown, stoppedWhyNoRecordedPID
	}
	cmd, present := f.pids[pid]
	if !present {
		if !f.selfSeen {
			return stopped.LivenessUnknown, stoppedWhyIncomplete
		}
		return stopped.LivenessGone, fmt.Sprintf(
			"recorded driver pid %d is absent from a process table that can see its own reader", pid)
	}
	if cmd == "" {
		return stopped.LivenessUnknown, fmt.Sprintf(
			"recorded driver pid %d is running but its command line was not readable — not examined", pid)
	}
	if strings.Contains(cmd, sid) || strings.Contains(cmd, "claude") {
		return stopped.LivenessLive, fmt.Sprintf("recorded driver pid %d is still running", pid)
	}
	// The pid exists but belongs to some other program. An operating system does not hand a
	// pid to a second process while the first still holds it, so this is pid REUSE — which
	// says nothing about where the original driver went, and must not be read as death.
	return stopped.LivenessUnknown, fmt.Sprintf(
		"pid %d is running but is not the recorded driver (pid reuse) — neither life nor death is witnessed", pid)
}

// summary is the one operator-visible line describing what the snapshot could and could not
// see, so a host on which liveness is unobservable says so out loud instead of silently
// deferring everything.
func (f stoppedDriverFacts) summary() string {
	if !f.readable {
		return fmt.Sprintf("driver liveness: NOT OBSERVABLE on this host (%s; rows=%d err=%q) — every mid-tool row stays MIDTOOL_UNKNOWN",
			stoppedWhyTableUnread, f.scanned, f.scanErr)
	}
	s := fmt.Sprintf("driver liveness: %d processes scanned, %d recorded driver pids", f.scanned, len(f.launchPIDs))
	if f.identityPIDs > 0 {
		s += fmt.Sprintf(" (%d from session-start identity rows)", f.identityPIDs)
	}
	if !f.selfSeen {
		s += "; table does NOT contain its own reader (incomplete) — no row can be witnessed gone"
	}
	if f.cmdlineUnread > 0 {
		s += fmt.Sprintf("; %d command lines not examined (unreadable)", f.cmdlineUnread)
	}
	return s
}

// stoppedDriverProbe takes the process-table snapshot at most ONCE per invocation (the
// triage classifies many transcripts against one host state) and folds in the durable
// launch record. A run that classifies nothing never pays for the scan.
type stoppedDriverProbe struct {
	once       sync.Once
	ledgerPath string
	// regDir is the fleet registry holding the durable identity store — the SECOND source of
	// recorded driver pids, and the only one that covers a first-generation worker (#5542).
	regDir string
	folded stoppedDriverFacts
}

// newStoppedDriverProbe binds the probe to the host's real evidence: the launch ledger and the
// identity store. Both resolve through the SAME registry-dir rule (FLEET_REG_DIR, else the host
// Fleet registry), so the two producers and this reader cannot drift onto different files.
func newStoppedDriverProbe() *stoppedDriverProbe {
	return &stoppedDriverProbe{ledgerPath: defaultResumeLedger(), regDir: resolveSweepRegDir("")}
}

func (p *stoppedDriverProbe) facts() stoppedDriverFacts {
	p.once.Do(func() { p.folded = foldStoppedDriverFacts(p.ledgerPath, p.regDir) })
	return p.folded
}

// foldStoppedDriverFacts takes the snapshot and folds it, together with the launch ledger,
// into the evidence base. It never fails: an unobtainable table yields readable=false,
// which resolves every session to LivenessUnknown.
func foldStoppedDriverFacts(ledgerPath, regDir string) stoppedDriverFacts {
	procs, errStr := stoppedProcRelations()
	f := stoppedDriverFacts{
		pids:       map[int]string{},
		scanned:    len(procs),
		scanErr:    errStr,
		launchPIDs: stoppedLaunchDriverPIDs(ledgerPath),
	}
	f.identityPIDs = mergeStoppedIdentityDriverPIDs(f.launchPIDs, regDir)
	if errStr != "" || len(procs) == 0 {
		return f
	}
	f.readable = true
	self := stoppedSelfPID()
	for _, pr := range procs {
		if pr.PID <= 0 {
			continue
		}
		cmd := strings.ToLower(strings.TrimSpace(pr.Cmdline))
		if cmd == "" {
			f.cmdlineUnread++
		}
		f.pids[pr.PID] = cmd
		if pr.PID == self {
			f.selfSeen = true
		}
	}
	return f
}

// stoppedLaunchDriverPIDs folds the durable launch ledger into session id -> the LAST driver
// pid a launcher recorded for it. Non-launch phases are excluded through the single
// reader-side launch rule in this package (isNonLaunchPhase), so a deferral row never
// contributes a pid that was never spawned. Last row wins: the most recent spawn is the one
// that owns the transcript. A missing, unreadable or malformed ledger yields an empty map —
// no recorded pid means LivenessUnknown, which is the honest answer, not a fallback.
func stoppedLaunchDriverPIDs(path string) map[string]int {
	out := map[string]int{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var row struct {
			Session string `json:"session"`
			Phase   string `json:"phase"`
			PID     int    `json:"pid"`
		}
		if json.Unmarshal(line, &row) != nil {
			continue
		}
		sid := strings.ToLower(strings.TrimSpace(row.Session))
		if sid == "" || row.PID <= 0 || isNonLaunchPhase(row.Phase) {
			continue
		}
		out[sid] = row.PID
	}
	return out
}

// mergeStoppedIdentityDriverPIDs folds the durable identity store's witnessed driver pids into
// an existing launch-ledger map and reports how many entries it ADDED. It is deliberately
// fill-only: a session the launch ledger already answers keeps that pid, because a launch row
// is written at the newest spawn while an identity row is written at the start of the
// generation that spawn replaced — preferring the identity row there could hand the resolver a
// stale pid, and a stale pid that has since exited is exactly the false `gone` this whole file
// exists to avoid. So the identity store can only cover sessions that had NO answer, which is
// the first-generation population (#5542) and nothing else.
//
// A row with no pid contributes nothing (the store's contract: absent means NOT RECORDED, never
// gone), and a missing/unreadable store yields zero additions — the same honest empty the
// launch-ledger reader returns.
func mergeStoppedIdentityDriverPIDs(launchPIDs map[string]int, regDir string) int {
	if launchPIDs == nil || strings.TrimSpace(regDir) == "" {
		return 0
	}
	added := 0
	for sid, pid := range resume.FoldIdentityDriverPIDs(resume.LoadIdentityRows(regDir)) {
		sid = strings.ToLower(strings.TrimSpace(sid))
		if sid == "" || pid <= 0 {
			continue
		}
		if _, ok := launchPIDs[sid]; ok {
			continue // the launch ledger already answers this session; it wins
		}
		launchPIDs[sid] = pid
		added++
	}
	return added
}
