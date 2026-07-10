package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// resumehistory.go — the fak_resume_history MCP tool (#3804): the worker-facing, in-session
// half of the resume self-observation. Its sibling `fak resume self`
// (cmd/fak/resume_self.go) answers the same first-person question from the CLI — "was I
// resumed, did that resume take, will another attempt fire, or does a human own me now?" —
// while this hook answers it WITHOUT the worker shelling out, from inside the guarded session.
//
// Both fold the SAME closed outcome folds (resume.FoldSelfObservation) over one session's
// durable launch ledger, so the CLI, this MCP tool, and the operator `resume status` table can
// never tell a worker a different story than they tell a human. It shares the ctxvalue.go
// posture exactly: self-scoped (the calling session by default), read-only, advice-only —
// nothing here feeds the request path.
//
// # Fail-closed on every axis
//
// An unresolved ledger path folds to the honest floor (Resolved=false, an empty observation:
// nothing to recover) with the reason, never a guessed path. A session with no launch rows
// folds to the same floor (has_history=false). And because this hook reads the LEDGER only —
// no transcript — its Outcome is Unknown and NewTurns zero: the same conservative burn-once
// reading the retry gate already takes when a terminal turn cannot be witnessed. The
// transcript-witnessed outcome/progress is `fak resume self`'s enrichment; the shared fold
// downstream makes the two agree wherever the CLI has no transcript to add.

const resumeHistorySchema = "fak.resume-history.v1"

// ResumeHistoryRequest is the fak_resume_history MCP argument shape. Every field is optional:
// the common in-session call passes nothing and self-observes the guarded session against the
// fleet ledger the guard's environment points at.
type ResumeHistoryRequest struct {
	// Session is the session id to self-observe; empty resolves to $CLAUDE_SESSION_ID, then the
	// gateway default trace (under fak guard, the wrapped session itself).
	Session string `json:"session"`
	// Ledger is an explicit resume ledger path; empty resolves the fleet default from the
	// environment ($FLEET_REG_DIR, then the Fleet registry conventions).
	Ledger string `json:"ledger"`
	// MaxAttempts is the give-up cap; <= 0 uses the progress-earned budget (EarnedResumeBudget),
	// the same convention `fak resume self`, RetryGate, and the operator fold take.
	MaxAttempts int `json:"max_attempts"`
}

// ResumeHistoryReport is the fak_resume_history return shape: the shared self-observation record
// plus the resolved provenance (which session, which ledger) so a reader sees WHAT was folded.
// Resolved is false with a Reason when no ledger path could be resolved — the fail-closed floor.
type ResumeHistoryReport struct {
	Schema      string                 `json:"schema"`
	Session     string                 `json:"session,omitempty"`
	LedgerPath  string                 `json:"ledger_path,omitempty"`
	Resolved    bool                   `json:"resolved"`
	Reason      string                 `json:"reason,omitempty"`
	Observation resume.SelfObservation `json:"observation"`
}

// ResumeHistoryFor folds one session's durable launch ledger into its self-observation record.
// It is the single-session read the fak_resume_history MCP tool serves: resolve the session id
// and ledger path (both fail-closed), read that session's ledger rows, and fold them through
// resume.FoldSelfObservation — the exact fold `fak resume self` and `resume status` use.
func (s *Server) ResumeHistoryFor(req ResumeHistoryRequest) ResumeHistoryReport {
	sid := strings.TrimSpace(req.Session)
	if sid == "" {
		sid = strings.TrimSpace(os.Getenv("CLAUDE_SESSION_ID"))
	}
	if sid == "" {
		sid = strings.TrimSpace(s.traceFor(""))
	}
	rep := ResumeHistoryReport{Schema: resumeHistorySchema, Session: sid}

	ledgerPath := strings.TrimSpace(req.Ledger)
	if ledgerPath == "" {
		ledgerPath = resolveResumeLedgerPath()
	}
	if ledgerPath == "" {
		// No ledger to read: fold the honest empty floor and say why, rather than guess a path.
		rep.Reason = "no resume ledger path resolvable — pass ledger, or set FLEET_REG_DIR/FLEET_STATE_DIR"
		rep.Observation = resume.FoldSelfObservation(resume.SelfFacts{Session: sid, MaxAttempts: req.MaxAttempts})
		return rep
	}
	rep.LedgerPath = ledgerPath
	rep.Resolved = true

	// Ledger-only fold: Outcome Unknown, NewTurns 0 (no transcript is read here). A session with
	// no rows for sid reads the same has_history=false floor the CLI takes when never resumed.
	hist := readResumeLedgerHistory(ledgerPath)[sid]
	rep.Observation = resume.FoldSelfObservation(resume.SelfFacts{
		Session:     sid,
		History:     hist,
		Outcome:     resume.OutcomeUnknown,
		MaxAttempts: req.MaxAttempts,
	})
	return rep
}

// resolveResumeLedgerPath resolves the fleet resume ledger from the environment the guard runs
// under, mirroring the CLI's defaultResumeLedger (FLEET_REG_DIR, else the Fleet registry
// conventions). It is the gateway's I/O-shell copy of that resolution — internal/resume stays
// I/O-free by design, so the shell owns the path. $FLEET_REG_DIR wins, then
// $FLEET_STATE_DIR/registry, then %LOCALAPPDATA%/Fleet/registry when it exists. Returns "" when
// none resolves: the caller fails closed rather than guess the repo-root fallback the CLI takes
// (findRepoRoot lives in cmd/fak and is not worth importing here for a best-effort self-read).
func resolveResumeLedgerPath() string {
	if reg := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); reg != "" {
		return filepath.Join(reg, "resume_ledger.jsonl")
	}
	if v := strings.TrimSpace(os.Getenv("FLEET_STATE_DIR")); v != "" {
		return filepath.Join(v, "registry", "resume_ledger.jsonl")
	}
	if v := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); v != "" {
		cand := filepath.Join(v, "Fleet", "registry")
		if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
			return filepath.Join(cand, "resume_ledger.jsonl")
		}
	}
	return ""
}

// readResumeLedgerHistory parses the durable resume ledger JSONL into per-session launch
// history, in ledger order (oldest first — the order RetryGate/EarnedResumeBudget expect). It
// mirrors the CLI's loadResumeHistory row-for-row — same fields, same Attempt shape — so the two
// readers hand FoldSelfObservation identical facts; the shared fold downstream is the contract.
// A missing or unreadable ledger is no history, never an error: a worker never resumed reads the
// honest empty floor.
func readResumeLedgerHistory(path string) map[string][]resume.Attempt {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	type lrec struct {
		Ts             string `json:"ts"`
		Session        string `json:"session"`
		Phase          string `json:"phase"`
		Action         string `json:"action"`
		ManualOverride bool   `json:"manual_override"`
	}
	out := make(map[string][]resume.Attempt)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r lrec
		if json.Unmarshal(line, &r) != nil || r.Session == "" {
			continue
		}
		out[r.Session] = append(out[r.Session], resume.Attempt{
			UnixSeconds:    parseLedgerUnix(r.Ts),
			Phase:          r.Phase,
			Action:         r.Action,
			ManualOverride: r.ManualOverride,
		})
	}
	return out
}

// parseLedgerUnix parses a ledger row's RFC3339(/Nano) timestamp to unix seconds, mirroring the
// CLI's parseTranscriptUnix. An empty or unparseable stamp is zero — a neutral, never-penalized
// gap in EarnedResumeBudget's spacing read.
func parseLedgerUnix(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.Unix()
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.Unix()
	}
	return 0
}
