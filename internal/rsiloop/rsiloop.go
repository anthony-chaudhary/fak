// Package rsiloop closes fak's recursive-self-improvement loop. internal/shipgate
// is the non-forgeable keep-bit and cmd/rsicycle is a ONE-SHOT that takes the
// before/after/suite-green/truth-clean witnesses AS FLAGS — the loop author hand-
// feeds them. That is the gap this package fills: a TRUE loop DERIVES every witness
// field from a measurement it runs itself, in an isolated worktree, so the author
// cannot forge the numbers that drive the keep-bit.
//
// The four modular parts of the cycle (the "process already known in repo") map to
// four seams, and the engine runs them in order, recursively:
//
//  1. PROPOSE       — Harness.Candidates() yields the proposals to try.
//  2. VERIFY-CORRECT — Harness.Measure(c).SuiteGreen, from a real suite/build run.
//  3. MEASURE-FASTER — Harness.Measure(c).Metric + .TruthClean, from a real KPI run.
//  4. KEEP-OR-REVERT — shipgate.Evaluate (the keep-bit) + shipgate.Gate (the breaker).
//
// The baseline is re-derived from `main` every Run (Harness.BaselineMetric), so a
// keep is always a gain over LATEST main, never a stale local number — and `track`
// mode records that same main measurement as an append-only time series, the
// "ongoing thing that benchmarks against latest main."
package rsiloop

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// Candidate is one proposed change. The engine treats Payload opaquely — the
// Proposer constructs it, the Measure seam interprets it (e.g. a new cache size) —
// so the engine's keep/revert logic stays independent of WHAT is being tuned.
type Candidate struct {
	Label   string
	Payload any
}

// Measurement is the witness a candidate earns, every field DERIVED from a real
// run — never supplied by the loop author. Metric is the candidate's absolute KPI;
// SuiteGreen is the suite's verdict on the candidate; TruthClean is the truth
// syscall's verdict (clean worktree / dos verify). These are exactly the three
// inputs shipgate.Evaluate folds into the non-forgeable keep-bit.
type Measurement struct {
	Metric     float64
	SuiteGreen bool
	TruthClean bool
	// Note carries an optional human-readable detail the engine folds into the
	// journal row (e.g. a suite-red diagnostic) — so a REVERT is diagnosable, not a
	// silent false with no signal.
	Note string
}

// Harness wires the four seams the engine drives. A test supplies in-process fakes;
// cmd/rsiloop supplies the real worktree/probe impls (worktree.go).
type Harness struct {
	// MetricName labels the KPI in the journal (e.g. "vdso_hit_rate").
	MetricName string
	// LowerBetter declares the metric direction (true: a smaller metric wins).
	LowerBetter bool
	// BaselineRefName is the SYMBOLIC ref the baseline forks from (e.g. "main"). It
	// is recorded alongside the resolved SHA so the tracker can refuse to compare two
	// points measured against DIFFERENT refs (a meaningless delta).
	BaselineRefName string
	// BaselineMetric measures the baseline KPI and reports the resolved SHA it was
	// measured at. Called once per Run; a real harness PINS that SHA so every
	// candidate forks from the identical commit (the baselineRef is then truthful for
	// the whole run, and before/after are measured on the same tree).
	BaselineMetric func() (metric float64, baselineRef string, err error)
	// Candidates yields the proposals to try, in order.
	Candidates func() []Candidate
	// Measure applies a candidate in isolation and returns its measured witness. An
	// error here is NOT fatal: a candidate that won't even build is simply reverted
	// (recorded as a non-keep with SuiteGreen=false), and the loop continues.
	Measure func(c Candidate) (Measurement, error)
}

// Row is one append-only journal record. The schema is stable so a downstream
// tracker can diff runs (regression detection vs the last recorded `main` KPI).
type Row struct {
	Cycle        int     `json:"cycle"`
	Mode         string  `json:"mode"` // "improve" | "track"
	Candidate    string  `json:"candidate"`
	MetricName   string  `json:"metric_name"`
	Baseline     float64 `json:"baseline"`
	Candidate_   float64 `json:"candidate_metric"`
	Measured     bool    `json:"measured"` // false => candidate_metric is NOT a real measurement (build/probe failed)
	LowerBetter  bool    `json:"lower_better"`
	Improved     bool    `json:"improved"`    // strict directional gain vs the running baseline
	SuiteGreen   bool    `json:"suite_green"` // derived from a real suite run
	TruthClean   bool    `json:"truth_clean"` // derived from a real truth syscall
	Decision     string  `json:"decision"`    // KEEP | REVERT | ESCALATE | TRACK
	Kept         bool    `json:"kept"`        // the non-forgeable keep-bit (shipgate)
	BreakerCount int     `json:"breaker_nonkeeps"`
	BaselineRef  string  `json:"baseline_ref"`       // the resolved SHA the baseline + every candidate forked from
	RefName      string  `json:"ref_name,omitempty"` // the symbolic ref that SHA was resolved from (e.g. "main")
	Note         string  `json:"note,omitempty"`
}

// Result summarizes a Run.
type Result struct {
	Cycles        int
	Kept          int
	Final         shipgate.Decision
	FinalBaseline float64 // the running baseline after the last KEEP (the recursion's product)
	BaselineRef   string
	Escalated     bool
	Rows          []Row
}

// Journal is an append-only JSONL sink. It is the durable ledger the loop writes —
// every keep/revert and every track measurement is one line, so the file is a
// replayable record of how the KPI moved against main over time.
type Journal struct {
	w      io.WriteCloser
	closer bool
}

// NewJournal opens (creating, appending) a JSONL journal at path. A path of "-" or
// "" writes to stdout and is not closed.
func NewJournal(path string) (*Journal, error) {
	if path == "" || path == "-" {
		return &Journal{w: nopWriteCloser{os.Stdout}}, nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &Journal{w: f, closer: true}, nil
}

// Append writes one row as a single JSON line.
func (j *Journal) Append(r Row) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = j.w.Write(b)
	return err
}

// Close releases the underlying file (a stdout journal is left open).
func (j *Journal) Close() error {
	if j.closer {
		return j.w.Close()
	}
	return nil
}

type nopWriteCloser struct{ io.Writer }

// Close is a no-op so a bare io.Writer (e.g. stdout) satisfies io.WriteCloser without being closed.
func (nopWriteCloser) Close() error { return nil }

// Observer receives each journal Row as it is produced — AFTER the keep/revert
// decision is computed and journaled. It is a pure telemetry side-channel: Run reads
// nothing back from it, so observing a verdict can NEVER move a KEEP to a REVERT or
// vice-versa. It is the seam #588 uses to emit a `dos improve --observe` receipt of
// each verdict without that external command re-gating the loop's own non-forgeable
// keep-bit. A nil Observer is a no-op.
type Observer func(Row)

// Run drives the closed RSI loop. It measures the baseline from `main`, then folds
// each candidate through shipgate's keep-bit and breaker, advancing the running
// baseline on every KEEP (the recursion: the next candidate competes against the
// improved metric). maxCycles<=0 means "all proposed candidates". k is the
// escalation breaker threshold (consecutive non-keeps); k<=0 defaults to 3.
//
// The loop NEVER mutates `main`: kept candidates advance only the in-memory
// baseline and are journaled; landing the final kept change on main is a separate,
// human/gated step (EXTENDING.md "Land it"). It stops early on ESCALATE.
func Run(h Harness, j *Journal, k, maxCycles int) (Result, error) {
	return RunObserved(h, j, k, maxCycles, nil)
}

// RunObserved is Run with a telemetry Observer invoked once per cycle, AFTER the row
// is journaled. The observer sees the SAME Row the journal received and is called for
// its side effect only — the keep-bit stays exactly where Run computes it (the loop's
// decision is never read back from, nor re-gated by, the observer). This is the seam
// #588 uses to emit a `dos improve --observe` receipt of every keep/revert verdict.
func RunObserved(h Harness, j *Journal, k, maxCycles int, obs Observer) (Result, error) {
	base, baseRef, err := h.BaselineMetric()
	if err != nil {
		return Result{}, fmt.Errorf("baseline measure: %w", err)
	}
	gate := shipgate.NewGate(k)
	res := Result{Final: shipgate.REVERT, FinalBaseline: base, BaselineRef: baseRef}
	running := base

	cands := h.Candidates()
	for i, c := range cands {
		if maxCycles > 0 && i >= maxCycles {
			break
		}
		cycle := i + 1

		m, merr := h.Measure(c)
		note := m.Note // carry a measurement-supplied detail (e.g. a suite-red diagnostic)
		measured := merr == nil
		if merr != nil {
			// A candidate that won't build/measure can't be kept — record it as a
			// hard non-keep (suite RED) and let the breaker advance. Measured=false
			// marks Candidate_ as NOT a real measurement (it is the baseline, set only
			// so the witness reverts) — a downstream reader must not trust it.
			m = Measurement{Metric: running, SuiteGreen: false, TruthClean: false}
			note = "measure error: " + merr.Error()
		}

		w := shipgate.Witness{
			Metric:      h.MetricName,
			Before:      running,
			After:       m.Metric,
			LowerBetter: h.LowerBetter,
			SuiteGreen:  m.SuiteGreen,
			TruthClean:  m.TruthClean,
		}
		decision, ev := shipgate.Evaluate(w)
		final := gate.Record(decision) // may upgrade REVERT -> ESCALATE

		row := Row{
			Cycle:        cycle,
			Mode:         "improve",
			Candidate:    c.Label,
			MetricName:   h.MetricName,
			Baseline:     running,
			Candidate_:   m.Metric,
			Measured:     measured,
			LowerBetter:  h.LowerBetter,
			Improved:     improvedDir(running, m.Metric, h.LowerBetter),
			SuiteGreen:   m.SuiteGreen,
			TruthClean:   m.TruthClean,
			Decision:     final.String(),
			Kept:         ev.Kept(),
			BreakerCount: gate.ConsecutiveNonKeeps(),
			BaselineRef:  baseRef,
			RefName:      h.BaselineRefName,
			Note:         note,
		}
		if j != nil {
			if err := j.Append(row); err != nil {
				return res, fmt.Errorf("journal append: %w", err)
			}
		}
		res.Rows = append(res.Rows, row)
		res.Cycles++
		if obs != nil {
			obs(row) // telemetry only — Run ignores any effect; the verdict is already set
		}

		if ev.Kept() {
			running = m.Metric // recursion: the kept gain is the new bar to beat
			res.Kept++
			res.FinalBaseline = running
		}
		res.Final = final
		if final == shipgate.ESCALATE {
			res.Escalated = true
			break
		}
	}
	return res, nil
}

// Track records ONE baseline measurement of the KPI on `main` as a journal row —
// the ongoing benchmark against latest main. It returns the measured metric, the
// ref, and (if a prior track row was loaded) a regression flag against it.
func Track(h Harness, j *Journal) (Row, error) {
	metric, ref, err := h.BaselineMetric()
	if err != nil {
		return Row{}, fmt.Errorf("baseline measure: %w", err)
	}
	row := Row{
		Mode:        "track",
		Candidate:   "(baseline@" + h.BaselineRefName + ")",
		MetricName:  h.MetricName,
		Baseline:    metric,
		Candidate_:  metric,
		Measured:    true,
		LowerBetter: h.LowerBetter,
		Decision:    "TRACK",
		BaselineRef: ref,
		RefName:     h.BaselineRefName,
	}
	if j != nil {
		if err := j.Append(row); err != nil {
			return row, fmt.Errorf("journal append: %w", err)
		}
	}
	return row, nil
}

// improvedDir reports a STRICT directional gain (mirrors shipgate.Witness.improved,
// re-exposed here so a journal row's Improved field is computed the same way the
// keep-bit computes it — one definition of "better", used in both places).
func improvedDir(before, after float64, lowerBetter bool) bool {
	if lowerBetter {
		return after < before
	}
	return after > before
}

// LastTrack scans a journal file for the most recent "track" row, returning it and
// whether one was found. It is CORRUPTION-TOLERANT by design: the journal is a
// durable O_APPEND artifact, so a process killed mid-write leaves a torn final line.
// A line-oriented scan that SKIPS unparseable lines (rather than aborting the whole
// read on the first one) is what keeps the regression guard from failing OPEN — a
// single bad line must not silently discard every valid track point before it. Only
// a file-open failure (other than not-exist) is returned as an error.
func LastTrack(path string) (Row, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Row{}, false, nil
		}
		return Row{}, false, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024) // rows are small; allow a generous line
	var last Row
	found := false
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			continue // skip a torn / non-JSON line rather than fail the whole read open
		}
		if r.Mode == "track" {
			last, found = r, true
		}
	}
	// A scanner error (e.g. an over-long line) is non-fatal for the guard: return what
	// we found so a real prior point still drives regression detection.
	return last, found, nil
}
