package naivecontrol

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	// Schema is the ledger row schema.
	Schema = "fak-naive-control/1"
	// CompareSchema is the side-by-side report schema.
	CompareSchema = "fak-naive-control-compare/1"
	// DefaultLedgerRel is the ledger path under the workspace. .fak/ is gitignored
	// runtime state, beside the loop ledger (.fak/loops.jsonl).
	DefaultLedgerRel = ".fak/naive-control.jsonl"
	// NaiveCommand is the naive arm, verbatim.
	NaiveCommand = "fak issue-orchestrator --top 10 --max-waves 1"
)

// Arm names which dispatch path produced a row, so both arms share one ledger.
type Arm string

const (
	ArmNaive        Arm = "naive"
	ArmOrchestrated Arm = "orchestrated"
)

// Arms is the fixed display order of the comparison.
var Arms = []Arm{ArmNaive, ArmOrchestrated}

// Valid reports whether a is in the closed arm vocabulary.
func (a Arm) Valid() bool { return a == ArmNaive || a == ArmOrchestrated }

// Stage says whether a row carries git evidence yet.
type Stage string

const (
	// StagePlanned: the arm chose its picks; nothing has been verified, so ship
	// counts are MISSING_MEASUREMENT, not 0.
	StagePlanned Stage = "PLANNED"
	// StageHarvested: the git-ancestry pass ran over the picks.
	StageHarvested Stage = "HARVESTED"
)

// PickStatus is the git-derived outcome for one pick.
type PickStatus string

const (
	PickShipped    PickStatus = "SHIPPED"
	PickNotShipped PickStatus = "NOT_SHIPPED"
	PickUnknown    PickStatus = "UNKNOWN"
)

// PickRecord is one pick and what git said about it.
type PickRecord struct {
	Issue  int        `json:"issue"`
	Base   string     `json:"base_sha,omitempty"`
	Status PickStatus `json:"status"`
	Reason string     `json:"reason,omitempty"`
}

// Row is one run of one arm. Later rows for the same (arm, run_id) supersede earlier
// ones, so a PLANNED row is replaced by its HARVESTED row without rewriting history.
//
// Field names follow the orchestrated loop ledger where the concepts align:
// ts_unix_nano and run_id are loopmgr.Event's; controller_ms is the counterpart of
// the dispatch tick's tick_total_ms.
type Row struct {
	Schema          string          `json:"schema"`
	Arm             Arm             `json:"arm"`
	RunID           string          `json:"run_id"`
	Stage           Stage           `json:"stage"`
	TSUnixNano      int64           `json:"ts_unix_nano"`
	StartedUnixNano int64           `json:"started_unix_nano,omitempty"`
	Command         string          `json:"command,omitempty"`
	HeadSHA         string          `json:"verified_against,omitempty"`
	PicksOffered    Metric[int64]   `json:"picks_offered"`
	PicksShipped    Metric[int64]   `json:"picks_shipped"`
	CommitsClaimed  Metric[int64]   `json:"commits_claimed"`
	CommitsLanded   Metric[int64]   `json:"commits_landed"`
	WallMS          Metric[int64]   `json:"wall_ms"`
	ControllerMS    Metric[int64]   `json:"controller_ms"`
	CostUSD         Metric[float64] `json:"cost_usd"`
	Picks           []PickRecord    `json:"picks,omitempty"`
	Commits         []CommitCheck   `json:"commits,omitempty"`
}

// RunInput is everything a recorder knows about one run. Build turns it into a Row
// without reading a clock, a process, or a file.
type RunInput struct {
	Arm        Arm
	RunID      string
	RecordedAt time.Time
	StartedAt  time.Time
	Command    string

	// Picks is the offered set. PicksKnown=false means the offered set could not be
	// read (the arm failed, or its output did not parse); PicksWhy says why.
	Picks      []Pick
	PicksKnown bool
	PicksWhy   string

	// SelfReported is what the arm claimed about itself. SelfReportRead=false means
	// no self-report was read, which makes commits_claimed unknown, not 0.
	SelfReported   []Claim
	SelfReportRead bool

	// Verification is the git evidence; nil means not harvested yet.
	Verification *Verification

	WallMS       Metric[int64]
	ControllerMS Metric[int64]
	CostUSD      Metric[float64]
}

// Build folds a RunInput into a Row. Pure.
func Build(in RunInput) Row {
	r := Row{
		Schema:       Schema,
		Arm:          in.Arm,
		RunID:        in.RunID,
		Stage:        StagePlanned,
		TSUnixNano:   unixNano(in.RecordedAt),
		Command:      in.Command,
		WallMS:       orMissing(in.WallMS, "wall time not observed"),
		ControllerMS: orMissing(in.ControllerMS, "controller time not observed"),
		CostUSD:      orMissing(in.CostUSD, "no cost source was read for this run"),
	}
	r.StartedUnixNano = unixNano(in.StartedAt)

	picks := dedupPicks(in.Picks)
	if in.PicksKnown {
		r.PicksOffered = Measured(int64(len(picks)))
	} else {
		r.PicksOffered = Missing[int64]("offered picks unreadable: " + orDefault(in.PicksWhy, "no plan output"))
	}

	if in.SelfReportRead {
		r.CommitsClaimed = Measured(int64(len(in.SelfReported)))
	} else {
		r.CommitsClaimed = Missing[int64]("no self-report was read for this run")
	}

	v := in.Verification
	switch {
	case !in.PicksKnown:
		why := "offered picks unreadable, so no shipped pick can be attributed"
		r.PicksShipped = Missing[int64](why)
		r.CommitsLanded = Missing[int64](why)
		if v != nil {
			r.Stage = StageHarvested
			r.HeadSHA = v.Head
			r.Commits = v.Checks
		}
		return r
	case v == nil:
		why := "not harvested: ship counts come only from a git-ancestry pass"
		r.PicksShipped = Missing[int64](why)
		r.CommitsLanded = Missing[int64](why)
		for _, p := range picks {
			r.Picks = append(r.Picks, PickRecord{Issue: p.Issue, Base: p.Base, Status: PickUnknown, Reason: "not harvested"})
		}
		return r
	}

	r.Stage = StageHarvested
	r.HeadSHA = v.Head
	r.Commits = v.Checks
	r.Picks, r.PicksShipped, r.CommitsLanded = foldPicks(picks, v)
	return r
}

// foldPicks applies the ship rule. A pick is SHIPPED when any claim for it LANDED;
// NOT_SHIPPED only when this package's own scan of its base..HEAD completed and no
// claim for it is UNVERIFIABLE; UNKNOWN otherwise. A totals metric is MEASURED only
// when no pick behind it is unknown.
func foldPicks(picks []Pick, v *Verification) ([]PickRecord, Metric[int64], Metric[int64]) {
	landed := map[int]bool{}
	unverifiable := map[int]int{}
	landedSHAs := map[string]bool{}
	for _, c := range v.Checks {
		switch c.Verdict {
		case VerdictLanded:
			landed[c.Issue] = true
			landedSHAs[c.Resolved] = true
		case VerdictUnverifiable:
			unverifiable[c.Issue]++
		}
	}
	var recs []PickRecord
	var shipped, unknown, incomplete int64
	unknownBy := map[string][]int{}
	for _, p := range picks {
		rec := PickRecord{Issue: p.Issue, Base: p.Base}
		switch {
		case landed[p.Issue]:
			rec.Status = PickShipped
			shipped++
		case unverifiable[p.Issue] > 0:
			rec.Status = PickUnknown
			rec.Reason = fmt.Sprintf("%d claim(s) git could not verify", unverifiable[p.Issue])
		case !v.Scanned[p.Issue]:
			rec.Status = PickUnknown
			rec.Reason = orDefault(v.ScanErrs[p.Issue], "base..HEAD scan did not run")
		default:
			rec.Status = PickNotShipped
		}
		if rec.Status == PickUnknown {
			unknown++
			unknownBy[rec.Reason] = append(unknownBy[rec.Reason], p.Issue)
		}
		// commits_landed is complete only when every pick was scanned and no claim
		// is unverifiable; a shipped-but-unscanned pick may have landed more.
		if !v.Scanned[p.Issue] || unverifiable[p.Issue] > 0 {
			incomplete++
		}
		recs = append(recs, rec)
	}

	var shippedM, landedM Metric[int64]
	if v.Head == "" {
		why := "git ancestry unreadable: " + orDefault(v.HeadErr, "HEAD unresolved")
		return recs, Missing[int64](why), Missing[int64](why)
	}
	if unknown == 0 {
		shippedM = Measured(shipped)
	} else {
		shippedM = Missing[int64](fmt.Sprintf("%d of %d pick(s) unknown (%d verified shipped): %s",
			unknown, len(picks), shipped, groupReasons(unknownBy)))
	}
	if incomplete == 0 {
		landedM = Measured(int64(len(landedSHAs)))
	} else {
		landedM = Missing[int64](fmt.Sprintf("%d of %d pick(s) lack a complete git scan (%d commit(s) verified landed so far)",
			incomplete, len(picks), len(landedSHAs)))
	}
	return recs, shippedM, landedM
}

// groupReasons renders pick-level reasons one cause at a time, most common first,
// naming at most three issues per cause so the line stays readable on one screen.
func groupReasons(by map[string][]int) string {
	reasons := make([]string, 0, len(by))
	for r := range by {
		reasons = append(reasons, r)
	}
	sort.Slice(reasons, func(i, j int) bool {
		if len(by[reasons[i]]) != len(by[reasons[j]]) {
			return len(by[reasons[i]]) > len(by[reasons[j]])
		}
		return reasons[i] < reasons[j]
	})
	parts := make([]string, 0, len(reasons))
	for _, r := range reasons {
		issues := by[r]
		var named []string
		for i, n := range issues {
			if i == 3 {
				named = append(named, fmt.Sprintf("+%d more", len(issues)-3))
				break
			}
			named = append(named, fmt.Sprintf("#%d", n))
		}
		parts = append(parts, fmt.Sprintf("%d x %s (%s)", len(issues), r, strings.Join(named, ", ")))
	}
	return strings.Join(parts, "; ")
}

// Since keeps the rows recorded at or after from, so a comparison can cover a recent
// window. Filtering is by recording time only; it never drops a run for its values.
func Since(rows []Row, from time.Time) []Row {
	if from.IsZero() {
		return rows
	}
	var out []Row
	for _, r := range rows {
		if r.TSUnixNano >= from.UnixNano() {
			out = append(out, r)
		}
	}
	return out
}

// Validate rejects a row this package did not write or that contradicts itself.
func Validate(r Row) error {
	if r.Schema != Schema {
		return fmt.Errorf("schema %q, want %q", r.Schema, Schema)
	}
	if !r.Arm.Valid() {
		return fmt.Errorf("arm %q is not one of naive|orchestrated", r.Arm)
	}
	if strings.TrimSpace(r.RunID) == "" {
		return errors.New("run_id is empty")
	}
	if r.Stage != StagePlanned && r.Stage != StageHarvested {
		return fmt.Errorf("stage %q is not one of PLANNED|HARVESTED", r.Stage)
	}
	if r.TSUnixNano <= 0 {
		return errors.New("ts_unix_nano is not set")
	}
	for _, e := range []error{
		r.PicksOffered.validate("picks_offered"),
		r.PicksShipped.validate("picks_shipped"),
		r.CommitsClaimed.validate("commits_claimed"),
		r.CommitsLanded.validate("commits_landed"),
		r.WallMS.validate("wall_ms"),
		r.ControllerMS.validate("controller_ms"),
		r.CostUSD.validate("cost_usd"),
	} {
		if e != nil {
			return e
		}
	}
	offered, okO := r.PicksOffered.Get()
	shipped, okS := r.PicksShipped.Get()
	if okO && okS && shipped > offered {
		return fmt.Errorf("picks_shipped %d exceeds picks_offered %d", shipped, offered)
	}
	return nil
}

// LedgerHealth discloses what a read dropped, so a malformed ledger is visible
// instead of silently shrinking an arm.
type LedgerHealth struct {
	Path     string   `json:"path,omitempty"`
	Rows     int      `json:"rows"`
	Rejected int      `json:"rejected"`
	Missing  bool     `json:"missing,omitempty"`
	Errors   []string `json:"errors,omitempty"`
}

const maxHealthErrors = 5

// ParseLedger reads JSONL rows, keeping valid ones and counting the rest.
func ParseLedger(content string) ([]Row, LedgerHealth) {
	var rows []Row
	var h LedgerHealth
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var r Row
		err := json.Unmarshal([]byte(text), &r)
		if err == nil {
			err = Validate(r)
		}
		if err != nil {
			h.Rejected++
			if len(h.Errors) < maxHealthErrors {
				h.Errors = append(h.Errors, fmt.Sprintf("line %d: %v", line, err))
			}
			continue
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		h.Rejected++
		h.Errors = append(h.Errors, "scan: "+err.Error())
	}
	h.Rows = len(rows)
	return rows, h
}

// ReadLedger reads path. A missing file is an empty ledger, reported as Missing.
func ReadLedger(path string) ([]Row, LedgerHealth, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, LedgerHealth{Path: path, Missing: true}, nil
	}
	if err != nil {
		return nil, LedgerHealth{Path: path}, err
	}
	rows, h := ParseLedger(string(b))
	h.Path = path
	return rows, h, nil
}

// Append validates r and appends it as one JSONL line.
func Append(path string, r Row) error {
	if err := Validate(r); err != nil {
		return fmt.Errorf("naivecontrol: refusing invalid row: %w", err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Latest keeps the last row per (arm, run_id), by ts_unix_nano then ledger order.
func Latest(rows []Row) []Row {
	type key struct {
		arm Arm
		run string
	}
	idx := map[key]int{}
	var out []Row
	for _, r := range rows {
		k := key{r.Arm, r.RunID}
		if i, ok := idx[k]; ok {
			if r.TSUnixNano >= out[i].TSUnixNano {
				out[i] = r
			}
			continue
		}
		idx[k] = len(out)
		out = append(out, r)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].TSUnixNano < out[j].TSUnixNano })
	return out
}

// FindRun returns the latest row for (arm, runID).
func FindRun(rows []Row, arm Arm, runID string) (Row, bool) {
	for _, r := range Latest(rows) {
		if r.Arm == arm && r.RunID == runID {
			return r, true
		}
	}
	return Row{}, false
}

// PicksOf rebuilds the Pick list a row was offered.
func PicksOf(r Row) []Pick {
	out := make([]Pick, 0, len(r.Picks))
	for _, p := range r.Picks {
		out = append(out, Pick{Issue: p.Issue, Base: p.Base})
	}
	return out
}

func dedupPicks(in []Pick) []Pick {
	seen := map[int]bool{}
	var out []Pick
	for _, p := range in {
		if p.Issue <= 0 || seen[p.Issue] {
			continue
		}
		seen[p.Issue] = true
		out = append(out, p)
	}
	return out
}

// orMissing normalizes a caller-supplied metric: a measured one passes through, an
// explicitly missing one keeps its reason, and an unset one gets the field default.
func orMissing[T Number](m Metric[T], def string) Metric[T] {
	switch {
	case m.IsMeasured():
		return m
	case m.State == "" && m.Reason == "":
		return Missing[T](def)
	}
	return Missing[T](m.WhyMissing())
}

func orDefault(s, def string) string {
	if strings.TrimSpace(s) == "" {
		return def
	}
	return s
}

func unixNano(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixNano()
}
