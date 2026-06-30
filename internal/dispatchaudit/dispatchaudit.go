// Package dispatchaudit classifies dispatch-fleet worker outcomes and rolls up
// the wasted-spawn / wasted-wall-clock that `fak dispatch status` (backend health)
// does not surface.
//
// The CORE here (Classify, Fold) is a PURE fold: it takes the already-parsed worker
// records IN and emits the classification + per-backend rollup + fingerprinted
// findings OUT. It touches no disk, no clock, no GitHub. The I/O that reads
// .dispatch-runs/ logs, the .backend sidecars, the progress ledger, and the
// .fak/loops.jsonl journal lives in the thin shell (shell.go), so the
// classification logic stays trivially unit-testable over fixture inputs.
//
// The taxonomy (issue #1454):
//
//	SHIPPED       — the worker's log carries a commit witness (a created/shipped SHA).
//	WASTED_SPAWN  — the worker hit a provider WEEKLY/MONTHLY cap (a hold for that
//	                backend was, or should have been, live) and produced no ship.
//	QUOTA_WALLED  — an explicit upstream cap/limit banner, distinct from a transient
//	                retry storm; the account is hard-walled.
//	RETRY_STORM   — repeated provider-error lines spanning real wall-clock with no ship.
//	NO_OP         — the log is only a startup banner / a progress tick that moved nothing.
//	ERRORED       — exited with an error that is not a recognized cap or storm.
//
// Each finding carries a STABLE fingerprint (outcome + backend + code-site) so a
// thin filer files only fingerprints it has not already filed.
package dispatchaudit

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
	"time"
)

// Outcome is the closed worker-outcome vocabulary.
type Outcome string

const (
	OutcomeShipped     Outcome = "SHIPPED"
	OutcomeWastedSpawn Outcome = "WASTED_SPAWN"
	OutcomeQuotaWalled Outcome = "QUOTA_WALLED"
	OutcomeRetryStorm  Outcome = "RETRY_STORM"
	OutcomeNoOp        Outcome = "NO_OP"
	OutcomeErrored     Outcome = "ERRORED"
)

// Backend is the dispatch backend a worker ran on. An empty/unknown sidecar
// resolves to BackendUnknown so a missing .backend never silently merges into
// another backend's rollup.
type Backend string

const (
	BackendClaude   Backend = "claude"
	BackendOpencode Backend = "opencode"
	BackendCodex    Backend = "codex"
	BackendUnknown  Backend = "unknown"
)

// NormalizeBackend maps a free-text backend token (from a .backend sidecar or a
// `# fak-spawn backend=X` header) to a known Backend, or BackendUnknown.
func NormalizeBackend(s string) Backend {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "claude":
		return BackendClaude
	case "opencode":
		return BackendOpencode
	case "codex":
		return BackendCodex
	default:
		return BackendUnknown
	}
}

// Worker is one already-parsed dispatch-worker record — the PURE input to the
// fold. The shell builds these from the on-disk logs/sidecars; the core never
// reads a file.
type Worker struct {
	// Log is the resolve-*.log base name (the stable code-site anchor for the
	// fingerprint).
	Log string
	// Issue is the issue number the worker targeted ("" if unknown).
	Issue string
	// Lane is the dispatch lane ("" if unknown).
	Lane string
	// HeaderBackend is the backend named in the `# fak-spawn backend=X` log header.
	HeaderBackend Backend
	// SidecarBackend is the backend resolved from the .backend sidecar.
	// SidecarMissing is true when no sidecar was present.
	SidecarBackend Backend
	SidecarMissing bool
	// CommitSHA is a witnessed commit SHA found in the log ("" if none).
	CommitSHA string
	// CapHit is true when the log carries an explicit provider weekly/monthly
	// cap / limit-exhausted banner.
	CapHit bool
	// ErrorLines is the count of provider ERROR/stream-error lines.
	ErrorLines int
	// FirstError / LastError bound the error span (zero values when none).
	FirstError time.Time
	LastError  time.Time
	// BannerOnly is true when the log is only a startup banner (no real work).
	BannerOnly bool
	// ProgressTicks / ProgressMoved describe the worker's progress ledger
	// contribution: ticks emitted and whether ANY tick moved the resolved count.
	ProgressTicks int
	ProgressMoved bool
}

// errSpan returns the wall-clock the error storm spanned, in minutes.
func (w Worker) errSpan() float64 {
	if w.FirstError.IsZero() || w.LastError.IsZero() || !w.LastError.After(w.FirstError) {
		return 0
	}
	return w.LastError.Sub(w.FirstError).Minutes()
}

// Backend resolves the worker's effective backend, preferring the sidecar and
// falling back to the header. BackendUnknown when neither is known.
func (w Worker) Backend() Backend {
	if !w.SidecarMissing && w.SidecarBackend != BackendUnknown && w.SidecarBackend != "" {
		return w.SidecarBackend
	}
	if w.HeaderBackend != BackendUnknown && w.HeaderBackend != "" {
		return w.HeaderBackend
	}
	return BackendUnknown
}

// Thresholds parameterize the RETRY_STORM heuristic. Defaults match the issue's
// worked example (resolve-1346: 15 errors / 17 min).
type Thresholds struct {
	StormMinErrors int     // ≥ this many provider-error lines …
	StormMinMins   float64 // … spanning ≥ this many minutes → RETRY_STORM (absent a cap).
}

// DefaultThresholds is the calibrated default.
func DefaultThresholds() Thresholds {
	return Thresholds{StormMinErrors: 5, StormMinMins: 5}
}

// Classification is the verdict for one worker.
type Classification struct {
	Worker        Worker  `json:"-"`
	Log           string  `json:"log"`
	Issue         string  `json:"issue,omitempty"`
	Lane          string  `json:"lane,omitempty"`
	Backend       Backend `json:"backend"`
	Outcome       Outcome `json:"outcome"`
	WallMinutes   float64 `json:"wall_minutes"`
	Misattributed bool    `json:"misattributed"`
	Reason        string  `json:"reason"`
}

// Classify is the PURE per-worker decision. Order matters: a witnessed ship wins
// over everything; an explicit cap banner is the strongest waste signal; a true
// retry storm beats a generic error; banner-only / no-progress collapses to NO_OP.
func Classify(w Worker, th Thresholds) Classification {
	c := Classification{
		Worker:      w,
		Log:         w.Log,
		Issue:       w.Issue,
		Lane:        w.Lane,
		Backend:     w.Backend(),
		WallMinutes: w.errSpan(),
	}
	// Misattribution flag: a present sidecar that disagrees with the header, or a
	// missing sidecar when a header backend is declared. Advisory — it annotates
	// the verdict, it does not replace it.
	if w.SidecarMissing && w.HeaderBackend != BackendUnknown && w.HeaderBackend != "" {
		c.Misattributed = true
	} else if !w.SidecarMissing && w.HeaderBackend != BackendUnknown && w.HeaderBackend != "" &&
		w.SidecarBackend != BackendUnknown && w.SidecarBackend != "" &&
		w.SidecarBackend != w.HeaderBackend {
		c.Misattributed = true
	}

	switch {
	case w.CommitSHA != "":
		c.Outcome = OutcomeShipped
		c.Reason = "commit witness " + w.CommitSHA
	case w.CapHit && w.CommitSHA == "":
		// A provider weekly/monthly wall with no ship: the spawn produced nothing
		// but burned the wall-clock between first and last cap line. A cap with a
		// long error span reads as a true hard-wall (QUOTA_WALLED); a brief one is
		// a wasted spawn that should have been gated behind the live hold.
		if w.errSpan() >= th.StormMinMins && w.ErrorLines >= th.StormMinErrors {
			c.Outcome = OutcomeQuotaWalled
			c.Reason = "provider cap banner; hard-walled account"
		} else {
			c.Outcome = OutcomeWastedSpawn
			c.Reason = "spawned into a capped backend; no ship"
		}
	case w.ErrorLines >= th.StormMinErrors && w.errSpan() >= th.StormMinMins:
		c.Outcome = OutcomeRetryStorm
		c.Reason = "provider-error storm with no ship"
	case w.BannerOnly || (w.ProgressTicks > 0 && !w.ProgressMoved):
		c.Outcome = OutcomeNoOp
		c.Reason = "banner-only or no-progress tick"
	case w.ErrorLines > 0:
		c.Outcome = OutcomeErrored
		c.Reason = "exited with provider errors"
	default:
		// No ship, no cap, no storm, no errors, but it ran — a quiet wasted spawn.
		c.Outcome = OutcomeWastedSpawn
		c.Reason = "no ship and no witnessed work"
	}
	return c
}

// Finding is a fingerprinted, fileable record of one worker outcome worth
// surfacing. SHIPPED workers do not produce findings (nothing to file).
type Finding struct {
	Fingerprint string  `json:"fingerprint"`
	Outcome     Outcome `json:"outcome"`
	Backend     Backend `json:"backend"`
	CodeSite    string  `json:"code_site"`
	Log         string  `json:"log"`
	Issue       string  `json:"issue,omitempty"`
	Title       string  `json:"title"`
	Detail      string  `json:"detail"`
}

// codeSite is the stable, log-name-independent anchor for a fingerprint: the
// (outcome, backend, lane) tuple. Two walled opencode workers on the same lane
// collapse to one fingerprint so the filer does not spam an issue per log.
func codeSite(c Classification) string {
	lane := c.Lane
	if lane == "" {
		lane = "unknown-lane"
	}
	return string(c.Outcome) + "/" + string(c.Backend) + "/" + lane
}

// Fingerprint is the stable hash over the code-site. Same outcome+backend+lane →
// same fingerprint across runs and across log timestamps.
func Fingerprint(c Classification) string {
	sum := sha256.Sum256([]byte(codeSite(c)))
	return hex.EncodeToString(sum[:])[:16]
}

// findingFor builds the fileable Finding for a non-shipped, non-trivial
// classification. It returns ok=false for outcomes not worth filing (SHIPPED).
func findingFor(c Classification) (Finding, bool) {
	if c.Outcome == OutcomeShipped {
		return Finding{}, false
	}
	site := codeSite(c)
	f := Finding{
		Fingerprint: Fingerprint(c),
		Outcome:     c.Outcome,
		Backend:     c.Backend,
		CodeSite:    site,
		Log:         c.Log,
		Issue:       c.Issue,
		Title:       "dispatch audit: " + string(c.Outcome) + " on " + string(c.Backend) + " (lane " + laneOrUnknown(c.Lane) + ")",
		Detail:      c.Reason + " — first seen in " + c.Log,
	}
	if c.Misattributed {
		f.Detail += " [.backend misattributed/missing]"
	}
	return f, true
}

func laneOrUnknown(lane string) string {
	if lane == "" {
		return "unknown"
	}
	return lane
}

// BackendRollup is the per-backend value KPI the status verb lacks.
type BackendRollup struct {
	Backend       Backend `json:"backend"`
	Workers       int     `json:"workers"`
	Shipped       int     `json:"shipped"`
	WastedSpawns  int     `json:"wasted_spawns"`
	QuotaWalled   int     `json:"quota_walled"`
	RetryStorms   int     `json:"retry_storms"`
	NoOps         int     `json:"no_ops"`
	Errored       int     `json:"errored"`
	Misattributed int     `json:"misattributed"`
	WastedMinutes float64 `json:"wasted_minutes"`
}

// Report is the full PURE fold output: per-worker classifications, per-backend
// rollups, and deduped findings (one per distinct fingerprint, first occurrence
// wins).
type Report struct {
	Classifications []Classification `json:"classifications"`
	Rollups         []BackendRollup  `json:"rollups"`
	Findings        []Finding        `json:"findings"`
}

// Fold is the PURE entry point: workers in, the full Report out. Deterministic —
// same workers + thresholds in, same Report out (rollups and findings are sorted
// by backend / fingerprint).
func Fold(workers []Worker, th Thresholds) Report {
	if th.StormMinErrors <= 0 && th.StormMinMins <= 0 {
		th = DefaultThresholds()
	}
	rep := Report{}
	roll := map[Backend]*BackendRollup{}
	seen := map[string]bool{}

	for _, w := range workers {
		c := Classify(w, th)
		rep.Classifications = append(rep.Classifications, c)

		r := roll[c.Backend]
		if r == nil {
			r = &BackendRollup{Backend: c.Backend}
			roll[c.Backend] = r
		}
		r.Workers++
		if c.Misattributed {
			r.Misattributed++
		}
		switch c.Outcome {
		case OutcomeShipped:
			r.Shipped++
		case OutcomeWastedSpawn:
			r.WastedSpawns++
			r.WastedMinutes += c.WallMinutes
		case OutcomeQuotaWalled:
			r.QuotaWalled++
			r.WastedMinutes += c.WallMinutes
		case OutcomeRetryStorm:
			r.RetryStorms++
			r.WastedMinutes += c.WallMinutes
		case OutcomeNoOp:
			r.NoOps++
			r.WastedMinutes += c.WallMinutes
		case OutcomeErrored:
			r.Errored++
			r.WastedMinutes += c.WallMinutes
		}

		if f, ok := findingFor(c); ok && !seen[f.Fingerprint] {
			seen[f.Fingerprint] = true
			rep.Findings = append(rep.Findings, f)
		}
	}

	for _, r := range roll {
		rep.Rollups = append(rep.Rollups, *r)
	}
	sort.Slice(rep.Rollups, func(i, j int) bool {
		return rep.Rollups[i].Backend < rep.Rollups[j].Backend
	})
	sort.Slice(rep.Findings, func(i, j int) bool {
		return rep.Findings[i].Fingerprint < rep.Findings[j].Fingerprint
	})
	return rep
}
