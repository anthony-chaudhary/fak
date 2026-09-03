// corpus.go — the READ-BACK half of #4785: fold a whole rollout STORE
// (~/.codex/sessions/**/*.jsonl) through the exactly-once reconciler and report the
// before/after integrity counts by provider/version.
//
// WHY IT IS THE WITNESS. #4785's evidence is a corpus claim, so the fix needs a
// corpus proof: "zero UNCLASSIFIED mid-session starts after reconciliation". The
// audit's 144 unmatched starts decompose exactly as this fold types them —
//
//	superseded              = a later task started while this one was open   (audit: 37 mid-session)
//	process_death + live    = the rollout's final start, never terminated    (audit: 107 open-at-end)
//
// — so UnmatchedBefore is the population the naive boolean fold leaves dangling, and
// UnclassifiedAfter is what survives reconciliation. The contract is that the second
// number is ZERO while the first stays whatever the corpus really contains: this
// package classifies the gaps, it does not make them disappear.
//
// FRESHNESS IS READ FROM THE FILE, NOT GUESSED. A rollout whose mtime is inside
// freshWithin may still be running, so its open final start is Live; an older one is
// ProcessDeath. That is the archive/read-back distinction #4785 requires, and it is
// the only place this package touches a clock — Fold itself stays pure.
package codexlifecycle

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/walkfiles"
)

// Meta identifies one rollout, read from its FIRST session_meta record only. A
// subagent rollout starts with its own metadata and then carries the PARENT session
// metadata in the inherited context, so letting a later record win would make every
// child look like the same parent session (the same trap cmd/fak's
// applyCodexLoopSessionMeta guards).
type Meta struct {
	RolloutID    string `json:"rollout_id,omitempty"`
	Provider     string `json:"provider,omitempty"`    // model_provider, e.g. "fak" / "openai"
	CLIVersion   string `json:"cli_version,omitempty"` // e.g. "0.144.4"
	CWD          string `json:"cwd,omitempty"`
	Originator   string `json:"originator,omitempty"`
	Source       string `json:"source,omitempty"`
	ThreadSource string `json:"thread_source,omitempty"`
}

// ProviderVersion is the table axis #4785 reports against ("fak 0.144.1").
func (m Meta) ProviderVersion() string {
	p, v := strings.TrimSpace(m.Provider), strings.TrimSpace(m.CLIVersion)
	if p == "" {
		p = "unknown"
	}
	if v == "" {
		v = "unknown"
	}
	return p + " " + v
}

// Counts is one provider/version row: the before/after integrity numbers.
type Counts struct {
	Rollouts int `json:"rollouts"`
	Starts   int `json:"starts"`

	// UnmatchedBefore is starts with NO observed terminal — what a naive fold leaves
	// dangling (audit's "unmatched starts" column).
	UnmatchedBefore int `json:"unmatched_before"`
	// RolloutsWithGap is rollouts carrying at least one mid-rollout gap.
	RolloutsWithGap int `json:"rollouts_with_gap"`

	// After reconciliation, each formerly-unmatched start carries one typed terminal:
	Superseded   int `json:"superseded"`    // mid-session gap (audit: 37)
	ProcessDeath int `json:"process_death"` // final start, stale rollout
	Live         int `json:"live"`          // final start, fresh rollout
	Complete     int `json:"complete"`
	Aborted      int `json:"aborted"`

	CompletedWithTrailingAbort int `json:"completed_with_trailing_abort,omitempty"`
	SubstantiveCompleted       int `json:"substantive_completed,omitempty"`

	// UnclassifiedAfter MUST be zero — the acceptance criterion.
	UnclassifiedAfter int `json:"unclassified_after"`

	Orphans            int `json:"orphans,omitempty"`
	Reused             int `json:"reused,omitempty"`
	MultiplyTerminated int `json:"multiply_terminated,omitempty"`
}

// CorpusReport is the whole-store report.
type CorpusReport struct {
	Root       string             `json:"root"`
	Scanned    int                `json:"scanned"`
	Unreadable int                `json:"unreadable,omitempty"`
	Totals     Counts             `json:"totals"`
	ByProvider map[string]*Counts `json:"by_provider_version"`
	// AllStartsTyped is the acceptance criterion: every start carries a typed
	// terminal after reconciliation (i.e. UnclassifiedAfter == 0).
	AllStartsTyped bool `json:"all_starts_typed"`
}

// ProviderVersions returns the table's row keys in a stable order.
func (w CorpusReport) ProviderVersions() []string {
	keys := make([]string, 0, len(w.ByProvider))
	for k := range w.ByProvider {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (c *Counts) fold(rep Report) {
	c.Rollouts++
	c.Starts += len(rep.Tasks)
	c.Orphans += len(rep.Orphans)
	c.Reused += len(rep.Reused)
	c.MultiplyTerminated += len(rep.MultiplyTerminated)
	c.UnclassifiedAfter += len(rep.Unclassified())
	gap := false
	for _, t := range rep.Tasks {
		switch t.Outcome {
		case Complete:
			c.Complete++
		case Aborted:
			c.Aborted++
		case Superseded:
			c.Superseded++
			gap = true
		case ProcessDeath:
			c.ProcessDeath++
		case Live:
			c.Live++
		}
		// A start with no OBSERVED terminal is exactly what the audit counted as
		// unmatched; provenance is the discriminator, not the outcome name.
		if t.Provenance == Synthesized {
			c.UnmatchedBefore++
		}
	}
	if gap {
		c.RolloutsWithGap++
	}
	if rep.CompletedWithTrailingAbort {
		c.CompletedWithTrailingAbort++
	}
	if rep.SubstantiveCompleted {
		c.SubstantiveCompleted++
	}
}

// ScanOptions bounds a corpus scan.
type ScanOptions struct {
	// CWD, when set, keeps only rollouts whose session_meta.cwd matches it — the
	// repository scoping #4785's evidence used.
	CWD string
	// FreshWithin decides Live vs ProcessDeath for an open final start, measured
	// against Now and the file's mtime.
	FreshWithin time.Duration
	// Now is injected so a scan is reproducible; zero means time.Now().
	Now time.Time
	// Limit caps files scanned (0 = all), newest first when set.
	Limit int
}

// ScanCorpus folds every rollout under root through the reconciler. A rollout that
// cannot be read or parsed is counted (Unreadable) rather than failing the scan: the
// store is append-only and a torn tail is normal, and refusing to report the other
// 2,900 sessions because one file is mangled would defeat the witness.
func ScanCorpus(root string, opt ScanOptions) (CorpusReport, error) {
	now := opt.Now
	if now.IsZero() {
		now = time.Now()
	}
	w := CorpusReport{Root: root, ByProvider: map[string]*Counts{}}

	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return w, err
	}
	for _, p := range paths {
		info, statErr := os.Stat(p)
		if statErr != nil {
			w.Unreadable++
			continue
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			w.Unreadable++
			continue
		}
		meta, events, parseErr := ReadRollout(fh)
		_ = fh.Close()
		if parseErr != nil {
			w.Unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		fresh := opt.FreshWithin > 0 && now.Sub(info.ModTime()) <= opt.FreshWithin
		rep := Fold(events, fresh)
		if len(rep.Tasks) == 0 && len(rep.Orphans) == 0 {
			continue // not a task-bearing rollout
		}
		w.Scanned++
		w.Totals.fold(rep)
		key := meta.ProviderVersion()
		row := w.ByProvider[key]
		if row == nil {
			row = &Counts{}
			w.ByProvider[key] = row
		}
		row.fold(rep)
	}
	w.AllStartsTyped = w.Totals.UnclassifiedAfter == 0
	return w, nil
}

func sameDir(a, b string) bool {
	return strings.EqualFold(filepath.Clean(strings.TrimSpace(a)), filepath.Clean(strings.TrimSpace(b)))
}

func rolloutPaths(root string, limit int) ([]string, error) {
	type cand struct {
		path  string
		mtime time.Time
	}
	var out []cand
	err := walkfiles.Files(root, func(p string, d os.DirEntry) error {
		if !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		out = append(out, cand{p, info.ModTime()})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].mtime.After(out[j].mtime) })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	paths := make([]string, 0, len(out))
	for _, c := range out {
		paths = append(paths, c.path)
	}
	return paths, nil
}
