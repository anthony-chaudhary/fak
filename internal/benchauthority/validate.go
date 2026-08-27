package benchauthority

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
)

// Validate checks the registered claim ledger for the structural invariants a
// benchmark number must satisfy to be admissible, returning one error per violation
// (a zero-length slice means the ledger is clean). It is the package-level entry the
// freshness gate / authorityledger generator call before rendering, and the record
// the commit-audit perf-claim rung (#4101) is designed to resolve.
//
// It is deliberately PURE over the claim data — no disk, no process state — so the
// ledger's admissibility is unit-testable in isolation. The on-disk half (an
// Artifact path must exist) is a separate, capability-scoped call: ValidateArtifacts.
func Validate() []error { return ValidateClaims(registry) }

// ValidateClaims is Validate over an arbitrary slice — the testable core, with no
// dependency on the package-global registry. It aggregates every claim's problems
// and adds the cross-claim check that IDs are unique (duplicate anchors would break
// the ledger's deep-link targets).
func ValidateClaims(claims []Claim) []error {
	var errs []error
	seen := make(map[string]bool, len(claims))
	for i, c := range claims {
		for _, e := range c.Validate() {
			errs = append(errs, fmt.Errorf("claim %d %q: %w", i, c.ID, e))
		}
		id := strings.TrimSpace(c.ID)
		if id == "" {
			continue // the empty-ID problem is already reported by c.Validate()
		}
		if seen[id] {
			errs = append(errs, fmt.Errorf("claim %d: duplicate ID %q (card anchors must be unique)", i, id))
		}
		seen[id] = true
	}
	return errs
}

// Validate reports every admissibility problem with a single claim, so a caller (or
// the commit-audit rung) can resolve one ingested claim in isolation. The rules are
// STATUS-AWARE: a status that asserts a number (MEASURED/VERIFIED/THEORETICAL) must
// carry the full checkable {Metric, numeric Value, Unit, Baseline} record; a status
// that withholds a number (GATED/PENDING) must NOT carry one and must name the
// missing witness. This is the mechanism that turns a headline adjective into a
// resolvable record — "specificity is the deliverable" (#4101).
func (c Claim) Validate() []error {
	var errs []error
	if strings.TrimSpace(c.ID) == "" {
		errs = append(errs, errors.New("empty ID (the card anchor and cross-link target)"))
	}
	if strings.TrimSpace(c.Title) == "" {
		errs = append(errs, errors.New("empty Title"))
	}
	if !knownStatus(c.Status) {
		errs = append(errs, fmt.Errorf("unknown Status %q (want MEASURED/VERIFIED/THEORETICAL/GATED/PENDING/RETRACTED)", c.Status))
		return errs // status-keyed rules below are meaningless on an unknown status
	}

	switch {
	case assertsNumber(c.Status):
		// A real or projected number must be a resolvable record, not prose.
		errs = append(errs, c.measurementProblems()...)
		if strings.TrimSpace(c.Baseline) == "" {
			errs = append(errs, fmt.Errorf("%s claim needs a Baseline (a ratio without a baseline is meaningless)", c.Status))
		}
		if requiresArtifact(c.Status) && strings.TrimSpace(c.Artifact) == "" {
			errs = append(errs, fmt.Errorf("%s claim needs an Artifact (the committed evidence path)", c.Status))
		}
	case withholdsNumber(c.Status):
		// The honest inversion: a withheld-number claim must not smuggle a figure in,
		// and must say which witness it is waiting on.
		if strings.TrimSpace(c.Value) != "" {
			errs = append(errs, fmt.Errorf("%s claim must not assert a numeric Value %q (record the missing witness in Fences instead)", c.Status, c.Value))
		}
		if len(c.Fences) == 0 {
			errs = append(errs, fmt.Errorf("%s claim needs at least one Fence naming the missing witness", c.Status))
		}
	}

	if c.Status == Retracted && strings.TrimSpace(c.Replacement) == "" {
		errs = append(errs, errors.New("RETRACTED claim needs a Replacement (never a silent tombstone)"))
	}
	return errs
}

// measurementProblems enforces the checkable {Metric, Value, Unit} record: the Value
// must parse as a FINITE number, and Metric/Unit must be present so the figure is
// labelled and dimensioned. Storing Value as text is deliberate — it is exactly what
// lets this rung catch a non-number ("≈60x", "a lot faster", "") that a float field
// would either reject at compile time or silently coerce.
func (c Claim) measurementProblems() []error {
	var errs []error
	if strings.TrimSpace(c.Metric) == "" {
		errs = append(errs, errors.New("missing Metric (what was measured)"))
	}
	if strings.TrimSpace(c.Unit) == "" {
		errs = append(errs, errors.New("missing Unit (e.g. x, %, ms, tok/s)"))
	}
	v := strings.TrimSpace(c.Value)
	if v == "" {
		errs = append(errs, errors.New("missing numeric Value"))
		return errs
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		errs = append(errs, fmt.Errorf("Value %q is not numeric (a perf number must be a figure, not prose)", c.Value))
		return errs
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		errs = append(errs, fmt.Errorf("Value %q is not finite", c.Value))
	}
	return errs
}

// ValidateArtifacts is the on-disk half of Validate: every claim's non-empty Artifact
// path must resolve to an existing file under root. Benchmark artifacts also pass the
// shared benchcli evidence validator, then the authority-specific promotion boundary:
// non-hardware evidence can support a theoretical noncompetitive row, but cannot be
// promoted into a MEASURED/VERIFIED or competitive row.
//
// It is kept separate from Validate() because it needs a filesystem capability (a
// repo root) the pure ledger check does not — and it catches exactly the
// dropped-directory-prefix drift the old hand-typed doc admitted to.
func ValidateArtifacts(root string, claims []Claim) []error {
	var errs []error
	for i, c := range claims {
		p := strings.TrimSpace(c.Artifact)
		if p == "" {
			continue
		}
		full := filepath.Join(root, filepath.FromSlash(p))
		if _, err := os.Stat(full); err != nil {
			errs = append(errs, fmt.Errorf("claim %d %q: Artifact %q not found under %q: %v", i, c.ID, p, root, err))
			continue
		}
		raw, err := os.ReadFile(full)
		if err != nil {
			errs = append(errs, fmt.Errorf("claim %d %q: Artifact %q cannot be read: %v", i, c.ID, p, err))
			continue
		}
		art, ok := benchcli.DecodeArtifact(raw)
		if !ok {
			// Legacy evidence may be a log, markdown dossier, or pre-envelope JSON.
			// Preserve it. A document that claims to carry the shared envelope,
			// however, must decode; otherwise malformed simulation metadata could
			// evade the promotion check by making DecodeArtifact return false.
			if bytes.Contains(raw, []byte(`"benchmark_artifact"`)) {
				errs = append(errs, fmt.Errorf("claim %d %q: Artifact %q has a malformed benchmark_artifact envelope", i, c.ID, p))
			}
			continue
		}
		if err := benchcli.ValidateBenchmarkArtifact(art); err != nil {
			errs = append(errs, fmt.Errorf("claim %d %q: Artifact %q has invalid simulation evidence: %v", i, c.ID, p, err))
			continue
		}
		ev := art.SimulationEvidence
		if ev == nil || ev.EvidenceType == benchcli.EvidenceHardwareMeasurement {
			continue
		}
		if c.Status == Measured || c.Status == Verified {
			errs = append(errs, fmt.Errorf("claim %d %q: %s row cannot use non-hardware %q evidence", i, c.ID, c.Status, ev.EvidenceType))
		}
		if c.Competitive {
			errs = append(errs, fmt.Errorf("claim %d %q: competitive row cannot use non-hardware %q evidence", i, c.ID, ev.EvidenceType))
		}
	}
	return errs
}

// --- status classification -------------------------------------------------

func knownStatus(s Status) bool {
	switch s {
	case Measured, Verified, Theoretical, Gated, Pending, Retracted:
		return true
	}
	return false
}

// assertsNumber is true for the statuses that PUT A NUMBER on the table — a measured
// result, an independently reproduced one, or a projection — each of which must carry
// the checkable measurement record.
func assertsNumber(s Status) bool {
	return s == Measured || s == Verified || s == Theoretical
}

// requiresArtifact is true for the statuses whose number rests on committed evidence
// (a real measurement). A THEORETICAL projection's derivation may be inline, so it is
// not forced to name an on-disk artifact.
func requiresArtifact(s Status) bool {
	return s == Measured || s == Verified
}

// withholdsNumber is true for the statuses that DELIBERATELY carry no number yet;
// they must not assert a Value and must record the missing witness in a Fence.
func withholdsNumber(s Status) bool {
	return s == Gated || s == Pending
}
