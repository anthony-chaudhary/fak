package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// PRODUCING fak.modelroute.serving.v1 FROM A LIVE GATEWAY (issue #5636, epic #5632).
//
// internal/modelroute has a complete serving-liveness gate and, until this file,
// nothing in fak produced its input. Grepping the tree for the schema string returned
// only test fixtures and one doc sample, so the placement ladder's FLEET RUNG was
// gated by a JSON file a human remembered to refresh. That is worse than no signal at
// all: PlaceWithServing reports a zone verdict in a closed reason vocabulary, which
// reads as MEASURED when it was in fact asserted by whoever last edited the file.
//
// Meanwhile a real prober sat one package away. FleetMembership already runs a health
// loop with hysteresis and drain and exposes Snapshot(). The two never met. This file
// is the join, in the only direction that keeps the packages acyclic.
//
// THE DEPENDENCY EDGE IS DELIBERATELY ONE-WAY. internal/gateway (tier 4) already
// imports internal/modelroute (tier 1); modelroute imports no gateway. Producing a
// report is an EMISSION — we build modelroute's value type and write it out. CONSUMING
// one is the other direction and is deliberately not done here: nothing in this file
// calls Place or PlaceWithServing, so the gateway never takes a placement decision to
// write a liveness file. See the issue's third confusion risk.
//
// WHAT THIS PRODUCER DOES NOT KNOW, STATED OUT LOUD:
//
//   - IT DOES NOT ALWAYS KNOW THE ROUTED MODEL IDS. modelroute keys observations by the
//     routed model id the roster binds; FleetMembership keys workers by worker id.
//     WorkerSpec.Models (#5635) is that label WHEN THE DEPLOYMENT SETS IT, so the
//     default reads it and only falls back to the worker id for an unlabeled worker —
//     which gates NOTHING unless the two happen to coincide, exactly as cmd/fak's
//     "unbound observation" diagnostic already reports. That is the honest failure (a
//     report that gates nothing) rather than the dishonest one (a report that gates the
//     wrong candidate). ModelsFor stays the seam for a deployment whose routed ids are
//     not the registered ones.
//   - IT DOES NOT KNOW PER-WORKER PROBE TIMES. FleetMembership records health, not when
//     each health was last established, so every observation is stamped with the
//     SNAPSHOT time. The stamp is therefore accurate to within one probe interval, and
//     MaxAge must be sized against that: a bound tighter than the probe interval marks
//     a live fleet stale.
//   - IT DOES NOT KNOW THE SHAPE OF THE FLEET. See Covers below.
//
// LIVENESS ONLY. Inflight depth is on hand here and is deliberately not read: this is
// a liveness signal, not a performance one, and mapping queue depth to DEGRADED would
// make the placer shed load to a paid vendor at exactly the busy hour when the bulk of
// the tokens are. Whether saturation should cost money is an operator policy call.

// ServingReportOptions is what the caller must supply that the membership cannot know.
type ServingReportOptions struct {
	// Now stamps the report and every observation in it. Required: a snapshot with no
	// as-of stamp can never be shown stale, so it would be honored forever.
	Now time.Time

	// MaxAge is the freshness bound the report declares about ITSELF — the TTL past
	// which a reader degrades it to unknown. Required whenever there is anything to go
	// stale, and it must be at least the health-loop interval; see the header.
	MaxAge time.Duration

	// Covers names the rungs where SILENCE about a candidate is meaningful, and it is
	// empty by default ON PURPOSE. Declaring ZoneFleet asserts that this membership is
	// the WHOLE fleet rung — an assertion a single gateway process cannot make for
	// itself, because another host's workers are invisible to it. Claimed wrongly, it
	// passes over every candidate this instance merely does not happen to serve. It is
	// an operator's declaration, so it is an operator's input.
	Covers []modelroute.PlacementZone

	// ModelsFor maps one worker to the routed model ids it serves. Nil uses
	// defaultModelsFor: the worker's declared Models, or its own id when it declares
	// none. Supply one only when a deployment's routed ids are neither.
	ModelsFor func(WorkerStatus) []string
}

// defaultModelsFor is the keying used when the caller supplies no ModelsFor.
//
// A LABELED worker reports under the routed ids it was registered to serve — the same
// WorkerSpec.Models the placement primitive filters on (#5635), so the report and the
// router agree on what a model id means instead of each holding its own opinion.
//
// An UNLABELED worker reports under its own id. Empty Models means UNCONSTRAINED in the
// registry, and the temptation is to mirror that here by reporting the worker as
// evidence about EVERY model. That would be the dishonest direction: one unlabeled host
// going down would mark the whole roster down, because the worst state wins. Keying by
// worker id instead produces an entry the roster probably does not bind, which gates
// nothing and is visible in cmd/fak's "unbound observation" diagnostic — a report that
// is merely useless rather than one that is wrong.
func defaultModelsFor(st WorkerStatus) []string {
	if len(st.Spec.Models) > 0 {
		return st.Spec.Models
	}
	return []string{st.Spec.ID}
}

// servingStateFor maps one worker's membership row to what a probe concluded.
//
// DRAINING IS DOWN, NOT DEGRADED, and that is the mapping most likely to be written
// backwards. modelroute defines DEGRADED as "answering under strain, still takes
// work" — a loaded host keeps its tokens. Drain means the opposite: the worker is
// being removed from service and must receive no NEW work, which is precisely what
// admissible() already enforces inside the registry. Calling it degraded would keep
// routing at a host somebody is trying to empty.
//
// UNKNOWN HEALTH STAYS UNKNOWN. A worker registered but never probed has been
// concluded about by nobody, and modelroute's ServingUnknown is the state that means
// exactly that. Reporting it up would manufacture the stale positive this whole issue
// exists to prevent; reporting it down would announce an outage nobody observed.
func servingStateFor(st WorkerStatus) modelroute.ServingState {
	if st.Draining {
		return modelroute.ServingDown
	}
	switch st.Health {
	case HealthHealthy:
		return modelroute.ServingUp
	case HealthUnhealthy:
		return modelroute.ServingDown
	default:
		return modelroute.ServingUnknown
	}
}

// ServingReportFromSnapshot folds a FleetMembership snapshot into the liveness report
// the placement ladder reads.
//
// It REFUSES to emit a report that cannot go stale. A document with no as-of stamp, or
// none declaring a TTL, is honored at any age by every reader — so a producer that
// dies on a Sunday leaves behind a permanent "the fleet is up", which is the failure
// mode the ladder is least able to explain. Refusing here costs an operator one
// configuration error; the alternative costs them an afternoon of dispatch failures
// against a host that has been dead since morning.
//
// An EMPTY snapshot is the one exemption: with no observations there is nothing to go
// stale, and the zero report is a structural no-op in modelroute by construction.
func ServingReportFromSnapshot(statuses []WorkerStatus, opts ServingReportOptions) (modelroute.ServingReport, error) {
	var rep modelroute.ServingReport
	if len(statuses) == 0 {
		// The honest zero report: it gates nothing anywhere, so it needs no stamp and
		// carries no schema (modelroute.Validate exempts the empty report for the same
		// reason — there is no meaning to get wrong).
		return rep, nil
	}
	if opts.Now.IsZero() {
		return rep, errors.New("gateway: a serving report needs an as-of stamp; without one nothing can measure its age and it is honored forever")
	}
	if opts.MaxAge <= 0 {
		return rep, errors.New("gateway: a serving report needs a positive MaxAge; a report declaring no freshness bound is honored at any age, so a dead producer would pin the fleet rung open")
	}
	now := opts.Now.Unix()
	modelsFor := opts.ModelsFor
	if modelsFor == nil {
		modelsFor = defaultModelsFor
	}
	rep = modelroute.ServingReport{
		Schema:        modelroute.ServingReportSchema,
		AsOfUnix:      now,
		MaxAgeSeconds: int64(opts.MaxAge / time.Second),
		Covers:        append([]modelroute.PlacementZone(nil), opts.Covers...),
		Models:        make(map[string]modelroute.ServingObservation, len(statuses)),
	}
	// A sub-second bound would truncate to zero, which modelroute reads as "no bound
	// declared" — silently switching the freshness gate off. Refuse instead.
	if rep.MaxAgeSeconds <= 0 {
		return modelroute.ServingReport{}, fmt.Errorf("gateway: MaxAge %s truncates to zero seconds, which reads as NO freshness bound at all; declare at least one second", opts.MaxAge)
	}
	for _, st := range statuses {
		state := servingStateFor(st)
		for _, model := range modelsFor(st) {
			model = strings.TrimSpace(model)
			if model == "" {
				continue
			}
			// Two workers can serve one routed model. The WORST state wins: one healthy
			// replica does not make a model healthy if another is known down, and the
			// ladder is meant to fail over WITHIN the rung on exactly that signal.
			if prior, seen := rep.Models[model]; seen && servingWorseOrEqual(prior.State, state) {
				continue
			}
			rep.Models[model] = modelroute.ServingObservation{State: state, ObservedUnix: now}
		}
	}
	if err := rep.Validate(); err != nil {
		return modelroute.ServingReport{}, fmt.Errorf("gateway: produced serving report is invalid: %w", err)
	}
	return rep, nil
}

// servingSeverity ranks the states so a model served by several workers takes the
// worst one. Down outranks unknown outranks degraded outranks up: a model with one
// known-dead replica must not read as healthy because a sibling answered.
func servingSeverity(s modelroute.ServingState) int {
	switch s {
	case modelroute.ServingDown:
		return 3
	case modelroute.ServingUnknown:
		return 2
	case modelroute.ServingDegraded:
		return 1
	default:
		return 0
	}
}

func servingWorseOrEqual(a, b modelroute.ServingState) bool {
	return servingSeverity(a) >= servingSeverity(b)
}

// WriteServingReport publishes the report at path, temp-then-rename so a reader never
// observes a half-written document.
//
// Atomicity is not belt-and-braces here. The consumer decodes with
// DisallowUnknownFields and refuses a trailing document, so a torn file does not
// degrade gracefully — it becomes a hard refusal on a placement path that had been
// working. The temp name is UNIQUE PER WRITE rather than a fixed ".tmp" (two emitters
// wearing one identity is a real state, and a shared temp path lets them interleave
// into the same file and rename the torn result into place), and it does not end in
// ".json" so a strand left by a crash is not mistaken for a report. This mirrors
// fleetbus.Announce, which solved the same problem for presence records.
func WriteServingReport(path string, rep modelroute.ServingReport) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("gateway: no serving report path; set FLEET_STATE_DIR or pass one explicitly")
	}
	if err := rep.Validate(); err != nil {
		return fmt.Errorf("gateway: refusing to publish an invalid serving report: %w", err)
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".*.publishing")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		_ = os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
		return err
	}
	return nil
}

// DefaultServingReportPath is the well-known location, DERIVED from the fleet state
// dir the deployment already declares rather than introduced as a new flag. It mirrors
// resolveResumeLedgerPath's chain so one deployment convention places both files.
//
// It returns "" when nothing declares a state dir, and the caller fails closed rather
// than inventing a path. A producer writing to a guessed location and a consumer
// reading from a different guessed location would together look exactly like a fleet
// that is never observed.
func DefaultServingReportPath() string {
	if v := strings.TrimSpace(os.Getenv("FLEET_STATE_DIR")); v != "" {
		return filepath.Join(v, "modelroute", "serving.json")
	}
	if v := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); v != "" {
		return filepath.Join(v, "modelroute", "serving.json")
	}
	if v := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); v != "" {
		cand := filepath.Join(v, "Fleet")
		if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
			return filepath.Join(cand, "modelroute", "serving.json")
		}
	}
	return ""
}
