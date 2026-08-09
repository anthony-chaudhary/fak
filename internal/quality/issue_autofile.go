package quality

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// issue_autofile.go is the regression AUTO-FILER (#4584, under epic #4509): the
// layer that turns a stream of quality runs into a small, stable set of tracked
// issues. The nightly matrix (#4577) already deduplicates the cells of ONE run
// into one finding per defect signature; this file supplies the half a matrix
// cannot: the lifecycle ACROSS runs. Without it a nightly that files what it
// found re-files the same regression every night, and nothing ever closes when
// the defect is fixed — the backlog stops being read, which is the same failure
// mode as filing nothing.
//
// Four properties make an auto-filed backlog trustworthy rather than merely
// automatic:
//
//   - Findings are KEYED by case × model × backend × mode × metric × first-bad
//     (RegressionKey). The key is the identity, so the same regression seen on
//     ten nights is ONE issue updated ten times, not ten issues. The key is
//     rendered into a marker comment carried in the issue body, so a driver
//     deduplicates against the real tracker by exact string match rather than by
//     fuzzy title similarity.
//   - Recovery closes on a GREEN WINDOW, never on a single green run. One green
//     run is indistinguishable from a flaky pass; N consecutive green runs of the
//     same coordinates is a recovery. An issue whose coordinates were not
//     observed at all does not advance its window — silence is not a green run.
//   - Every filed regression carries a SCRUBBED, REPLAY-COMPLETE artifact inside
//     its own body. The issue reproduces from its own text (`fak quality replay
//     --bundle -`), so the evidence survives the log rotation that ate the run.
//   - Missing or inconclusive evidence is NEVER a pass and never silently
//     dropped: an observation whose provenance is incomplete, whose revision does
//     not match the run, whose artifact is absent or unscrubbed, or whose case
//     carries no tier/cost routing header, files an evidence-gap issue of its own
//     and holds every open issue on those coordinates open.
//
// It is additive in the cohort idiom: it registers no oracle and edits no core,
// consuming only the QualityCase envelope (case.go) and the Evidence / Divergence
// / FailureBundle the spine (run.go) and release aggregator (release_gate.go)
// already emit. It stays stdlib-only and hermetic: Plan is a pure function of
// (tracker state, run, observations), and the one impure step — actually writing
// to the tracker of record — is behind the Filer seam, exactly as Probe (bisect)
// and MatrixEngine (matrix) fence off their expensive halves.

// IssueFilingSchema is the versioned tag on a filing plan, report, and tracker.
// Consumers pin the major so a schema bump is a conscious migration (the #4519
// house rule), not a silent field drift.
const IssueFilingSchema = "fak-quality-filing/1"

// DefaultGreenWindow is how many CONSECUTIVE green runs of a finding's own
// coordinates close it. Two, not one: a single green run after a failure is
// equally consistent with a fix and with a flake, and closing on a flake loses
// the issue that was the only record of the defect.
const DefaultGreenWindow = 2

// Filing kinds. A regression is a localized failure with a replayable artifact;
// an evidence gap is an ABSENCE of trustworthy evidence. Both file, because the
// second is exactly the state a suite is most likely to mistake for a pass.
const (
	FilingRegression  = "regression"
	FilingEvidenceGap = "evidence-gap"
)

// Causes recorded on an evidence-gap finding. They are part of the gap's key, so
// "the rocm build was missing on four runs" stays one issue rather than four.
// causeNoArtifact / causeUnscrubbed are shared with the nightly matrix so both
// layers name the same defect the same way.
const (
	causeCaseHeaderIncomplete = "incomplete-case-header"
	causeProvenanceIncomplete = "incomplete-provenance"
	causeStaleEvidence        = "stale-evidence"
	causeRunUnattributed      = "run-revision-missing"
)

// RegressionKey is the identity an auto-filed finding is keyed by — the #4584
// scope vocabulary exactly: which case, on which model, which backend, which
// engine mode, which metric regressed, and where it first went bad. Two findings
// with equal keys are the same defect observed again; two findings with different
// keys are different defects and get different issues.
//
// FirstBad is the first ACTIONABLE divergence coordinate, not a summary: the
// token index the streams first disagreed at when there is one, else the serving
// stage the bundle's own evidence attributes the failure to, else the failing
// kind. Keying on it means a regression that moves to a different token is a
// different issue — which is correct, because it is a different defect to fix.
type RegressionKey struct {
	CaseID   string `json:"case_id"`
	Model    string `json:"model"`
	Backend  string `json:"backend"`
	Mode     string `json:"mode"`
	Metric   string `json:"metric"`
	FirstBad string `json:"first_bad"`
}

// coord is the case×model×backend×mode prefix of a key: the COORDINATES a run
// observes, independent of what (if anything) went wrong there. It is what a
// green run exonerates — a passing run names no metric and no first-bad, so
// recovery is necessarily matched on the coordinates rather than on the defect.
func (k RegressionKey) coord() RegressionKey {
	return RegressionKey{CaseID: k.CaseID, Model: k.Model, Backend: k.Backend, Mode: k.Mode}
}

// String renders the key as its slash-joined coordinates, empty axes shown as "-"
// so a missing axis is visible rather than swallowed.
func (k RegressionKey) String() string {
	parts := []string{k.CaseID, k.Model, k.Backend, k.Mode, k.Metric, k.FirstBad}
	for i, p := range parts {
		if strings.TrimSpace(p) == "" {
			parts[i] = "-"
		}
	}
	return strings.Join(parts, "/")
}

// Marker is the deduplication handle a driver matches on in the tracker of
// record: an HTML comment carrying the key, stable across runs and safe to embed
// in an issue body. It is built per-FIELD and is INJECTIVE — distinct keys always
// produce distinct markers — which is the whole contract: a lossy marker would
// silently fold two different defects into one issue, the exact mistake
// deduplication is supposed to prevent. Field values are percent-escaped, so the
// joining "/" is unambiguously a separator even for a case id that contains one
// (the nightly matrix mints exactly those), and no value can smuggle in a "<",
// ">", newline, or "--" that would close the comment early.
func (k RegressionKey) Marker() string {
	fields := []string{k.CaseID, k.Model, k.Backend, k.Mode, k.Metric, k.FirstBad}
	for i, f := range fields {
		fields[i] = markerToken(f)
	}
	return "<!-- fak-quality-regression key=" + strings.Join(fields, "/") + " -->"
}

const markerHex = "0123456789ABCDEF"

// markerToken escapes one key field. Bytes that are safe and readable survive
// verbatim; every other byte — including "%" itself, which keeps the escaping
// reversible and therefore injective — becomes %XX. A "-" survives except
// immediately after another "-", where it is escaped so an embedded "--" can
// never terminate the enclosing comment.
func markerToken(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	lastDash := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9',
			c == '.', c == '_', c == ':', c == '+', c == '=', c == '@':
			b.WriteByte(c)
			lastDash = false
		case c == '-' && !lastDash:
			b.WriteByte('-')
			lastDash = true
		default:
			b.WriteByte('%')
			b.WriteByte(markerHex[c>>4])
			b.WriteByte(markerHex[c&0x0f])
			lastDash = false
		}
	}
	return b.String()
}

// FilingRun identifies one run of a quality suite: the run's own id (what an
// operator cites) and the code/module revision it qualified. The revision is
// load-bearing, not decorative — evidence attributed to a different revision than
// the run is stale, and a run that names no revision can attribute nothing, so
// both are evidence gaps rather than passes.
type FilingRun struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

// Observation is ONE case's outcome in ONE run: the case (which carries the
// model, tokenizer, backend, mode, tier, and cost header) and the evidence it
// produced (state, provenance, first divergence, scrubbed replay artifact). It is
// the unit a harness submits; the auto-filer never runs anything itself.
type Observation struct {
	Case     QualityCase `json:"case"`
	Evidence Evidence    `json:"evidence"`
}

// FilingAction is the closed lifecycle vocabulary. Open and Update and Close are
// the three things a driver does to the tracker of record; Hold is the internal
// bookkeeping step of a green run that has not yet completed the window, surfaced
// so an operator can see recovery in progress rather than silence.
type FilingAction string

const (
	ActionOpen   FilingAction = "open"
	ActionUpdate FilingAction = "update"
	ActionHold   FilingAction = "hold"
	ActionClose  FilingAction = "close"
)

// external reports whether an action must reach the tracker of record. A Hold has
// no external effect, so Apply advances it unconditionally; the other three are
// adopted only once the Filer says they landed.
func (a FilingAction) external() bool { return a != ActionHold }

// Filing is one lifecycle action against ONE keyed issue: what to do, which issue
// (by marker), the rendered title and body, the tier and cost the finding is
// assigned to, and — on a regression — the scrubbed replay artifact the body
// embeds. Body is empty on a Hold, which files nothing.
type Filing struct {
	Action FilingAction   `json:"action"`
	Key    RegressionKey  `json:"key"`
	Marker string         `json:"marker"`
	Kind   string         `json:"kind"`
	Title  string         `json:"title"`
	Body   string         `json:"body,omitempty"`
	Tier   Tier           `json:"tier,omitempty"`
	Cost   CostSpec       `json:"cost"`
	Replay *FailureBundle `json:"replay,omitempty"`
	Reason string         `json:"reason"`

	// next is the tracker state this filing advances to, carried unexported so it
	// cannot be forged by a caller: Apply adopts it only when the filing actually
	// landed, so a filing the driver could not write leaves the lifecycle exactly
	// where it was and the next run re-proposes it.
	next TrackedIssue
}

// FilingPlan is the machine-readable output of folding one run: the run it was
// computed for and every lifecycle action it implies, in marker order so a plan
// is stable across runs. It is a pure function of (tracker state, run,
// observations) — same inputs, same plan — so a filing decision replays.
type FilingPlan struct {
	Schema  string    `json:"schema"`
	Run     FilingRun `json:"run"`
	Filings []Filing  `json:"filings,omitempty"`
}

// TrackedIssue is the tracker's durable per-key record: the issue's identity,
// whether it is open, how many runs have observed it failing, and how many
// consecutive green runs of its coordinates it has accumulated toward closure.
type TrackedIssue struct {
	Key          RegressionKey `json:"key"`
	Marker       string        `json:"marker"`
	Kind         string        `json:"kind"`
	Open         bool          `json:"open"`
	Occurrences  int           `json:"occurrences"`
	GreenRuns    int           `json:"green_runs"`
	FirstSeenRun string        `json:"first_seen_run"`
	LastFailRun  string        `json:"last_fail_run"`
	LastRun      string        `json:"last_run"`
}

// Tracker is the durable, JSON-serializable ledger the lifecycle is computed
// against — the memory that makes "repeated runs update ONE issue" possible. A
// driver persists it beside the suite and reloads it on the next run.
type Tracker struct {
	Schema      string                  `json:"schema"`
	GreenWindow int                     `json:"green_window"`
	Issues      map[string]TrackedIssue `json:"issues,omitempty"`
}

// NewTracker returns an empty tracker closing on the given green window. A
// non-positive window is clamped to DefaultGreenWindow: closing on zero green
// runs would close every issue the moment it stopped being observed, which is the
// exact "silence is a pass" failure this file exists to refuse.
func NewTracker(greenWindow int) *Tracker {
	if greenWindow <= 0 {
		greenWindow = DefaultGreenWindow
	}
	return &Tracker{Schema: IssueFilingSchema, GreenWindow: greenWindow, Issues: map[string]TrackedIssue{}}
}

func (t *Tracker) window() int {
	if t.GreenWindow <= 0 {
		return DefaultGreenWindow
	}
	return t.GreenWindow
}

// Plan folds one run's observations into the lifecycle actions they imply,
// WITHOUT mutating the tracker. Findings sharing a key are deduplicated into one
// filing (a retried case in the same run is the same finding, not two), a keyed
// finding opens or updates its one issue, and an open issue whose coordinates
// went green this run holds — or closes, once the green window is complete.
//
// An open issue whose coordinates were not observed at all is deliberately
// untouched: an unrun case is not a green case, so it neither advances toward
// closure nor counts as a new occurrence.
func (t *Tracker) Plan(run FilingRun, obs []Observation) FilingPlan {
	plan := FilingPlan{Schema: IssueFilingSchema, Run: run}

	type finding struct {
		key    RegressionKey
		kind   string
		cause  string
		detail string
		tier   Tier
		cost   CostSpec
		prov   EvidenceProvenance
		tol    ToleranceSpec
		base   BaselineSpec
		tokrev Revision
		bundle *FailureBundle
	}
	var (
		order    []string
		findings = map[string]finding{}
		green    = map[RegressionKey]bool{}
	)

	for _, o := range obs {
		f, ok := classifyObservation(run, o)
		if !ok {
			// A trustworthy pass: it exonerates its coordinates this run.
			green[keyOf(o.Case, "", "").coord()] = true
			continue
		}
		fd := finding{
			key: f.key, kind: f.kind, cause: f.cause, detail: f.detail,
			tier: Tier(o.Case.Metadata.Tier.Name), cost: o.Case.Metadata.Cost,
			prov: o.Evidence.Provenance, tol: o.Case.Metadata.Tolerance,
			base: o.Case.Metadata.Baseline, tokrev: o.Case.Metadata.Tokenizer,
			bundle: f.bundle,
		}
		m := fd.key.Marker()
		if _, dup := findings[m]; !dup {
			order = append(order, m)
			findings[m] = fd
		}
	}

	// A coordinate that failed this run is NOT green, whatever else it also
	// reported: a retry that passed after a failure does not exonerate the run.
	for _, m := range order {
		delete(green, findings[m].key.coord())
	}

	for _, m := range order {
		fd := findings[m]
		cur, tracked := t.Issues[m]
		action := ActionUpdate
		if !tracked || !cur.Open {
			action = ActionOpen
		}
		next := TrackedIssue{
			Key: fd.key, Marker: m, Kind: fd.kind, Open: true,
			Occurrences: cur.Occurrences + 1, GreenRuns: 0,
			FirstSeenRun: cur.FirstSeenRun, LastFailRun: run.ID, LastRun: run.ID,
		}
		if next.FirstSeenRun == "" {
			next.FirstSeenRun = run.ID
		}
		// A CLOSED issue whose defect returns re-opens carrying its history, so a
		// recurring regression reads as recurring rather than as brand new — that
		// history is what `Occurrences` and `FirstSeenRun` above preserve.
		reason := fmt.Sprintf("%s observed in run %s (occurrence %d)", fd.kind, run.ID, next.Occurrences)
		if action == ActionUpdate {
			reason = fmt.Sprintf("%s recurred in run %s (occurrence %d) — updating the existing issue", fd.kind, run.ID, next.Occurrences)
		}
		plan.Filings = append(plan.Filings, Filing{
			Action: action, Key: fd.key, Marker: m, Kind: fd.kind,
			Title: filingTitle(fd.kind, fd.key),
			Body:  filingBody(fd.kind, fd.cause, fd.detail, next, fd.tier, fd.cost, fd.prov, fd.tol, fd.base, fd.tokrev, fd.bundle),
			Tier:  fd.tier, Cost: fd.cost, Replay: fd.bundle,
			Reason: reason, next: next,
		})
	}

	// Recovery: every OPEN issue whose coordinates went green this run advances
	// toward closure, and closes once the window is complete.
	openMarkers := make([]string, 0, len(t.Issues))
	for m := range t.Issues {
		openMarkers = append(openMarkers, m)
	}
	sort.Strings(openMarkers)
	for _, m := range openMarkers {
		cur := t.Issues[m]
		if !cur.Open || !green[cur.Key.coord()] {
			continue
		}
		next := cur
		next.GreenRuns = cur.GreenRuns + 1
		next.LastRun = run.ID
		if next.GreenRuns >= t.window() {
			next.Open = false
			plan.Filings = append(plan.Filings, Filing{
				Action: ActionClose, Key: cur.Key, Marker: m, Kind: cur.Kind,
				Title: filingTitle(cur.Kind, cur.Key),
				Body: fmt.Sprintf("%s\n\nRecovered: %d consecutive green run(s) of these coordinates (window %d), most recently run `%s`. Closing.\n",
					m, next.GreenRuns, t.window(), run.ID),
				Reason: fmt.Sprintf("recovered after %d consecutive green run(s) (window %d)", next.GreenRuns, t.window()),
				next:   next,
			})
			continue
		}
		plan.Filings = append(plan.Filings, Filing{
			Action: ActionHold, Key: cur.Key, Marker: m, Kind: cur.Kind,
			Title:  filingTitle(cur.Kind, cur.Key),
			Reason: fmt.Sprintf("green in run %s (%d of %d toward the close window) — holding the issue open", run.ID, next.GreenRuns, t.window()),
			next:   next,
		})
	}

	sort.SliceStable(plan.Filings, func(i, j int) bool { return plan.Filings[i].Marker < plan.Filings[j].Marker })
	return plan
}

// Filer writes ONE filing to the issue tracker of record — the single impure step
// of the auto-filer, and the seam a `gh issue create/comment/close` driver plugs
// into. A returned error means the filing did not land.
type Filer func(Filing) error

// FilingFailure is one filing the driver could not land, with the error it
// returned. Its issue's lifecycle state is left untouched, so the next run
// proposes it again: a dropped file is retried, never mistaken for filed.
type FilingFailure struct {
	Filing Filing `json:"filing"`
	Error  string `json:"error"`
}

// FilingReport is what actually happened when a plan was applied.
type FilingReport struct {
	Schema string          `json:"schema"`
	Run    FilingRun       `json:"run"`
	Landed []Filing        `json:"landed,omitempty"`
	Failed []FilingFailure `json:"failed,omitempty"`
}

// Apply executes a plan through a Filer and advances the tracker for exactly the
// filings that took effect. A Hold has no external effect and always advances; an
// Open, Update, or Close advances only once the Filer reports it landed. A nil
// Filer lands nothing external — a dry run that still advances holds, so an
// operator can rehearse a plan without inventing filings that do not exist.
func (t *Tracker) Apply(plan FilingPlan, file Filer) FilingReport {
	rep := FilingReport{Schema: IssueFilingSchema, Run: plan.Run}
	if t.Issues == nil {
		t.Issues = map[string]TrackedIssue{}
	}
	if t.Schema == "" {
		t.Schema = IssueFilingSchema
	}
	for _, f := range plan.Filings {
		// A filing carries its lifecycle advance only when it came straight from
		// Plan: `next` is unexported, so a plan that round-tripped through JSON
		// (the shape FilingPlan is explicitly built for) arrives with an empty
		// one. Adopting that would blank the very issue the filing is about —
		// silently reopening a closed defect or losing a recurrence's history —
		// so it is refused and reported, exactly like a filing the driver could
		// not write. Re-plan against the tracker instead.
		if f.next.Marker != f.Marker {
			rep.Failed = append(rep.Failed, FilingFailure{Filing: f,
				Error: "filing carries no lifecycle advance: it did not come from Plan (a plan that round-tripped through JSON must be re-planned against the tracker)"})
			continue
		}
		if f.Action.external() {
			if file == nil {
				rep.Failed = append(rep.Failed, FilingFailure{Filing: f, Error: "no filer supplied: nothing was written to the tracker of record"})
				continue
			}
			if err := file(f); err != nil {
				rep.Failed = append(rep.Failed, FilingFailure{Filing: f, Error: err.Error()})
				continue
			}
		}
		t.Issues[f.Marker] = f.next
		rep.Landed = append(rep.Landed, f)
	}
	return rep
}

// OpenIssues returns the tracker's currently-open issues in marker order.
func (t *Tracker) OpenIssues() []TrackedIssue {
	out := make([]TrackedIssue, 0, len(t.Issues))
	for _, iss := range t.Issues {
		if iss.Open {
			out = append(out, iss)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Marker < out[j].Marker })
	return out
}

// classifiedObservation is one observation reduced to a fileable finding.
type classifiedObservation struct {
	key    RegressionKey
	kind   string
	cause  string
	detail string
	bundle *FailureBundle
}

// classifyObservation reduces one observation to a finding, or reports that it
// was a trustworthy pass. Every path that does not end in a localized failure
// with a scrubbed, replay-complete artifact ends as an evidence GAP with its
// cause named — never as a pass, and never as a filing whose evidence cannot be
// reproduced. The order is deliberate: routing header, then run attribution, then
// provenance, then state, then artifact — the earliest thing that makes the
// evidence uninterpretable is the thing reported.
func classifyObservation(run FilingRun, o Observation) (classifiedObservation, bool) {
	gap := func(cause, detail string) (classifiedObservation, bool) {
		return classifiedObservation{
			key:    keyOf(o.Case, "evidence", "gap:"+cause),
			kind:   FilingEvidenceGap,
			cause:  cause,
			detail: detail,
		}, true
	}
	if err := o.Case.ValidateCanonical(); err != nil {
		return gap(causeCaseHeaderIncomplete,
			"case carries no complete routing header, so this finding cannot be assigned a tier or costed: "+err.Error())
	}
	if strings.TrimSpace(run.Revision) == "" {
		return gap(causeRunUnattributed,
			"the run names no code/module revision, so its evidence cannot be attributed to what was under test")
	}
	if ok, why := o.Evidence.Provenance.complete(); !ok {
		return gap(causeProvenanceIncomplete, "incomplete provenance: "+why)
	}
	if o.Evidence.Provenance.Revision != run.Revision {
		return gap(causeStaleEvidence, fmt.Sprintf("evidence produced at revision %q != run revision %q",
			o.Evidence.Provenance.Revision, run.Revision))
	}
	switch o.Evidence.State {
	case StatePass:
		return classifiedObservation{}, false
	case StateFail:
		// fall through to the artifact checks below
	default:
		return gap("evidence-"+string(o.Evidence.State), "evidence state "+string(o.Evidence.State)+": "+o.Evidence.Detail)
	}
	fb := o.Evidence.Replay
	if fb == nil {
		return gap(causeNoArtifact, "case failed but emitted no replay artifact, so the failure cannot be reproduced")
	}
	if !fb.Scrubbed {
		return gap(causeUnscrubbed, "case emitted an artifact the spine did not redact; refusing to file an unscrubbed replay")
	}
	if err := fb.ReplayComplete(); err != nil {
		return gap(causeNoArtifact, "artifact is not replay-complete, so the filed issue could not reproduce it: "+err.Error())
	}
	return classifiedObservation{
		key:    keyOf(o.Case, metricOf(*fb), firstBadOf(*fb)),
		kind:   FilingRegression,
		detail: o.Evidence.Detail,
		bundle: fb,
	}, true
}

// keyOf builds a finding's key from the case's own coordinates plus the metric
// and first-bad the evidence localized. The backend and mode are read from the
// case's engine spec — the same place the nightly matrix writes them — so a
// matrix cell and a standalone case key identically.
func keyOf(c QualityCase, metric, firstBad string) RegressionKey {
	return RegressionKey{
		CaseID:   c.ID,
		Model:    c.Metadata.Model.Name,
		Backend:  c.Metadata.Engine.Backend,
		Mode:     c.Metadata.Engine.Flags["mode"],
		Metric:   metric,
		FirstBad: firstBad,
	}
}

// metricOf names the metric that regressed: the first failing oracle, falling
// back to its kind when a pre-oracle check (request fidelity) failed first.
func metricOf(fb FailureBundle) string {
	if s := strings.TrimSpace(fb.FailingOracle); s != "" {
		return s
	}
	if s := strings.TrimSpace(fb.FailingKind); s != "" {
		return s
	}
	return "unnamed"
}

// firstBadOf names the first actionable divergence: the token index the streams
// first disagreed at, else the serving stage the bundle's evidence attributes the
// failure to, else the failing kind. It never returns empty — a finding with no
// localization would collapse every defect on a case into one issue.
func firstBadOf(fb FailureBundle) string {
	if d := fb.FirstDivergence; d != nil {
		return fmt.Sprintf("token:%d", d.Index)
	}
	if c := fb.Classification; c != nil && !c.Abstained() {
		return "stage:" + c.Stage
	}
	if s := strings.TrimSpace(fb.FailingKind); s != "" {
		return "kind:" + s
	}
	return "kind:unlocalized"
}

func filingTitle(kind string, k RegressionKey) string {
	label := "quality regression"
	if kind == FilingEvidenceGap {
		label = "quality evidence gap"
	}
	coords := k.Model
	if k.Backend != "" {
		coords += "/" + k.Backend
	}
	if k.Mode != "" {
		coords += "/" + k.Mode
	}
	return fmt.Sprintf("%s: %s on %s — %s at %s", label, k.CaseID, coords, k.Metric, k.FirstBad)
}

// filingBody renders the issue body. It leads with the marker (the driver's
// deduplication handle), states the finding and its lifecycle, documents the tier
// and the runtime/resource cost the case was assigned, records the full
// provenance a reader needs to trust the evidence, and — for a regression —
// EMBEDS the scrubbed failure bundle so the issue reproduces from its own text.
// An evidence gap says plainly that it carries no artifact, because that absence
// is the finding.
func filingBody(kind, cause, detail string, iss TrackedIssue, tier Tier, cost CostSpec,
	prov EvidenceProvenance, tol ToleranceSpec, base BaselineSpec, tokenizer Revision, fb *FailureBundle) string {
	k := iss.Key
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n\n", iss.Marker)
	if kind == FilingEvidenceGap {
		fmt.Fprintf(&b, "**Quality evidence gap** — case `%s` produced no trustworthy evidence. Missing or inconclusive evidence is never a pass.\n\n", k.CaseID)
	} else {
		fmt.Fprintf(&b, "**Quality regression** — case `%s` diverged from its reference.\n\n", k.CaseID)
	}
	fmt.Fprintf(&b, "- key: `%s`\n", k)
	fmt.Fprintf(&b, "- model: `%s`  backend: `%s`  mode: `%s`\n", orDash(k.Model), orDash(k.Backend), orDash(k.Mode))
	fmt.Fprintf(&b, "- metric: `%s`\n", orDash(k.Metric))
	if kind == FilingEvidenceGap {
		fmt.Fprintf(&b, "- cause: `%s`\n", cause)
	} else if fb != nil && fb.FirstDivergence != nil {
		d := fb.FirstDivergence
		fmt.Fprintf(&b, "- first actionable divergence: token %d — reference %q, engine %q\n", d.Index, d.Reference, d.Engine)
	} else {
		fmt.Fprintf(&b, "- first actionable divergence: `%s`\n", k.FirstBad)
	}
	if fb != nil && fb.Classification != nil {
		fmt.Fprintf(&b, "- stage: `%s` — %s\n", fb.Classification.Stage, fb.Classification.Reason)
	}
	if detail != "" {
		fmt.Fprintf(&b, "- detail: %s\n", detail)
	}
	fmt.Fprintf(&b, "- tier: `%s` — runtime %ds, timeout %ds, cpu %d, memory %dMiB, accelerators %d\n",
		orDash(string(tier)), cost.RuntimeSeconds, cost.TimeoutSeconds, cost.CPU, cost.MemoryMiB, cost.Accelerators)
	fmt.Fprintf(&b, "- lifecycle: first seen in run `%s`, %d occurrence(s), last failing run `%s`\n",
		orDash(iss.FirstSeenRun), iss.Occurrences, orDash(iss.LastFailRun))

	fmt.Fprintf(&b, "\n**Provenance**\n\n")
	fmt.Fprintf(&b, "- model: `%s`\n", orDash(prov.Model))
	if tokenizer.Revision != "" {
		fmt.Fprintf(&b, "- tokenizer: `%s` @ `%s`\n", orDash(prov.Tokenizer), tokenizer.Revision)
	} else {
		fmt.Fprintf(&b, "- tokenizer: `%s`\n", orDash(prov.Tokenizer))
	}
	fmt.Fprintf(&b, "- engine/backend: `%s`\n", orDash(prov.Engine))
	if prov.Oracle != "" {
		fmt.Fprintf(&b, "- determinism: deterministic oracle `%s`\n", prov.Oracle)
	} else {
		fmt.Fprintf(&b, "- determinism: seed `%d`\n", prov.Seed)
	}
	fmt.Fprintf(&b, "- code/module revision: `%s`\n", orDash(prov.Revision))
	fmt.Fprintf(&b, "- tolerance: `%s` @ `%s`\n", orDash(tol.Metric), orDash(tol.Revision))
	fmt.Fprintf(&b, "- baseline: `%s` @ `%s`\n", orDash(base.ID), orDash(base.Revision))

	fmt.Fprintf(&b, "\n**Replay**\n\n")
	if fb == nil {
		fmt.Fprintf(&b, "None — this finding is an ABSENCE of evidence, which is never a pass. Re-run these coordinates and attach a scrubbed bundle before treating the case as green.\n")
		return b.String()
	}
	blob, err := json.MarshalIndent(fb, "", "  ")
	if err != nil {
		fmt.Fprintf(&b, "None — the scrubbed bundle could not be serialized (%v), so this issue cannot reproduce itself. Treat it as an evidence gap.\n", err)
		return b.String()
	}
	fmt.Fprintf(&b, "The scrubbed bundle below is replay-complete: save it and run\n\n")
	fmt.Fprintf(&b, "    fak quality replay --bundle bundle.json\n\n")
	fmt.Fprintf(&b, "```json\n%s\n```\n", blob)
	return b.String()
}

func orDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// ExplainFilingPlan renders a plan as the operator readout, mirroring Explain /
// ExplainMatrix / ExplainBisect: one line per lifecycle action with the key it
// acts on and the reason it was chosen, so "the nightly filed nothing new and
// closed one" is legible without opening the tracker.
func ExplainFilingPlan(p FilingPlan) string {
	var b strings.Builder
	opens, updates, holds, closes := 0, 0, 0, 0
	for _, f := range p.Filings {
		switch f.Action {
		case ActionOpen:
			opens++
		case ActionUpdate:
			updates++
		case ActionHold:
			holds++
		case ActionClose:
			closes++
		}
	}
	fmt.Fprintf(&b, "FILING  run %s @ %s — %d action(s): %d open, %d update, %d hold, %d close\n",
		orDash(p.Run.ID), orDash(p.Run.Revision), len(p.Filings), opens, updates, holds, closes)
	for _, f := range p.Filings {
		fmt.Fprintf(&b, "  %-6s %-13s %s\n", f.Action, f.Kind, f.Key)
		fmt.Fprintf(&b, "         %s\n", f.Reason)
		if f.Action != ActionHold && f.Kind == FilingRegression && f.Replay == nil {
			fmt.Fprintf(&b, "         replay: none — an issue that cannot reproduce itself is not evidence\n")
		}
	}
	return b.String()
}

// DemoFilingCase stamps the spine demo case with the canonical routing header and
// baseline provenance a filed finding must document (#4584 acceptance 2 and 4):
// model, tokenizer, engine/backend and mode, deterministic oracle, code revision,
// tolerance and baseline, an explicit tier, and the runtime/resource cost. It is
// the hermetic fixture the witness folds, and the shape a real corpus case has.
func DemoFilingCase(revision string) QualityCase {
	c := DemoCase()
	c.Metadata = CaseMetadata{
		Model:     Revision{Name: "demo-1b", Revision: "sha256:m-demo"},
		Tokenizer: Revision{Name: "demo-bpe", Revision: "sha256:t-demo"},
		Engine:    EngineSpec{Name: "fak", Backend: "cpu", Flags: map[string]string{"mode": "eager"}},
		Code:      Revision{Name: "github.com/anthony-chaudhary/fak", Revision: revision},
		Oracle:    OracleEvidence{Kind: "exact-greedy-trace", Revision: "sha256:o-demo"},
		Tolerance: ToleranceSpec{Metric: "exact-token", Revision: "policy:v1"},
		Baseline:  BaselineSpec{ID: "spine-demo-baseline", Revision: "sha256:b-demo"},
		Tier:      TierSpec{Name: string(TierPR)},
		Cost:      CostSpec{RuntimeSeconds: 2, TimeoutSeconds: 30, CPU: 1, MemoryMiB: 256},
		Owner:     "quality-team",
		Family:    string(FamilyDeterministic),
	}
	return c
}

// DemoFilingObservation runs the demo case through the REAL spine at the given
// revision with an optional injected defect ("" = clean, "decode"/"stop"/"report"
// = planted) and folds the Result into an Observation. The evidence is produced
// by RunCase rather than hand-written, so the auto-filer is proven against
// genuine first-divergence evidence and a genuinely scrubbed bundle — the
// planted-defect-fails / fix-passes witness the epic requires.
func DemoFilingObservation(revision, defect string) Observation {
	c := DemoFilingCase(revision)
	prov := EvidenceProvenance{
		Model:     c.Metadata.Model.Name,
		Tokenizer: c.Metadata.Tokenizer.Name,
		Engine:    c.Metadata.Engine.Name + "/" + c.Metadata.Engine.Backend,
		Oracle:    c.Metadata.Oracle.Kind,
		Revision:  revision,
		Baseline:  c.Metadata.Baseline.ID,
	}
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		return Observation{Case: c, Evidence: Evidence{CaseID: c.ID, State: StateInconclusive, Provenance: prov, Detail: err.Error()}}
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(defect), oracles)
	if err != nil {
		return Observation{Case: c, Evidence: Evidence{CaseID: c.ID, State: StateInconclusive, Provenance: prov, Detail: err.Error()}}
	}
	return Observation{Case: c, Evidence: EvidenceFromResult(prov, res)}
}
