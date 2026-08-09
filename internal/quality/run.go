package quality

import (
	"fmt"
	"regexp"
	"strings"
)

// ResultSchema and ManifestSchema are the versioned tags on the two
// machine-readable artifacts the spine emits. Consumers pin the major so a schema
// bump is a conscious migration, not a silent field drift (#4519).
const (
	ResultSchema   = "fak-quality-result/1"
	ManifestSchema = "fak-quality-manifest/1"
)

// RunManifest is the replay-complete record of HOW a result was produced: the case
// identity and version, which reference/engine runners ran, and which oracles were
// applied. It deliberately carries no wall-clock or host state — replay identity
// must not depend on when or where a run happened (#4514). A caller that wants a
// timestamp stamps it outside, onto the enclosing artifact.
type RunManifest struct {
	Schema          string   `json:"schema"`
	CaseID          string   `json:"case_id"`
	CaseVersion     int      `json:"case_version"`
	ReferenceRunner string   `json:"reference_runner"`
	EngineRunner    string   `json:"engine_runner"`
	Oracles         []string `json:"oracles"`
}

// Provenance is the captured request/config/token/logit/output evidence for a run
// (spine contract item 2). It is what makes a failure localizable: the exact params
// both paths ran under and the full reference vs engine traces.
type Provenance struct {
	CaseID    string         `json:"case_id"`
	Params    SamplingParams `json:"params"`
	Reference Trace          `json:"reference"`
	Engine    Trace          `json:"engine"`
	// Requests is what each side did with those params: which normalized request
	// fields it could not honor and how the request it actually executed differed
	// (#4518). Without it a passing run only asserts that two traces agree, not
	// that they answer the same question.
	Requests RequestRecord `json:"requests"`
}

// FailureBundle is the portable, replay-complete artifact emitted on any failing
// run (#4515). It embeds the full case so the failure reproduces from the bundle
// alone, names the first oracle that failed, and pins the first token divergence
// when one exists. It is scrubbed by construction — it carries only case text the
// author already committed, never ambient secrets.
type FailureBundle struct {
	CaseID          string      `json:"case_id"`
	Case            QualityCase `json:"case"`
	FailingOracle   string      `json:"failing_oracle"`
	FailingKind     string      `json:"failing_kind"`
	FirstDivergence *Divergence `json:"first_divergence,omitempty"`
	Reference       Trace       `json:"reference"`
	Engine          Trace       `json:"engine"`
	Detail          string      `json:"detail"`
	Scrubbed        bool        `json:"scrubbed"`
	// Requests travels with the bundle so a failure reproduces from the bundle
	// ALONE (#4515): a replayer that cannot see the engine dropped top_k would
	// re-run a different request than the one that failed.
	Requests RequestRecord `json:"requests"`
	// Classification is the machine-readable half of first-failure localization
	// (#4520): which layer of the serving path the bundle's own evidence points
	// at, or an explicit abstention when it points nowhere. It is Classify's
	// output, stamped here so a CI consumer routes on the artifact instead of
	// re-deriving it from prose.
	Classification *Classification `json:"classification,omitempty"`
}

// Result is the stable machine-readable outcome of one quality run (#4519): a
// pass/fail verdict, every oracle's verdict, the captured provenance, the replay
// manifest, and — iff the run failed — a portable failure bundle. Encoding it as
// JSON is the CI contract: a PR gate reads Pass, a human reads the bundle.
type Result struct {
	Schema        string         `json:"schema"`
	CaseID        string         `json:"case_id"`
	Pass          bool           `json:"pass"`
	Verdicts      []Verdict      `json:"verdicts"`
	Provenance    Provenance     `json:"provenance"`
	Manifest      RunManifest    `json:"manifest"`
	FailureBundle *FailureBundle `json:"failure_bundle,omitempty"`
}

// RunCase is the spine orchestrator (contract items 1–4): it runs the case through
// the reference and engine runners, captures provenance, applies every named
// oracle, folds the verdicts into a single pass/fail, and — on failure — attaches a
// replayable bundle localized to the first failing oracle. It is a pure function of
// (case, runners, oracles): same inputs, same Result.
func RunCase(c QualityCase, ref, eng Runner, oracles []Oracle) (Result, error) {
	if ok, why := c.Valid(); !ok {
		return Result{}, fmt.Errorf("invalid case %q: %s", c.ID, why)
	}
	if len(oracles) == 0 {
		return Result{}, fmt.Errorf("case %q has no executable oracle evidence", c.ID)
	}
	if ref == nil || eng == nil {
		return Result{}, fmt.Errorf("case %q requires reference and engine runners", c.ID)
	}
	refTrace, err := ref.Run(c)
	if err != nil {
		return Result{}, fmt.Errorf("reference runner %q: %w", ref.Name(), err)
	}
	engTrace, err := eng.Run(c)
	if err != nil {
		return Result{}, fmt.Errorf("engine runner %q: %w", eng.Name(), err)
	}
	requests := RequestRecord{Reference: requestFidelity(ref, c), Engine: requestFidelity(eng, c)}
	prov := Provenance{CaseID: c.ID, Params: c.Params, Reference: refTrace, Engine: engTrace, Requests: requests}

	verdicts := make([]Verdict, 0, len(oracles)+1)
	pass := true
	var firstFail *Verdict
	// Request fidelity is judged BEFORE any oracle and, when it fails, becomes the
	// first failure by construction: if the two sides were not handed the same
	// request, no downstream verdict is interpretable — agreement proves nothing
	// and disagreement indicts the wrong layer. On a faithful run nothing is
	// appended, so a runner that never declares a request keeps the exact verdict
	// list it had before this check existed.
	if v, drifted := requestFidelityVerdict(requests); drifted {
		verdicts = append(verdicts, v)
		pass = false
		fv := v
		firstFail = &fv
	}
	for i := range oracles {
		v := oracles[i].Judge(refTrace, engTrace, c)
		verdicts = append(verdicts, v)
		if !v.Pass {
			pass = false
			if firstFail == nil {
				fv := v
				firstFail = &fv
			}
		}
	}

	res := Result{
		Schema:     ResultSchema,
		CaseID:     c.ID,
		Pass:       pass,
		Verdicts:   verdicts,
		Provenance: prov,
		Manifest: RunManifest{
			Schema:          ManifestSchema,
			CaseID:          c.ID,
			CaseVersion:     c.Version,
			ReferenceRunner: ref.Name(),
			EngineRunner:    eng.Name(),
			Oracles:         oracleNames(oracles),
		},
	}
	if !pass && firstFail != nil {
		// Localize AFTER scrubbing: the reason quotes trace excerpts, so it must
		// quote the redacted ones or the bundle would leak through its own summary.
		bundle := scrubFailureBundle(FailureBundle{
			CaseID:          c.ID,
			Case:            c,
			FailingOracle:   firstFail.Oracle,
			FailingKind:     firstFail.Kind,
			FirstDivergence: firstFail.FirstDivergence,
			Reference:       refTrace,
			Engine:          engTrace,
			Detail:          firstFail.Detail,
			Requests:        requests,
		})
		stage := classifyFailure(*bundle)
		bundle.Classification = &stage
		res.FailureBundle = bundle
	}
	return res, nil
}

var replaySecret = regexp.MustCompile(`(?i)(api[_-]?key|authorization|token|password)\s*[:=]\s*[^\s,;]+`)

func scrubFailureBundle(f FailureBundle) *FailureBundle {
	redact := func(s string) string { return replaySecret.ReplaceAllString(s, "$1=[REDACTED]") }
	f.Case.Prompt = redact(f.Case.Prompt)
	f.Case.Reference.Text = redact(f.Case.Reference.Text)
	for i := range f.Case.Reference.Tokens {
		f.Case.Reference.Tokens[i] = redact(f.Case.Reference.Tokens[i])
	}
	f.Reference.Text = redact(f.Reference.Text)
	f.Engine.Text = redact(f.Engine.Text)
	for i := range f.Reference.Tokens {
		f.Reference.Tokens[i] = redact(f.Reference.Tokens[i])
	}
	for i := range f.Engine.Tokens {
		f.Engine.Tokens[i] = redact(f.Engine.Tokens[i])
	}
	f.Detail = redact(f.Detail)
	// A "prompt" delta echoes request text verbatim, so it is redacted like every
	// other quoted surface. Both records are rebuilt onto fresh slices rather than
	// edited in place: the bundle's RequestRecord shares its backing array with
	// the Result's own Provenance, and scrubbing must not reach back through it.
	f.Requests.Reference.Diff = redactDeltas(f.Requests.Reference.Diff, redact)
	f.Requests.Engine.Diff = redactDeltas(f.Requests.Engine.Diff, redact)
	if f.FirstDivergence != nil {
		d := *f.FirstDivergence
		d.Reference = redact(d.Reference)
		d.Engine = redact(d.Engine)
		f.FirstDivergence = &d
	}
	f.Scrubbed = true
	return &f
}

func redactDeltas(ds []FieldDelta, redact func(string) string) []FieldDelta {
	if len(ds) == 0 {
		return ds
	}
	out := make([]FieldDelta, len(ds))
	for i, d := range ds {
		d.Requested, d.Effective = redact(d.Requested), redact(d.Effective)
		out[i] = d
	}
	return out
}

func oracleNames(os []Oracle) []string {
	out := make([]string, len(os))
	for i, o := range os {
		out[i] = o.Name()
	}
	return out
}

// Explain renders a Result as human-readable first-failure localization (#4520):
// on pass it states what was verified; on failure it names the first failing
// oracle, the exact token index and the reference-vs-engine tokens there, and the
// STAGE of the serving path the bundle's evidence attributes that divergence to
// (or an explicit abstention when the evidence attributes it nowhere — see
// Classify). It is the `fak quality explain` body — the bridge from a machine
// verdict to "here is where, and in which layer, it first went wrong".
func Explain(r Result) string {
	var b strings.Builder
	if r.Pass {
		fmt.Fprintf(&b, "PASS  case %s — %d oracle(s) agreed with the reference\n", r.CaseID, len(r.Verdicts))
		for _, v := range r.Verdicts {
			fmt.Fprintf(&b, "  ok   %-18s %s\n", v.Oracle, v.Detail)
		}
		return b.String()
	}
	fmt.Fprintf(&b, "FAIL  case %s\n", r.CaseID)
	if fb := r.FailureBundle; fb != nil {
		fmt.Fprintf(&b, "  first failure: %s (%s)\n", fb.FailingOracle, fb.FailingKind)
		if d := fb.FirstDivergence; d != nil {
			fmt.Fprintf(&b, "  first divergence at token %d: reference %q, engine %q\n", d.Index, d.Reference, d.Engine)
		}
		if s := Classify(r); s.Abstained() {
			fmt.Fprintf(&b, "  stage: %s (abstained) — %s\n", s.Stage, s.Reason)
		} else {
			fmt.Fprintf(&b, "  stage: %s — %s\n", s.Stage, s.Reason)
		}
		fmt.Fprintf(&b, "  detail: %s\n", fb.Detail)
		fmt.Fprintf(&b, "  replay: quality case %s @ v%d, runner %s vs %s\n",
			fb.CaseID, fb.Case.Version, r.Manifest.ReferenceRunner, r.Manifest.EngineRunner)
	}
	for _, v := range r.Verdicts {
		state := "ok  "
		if !v.Pass {
			state = "FAIL"
		}
		fmt.Fprintf(&b, "  %s %-18s %s\n", state, v.Oracle, v.Detail)
	}
	return b.String()
}
