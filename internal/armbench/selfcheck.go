package armbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// SelfcheckSchema tags the selfcheck artifact.
const SelfcheckSchema = "fak.armbench.selfcheck/1"

// Check is one selfcheck assertion and its captured evidence.
type Check struct {
	Name     string `json:"name"`
	OK       bool   `json:"ok"`
	Detail   string `json:"detail"`
	Evidence string `json:"evidence,omitempty"`
}

// SelfcheckResult is the whole selfcheck artifact.
type SelfcheckResult struct {
	Schema string  `json:"schema"`
	OK     bool    `json:"ok"`
	Checks []Check `json:"checks"`
	// Report is the human summary of the spine run, captured so the selfcheck
	// artifact is itself the behavioural proof rather than a pass/fail bit that
	// asserts one happened.
	Report        string  `json:"report"`
	SpineIdentity string  `json:"spine_identity"`
	Spine         *Report `json:"spine"`
	// SpineRun carries the raw request/response/judgment evidence behind Spine.
	// The committed selfcheck capture is therefore an inspectable proof, not a
	// report whose measurements have already discarded their source artifacts.
	SpineRun *Run `json:"spine_run"`
}

// DemoManifest is the pinned four-arm demo: an untreated baseline, the upstream
// Caveman treatment at the SHA epic #6674 pins, fak passthrough, and exactly one
// isolated fak capability. It is the shape every later benchmark issue
// instantiates, so it doubles as the worked example in the docs.
func DemoManifest() *Manifest {
	tasks := DemoCorpus()
	return &Manifest{
		Schema: ManifestSchema,
		ID:     "armbench-spine-demo",
		Sources: []Source{{
			Name:        "caveman",
			Repo:        "JuliusBrussee/caveman",
			SHA:         "c72984e4392c7a154e55c11dbf445f01ce5c35d4",
			Path:        "benchmarks/run.py",
			ContentHash: "sha256:530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce",
			RetrievedAt: "2026-08-13",
		}, {
			Name:        "ponytail",
			Repo:        "DietrichGebert/ponytail",
			SHA:         "2ed6c52c9d7e5e56942508591085fd45dea277d3",
			Path:        "benchmarks/arms/baseline.js",
			ContentHash: "sha256:ef0f81f670425705ab3195609947aa64890890cb078d7669780afe2228da8740",
			RetrievedAt: "2026-08-13",
		}},
		Model: Model{
			Provider:  "fake",
			Snapshot:  "fake-deterministic-2026-08-13",
			Region:    "local",
			Sampling:  Sampling{Temperature: 0, TopP: 1, Seed: 7},
			MaxTokens: 4096,
		},
		Corpus:      Corpus{ID: "spine-demo-corpus", Hash: HashTasks(tasks), TaskCount: len(tasks)},
		Judge:       Judge{ID: "fake/contains-task-id", Hash: fixtureDigest("fake/contains-task-id/v1"), Kind: "deterministic"},
		Trials:      Trials{Count: 2, Seed: 11, Order: OrderCounterbalanced, Concurrency: 2},
		Environment: Environment{OS: "any", Arch: "any", HostClass: "dev", FakVersion: "armbench@r1", PricingDate: "2026-08-13"},
		Arms: []Arm{
			{ID: "baseline", Kind: ArmBaseline, PromptHash: fixtureDigest("prompt/baseline/v1")},
			{ID: "caveman-upstream", Kind: ArmUpstreamTreatment, PromptHash: fixtureDigest("prompt/caveman/v1"), SourceName: "caveman"},
			{ID: "fak-passthrough", Kind: ArmFakPassthrough, PromptHash: fixtureDigest("prompt/baseline/v1")},
			{ID: "fak-ctxmmu", Kind: ArmFakCapability, Capabilities: []string{"ctxmmu-paging"}, PromptHash: fixtureDigest("prompt/baseline/v1")},
		},
	}
}

// DemoCorpus is the pinned three-task corpus the demo manifest declares.
func DemoCorpus() []Task {
	return []Task{
		{ID: "task-alpha", Input: "summarize the build log", Expect: "task-alpha"},
		{ID: "task-bravo", Input: "explain the failing test", Expect: "task-bravo"},
		{ID: "task-charlie", Input: "list the changed files", Expect: "task-charlie"},
	}
}

// Selfcheck runs the deterministic spine and every fail-closed proof this
// package owes issue #6676, returning a captured artifact. It never touches the
// network or the clock, so it is safe on any host and in CI.
//
// The checks, in the order they build on each other:
//
//  1. the spine runs baseline + upstream treatment + fak arms end to end;
//  2. changing the model, the prompt, the judge, the corpus, or the capability
//     each moves the manifest identity (the five mutations #6676 names);
//  3. a trial with no raw response fails closed;
//  4. an arm that bundles two unnamed-apart capabilities fails closed;
//  5. resume re-executes nothing and duplicates nothing;
//  6. a drifted manifest is refused as incomparable;
//  7. the paired order is counterbalanced, not fixed.
func Selfcheck() (*SelfcheckResult, error) {
	ctx := context.Background()
	res := &SelfcheckResult{Schema: SelfcheckSchema, OK: true}
	add := func(name string, ok bool, detail, evidence string) {
		res.Checks = append(res.Checks, Check{Name: name, OK: ok, Detail: detail, Evidence: evidence})
		if !ok {
			res.OK = false
		}
	}

	m := DemoManifest()
	corpus := DemoCorpus()
	base := m.Identity()
	res.SpineIdentity = base

	run, err := Execute(ctx, m, corpus, &FakeProvider{SetupWallMS: 2500}, &FakeGrader{}, Options{})
	if err != nil {
		return nil, fmt.Errorf("spine run: %w", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		return nil, fmt.Errorf("spine summarize: %w", err)
	}
	res.Spine = rep
	res.SpineRun = run
	res.Report = Human(rep)

	wantRows := len(corpus) * m.Trials.Count * len(m.Arms)
	add("spine.end_to_end", len(run.Trials) == wantRows && run.Executed == wantRows,
		fmt.Sprintf("ran %d rows (%d executed), want %d", len(run.Trials), run.Executed, wantRows),
		strings.Join(armIDsOf(rep), ","))

	kinds := map[ArmKind]bool{}
	for _, a := range rep.Arms {
		kinds[a.Kind] = true
	}
	add("spine.covers_baseline_treatment_and_fak",
		kinds[ArmBaseline] && kinds[ArmUpstreamTreatment] && kinds[ArmFakCapability],
		fmt.Sprintf("arm kinds present: %v", sortedKinds(kinds)), "")

	// (2) identity sensitivity — the five mutations #6676 names.
	for _, mut := range identityMutations() {
		mm := DemoManifest()
		mut.apply(mm)
		moved := mm.Identity() != base
		add("identity.changes_on_"+mut.name, moved,
			fmt.Sprintf("%s: identity %s", mut.detail, movedWord(moved)), mm.Identity())
	}
	// The control: an untouched copy must hash the SAME, or "it always changes"
	// would pass every mutation check vacuously.
	add("identity.stable_when_unchanged", DemoManifest().Identity() == base,
		"an unmutated copy of the demo manifest hashes to the same identity", base)
	// The manifest is immutable as a whole: disclosed scheduling/environment
	// changes also move the identity even though the five checks above are the
	// acceptance-critical semantic terms.
	envChanged := DemoManifest()
	envChanged.Environment.HostClass = "ci"
	add("identity.changes_on_environment", envChanged.Identity() != base,
		"changing the recorded host class moves the immutable-manifest identity", envChanged.Identity())

	// (3) missing raw evidence fails closed.
	_, err = Execute(ctx, DemoManifest(), corpus, &FakeProvider{OmitRawResponse: "fak-ctxmmu"}, &FakeGrader{}, Options{})
	add("failclosed.missing_raw_evidence", isReason(err, ReasonMissingRawEvidence),
		describeRefusal(err, ReasonMissingRawEvidence), errText(err))

	// (4) a bundled, un-isolated capability arm fails closed.
	bundled := DemoManifest()
	for i := range bundled.Arms {
		if bundled.Arms[i].Kind == ArmFakCapability {
			bundled.Arms[i].Capabilities = []string{"ctxmmu-paging", "toon-encoding"}
		}
	}
	err = bundled.Validate()
	add("failclosed.bundled_capability", isReason(err, ReasonArmCapabilityBundled),
		describeRefusal(err, ReasonArmCapabilityBundled), errText(err))

	unnamed := DemoManifest()
	for i := range unnamed.Arms {
		if unnamed.Arms[i].Kind == ArmFakCapability {
			unnamed.Arms[i].Capabilities = nil
		}
	}
	err = unnamed.Validate()
	add("failclosed.unnamed_capability", isReason(err, ReasonArmCapabilityUnnamed),
		describeRefusal(err, ReasonArmCapabilityUnnamed), errText(err))

	// (5) resume executes nothing new and duplicates nothing.
	resumed, err := Execute(ctx, DemoManifest(), corpus, &FakeProvider{SetupWallMS: 2500}, &FakeGrader{}, Options{Resume: run})
	switch {
	case err != nil:
		add("resume.no_duplicate_trials", false, "resume run failed: "+err.Error(), "")
	default:
		dupes := duplicateKeys(resumed.Trials)
		add("resume.no_duplicate_trials", resumed.Executed == 0 && len(resumed.Trials) == wantRows && len(dupes) == 0,
			fmt.Sprintf("resume executed %d new trial(s), carried %d row(s), %d duplicate key(s)", resumed.Executed, len(resumed.Trials), len(dupes)),
			strings.Join(dupes, ","))
	}
	resumeDrift := DemoManifest()
	resumeDrift.Model.Snapshot = "fake-deterministic-2026-09-01"
	_, err = Execute(ctx, resumeDrift, corpus, &FakeProvider{}, &FakeGrader{}, Options{Resume: run})
	add("resume.refuses_foreign_ledger", isReason(err, ReasonResumeIdentityMismatch),
		describeRefusal(err, ReasonResumeIdentityMismatch), errText(err))

	// (6) an incomparable manifest is refused.
	drift := DemoManifest()
	drift.Corpus.Hash = fixtureDigest("different corpus")
	fields, err := CheckComparable(DemoManifest(), drift)
	add("comparability.refuses_drifted_manifest", isReason(err, ReasonIncomparableManifest) && len(fields) > 0,
		describeRefusal(err, ReasonIncomparableManifest), fieldNames(fields))
	_, err = CheckComparable(DemoManifest(), DemoManifest())
	add("comparability.admits_identical_manifest", err == nil,
		"two identical manifests are comparable", errText(err))

	// (7) the paired order is counterbalanced, not fixed.
	units := PlanUnits(m, corpus)
	firsts := map[string]int{}
	for _, u := range units {
		firsts[u.ArmOrder[0]]++
	}
	add("pairing.counterbalanced_order", len(firsts) == len(m.Arms),
		fmt.Sprintf("%d of %d arms occupied first position across %d paired units", len(firsts), len(m.Arms), len(units)),
		fmt.Sprint(firsts))

	return res, nil
}

type mutation struct {
	name   string
	detail string
	apply  func(*Manifest)
}

// identityMutations is the exact list #6676's acceptance names: model, prompt,
// judge, corpus, capability.
func identityMutations() []mutation {
	return []mutation{
		{"model", "model snapshot swapped", func(m *Manifest) { m.Model.Snapshot = "fake-deterministic-2026-09-01" }},
		{"prompt", "baseline arm prompt hash swapped", func(m *Manifest) { m.Arms[0].PromptHash = fixtureDigest("prompt/baseline/v2") }},
		{"judge", "judge hash swapped", func(m *Manifest) { m.Judge.Hash = fixtureDigest("fake/contains-task-id/v2") }},
		{"corpus", "corpus hash swapped", func(m *Manifest) { m.Corpus.Hash = fixtureDigest("different corpus") }},
		{"capability", "isolated capability swapped", func(m *Manifest) {
			for i := range m.Arms {
				if m.Arms[i].Kind == ArmFakCapability {
					m.Arms[i].Capabilities = []string{"toon-encoding"}
				}
			}
		}},
	}
}

func fixtureDigest(s string) string {
	sum := sha256.Sum256([]byte(s))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func duplicateKeys(rows []TrialResult) []string {
	seen := map[string]bool{}
	var dupes []string
	for _, t := range rows {
		k := t.Key()
		if seen[k] {
			dupes = append(dupes, k)
		}
		seen[k] = true
	}
	return dupes
}

func isReason(err error, reason string) bool {
	if err == nil {
		return false
	}
	var r *RefusalError
	if ok := asRefusal(err, &r); ok {
		return r.Reason == reason
	}
	return false
}

// asRefusal unwraps to a *RefusalError without pulling in errors.As's reflection
// path for a single concrete type.
func asRefusal(err error, out **RefusalError) bool {
	for err != nil {
		if r, ok := err.(*RefusalError); ok {
			*out = r
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func describeRefusal(err error, want string) string {
	if err == nil {
		return "expected refusal " + want + " but the call succeeded"
	}
	if isReason(err, want) {
		return "refused with " + want + " as required"
	}
	return "expected " + want + " but got: " + err.Error()
}

func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func movedWord(moved bool) string {
	if moved {
		return "moved"
	}
	return "did NOT move"
}

func armIDsOf(rep *Report) []string {
	out := make([]string, 0, len(rep.Arms))
	for _, a := range rep.Arms {
		out = append(out, a.ArmID)
	}
	return out
}

func sortedKinds(kinds map[ArmKind]bool) []string {
	out := []string{}
	for _, k := range KnownArmKinds() {
		if kinds[k] {
			out = append(out, string(k))
		}
	}
	return out
}

func fieldNames(fields []ComparabilityField) string {
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		out = append(out, f.Field)
	}
	return strings.Join(out, ",")
}

// MarshalSelfcheck renders the selfcheck artifact as strict, stable JSON.
func MarshalSelfcheck(r *SelfcheckResult) ([]byte, error) {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// HumanSelfcheck renders the operator-facing selfcheck summary.
func HumanSelfcheck(r *SelfcheckResult) string {
	var b strings.Builder
	b.WriteString(r.Report)
	b.WriteString("\n  selfcheck\n")
	for _, c := range r.Checks {
		mark := "ok  "
		if !c.OK {
			mark = "FAIL"
		}
		fmt.Fprintf(&b, "    [%s] %-44s %s\n", mark, c.Name, c.Detail)
	}
	verdict := "PASS"
	if !r.OK {
		verdict = "FAIL"
	}
	fmt.Fprintf(&b, "\n  %s — %d checks over identity %s\n", verdict, len(r.Checks), r.SpineIdentity)
	return b.String()
}
