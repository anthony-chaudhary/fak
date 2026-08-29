// Package armbench is the provenance-locked multi-arm benchmark runner (#6676,
// epic #6674). It replaces bespoke per-comparison scripts with ONE immutable
// manifest that pins every term a benchmark comparison can drift on — upstream
// repo/SHA/path, model snapshot, provider, sampling, max tokens, corpus hash,
// judge hash, trial count, seed, pairing order, concurrency, region, pricing
// date, and environment — and executes the declared arms against it.
//
// The load-bearing idea is MANIFEST IDENTITY. Two benchmark runs are comparable
// only if the terms that decide what was measured are byte-identical; a changed
// model, prompt, judge, corpus, or arm capability must produce a DIFFERENT
// identity so a drifted rerun can never be silently stacked next to the old
// number. Identity is a sha256 over a canonical encoding of exactly those
// terms, and Selfcheck proves each of the five mutations moves it.
//
// Everything here is pure and stdlib-only: no network, no provider SDK, no
// clock read except the one the caller injects. The provider and the judge are
// interfaces (see run.go), so the deterministic fake-provider spine that proves
// the runner end to end is the same code path a live provider will take.
//
// Fail-closed is the default, not an option. A trial with no raw request or no
// raw response is refused (a token count with no evidence behind it is not
// evidence), and an arm that bundles more than one named fak capability is
// refused at validation (a bundled arm cannot attribute a delta to anything).
package armbench

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ManifestSchema is the schema tag every manifest must carry. A manifest whose
// schema tag is absent or unknown is refused rather than best-effort parsed:
// this file's whole value is that a reader knows which fields were pinned.
const ManifestSchema = "fak.armbench.manifest/1"

// ArmKind is the closed vocabulary of arm roles. The four are deliberately
// distinct because they answer different questions: baseline is the untreated
// control, upstream_treatment reproduces the comparator's own published arm,
// fak_passthrough charges fak's plumbing cost with NO capability enabled (the
// honest zero point for a fak claim), and fak_capability isolates exactly one
// named capability on top of that.
type ArmKind string

const (
	// ArmBaseline is the untreated control arm.
	ArmBaseline ArmKind = "baseline"
	// ArmUpstreamTreatment reproduces the comparator's published treatment.
	ArmUpstreamTreatment ArmKind = "upstream_treatment"
	// ArmFakPassthrough routes through fak with no capability enabled — it
	// measures what fak COSTS before anything it saves is counted.
	ArmFakPassthrough ArmKind = "fak_passthrough"
	// ArmFakCapability enables exactly one named fak capability.
	ArmFakCapability ArmKind = "fak_capability"
)

// KnownArmKinds returns the closed vocabulary in declaration order.
func KnownArmKinds() []ArmKind {
	return []ArmKind{ArmBaseline, ArmUpstreamTreatment, ArmFakPassthrough, ArmFakCapability}
}

// OrderStrategy decides the within-pair execution order of the arms for one
// (task, trial) unit. Both settings are deterministic given the manifest seed —
// "randomized" means randomized ACROSS pairs, not irreproducible.
type OrderStrategy string

const (
	// OrderCounterbalanced rotates the arm order by (task index + trial), so
	// every arm occupies every position an equal number of times.
	OrderCounterbalanced OrderStrategy = "counterbalanced"
	// OrderRandomized shuffles the arm order with a seeded PRNG derived from
	// (seed, task id, trial), so the order is random across pairs and exactly
	// reproducible from the manifest.
	OrderRandomized OrderStrategy = "randomized"
)

// Source is one pinned upstream input: which repository, at which commit, which
// path, and the content hash of what was actually retrieved. The content hash
// is what makes the pin checkable — a repo/SHA pair alone still permits a
// hand-edited local copy.
type Source struct {
	Name                string `json:"name"`
	Repo                string `json:"repo"`
	URL                 string `json:"url,omitempty"`
	SHA                 string `json:"sha"`
	Path                string `json:"path"`
	ContentHash         string `json:"content_hash"`
	License             string `json:"license,omitempty"`
	LicenseBoundary     string `json:"license_boundary,omitempty"`
	LicenseBoundaryHash string `json:"license_boundary_hash,omitempty"`
	LicenseReview       string `json:"license_review,omitempty"`
	RetrievedAt         string `json:"retrieved_at,omitempty"`
	Normalization       string `json:"normalization,omitempty"`
	LocalPath           string `json:"local_path,omitempty"`
}

// Sampling pins the decode parameters. Temperature and TopP are pointers-free
// plain values because 0 is a meaningful setting (temperature 0 is exactly what
// the Caveman comparator publishes); Seed 0 means "provider default / unseeded"
// and is recorded as such rather than hidden.
type Sampling struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	Seed        int64   `json:"seed"`
}

// Model pins the generation side: which provider, which model SNAPSHOT (never a
// floating alias — an alias silently repoints and destroys comparability), the
// region it was served from, the sampling parameters, and the output cap.
type Model struct {
	Provider  string   `json:"provider"`
	Snapshot  string   `json:"snapshot"`
	Region    string   `json:"region"`
	Sampling  Sampling `json:"sampling"`
	MaxTokens int      `json:"max_tokens"`
}

// Corpus pins the task set by content hash. TaskCount is recorded so a truncated
// corpus is visible in the report even before the hash is recomputed.
type Corpus struct {
	ID        string `json:"id"`
	Hash      string `json:"hash"`
	TaskCount int    `json:"task_count"`
}

// Judge pins the grader: its identifier and the content hash of its definition
// (prompt, rubric, or deterministic checker source). A changed judge changes the
// manifest identity, because a score graded by a different judge is a different
// measurement.
type Judge struct {
	ID   string `json:"id"`
	Hash string `json:"hash"`
	Kind string `json:"kind,omitempty"`
}

// Trials pins the repetition and pairing plan. Concurrency is recorded,
// enforced, and identity-bearing: provider throttling and queueing can change
// measured latency, so a different concurrency is not silently comparable.
type Trials struct {
	Count       int           `json:"count"`
	Seed        int64         `json:"seed"`
	Order       OrderStrategy `json:"order"`
	Concurrency int           `json:"concurrency"`
}

// Environment records the host and pricing context. It is identity-bearing so
// a changed machine or price sheet cannot hide behind the old manifest id; a
// later comparison verb (#6680) can still classify the named differing fields.
type Environment struct {
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	HostClass   string `json:"host_class"`
	FakVersion  string `json:"fak_version"`
	PricingDate string `json:"pricing_date"`
}

// Arm is one comparison arm. Capabilities is the fail-closed field: an
// ArmFakCapability arm must name EXACTLY one, and every other kind must name
// none, so "which single thing is this arm testing" is always answerable.
type Arm struct {
	ID           string   `json:"id"`
	Kind         ArmKind  `json:"kind"`
	Capabilities []string `json:"capabilities,omitempty"`
	// GPUIndex optionally assigns this arm one host-local CUDA device. A pointer
	// keeps device zero distinct from an omitted assignment and lets legacy
	// manifests retain their byte shape and identity.
	GPUIndex *int `json:"gpu_index,omitempty"`
	// PromptHash pins the system/skill prompt this arm installs. It is
	// identity-bearing: a changed prompt is a changed experiment.
	PromptHash string `json:"prompt_hash"`
	// SourceName optionally binds this arm to one of Manifest.Sources by name
	// (required for upstream_treatment — a treatment with no pinned upstream
	// input is a hand-copied approximation wearing the comparator's name).
	SourceName string `json:"source_name,omitempty"`
	Notes      string `json:"notes,omitempty"`
}

// Manifest is the immutable description of one multi-arm comparison.
type Manifest struct {
	Schema      string      `json:"schema"`
	ID          string      `json:"id"`
	Sources     []Source    `json:"sources"`
	Model       Model       `json:"model"`
	Corpus      Corpus      `json:"corpus"`
	Judge       Judge       `json:"judge"`
	Trials      Trials      `json:"trials"`
	Environment Environment `json:"environment"`
	Arms        []Arm       `json:"arms"`
}

// RefusalError is a typed refusal carrying a token from a closed vocabulary, so
// a caller can branch on the reason instead of matching prose.
type RefusalError struct {
	Reason string
	Detail string
}

func (e *RefusalError) Error() string { return e.Reason + ": " + e.Detail }

func refuse(reason, format string, args ...any) *RefusalError {
	return &RefusalError{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// The closed refusal vocabulary. Each names a distinct failure a benchmark
// runner must never paper over.
const (
	// ReasonManifestInvalid — a required provenance field is missing or a
	// declared value is outside its closed vocabulary.
	ReasonManifestInvalid = "MANIFEST_INVALID"
	// ReasonArmCapabilityBundled — a fak_capability arm named more than one
	// capability, so no delta it produces can be attributed to a single thing.
	ReasonArmCapabilityBundled = "ARM_CAPABILITY_BUNDLED"
	// ReasonArmCapabilityUnnamed — a fak_capability arm named no capability (or
	// a non-capability arm named one), so what it enables is unstated.
	ReasonArmCapabilityUnnamed = "ARM_CAPABILITY_UNNAMED"
	// ReasonMissingRawEvidence — a trial produced a number with no raw
	// request/response behind it.
	ReasonMissingRawEvidence = "MISSING_RAW_EVIDENCE"
	// ReasonIncomparableManifest — two runs differ on a term that decides what
	// was measured, so putting their numbers side by side would be a category
	// error.
	ReasonIncomparableManifest = "INCOMPARABLE_MANIFEST"
	// ReasonResumeIdentityMismatch — a resume ledger was produced under a
	// different manifest identity, so resuming from it would silently mix two
	// experiments.
	ReasonResumeIdentityMismatch = "RESUME_IDENTITY_MISMATCH"
	// ReasonProviderUnknown — the requested provider is not registered.
	ReasonProviderUnknown = "PROVIDER_UNKNOWN"
	// ReasonDuplicateTrial — a ledger contains the same manifest/arm/task/trial
	// key more than once, so resume or reporting would silently double count it.
	ReasonDuplicateTrial = "DUPLICATE_TRIAL"
	// ReasonGPUAssignmentUnknown — concurrent execution was requested without a
	// usable explicit GPU assignment for every arm.
	ReasonGPUAssignmentUnknown = "GPU_ASSIGNMENT_UNKNOWN"
	// ReasonGPUAssignmentDuplicate — two arms claim the same host-local GPU, so
	// launching them together could silently oversubscribe one device.
	ReasonGPUAssignmentDuplicate = "GPU_ASSIGNMENT_DUPLICATE"
)

// Validate refuses a manifest that is not fully pinned. Every check here exists
// because the missing field makes a published number unreproducible or
// unattributable; there is no "warn" tier.
func (m *Manifest) Validate() error {
	if m == nil {
		return refuse(ReasonManifestInvalid, "manifest is nil")
	}
	if m.Schema != ManifestSchema {
		return refuse(ReasonManifestInvalid, "schema %q is not %q", m.Schema, ManifestSchema)
	}
	if strings.TrimSpace(m.ID) == "" {
		return refuse(ReasonManifestInvalid, "id is empty")
	}
	if err := m.validateModel(); err != nil {
		return err
	}
	if strings.TrimSpace(m.Corpus.ID) == "" {
		return refuse(ReasonManifestInvalid, "corpus.id is empty")
	}
	if strings.TrimSpace(m.Corpus.Hash) == "" {
		return refuse(ReasonManifestInvalid, "corpus.hash is empty — an unhashed corpus cannot be pinned")
	}
	if !validSHA256(m.Corpus.Hash) {
		return refuse(ReasonManifestInvalid, "corpus.hash %q is not sha256:<64 lowercase hex>", m.Corpus.Hash)
	}
	if m.Corpus.TaskCount <= 0 {
		return refuse(ReasonManifestInvalid, "corpus.task_count is %d, want > 0", m.Corpus.TaskCount)
	}
	if strings.TrimSpace(m.Judge.ID) == "" {
		return refuse(ReasonManifestInvalid, "judge.id is empty")
	}
	if strings.TrimSpace(m.Judge.Hash) == "" {
		return refuse(ReasonManifestInvalid, "judge.hash is empty — an unhashed judge cannot be pinned")
	}
	if !validSHA256(m.Judge.Hash) {
		return refuse(ReasonManifestInvalid, "judge.hash %q is not sha256:<64 lowercase hex>", m.Judge.Hash)
	}
	if err := m.validateTrials(); err != nil {
		return err
	}
	for _, field := range []struct{ name, value string }{
		{"os", m.Environment.OS}, {"arch", m.Environment.Arch},
		{"host_class", m.Environment.HostClass}, {"fak_version", m.Environment.FakVersion},
	} {
		if strings.TrimSpace(field.value) == "" {
			return refuse(ReasonManifestInvalid, "environment.%s is empty", field.name)
		}
	}
	if !validDate(m.Environment.PricingDate) {
		return refuse(ReasonManifestInvalid, "environment.pricing_date %q is not a YYYY-MM-DD date", m.Environment.PricingDate)
	}
	if err := m.validateSources(); err != nil {
		return err
	}
	return m.validateArms()
}

func (m *Manifest) validateModel() error {
	if strings.TrimSpace(m.Model.Provider) == "" {
		return refuse(ReasonManifestInvalid, "model.provider is empty")
	}
	if strings.TrimSpace(m.Model.Snapshot) == "" {
		return refuse(ReasonManifestInvalid, "model.snapshot is empty — pin a dated snapshot, never a floating alias")
	}
	if strings.TrimSpace(m.Model.Region) == "" {
		return refuse(ReasonManifestInvalid, "model.region is empty")
	}
	if m.Model.MaxTokens <= 0 {
		return refuse(ReasonManifestInvalid, "model.max_tokens is %d, want > 0", m.Model.MaxTokens)
	}
	if math.IsNaN(m.Model.Sampling.Temperature) || math.IsInf(m.Model.Sampling.Temperature, 0) || m.Model.Sampling.Temperature < 0 {
		return refuse(ReasonManifestInvalid, "model.sampling.temperature is %v, want >= 0", m.Model.Sampling.Temperature)
	}
	if math.IsNaN(m.Model.Sampling.TopP) || math.IsInf(m.Model.Sampling.TopP, 0) || m.Model.Sampling.TopP <= 0 || m.Model.Sampling.TopP > 1 {
		return refuse(ReasonManifestInvalid, "model.sampling.top_p is %v, want within (0,1]", m.Model.Sampling.TopP)
	}
	return nil
}

func (m *Manifest) validateTrials() error {
	if m.Trials.Count <= 0 {
		return refuse(ReasonManifestInvalid, "trials.count is %d, want > 0", m.Trials.Count)
	}
	switch m.Trials.Order {
	case OrderCounterbalanced, OrderRandomized:
	default:
		return refuse(ReasonManifestInvalid, "trials.order %q is not one of %q/%q", m.Trials.Order, OrderCounterbalanced, OrderRandomized)
	}
	if m.Trials.Order == OrderRandomized && m.Trials.Seed == 0 {
		return refuse(ReasonManifestInvalid, "trials.order is %q but trials.seed is 0 — a randomized order with no seed is not reproducible", OrderRandomized)
	}
	if m.Trials.Concurrency <= 0 {
		return refuse(ReasonManifestInvalid, "trials.concurrency is %d, want > 0", m.Trials.Concurrency)
	}
	return nil
}

func (m *Manifest) validateSources() error {
	seen := map[string]bool{}
	for i, s := range m.Sources {
		switch {
		case strings.TrimSpace(s.Name) == "":
			return refuse(ReasonManifestInvalid, "sources[%d].name is empty", i)
		case strings.TrimSpace(s.Repo) == "":
			return refuse(ReasonManifestInvalid, "sources[%d] (%s).repo is empty", i, s.Name)
		case !validCommitSHA(s.SHA):
			return refuse(ReasonManifestInvalid, "sources[%d] (%s).sha %q is not a full 40-hex commit", i, s.Name, s.SHA)
		case strings.TrimSpace(s.Path) == "":
			return refuse(ReasonManifestInvalid, "sources[%d] (%s).path is empty", i, s.Name)
		case !validSHA256(s.ContentHash):
			return refuse(ReasonManifestInvalid, "sources[%d] (%s).content_hash %q is not sha256:<64 lowercase hex>", i, s.Name, s.ContentHash)
		case s.RetrievedAt != "" && !validDate(s.RetrievedAt):
			return refuse(ReasonManifestInvalid, "sources[%d] (%s).retrieved_at %q is not a YYYY-MM-DD date", i, s.Name, s.RetrievedAt)
		case seen[s.Name]:
			return refuse(ReasonManifestInvalid, "sources[%d].name %q is declared twice", i, s.Name)
		}
		seen[s.Name] = true
	}
	return nil
}

func (m *Manifest) validateArms() error {
	if len(m.Arms) < 2 {
		return refuse(ReasonManifestInvalid, "manifest declares %d arm(s); a comparison needs at least 2", len(m.Arms))
	}
	sources := map[string]bool{}
	for _, s := range m.Sources {
		sources[s.Name] = true
	}
	seen := map[string]bool{}
	devices := map[int]string{}
	baselines := 0
	for i, a := range m.Arms {
		if strings.TrimSpace(a.ID) == "" {
			return refuse(ReasonManifestInvalid, "arms[%d].id is empty", i)
		}
		if seen[a.ID] {
			return refuse(ReasonManifestInvalid, "arms[%d].id %q is declared twice", i, a.ID)
		}
		seen[a.ID] = true
		if a.GPUIndex != nil {
			if *a.GPUIndex < 0 {
				return refuse(ReasonGPUAssignmentUnknown, "arms[%d] (%s).gpu_index is %d, want a host-local index >= 0", i, a.ID, *a.GPUIndex)
			}
			if owner, exists := devices[*a.GPUIndex]; exists {
				return refuse(ReasonGPUAssignmentDuplicate, "arms[%d] (%s).gpu_index %d is already assigned to arm %s", i, a.ID, *a.GPUIndex, owner)
			}
			devices[*a.GPUIndex] = a.ID
		}
		if !validSHA256(a.PromptHash) {
			return refuse(ReasonManifestInvalid, "arms[%d] (%s).prompt_hash %q is not sha256:<64 lowercase hex>", i, a.ID, a.PromptHash)
		}
		if err := validateArmKind(i, a, sources); err != nil {
			return err
		}
		if a.Kind == ArmBaseline {
			baselines++
		}
	}
	if baselines != 1 {
		return refuse(ReasonManifestInvalid, "manifest declares %d %q arm(s); exactly 1 control is required", baselines, ArmBaseline)
	}
	return nil
}

// validateArmKind is where the bundled-capability fence lives. An arm that turns
// on two capabilities at once cannot attribute its delta to either, so it is
// refused as a class rather than reported with a caveat.
func validateArmKind(i int, a Arm, sources map[string]bool) error {
	switch a.Kind {
	case ArmFakCapability:
		switch {
		case len(a.Capabilities) == 0:
			return refuse(ReasonArmCapabilityUnnamed, "arms[%d] (%s) is kind %q but names no capability", i, a.ID, a.Kind)
		case len(a.Capabilities) > 1:
			return refuse(ReasonArmCapabilityBundled, "arms[%d] (%s) bundles %d capabilities %v — one arm isolates exactly one capability; bundle only after each has an isolated arm", i, a.ID, len(a.Capabilities), a.Capabilities)
		case strings.TrimSpace(a.Capabilities[0]) == "":
			return refuse(ReasonArmCapabilityUnnamed, "arms[%d] (%s) names an empty capability", i, a.ID)
		}
	case ArmBaseline, ArmFakPassthrough, ArmUpstreamTreatment:
		if len(a.Capabilities) != 0 {
			return refuse(ReasonArmCapabilityUnnamed, "arms[%d] (%s) is kind %q but names capabilities %v — only %q arms enable a capability", i, a.ID, a.Kind, a.Capabilities, ArmFakCapability)
		}
	default:
		return refuse(ReasonManifestInvalid, "arms[%d] (%s) kind %q is outside the closed vocabulary %v", i, a.ID, a.Kind, KnownArmKinds())
	}
	if a.Kind == ArmUpstreamTreatment {
		if strings.TrimSpace(a.SourceName) == "" {
			return refuse(ReasonManifestInvalid, "arms[%d] (%s) is kind %q but binds no source_name — an unpinned treatment is a hand-copied approximation", i, a.ID, a.Kind)
		}
	}
	if a.SourceName != "" && !sources[a.SourceName] {
		return refuse(ReasonManifestInvalid, "arms[%d] (%s).source_name %q is not declared in sources", i, a.ID, a.SourceName)
	}
	return nil
}

// Identity is the sha256 over the canonical encoding of the identity-bearing
// terms. Two manifests share an identity exactly when they describe the same
// measurement; a changed model, prompt, judge, corpus, or arm capability moves
// it (proven by Selfcheck). Scheduling and environment are included too because
// they can move measured latency/cost even when prompts and model stay fixed.
func (m *Manifest) Identity() string {
	sum := sha256.Sum256([]byte(m.canonical()))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// canonical renders the identity-bearing terms as a deterministic line stream.
// It is hand-rolled rather than JSON-marshalled so a future struct-field
// reordering, an added non-identity field, or a map iteration cannot silently
// change every published identity.
func (m *Manifest) canonical() string {
	var b strings.Builder
	for _, f := range m.identityFields() {
		// Length-prefix BOTH terms. Separator escaping is easy to get subtly
		// wrong; a byte count makes arbitrary field content unambiguous.
		fmt.Fprintf(&b, "%d:%s%d:%s", len(f.Field), f.Field, len(f.A), f.A)
	}
	return b.String()
}

// identityFields is the single source for both hashing and comparability. Every
// execution/provenance term recorded by the immutable manifest is included;
// only display-only ID/notes and source license/retrieval annotations are not.
func (m *Manifest) identityFields() []ComparabilityField {
	fields := []ComparabilityField{}
	add := func(name, value string) { fields = append(fields, ComparabilityField{Field: name, A: value}) }
	add("schema", m.Schema)
	add("model.provider", m.Model.Provider)
	add("model.snapshot", m.Model.Snapshot)
	add("model.region", m.Model.Region)
	add("model.max_tokens", strconv.Itoa(m.Model.MaxTokens))
	add("model.sampling.temperature", strconv.FormatFloat(m.Model.Sampling.Temperature, 'g', -1, 64))
	add("model.sampling.top_p", strconv.FormatFloat(m.Model.Sampling.TopP, 'g', -1, 64))
	add("model.sampling.seed", strconv.FormatInt(m.Model.Sampling.Seed, 10))
	add("corpus.id", m.Corpus.ID)
	add("corpus.hash", m.Corpus.Hash)
	add("corpus.task_count", strconv.Itoa(m.Corpus.TaskCount))
	add("judge.id", m.Judge.ID)
	add("judge.hash", m.Judge.Hash)
	add("judge.kind", m.Judge.Kind)
	add("trials.count", strconv.Itoa(m.Trials.Count))
	add("trials.seed", strconv.FormatInt(m.Trials.Seed, 10))
	add("trials.order", string(m.Trials.Order))
	add("trials.concurrency", strconv.Itoa(m.Trials.Concurrency))
	add("environment.os", m.Environment.OS)
	add("environment.arch", m.Environment.Arch)
	add("environment.host_class", m.Environment.HostClass)
	add("environment.fak_version", m.Environment.FakVersion)
	add("environment.pricing_date", m.Environment.PricingDate)

	srcs := append([]Source(nil), m.Sources...)
	sort.Slice(srcs, func(i, j int) bool { return srcs[i].Name < srcs[j].Name })
	add("sources.count", strconv.Itoa(len(srcs)))
	for _, s := range srcs {
		prefix := "sources[" + strconv.Quote(s.Name) + "]."
		add(prefix+"repo", s.Repo)
		add(prefix+"sha", s.SHA)
		add(prefix+"path", s.Path)
		add(prefix+"content_hash", s.ContentHash)
	}

	arms := append([]Arm(nil), m.Arms...)
	sort.Slice(arms, func(i, j int) bool { return arms[i].ID < arms[j].ID })
	add("arms.count", strconv.Itoa(len(arms)))
	for _, a := range arms {
		prefix := "arms[" + strconv.Quote(a.ID) + "]."
		add(prefix+"kind", string(a.Kind))
		add(prefix+"prompt_hash", a.PromptHash)
		add(prefix+"source_name", a.SourceName)
		// Omitted assignments are deliberately absent from the canonical stream,
		// preserving every manifest/1 identity published before gpu_index existed.
		if a.GPUIndex != nil {
			add(prefix+"gpu_index", strconv.Itoa(*a.GPUIndex))
		}
		caps := append([]string(nil), a.Capabilities...)
		sort.Strings(caps)
		add(prefix+"capabilities", strings.Join(caps, ","))
	}
	return fields
}

// ComparabilityField names one term two manifests disagreed on.
type ComparabilityField struct {
	Field string `json:"field"`
	A     string `json:"a"`
	B     string `json:"b"`
}

// CheckComparable refuses two manifests whose reports must not be treated as
// repeat measurements of one experiment. Arms are compared WITHIN one run; a
// changed arm contract, model, corpus, schedule, or environment is a different
// manifest and the returned field list makes that drift explicit.
//
// It reports every disagreeing field rather than the first, because the operator
// fixing a drifted manifest wants the whole list in one pass.
func CheckComparable(a, b *Manifest) ([]ComparabilityField, error) {
	if a == nil || b == nil {
		return nil, refuse(ReasonIncomparableManifest, "a nil manifest is comparable to nothing")
	}
	if err := a.Validate(); err != nil {
		return nil, refuse(ReasonIncomparableManifest, "manifest a is invalid: %v", err)
	}
	if err := b.Validate(); err != nil {
		return nil, refuse(ReasonIncomparableManifest, "manifest b is invalid: %v", err)
	}
	af, bf := a.identityFields(), b.identityFields()
	bByName := make(map[string]string, len(bf))
	for _, f := range bf {
		bByName[f.Field] = f.A
	}
	aByName := make(map[string]string, len(af))
	for _, f := range af {
		aByName[f.Field] = f.A
	}
	cmp := []ComparabilityField{}
	for _, f := range af {
		bv, ok := bByName[f.Field]
		if !ok {
			bv = "<missing>"
		}
		if f.A != bv {
			cmp = append(cmp, ComparabilityField{Field: f.Field, A: f.A, B: bv})
		}
	}
	for _, f := range bf {
		if _, ok := aByName[f.Field]; !ok {
			cmp = append(cmp, ComparabilityField{Field: f.Field, A: "<missing>", B: f.A})
		}
	}
	if len(cmp) > 0 {
		names := make([]string, 0, len(cmp))
		for _, c := range cmp {
			names = append(names, fmt.Sprintf("%s (%q vs %q)", c.Field, c.A, c.B))
		}
		return cmp, refuse(ReasonIncomparableManifest, "%d term(s) decide what was measured and disagree: %s", len(cmp), strings.Join(names, "; "))
	}
	return nil, nil
}

func validCommitSHA(s string) bool {
	if len(s) != 40 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil && strings.ToLower(s) == s
}

func validSHA256(s string) bool {
	if len(s) != len("sha256:")+64 || !strings.HasPrefix(s, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(s, "sha256:"))
	return err == nil && strings.ToLower(s) == s
}

func validDate(s string) bool {
	if len(s) != len("2006-01-02") {
		return false
	}
	_, err := time.Parse("2006-01-02", s)
	return err == nil
}

// ArmByID returns the declared arm with the given id.
func (m *Manifest) ArmByID(id string) (Arm, bool) {
	for _, a := range m.Arms {
		if a.ID == id {
			return a, true
		}
	}
	return Arm{}, false
}
