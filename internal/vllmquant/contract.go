package vllmquant

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// SchemaVersion is the only request schema this leaf reads. A request carrying
// any other schema abstains rather than being interpreted under assumptions the
// producer never agreed to.
const SchemaVersion = "vllmquant/v1"

// MethodUnknown is what a selection reports for artifact_method when the
// artifact never named a method this contract knows. It is a label, never a
// Method: nothing downstream can mistake it for a servable method.
const MethodUnknown = "unknown"

// Outcome is the typed verdict for one adjudication. Every input lands on
// exactly one of these; there is no silent fallback and no untyped "maybe".
//
//   - OutcomeSupported: exactly one advertised kernel is admissible (or a
//     declared preference resolved the tie) and the returned Args are licensed.
//   - OutcomeDelegate: more than one kernel is admissible — or the server said
//     it resolves the kernel itself — so the candidates are reported and no
//     kernel is named here.
//   - OutcomeUnsupported: the evidence is well formed, and this build on this
//     device cannot serve the artifact. Every kernel that could have served the
//     method is in Excluded with the reason it was dropped.
//   - OutcomeAbstain: the evidence does not contain an answer (something is
//     undeclared or outside this contract's vocabulary). Not a refusal — a
//     missing witness.
//   - OutcomeRefuse: the evidence contradicts itself or is malformed (fp8 at 4
//     bits, a group size of 0, a compute capability of "eight").
//
// The unsupported/refuse split is load bearing: "your build cannot serve this"
// is a fact about the build that a caller can act on by changing the build,
// while "this declaration is malformed" is a fact about the input that can only
// be fixed at the producer.
type Outcome string

const (
	OutcomeSupported   Outcome = "supported"
	OutcomeDelegate    Outcome = "delegate"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeAbstain     Outcome = "abstain"
	OutcomeRefuse      Outcome = "refuse"
)

// Known reports whether o is one of the five published outcomes, so a caller can
// reject a corrupt serialized selection instead of switching on a typo.
func (o Outcome) Known() bool {
	switch o {
	case OutcomeSupported, OutcomeDelegate, OutcomeUnsupported, OutcomeAbstain, OutcomeRefuse:
		return true
	}
	return false
}

// Claim names what a selection actually licenses. It is deliberately narrow, and
// it is a ladder rather than a label per outcome:
//
//   - ClaimNone: nothing was understood about the artifact, so nothing is
//     claimed about anything.
//   - ClaimArtifactDescribed: the artifact's own declaration was read and is
//     coherent; it was the server/device side that produced no launch recipe.
//   - ClaimRuntimeDelegated: candidate kernels were enumerated from declared
//     evidence. It stays "delegated" even for a single supported candidate,
//     because vLLM still resolves the concrete kernel at load time — passing
//     --quantization gptq on Ampere may still run the marlin kernel. This leaf
//     licenses the launch arguments, never the kernel that ultimately runs.
type Claim string

const (
	ClaimNone              Claim = ""
	ClaimArtifactDescribed Claim = "artifact-described"
	ClaimRuntimeDelegated  Claim = "runtime-delegated"
)

// Reason is a stable, routable code. The vocabulary is closed: every code a
// caller can receive is published by Known, and every one carries the leaf
// prefix so it stays greppable in a mixed decision log.
type Reason string

const (
	ReasonAdmitted                   Reason = "VLLMQUANT_ADMITTED"
	ReasonKernelChoiceDelegated      Reason = "VLLMQUANT_KERNEL_CHOICE_DELEGATED"
	ReasonSelectedByServerPreference Reason = "VLLMQUANT_SELECTED_BY_SERVER_PREFERENCE"

	ReasonSchemaUnknown Reason = "VLLMQUANT_SCHEMA_UNKNOWN"
	ReasonInvalidJSON   Reason = "VLLMQUANT_INVALID_JSON"

	ReasonArtifactUndeclared       Reason = "VLLMQUANT_ARTIFACT_UNDECLARED"
	ReasonMethodUnknown            Reason = "VLLMQUANT_METHOD_UNKNOWN"
	ReasonMethodBitsConflict       Reason = "VLLMQUANT_METHOD_BITS_CONFLICT"
	ReasonWeightBitsUndeclared     Reason = "VLLMQUANT_WEIGHT_BITS_UNDECLARED"
	ReasonWeightBitsUnsupported    Reason = "VLLMQUANT_WEIGHT_BITS_UNSUPPORTED"
	ReasonGroupSizeUndeclared      Reason = "VLLMQUANT_GROUP_SIZE_UNDECLARED"
	ReasonGroupSizeMalformed       Reason = "VLLMQUANT_GROUP_SIZE_MALFORMED"
	ReasonGroupSizeUnsupported     Reason = "VLLMQUANT_GROUP_SIZE_UNSUPPORTED"
	ReasonSymmetryUndeclared       Reason = "VLLMQUANT_SYMMETRY_UNDECLARED"
	ReasonAsymmetricUnsupported    Reason = "VLLMQUANT_ASYMMETRIC_UNSUPPORTED"
	ReasonMarlinRequirementUnmet   Reason = "VLLMQUANT_MARLIN_REQUIREMENT_UNMET"
	ReasonActivationSchemeUnknown  Reason = "VLLMQUANT_ACTIVATION_SCHEME_UNKNOWN"
	ReasonKVCacheDtypeUnknown      Reason = "VLLMQUANT_KV_CACHE_DTYPE_UNKNOWN"
	ReasonCheckpointFormatUnknown  Reason = "VLLMQUANT_CHECKPOINT_FORMAT_UNKNOWN"
	ReasonCheckpointFormatRepacked Reason = "VLLMQUANT_CHECKPOINT_FORMAT_REPACKED"

	ReasonServerUndeclared         Reason = "VLLMQUANT_SERVER_UNDECLARED"
	ReasonVersionUnknown           Reason = "VLLMQUANT_SERVER_VERSION_UNKNOWN"
	ReasonVersionBelowMinimum      Reason = "VLLMQUANT_SERVER_VERSION_BELOW_MINIMUM"
	ReasonNoKernelAdvertised       Reason = "VLLMQUANT_NO_KERNEL_ADVERTISED"
	ReasonKernelUnknown            Reason = "VLLMQUANT_KERNEL_UNKNOWN"
	ReasonMethodNotInBuild         Reason = "VLLMQUANT_METHOD_NOT_IN_BUILD"
	ReasonCapabilityMalformed      Reason = "VLLMQUANT_CAPABILITY_MALFORMED"
	ReasonComputeCapabilityUnknown Reason = "VLLMQUANT_CAPABILITY_UNKNOWN"
	ReasonComputeCapabilityTooLow  Reason = "VLLMQUANT_CAPABILITY_TOO_LOW"
)

var publishedReasons = map[Reason]bool{
	ReasonAdmitted:                   true,
	ReasonKernelChoiceDelegated:      true,
	ReasonSelectedByServerPreference: true,
	ReasonSchemaUnknown:              true,
	ReasonInvalidJSON:                true,
	ReasonArtifactUndeclared:         true,
	ReasonMethodUnknown:              true,
	ReasonMethodBitsConflict:         true,
	ReasonWeightBitsUndeclared:       true,
	ReasonWeightBitsUnsupported:      true,
	ReasonGroupSizeUndeclared:        true,
	ReasonGroupSizeMalformed:         true,
	ReasonGroupSizeUnsupported:       true,
	ReasonSymmetryUndeclared:         true,
	ReasonAsymmetricUnsupported:      true,
	ReasonMarlinRequirementUnmet:     true,
	ReasonActivationSchemeUnknown:    true,
	ReasonKVCacheDtypeUnknown:        true,
	ReasonCheckpointFormatUnknown:    true,
	ReasonCheckpointFormatRepacked:   true,
	ReasonServerUndeclared:           true,
	ReasonVersionUnknown:             true,
	ReasonVersionBelowMinimum:        true,
	ReasonNoKernelAdvertised:         true,
	ReasonKernelUnknown:              true,
	ReasonMethodNotInBuild:           true,
	ReasonCapabilityMalformed:        true,
	ReasonComputeCapabilityUnknown:   true,
	ReasonComputeCapabilityTooLow:    true,
}

// Known reports whether r is in the published vocabulary. A code a caller can
// receive but cannot look up is not routable, so nothing outside this set is
// ever emitted.
func (r Reason) Known() bool { return publishedReasons[r] }

// PublishedReasons returns the closed reason vocabulary in sorted order, so a
// caller (or a doc generator) can enumerate what it may have to route.
func PublishedReasons() []Reason {
	out := make([]Reason, 0, len(publishedReasons))
	for r := range publishedReasons {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Method is the quantization method a checkpoint declares for itself.
type Method string

const (
	MethodAWQ          Method = "awq"
	MethodGPTQ         Method = "gptq"
	MethodFP8          Method = "fp8"
	MethodBitsAndBytes Method = "bitsandbytes"
)

// Kernel is a compiled vLLM quantization kernel a server build advertises.
// Several kernels can serve one method; that is exactly the ambiguity this leaf
// refuses to resolve by itself.
type Kernel string

const (
	KernelAWQ          Kernel = "awq"
	KernelAWQMarlin    Kernel = "awq_marlin"
	KernelGPTQ         Kernel = "gptq"
	KernelGPTQMarlin   Kernel = "gptq_marlin"
	KernelFP8          Kernel = "fp8"
	KernelBitsAndBytes Kernel = "bitsandbytes"
)

// Capability is a CUDA compute capability in packed form: 8.0 is 80, 7.5 is 75.
// A producer may declare it either the way vLLM prints it ("8.0") or already
// packed (80); both read to the same value. The two sentinels keep "nobody said"
// separate from "somebody said something unreadable", because those are an
// abstain and a refusal respectively.
type Capability int

const (
	CapabilityUndeclared Capability = 0
	CapabilityMalformed  Capability = -1
)

// UnmarshalJSON never fails: an unreadable capability becomes CapabilityMalformed
// so the adjudication can refuse it with a typed reason instead of the caller
// having to string-match a json.SyntaxError.
func (c *Capability) UnmarshalJSON(b []byte) error {
	trimmed := strings.TrimSpace(string(b))
	if trimmed == "null" || trimmed == "" {
		*c = CapabilityUndeclared
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			*c = CapabilityMalformed
			return nil
		}
		*c = parseCapability(s)
		return nil
	}
	var n json.Number
	if err := json.Unmarshal(b, &n); err != nil {
		*c = CapabilityMalformed
		return nil
	}
	packed, err := strconv.Atoi(n.String())
	if err != nil || packed < 0 {
		*c = CapabilityMalformed
		return nil
	}
	*c = Capability(packed)
	return nil
}

// MarshalJSON emits the dotted form vLLM itself reports, so a round trip through
// this contract does not silently re-key the producer's own vocabulary.
func (c Capability) MarshalJSON() ([]byte, error) {
	switch {
	case c == CapabilityUndeclared:
		return []byte(`""`), nil
	case c < 0:
		return []byte(`"malformed"`), nil
	}
	return json.Marshal(strconv.Itoa(int(c)/10) + "." + strconv.Itoa(int(c)%10))
}

// parseCapability reads "8.0" (major.minor) or "80" (already packed). Anything
// else is malformed rather than coerced into the nearest number.
func parseCapability(s string) Capability {
	s = strings.TrimSpace(s)
	if s == "" {
		return CapabilityUndeclared
	}
	major, minor, dotted := strings.Cut(s, ".")
	if !dotted {
		packed, err := strconv.Atoi(s)
		if err != nil || packed < 0 {
			return CapabilityMalformed
		}
		return Capability(packed)
	}
	maj, err := strconv.Atoi(major)
	if err != nil || maj < 0 {
		return CapabilityMalformed
	}
	min, err := strconv.Atoi(minor)
	if err != nil || min < 0 || min > 9 {
		return CapabilityMalformed
	}
	return Capability(maj*10 + min)
}

// Artifact is the producer's own checkpoint metadata, read as-is: the field
// names are the ones a Hugging Face quantization_config actually carries (bits,
// group_size, sym), with this contract's older spellings accepted as aliases.
// A zero field means "the producer did not declare it" — never "the usual
// default". GroupSize is a pointer because an explicitly declared 0 is a
// malformed grid value, which is not the same fact as no grid at all.
type Artifact struct {
	Model            string `json:"model,omitempty"`
	QuantMethod      Method `json:"quant_method,omitempty"`
	WeightBits       int    `json:"bits,omitempty"`
	GroupSize        *int   `json:"group_size,omitempty"`
	Symmetric        *bool  `json:"sym,omitempty"`
	ActivationScheme string `json:"activation_scheme,omitempty"`
	KVCacheDtype     string `json:"kv_cache_dtype,omitempty"`
	CheckpointFormat string `json:"checkpoint_format,omitempty"`
}

// UnmarshalJSON accepts both spellings of the two renamed fields: bits/weight_bits
// and sym/symmetric. A producer's own key wins; the alias only fills a field the
// producer left unset.
func (a *Artifact) UnmarshalJSON(b []byte) error {
	var raw struct {
		Model            string `json:"model"`
		QuantMethod      Method `json:"quant_method"`
		Bits             *int   `json:"bits"`
		WeightBits       *int   `json:"weight_bits"`
		GroupSize        *int   `json:"group_size"`
		Sym              *bool  `json:"sym"`
		Symmetric        *bool  `json:"symmetric"`
		ActivationScheme string `json:"activation_scheme"`
		KVCacheDtype     string `json:"kv_cache_dtype"`
		CheckpointFormat string `json:"checkpoint_format"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*a = Artifact{
		Model:            raw.Model,
		QuantMethod:      raw.QuantMethod,
		GroupSize:        raw.GroupSize,
		Symmetric:        firstBool(raw.Sym, raw.Symmetric),
		ActivationScheme: raw.ActivationScheme,
		KVCacheDtype:     raw.KVCacheDtype,
		CheckpointFormat: raw.CheckpointFormat,
	}
	if bits := firstInt(raw.Bits, raw.WeightBits); bits != nil {
		a.WeightBits = *bits
	}
	return nil
}

// Server is what one vLLM build advertises about itself plus the device it runs
// on. Kernels is the compiled kernel set, in the build's own order; a build
// reports it as "methods", and this contract's older "kernels" spelling is
// accepted as an alias.
//
// KernelOrderIsPreference and RuntimeSelects are the only two ways a tie between
// admissible kernels is ever broken in favour of one kernel. With neither set,
// this leaf delegates the choice rather than ranking kernels by an opinion it
// does not own.
type Server struct {
	Version                 string     `json:"version,omitempty"`
	Kernels                 []Kernel   `json:"methods,omitempty"`
	ComputeCapability       Capability `json:"compute_capability,omitempty"`
	Dtype                   string     `json:"dtype,omitempty"`
	KernelOrderIsPreference bool       `json:"kernel_order_is_preference,omitempty"`
	RuntimeSelects          bool       `json:"runtime_selects,omitempty"`
}

// UnmarshalJSON accepts methods/kernels for the advertised kernel set.
func (s *Server) UnmarshalJSON(b []byte) error {
	var raw struct {
		Version                 string     `json:"version"`
		Methods                 []Kernel   `json:"methods"`
		Kernels                 []Kernel   `json:"kernels"`
		ComputeCapability       Capability `json:"compute_capability"`
		Dtype                   string     `json:"dtype"`
		KernelOrderIsPreference bool       `json:"kernel_order_is_preference"`
		RuntimeSelects          bool       `json:"runtime_selects"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	kernels := raw.Methods
	if len(kernels) == 0 {
		kernels = raw.Kernels
	}
	*s = Server{
		Version:                 raw.Version,
		Kernels:                 kernels,
		ComputeCapability:       raw.ComputeCapability,
		Dtype:                   raw.Dtype,
		KernelOrderIsPreference: raw.KernelOrderIsPreference,
		RuntimeSelects:          raw.RuntimeSelects,
	}
	return nil
}

// declared reports whether the producer said anything at all about the server.
// A wholly silent server is one fact ("nobody described the build"), not a pile
// of independent missing fields.
func (s Server) declared() bool {
	return strings.TrimSpace(s.Version) != "" || len(s.Kernels) > 0 ||
		s.ComputeCapability != CapabilityUndeclared || s.Dtype != "" ||
		s.KernelOrderIsPreference || s.RuntimeSelects
}

// advertises reports whether the build compiled kernel k.
func (s Server) advertises(k Kernel) bool {
	for _, got := range s.Kernels {
		if got == k {
			return true
		}
	}
	return false
}

// Request is one adjudication input: a schema tag plus the two independent
// evidence sources.
type Request struct {
	Schema   string   `json:"schema"`
	Artifact Artifact `json:"artifact"`
	Server   Server   `json:"server"`
}

// Candidate is one admissible kernel and the launch arguments that would pick
// it. Every candidate carries its own arguments, so a caller that has its own
// preference does not have to re-derive them.
type Candidate struct {
	Method Kernel   `json:"method"`
	Args   []string `json:"args"`
}

// Exclusion is one kernel that could have served the artifact's method and the
// single reason it was dropped. The excluded set is the audit trail behind an
// unsupported verdict: it names what the build would have needed.
type Exclusion struct {
	Method Kernel `json:"method"`
	Reason Reason `json:"reason"`
}

// Selection is the typed result. Every field is always present on the wire —
// an empty candidate list serializes as [] rather than null — so a consumer
// never has to distinguish "absent" from "empty".
type Selection struct {
	Outcome        Outcome     `json:"outcome"`
	Reasons        []Reason    `json:"reasons"`
	Claim          Claim       `json:"claim"`
	ArtifactMethod string      `json:"artifact_method"`
	Candidates     []Candidate `json:"candidates"`
	Excluded       []Exclusion `json:"excluded"`
	Args           []string    `json:"args"`
}

// HasReason reports whether r was emitted, so a caller routes on a code rather
// than on the position of a reason in the slice.
func (s Selection) HasReason(r Reason) bool {
	for _, got := range s.Reasons {
		if got == r {
			return true
		}
	}
	return false
}

// Kernel is the single kernel a supported selection resolved to. It is derived
// rather than stored: only a supported outcome names a kernel, and it is always
// the head of the candidate list (the sole admissible kernel, or the winner of
// the server's own declared order). Delegate reports candidates and no kernel.
func (s Selection) Kernel() Kernel {
	if s.Outcome != OutcomeSupported || len(s.Candidates) == 0 {
		return ""
	}
	return s.Candidates[0].Method
}

// CandidateMethods returns just the admissible kernels, in selection order.
func (s Selection) CandidateMethods() []Kernel {
	out := make([]Kernel, 0, len(s.Candidates))
	for _, c := range s.Candidates {
		out = append(out, c.Method)
	}
	return out
}

// ExcludedReason returns the reason kernel k was dropped, if it was.
func (s Selection) ExcludedReason(k Kernel) (Reason, bool) {
	for _, e := range s.Excluded {
		if e.Method == k {
			return e.Reason, true
		}
	}
	return "", false
}

// methodRule is what the METHOD itself constrains, independent of any kernel: a
// declaration that violates it is a self-contradictory artifact.
//
// weightBits with a single entry is a width the method entails (awq is 4-bit,
// fp8 is 8-bit by construction), so an artifact may leave `bits` undeclared;
// a method with several legal widths must declare which one it used.
type methodRule struct {
	weightBits               []int
	requiresGroupSize        bool
	requiresActivationScheme bool
	kernels                  []Kernel // the family, in this contract's report order
}

var methodTable = map[Method]methodRule{
	//enumlint:exempt Each method lists only its compatible kernel family, not every kernel in the global vocabulary.
	MethodAWQ:          {weightBits: []int{4}, requiresGroupSize: true, kernels: []Kernel{KernelAWQ, KernelAWQMarlin}},
	MethodGPTQ:         {weightBits: []int{2, 3, 4, 8}, requiresGroupSize: true, kernels: []Kernel{KernelGPTQ, KernelGPTQMarlin}},
	MethodFP8:          {weightBits: []int{8}, requiresActivationScheme: true, kernels: []Kernel{KernelFP8}},
	MethodBitsAndBytes: {weightBits: []int{4, 8}, kernels: []Kernel{KernelBitsAndBytes}},
}

// kernelRule is one kernel's admission threshold. The version and
// compute-capability floors are this contract's own conservative thresholds
// under SchemaVersion — data, not a claim about the exact upstream release a
// kernel first appeared in.
type kernelRule struct {
	name              Kernel // filled in when a rule is carried to argument building
	method            Method
	minVersion        version
	minCapability     Capability
	weightBits        []int
	requiresSymmetric bool     // needs symmetry DECLARED, and declared symmetric
	consumesRepacked  bool     // can consume a marlin-repacked checkpoint
	marlin            bool     // a marlin kernel: its extra requirements report as one code
	extraArgs         []string // launch arguments this kernel additionally needs
}

var kernelTable = map[Kernel]kernelRule{
	KernelAWQ:          {method: MethodAWQ, minVersion: version{0, 2, 0}, minCapability: 75, weightBits: []int{4}},
	KernelAWQMarlin:    {method: MethodAWQ, minVersion: version{0, 5, 0}, minCapability: 80, weightBits: []int{4}, consumesRepacked: true, marlin: true},
	KernelGPTQ:         {method: MethodGPTQ, minVersion: version{0, 2, 0}, minCapability: 60, weightBits: []int{2, 3, 4, 8}},
	KernelGPTQMarlin:   {method: MethodGPTQ, minVersion: version{0, 4, 0}, minCapability: 80, weightBits: []int{4, 8}, requiresSymmetric: true, consumesRepacked: true, marlin: true},
	KernelFP8:          {method: MethodFP8, minVersion: version{0, 5, 0}, minCapability: 89, weightBits: []int{8}},
	KernelBitsAndBytes: {method: MethodBitsAndBytes, minVersion: version{0, 5, 0}, minCapability: 70, weightBits: []int{4, 8}, extraArgs: []string{"--load-format", "bitsandbytes"}},
}

// groupSizes is the declared group-size grid. -1 is the producer's explicit
// "no grouping"; a declared 0 is not a grid value at all, and absence abstains
// rather than being read as either.
var groupSizes = []int{-1, 32, 64, 128}

var (
	activationSchemes = map[string]bool{"static": true, "dynamic": true}
	kvCacheDtypes     = map[string]bool{"auto": true, "fp8": true, "fp8_e4m3": true, "fp8_e5m2": true}
	checkpointFormats = map[string]bool{"awq": true, "gptq": true, "marlin": true}
)

// The release window this contract's kernel data actually describes. A build
// outside it is reported unknown rather than judged against thresholds that were
// never written for it — the ceiling is why a 1.0.0 server abstains instead of
// being silently admitted under 0.x rules.
var (
	minKnownVersion = version{0, 2, 0}
	maxKnownVersion = version{1, 0, 0} // exclusive
)

// severity separates "the evidence is missing" from "the evidence rules the
// kernel out". It is what decides abstain vs unsupported when the candidate set
// empties out.
type severity int

const (
	sevNone severity = iota
	sevSoft          // undeclared: a missing witness, so abstain
	sevHard          // inadmissible: this build cannot serve it, so unsupported
)

// admit reports whether one kernel may serve this artifact on this build, and
// the first reason it may not.
func (kr kernelRule) admit(a Artifact, bits int, sv version, capability Capability) (Reason, severity) {
	if sv.below(kr.minVersion) {
		return ReasonVersionBelowMinimum, sevHard
	}
	if capability < kr.minCapability {
		return ReasonComputeCapabilityTooLow, sevHard
	}
	if !containsInt(kr.weightBits, bits) {
		if kr.marlin {
			return ReasonMarlinRequirementUnmet, sevHard
		}
		return ReasonWeightBitsUnsupported, sevHard
	}
	if a.CheckpointFormat == "marlin" && !kr.consumesRepacked {
		return ReasonCheckpointFormatRepacked, sevHard
	}
	if kr.requiresSymmetric {
		if a.Symmetric == nil {
			return ReasonSymmetryUndeclared, sevSoft
		}
		if !*a.Symmetric {
			return ReasonAsymmetricUnsupported, sevHard
		}
	}
	return "", sevNone
}

// Invariant: vLLM quantization kernel selection is fail-closed and deterministic.
// No winner is picked among ambiguous kernels without explicit server preference.
//
// Guard: undeclared configurations, unknown methods, or contradictory descriptor
// parameters immediately reject admission with typed reasons rather than guessing.
//
// Adjudicate decides which advertised kernel may serve the artifact, or reports
// precisely why no answer is available. It never ranks kernels: a tie between
// admissible kernels is broken only by Server.KernelOrderIsPreference or
// Server.RuntimeSelects, and otherwise delegates with the candidate set.
func Adjudicate(r Request) Selection {
	if r.Schema != SchemaVersion {
		return verdict(OutcomeAbstain, MethodUnknown, ClaimNone, ReasonSchemaUnknown)
	}
	a, s := r.Artifact, r.Server

	// --- what the artifact declares about itself -------------------------
	if strings.TrimSpace(string(a.QuantMethod)) == "" {
		return verdict(OutcomeAbstain, MethodUnknown, ClaimNone, ReasonArtifactUndeclared)
	}
	mr, known := methodTable[a.QuantMethod]
	if !known {
		return verdict(OutcomeAbstain, MethodUnknown, ClaimNone, ReasonMethodUnknown)
	}
	method := string(a.QuantMethod)

	bits := a.WeightBits
	switch {
	case bits == 0 && len(mr.weightBits) == 1:
		bits = mr.weightBits[0] // the method entails exactly one width
	case bits == 0:
		return verdict(OutcomeAbstain, method, ClaimNone, ReasonWeightBitsUndeclared)
	case !containsInt(mr.weightBits, bits):
		return verdict(OutcomeRefuse, method, ClaimNone, ReasonMethodBitsConflict)
	}
	if mr.requiresGroupSize {
		switch {
		case a.GroupSize == nil:
			return verdict(OutcomeAbstain, method, ClaimNone, ReasonGroupSizeUndeclared)
		case *a.GroupSize == 0:
			return verdict(OutcomeRefuse, method, ClaimNone, ReasonGroupSizeMalformed)
		case !containsInt(groupSizes, *a.GroupSize):
			return verdict(OutcomeRefuse, method, ClaimNone, ReasonGroupSizeUnsupported)
		}
	}
	// A declared-but-unreadable value is reported before a missing one: the
	// producer said something wrong, which is the more actionable fact.
	if a.KVCacheDtype != "" && !kvCacheDtypes[a.KVCacheDtype] {
		return verdict(OutcomeAbstain, method, ClaimNone, ReasonKVCacheDtypeUnknown)
	}
	if a.CheckpointFormat != "" && !checkpointFormats[a.CheckpointFormat] {
		return verdict(OutcomeAbstain, method, ClaimNone, ReasonCheckpointFormatUnknown)
	}
	if mr.requiresActivationScheme && !activationSchemes[a.ActivationScheme] {
		return verdict(OutcomeAbstain, method, ClaimNone, ReasonActivationSchemeUnknown)
	}

	// --- what the build and the device advertise --------------------------
	// From here the artifact itself was read and is coherent, so every verdict
	// carries ClaimArtifactDescribed: it was the build side that fell short.
	if !s.declared() {
		return verdict(OutcomeAbstain, method, ClaimArtifactDescribed, ReasonServerUndeclared)
	}
	sv, ok := parseVersion(s.Version)
	if !ok || sv.below(minKnownVersion) || !sv.below(maxKnownVersion) {
		return verdict(OutcomeAbstain, method, ClaimArtifactDescribed, ReasonVersionUnknown)
	}
	if len(s.Kernels) == 0 {
		return verdict(OutcomeRefuse, method, ClaimArtifactDescribed, ReasonNoKernelAdvertised)
	}
	for _, k := range s.Kernels {
		if _, ok := kernelTable[k]; !ok {
			// An unrecognized kernel is not guessed into the nearest known
			// shape, and it may be the very one that would have served.
			return verdict(OutcomeAbstain, method, ClaimArtifactDescribed, ReasonKernelUnknown)
		}
	}
	switch {
	case s.ComputeCapability == CapabilityMalformed:
		return verdict(OutcomeRefuse, method, ClaimArtifactDescribed, ReasonCapabilityMalformed)
	case s.ComputeCapability <= CapabilityUndeclared:
		return verdict(OutcomeAbstain, method, ClaimArtifactDescribed, ReasonComputeCapabilityUnknown)
	}

	// --- the admissible set, over the method's whole kernel family --------
	// The family is walked (not just the build's list) so that a kernel the
	// build never compiled is reported as excluded rather than silently absent.
	var (
		candidates []Kernel
		excluded   []Exclusion
		anySoft    bool
	)
	for _, k := range mr.kernels {
		if !s.advertises(k) {
			excluded = append(excluded, Exclusion{Method: k, Reason: ReasonMethodNotInBuild})
			continue
		}
		reason, sev := kernelTable[k].admit(a, bits, sv, s.ComputeCapability)
		switch sev {
		case sevNone:
			candidates = append(candidates, k)
		case sevSoft:
			excluded = append(excluded, Exclusion{Method: k, Reason: reason})
			anySoft = true
		default:
			excluded = append(excluded, Exclusion{Method: k, Reason: reason})
		}
	}
	if len(candidates) == 0 {
		// A missing witness dominates: if even one kernel was dropped only
		// because something was never declared, the set is not provably empty
		// and "this build cannot serve it" is not yet a fact.
		out := OutcomeUnsupported
		if anySoft {
			out = OutcomeAbstain
		}
		sel := verdict(out, method, ClaimArtifactDescribed, exclusionReasons(excluded)...)
		sel.Excluded = excluded
		return sel
	}

	// --- who gets to choose ----------------------------------------------
	if s.KernelOrderIsPreference {
		candidates = byBuildOrder(candidates, s.Kernels)
	}
	sel := Selection{
		Claim:          ClaimRuntimeDelegated,
		ArtifactMethod: method,
		Candidates:     candidatesFor(candidates, a, s),
		Excluded:       excluded,
	}
	switch {
	case s.RuntimeSelects || (len(candidates) > 1 && !s.KernelOrderIsPreference):
		// Nobody licensed a ranking, so the runtime owns the kernel choice and
		// only the non-kernel arguments are licensed here.
		sel.Outcome = OutcomeDelegate
		sel.Reasons = []Reason{ReasonKernelChoiceDelegated}
		sel.Args = launchArgs(kernelRule{}, a, s)
	case len(candidates) > 1:
		sel.Outcome = OutcomeSupported
		sel.Reasons = []Reason{ReasonAdmitted, ReasonSelectedByServerPreference}
		sel.Args = append([]string{}, sel.Candidates[0].Args...)
	default:
		sel.Outcome = OutcomeSupported
		sel.Reasons = []Reason{ReasonAdmitted}
		sel.Args = append([]string{}, sel.Candidates[0].Args...)
	}
	return normalize(sel)
}

// ParseAndAdjudicate reads a serialized request and adjudicates it. A caller
// never has to string-match a Go error to tell "could not read" from "read and
// refused": unreadable bytes come back as a typed refusal.
func ParseAndAdjudicate(raw []byte) Selection {
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		return verdict(OutcomeRefuse, MethodUnknown, ClaimNone, ReasonInvalidJSON)
	}
	return Adjudicate(r)
}

// candidatesFor pairs each admissible kernel with the arguments that pick it.
func candidatesFor(kernels []Kernel, a Artifact, s Server) []Candidate {
	out := make([]Candidate, 0, len(kernels))
	for _, k := range kernels {
		kr := kernelTable[k]
		kr.name = k
		out = append(out, Candidate{Method: k, Args: launchArgs(kr, a, s)})
	}
	return out
}

// launchArgs builds the exact arguments a selection licenses. A zero kernelRule
// (the delegate case) omits --quantization and any kernel-specific argument:
// those belong to whoever ends up choosing the kernel.
func launchArgs(kr kernelRule, a Artifact, s Server) []string {
	args := []string{}
	if kr.name != "" {
		args = append(args, "--quantization", string(kr.name))
		args = append(args, kr.extraArgs...)
	}
	if s.Dtype != "" {
		args = append(args, "--dtype", s.Dtype)
	}
	if a.KVCacheDtype != "" {
		args = append(args, "--kv-cache-dtype", a.KVCacheDtype)
	}
	return args
}

// byBuildOrder re-orders admissible kernels into the build's own advertised
// order, which is what KernelOrderIsPreference declares to be a preference.
func byBuildOrder(candidates []Kernel, advertised []Kernel) []Kernel {
	index := make(map[Kernel]int, len(advertised))
	for i, k := range advertised {
		if _, seen := index[k]; !seen {
			index[k] = i
		}
	}
	out := append([]Kernel(nil), candidates...)
	sort.SliceStable(out, func(i, j int) bool { return index[out[i]] < index[out[j]] })
	return out
}

// exclusionReasons collapses the per-kernel exclusions into the deduplicated
// reason list, in the order they were found.
func exclusionReasons(excluded []Exclusion) []Reason {
	var out []Reason
	for _, e := range excluded {
		out = appendReason(out, e.Reason)
	}
	return out
}

// verdict builds a selection that licensed nothing: no candidates, no arguments,
// just the outcome and why.
func verdict(o Outcome, method string, c Claim, reasons ...Reason) Selection {
	return normalize(Selection{Outcome: o, Reasons: reasons, Claim: c, ArtifactMethod: method})
}

// normalize guarantees the wire shape: every list is present and empty rather
// than null, so a consumer never distinguishes absent from empty.
func normalize(s Selection) Selection {
	if s.Reasons == nil {
		s.Reasons = []Reason{}
	}
	if s.Candidates == nil {
		s.Candidates = []Candidate{}
	}
	if s.Excluded == nil {
		s.Excluded = []Exclusion{}
	}
	if s.Args == nil {
		s.Args = []string{}
	}
	return s
}

func appendReason(reasons []Reason, r Reason) []Reason {
	for _, got := range reasons {
		if got == r {
			return reasons
		}
	}
	return append(reasons, r)
}

func containsInt(values []int, target int) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}

func firstInt(values ...*int) *int {
	for _, v := range values {
		if v != nil {
			return v
		}
	}
	return nil
}

func firstBool(values ...*bool) *bool {
	for _, v := range values {
		if v != nil {
			return v
		}
	}
	return nil
}

// version is a three-field release number. Anything that is not exactly three
// decimal fields (a ".post1" suffix, a git describe, an empty string) is not
// parsed into the nearest known shape — it is reported unknown.
type version struct{ major, minor, patch int }

func parseVersion(s string) (version, bool) {
	fields := strings.Split(strings.TrimSpace(s), ".")
	if len(fields) != 3 {
		return version{}, false
	}
	var v version
	dst := []*int{&v.major, &v.minor, &v.patch}
	for i, f := range fields {
		if f == "" || strings.ContainsFunc(f, func(r rune) bool { return r < '0' || r > '9' }) {
			return version{}, false
		}
		n, err := strconv.Atoi(f)
		if err != nil {
			return version{}, false
		}
		*dst[i] = n
	}
	return v, true
}

func (v version) below(o version) bool {
	if v.major != o.major {
		return v.major < o.major
	}
	if v.minor != o.minor {
		return v.minor < o.minor
	}
	return v.patch < o.patch
}
