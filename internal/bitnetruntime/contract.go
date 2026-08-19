package bitnetruntime

// ContractVersion names this leaf's public adjudication contract. It is stamped
// on nothing and stored nowhere: it exists so a caller can pin the reason
// vocabulary it switches on.
const ContractVersion = "bitnetruntime/v1"

// RuntimeName is the delegate this contract admits. It is a fixed string rather
// than a parsed field because a probe report that came from something else is
// not evidence about bitnet.cpp.
const RuntimeName = "bitnet.cpp"

// MinRuntimeVersion is the oldest runtime this contract adjudicates. Anything
// older is reported unsupported rather than assumed compatible: this leaf has
// no evidence about pre-1.0 builds, and inventing some is exactly the silent
// fallback it exists to prevent.
const MinRuntimeVersion = "1.0.0"

// Kernel is a bitnet.cpp compute kernel. Weights are packed FOR a kernel, so it
// is a property of the artifact as much as of the build.
type Kernel string

const (
	// KernelI2S is the 2-bit signed ternary kernel, carried on both supported
	// architectures.
	KernelI2S Kernel = "i2_s"
	// KernelTL1 is the ARM lookup-table kernel.
	KernelTL1 Kernel = "tl1"
	// KernelTL2 is the x86 lookup-table kernel, which packs three ternary
	// weights into five stored bits.
	KernelTL2 Kernel = "tl2"
	// KernelUnknown is what a delegation reports when none was selected. It is a
	// terminal answer, not a placeholder to be resolved by guessing.
	KernelUnknown Kernel = "unknown"
)

// Alphabet is the weight alphabet an artifact was produced in. Only the
// three-level ternary alphabet is servable by bitnet.cpp; the rest are named so
// they can be refused by name instead of being lumped in as "low-bit".
const (
	// AlphabetTernary is the three-level signed alphabet {-1,0,+1} — the
	// "1.58-bit" family bitnet.cpp serves.
	AlphabetTernary = "ternary"
)

// Outcome is the typed verdict every request receives.
type Outcome string

const (
	// OutcomeDelegate: the named external runtime can execute this artifact on
	// this host. This is the success outcome; fak still owns none of it.
	OutcomeDelegate Outcome = "delegate"
	// OutcomeUnsupported: a definite, named no — the evidence is complete and it
	// rules the delegation out.
	OutcomeUnsupported Outcome = "unsupported"
	// OutcomeAbstain: not enough evidence to decide. Never resolved by assuming
	// the usual answer.
	OutcomeAbstain Outcome = "abstain"
	// OutcomeRefuse: the evidence contradicts itself, so no reading of it is
	// trustworthy.
	OutcomeRefuse Outcome = "refuse"
)

// ClaimClass names what a result licenses a caller to say. The four are kept
// apart deliberately, and this leaf emits only one of them: describing a
// delegation is not describing an artifact, is not describing how it was
// produced, and is certainly not a measured hardware envelope.
type ClaimClass string

const (
	// ClaimNone licenses nothing.
	ClaimNone ClaimClass = ""
	// ClaimRuntimeDelegated licenses only the statement that a named external
	// runtime owns the artifact's realization on this host. It is the only class
	// this leaf ever emits.
	ClaimRuntimeDelegated ClaimClass = "runtime-delegated"
	// ClaimArtifactDescribed licenses statements about what the artifact is. It
	// is part of the vocabulary so callers can route on it; internal/bitnetmeta
	// emits it and this leaf never does.
	ClaimArtifactDescribed ClaimClass = "artifact-described"
	// ClaimRecipeDescribed additionally licenses statements about how the
	// artifact was produced. Same owner, same exclusion.
	ClaimRecipeDescribed ClaimClass = "recipe-described"
	// ClaimHardwareEnvelope names a measured device/runtime envelope. This leaf
	// never emits it: probing a capability is not measuring hardware.
	ClaimHardwareEnvelope ClaimClass = "hardware-envelope"
)

// Reason is a stable machine-routable code for a decision.
type Reason string

const (
	ReasonAdmitted                  Reason = "BITNETRUNTIME_ADMITTED"
	ReasonProbeFailed               Reason = "BITNETRUNTIME_PROBE_FAILED"
	ReasonProbeEmpty                Reason = "BITNETRUNTIME_PROBE_EMPTY"
	ReasonProbeConflict             Reason = "BITNETRUNTIME_PROBE_CONFLICT"
	ReasonVersionUndeclared         Reason = "BITNETRUNTIME_VERSION_UNDECLARED"
	ReasonVersionMalformed          Reason = "BITNETRUNTIME_VERSION_MALFORMED"
	ReasonVersionTooOld             Reason = "BITNETRUNTIME_VERSION_TOO_OLD"
	ReasonKernelsUndeclared         Reason = "BITNETRUNTIME_KERNELS_UNDECLARED"
	ReasonKernelNotBuilt            Reason = "BITNETRUNTIME_KERNEL_NOT_BUILT"
	ReasonKernelArchMismatch        Reason = "BITNETRUNTIME_KERNEL_ARCH_MISMATCH"
	ReasonCPUFeatureMissing         Reason = "BITNETRUNTIME_CPU_FEATURE_MISSING"
	ReasonModelFamilyUndeclared     Reason = "BITNETRUNTIME_MODEL_FAMILY_UNDECLARED"
	ReasonModelFamilyUnsupported    Reason = "BITNETRUNTIME_MODEL_FAMILY_UNSUPPORTED"
	ReasonModelKernelUndeclared     Reason = "BITNETRUNTIME_MODEL_KERNEL_UNDECLARED"
	ReasonModelKernelUnknown        Reason = "BITNETRUNTIME_MODEL_KERNEL_UNKNOWN"
	ReasonPackingNarrowerThanKernel Reason = "BITNETRUNTIME_PACKING_NARROWER_THAN_KERNEL"
	ReasonHostOSUndeclared          Reason = "BITNETRUNTIME_HOST_OS_UNDECLARED"
	ReasonHostOSUnsupported         Reason = "BITNETRUNTIME_HOST_OS_UNSUPPORTED"
	ReasonHostArchUndeclared        Reason = "BITNETRUNTIME_HOST_ARCH_UNDECLARED"
	ReasonHostArchUnsupported       Reason = "BITNETRUNTIME_HOST_ARCH_UNSUPPORTED"
)

// severity maps a reason to the outcome it forces. The worst severity across a
// result's reasons wins, so one refusal is never masked by a green neighbour.
//
// Unsupported outranks abstain on purpose: a definite no derived from complete
// evidence on ONE axis stays true however little is known about another, while
// the reverse is not so. A contradiction outranks both, because it makes every
// other reading of the same report untrustworthy.
func (r Reason) severity() Outcome {
	switch r {
	case ReasonAdmitted:
		return OutcomeDelegate
	case ReasonProbeConflict, ReasonPackingNarrowerThanKernel:
		return OutcomeRefuse
	case ReasonVersionTooOld, ReasonKernelNotBuilt, ReasonKernelArchMismatch,
		ReasonCPUFeatureMissing, ReasonModelFamilyUnsupported,
		ReasonHostOSUnsupported, ReasonHostArchUnsupported:
		return OutcomeUnsupported
	case ReasonProbeFailed, ReasonProbeEmpty, ReasonVersionUndeclared,
		ReasonVersionMalformed, ReasonKernelsUndeclared, ReasonModelFamilyUndeclared,
		ReasonModelKernelUndeclared, ReasonModelKernelUnknown,
		ReasonHostOSUndeclared, ReasonHostArchUndeclared:
		return OutcomeAbstain
	}
	return OutcomeDelegate
}

// Known reports whether r is part of the published vocabulary.
func (r Reason) Known() bool {
	switch r {
	case ReasonAdmitted, ReasonProbeFailed, ReasonProbeEmpty, ReasonProbeConflict,
		ReasonVersionUndeclared, ReasonVersionMalformed, ReasonVersionTooOld,
		ReasonKernelsUndeclared, ReasonKernelNotBuilt, ReasonKernelArchMismatch,
		ReasonCPUFeatureMissing, ReasonModelFamilyUndeclared, ReasonModelFamilyUnsupported,
		ReasonModelKernelUndeclared, ReasonModelKernelUnknown, ReasonPackingNarrowerThanKernel,
		ReasonHostOSUndeclared, ReasonHostOSUnsupported,
		ReasonHostArchUndeclared, ReasonHostArchUnsupported:
		return true
	}
	return false
}

// Host is the machine the delegate would run on. Features is the CPU feature
// set the caller detected, lowercase; an empty set means "none detected", which
// is a different answer from an undeclared architecture.
type Host struct {
	OS       string   `json:"os,omitempty"`
	Arch     string   `json:"arch,omitempty"`
	Features []string `json:"features,omitempty"`
}

// Model is the artifact a caller wants served. Alphabet is the weight alphabet
// it was produced in; Kernel is the kernel its weights were PACKED for, which
// is a property of the file and not a preference. BitsPerWeightStored is a
// container width the caller read off the artifact, kept so a file that cannot
// physically hold its declared kernel's packing is caught rather than served.
type Model struct {
	ID                  string  `json:"id,omitempty"`
	Alphabet            string  `json:"alphabet,omitempty"`
	Kernel              Kernel  `json:"kernel,omitempty"`
	BitsPerWeightStored float64 `json:"bits_per_weight_stored,omitempty"`
}

// Runtime is what a probe reported about one bitnet.cpp build. Kernels is the
// build's own list; this contract never extends it from a version number.
type Runtime struct {
	Name    string   `json:"name"`
	Version string   `json:"version"`
	Build   string   `json:"build"`
	Kernels []Kernel `json:"kernels"`
}

// HasKernel reports whether the build carries k.
func (rt Runtime) HasKernel(k Kernel) bool {
	for _, got := range rt.Kernels {
		if got == k {
			return true
		}
	}
	return false
}

// Delegation is the executable half of an admitted result: which external
// runtime, which kernel, on which host. It is populated ONLY on a delegation —
// a request that was not admitted selected nothing, and reporting a kernel for
// it would invite a caller to dispatch on a decision that was never made.
type Delegation struct {
	Runtime  string `json:"runtime"`
	Version  string `json:"version"`
	Build    string `json:"build"`
	Kernel   Kernel `json:"kernel"`
	HostOS   string `json:"host_os"`
	HostArch string `json:"host_arch"`
}

// Result is the typed outcome of adjudicating one delegation request.
type Result struct {
	Outcome    Outcome    `json:"outcome"`
	Reasons    []Reason   `json:"reasons"`
	Claim      ClaimClass `json:"claim"`
	Delegation Delegation `json:"delegation"`
}

// HasReason reports whether want is among the result's reasons.
func (r Result) HasReason(want Reason) bool {
	for _, got := range r.Reasons {
		if got == want {
			return true
		}
	}
	return false
}

// kernelSpec is what a kernel needs from a host. byArch is both the set of
// architectures the kernel exists for and, per architecture, the CPU feature it
// dispatches on — the two are one table because tl1 and tl2 are single-
// architecture kernels and splitting them invites an arch-blind feature check.
type kernelSpec struct {
	storedBits float64
	byArch     map[string]string
}

// kernelSpecs is the published dispatch matrix. storedBits is the container
// width the kernel's packing requires per weight: i2_s and tl1 index two bits
// per weight, tl2 packs three ternary weights into five bits.
var kernelSpecs = map[Kernel]kernelSpec{
	KernelI2S:     {storedBits: 2, byArch: map[string]string{"amd64": "avx2", "arm64": "neon"}},
	KernelTL1:     {storedBits: 2, byArch: map[string]string{"arm64": "neon"}},
	KernelTL2:     {storedBits: 5.0 / 3.0, byArch: map[string]string{"amd64": "avx2"}},
	KernelUnknown: {}, // terminal no-selection sentinel; never dispatchable
}

// supportedOS is the set of platforms bitnet.cpp builds for. It is a closed set
// so an unlisted platform gets a named unsupported instead of an optimistic try.
var supportedOS = map[string]bool{"linux": true, "darwin": true, "windows": true}

// packingEpsilon absorbs float noise when comparing a container width read off
// an artifact against a kernel's required packing (a tl2 artifact declaring
// 1.667 bits per stored weight must not read as narrower than 5/3).
const packingEpsilon = 1e-3

// Admit decides whether one artifact may be delegated to one discovered runtime
// on one host. Every axis is evaluated so a caller sees all the obstacles at
// once, except where an earlier answer makes a later question meaningless — an
// undeclared architecture makes "does this host have avx2" unanswerable, not
// false.
func Admit(rt Runtime, host Host, model Model) Result {
	var reasons []Reason

	archUsable := true
	switch {
	case host.OS == "":
		reasons = append(reasons, ReasonHostOSUndeclared)
	case !supportedOS[host.OS]:
		reasons = append(reasons, ReasonHostOSUnsupported)
	}
	switch {
	case host.Arch == "":
		reasons = append(reasons, ReasonHostArchUndeclared)
		archUsable = false
	case !archSupported(host.Arch):
		reasons = append(reasons, ReasonHostArchUnsupported)
		archUsable = false
	}

	switch {
	case model.Alphabet == "":
		reasons = append(reasons, ReasonModelFamilyUndeclared)
	case model.Alphabet != AlphabetTernary:
		// Binary, int2 and int4 artifacts are all honestly outside this
		// runtime's family; "low-bit" is not one bucket.
		reasons = append(reasons, ReasonModelFamilyUnsupported)
	}

	spec, kernelKnown := kernelSpecs[model.Kernel]
	switch {
	case model.Kernel == "":
		reasons = append(reasons, ReasonModelKernelUndeclared)
	case !kernelKnown:
		reasons = append(reasons, ReasonModelKernelUnknown)
	}

	if kernelKnown {
		if model.BitsPerWeightStored > 0 && model.BitsPerWeightStored+packingEpsilon < spec.storedBits {
			// A container narrower than the kernel's packing cannot hold the
			// weights the artifact claims to carry for it.
			reasons = append(reasons, ReasonPackingNarrowerThanKernel)
		}
		if !rt.HasKernel(model.Kernel) {
			// bitnet.cpp selects its kernels at compile time, so this is a
			// property of the build in hand, not of the project.
			reasons = append(reasons, ReasonKernelNotBuilt)
		}
		if archUsable {
			feature, onThisArch := spec.byArch[host.Arch]
			switch {
			case !onThisArch:
				reasons = append(reasons, ReasonKernelArchMismatch)
			case !hasFeature(host.Features, feature):
				reasons = append(reasons, ReasonCPUFeatureMissing)
			}
		}
	}

	if len(reasons) > 0 {
		return finish(reasons, Delegation{Kernel: KernelUnknown})
	}
	return finish(nil, Delegation{
		Runtime:  RuntimeName,
		Version:  rt.Version,
		Build:    rt.Build,
		Kernel:   model.Kernel,
		HostOS:   host.OS,
		HostArch: host.Arch,
	})
}

// archSupported reports whether any kernel exists for this architecture, which
// is what makes the architecture itself adjudicable.
func archSupported(arch string) bool {
	for _, spec := range kernelSpecs {
		if _, ok := spec.byArch[arch]; ok {
			return true
		}
	}
	return false
}

func hasFeature(features []string, want string) bool {
	for _, got := range features {
		if got == want {
			return true
		}
	}
	return false
}

// outcomeRank orders the outcomes by severity so the worst one wins.
var outcomeRank = map[Outcome]int{OutcomeDelegate: 0, OutcomeAbstain: 1, OutcomeUnsupported: 2, OutcomeRefuse: 3}

// finish picks the worst outcome across the collected reasons and the claim
// class that outcome licenses.
func finish(reasons []Reason, delegation Delegation) Result {
	outcome := OutcomeDelegate
	for _, r := range reasons {
		if outcomeRank[r.severity()] > outcomeRank[outcome] {
			outcome = r.severity()
		}
	}
	if len(reasons) == 0 {
		reasons = []Reason{ReasonAdmitted}
	}
	claim := ClaimNone
	if outcome == OutcomeDelegate {
		claim = ClaimRuntimeDelegated
	}
	return Result{Outcome: outcome, Reasons: reasons, Claim: claim, Delegation: delegation}
}
