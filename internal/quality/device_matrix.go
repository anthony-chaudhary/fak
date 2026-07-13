package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// device_matrix.go is the device-matrix parity child of the quality spine
// (#4544): the SAME case decoded on every backend in the supported device
// matrix — CPU, CUDA, ROCm, Metal, Vulkan — must agree with the reference. A
// kernel that is correct on four backends and subtly wrong on the fifth passes
// every single-backend gate and ships a device-dependent regression; this file
// makes the whole matrix ONE fan-in verdict. Each backend is modeled as a
// labeled trace (tokens exact, per-step logits within a numeric tolerance),
// the labeled traces are serialized into Trace.Text as JSON — the additive
// seam, Judge's signature untouched — and the "device-matrix-parity"
// differential oracle parses them back out and pins the first backend and
// token index that departed from the reference.

// devMatrixBackends is the required device matrix, in the fixed order the
// oracle reports in. Every backend listed here must appear in the engine's
// fan-in payload and agree with the reference — a missing backend is a
// failure, not a skip (an unchecked backend is not a green backend).
var devMatrixBackends = []string{"cpu", "cuda", "rocm", "metal", "vulkan"}

// devMatrixTolerance is the per-logit absolute tolerance backends may differ
// from the reference by. Distinct device kernels legitimately reorder float
// operations, so bitwise logit equality across a device matrix is the wrong
// contract — but tokens are argmax decisions and must match EXACTLY.
const devMatrixTolerance = 1e-6

// devMatrixJitterScale bounds the deterministic per-backend numeric jitter the
// faithful model applies: three orders of magnitude inside the tolerance, so a
// clean matrix passes because the tolerance absorbs device-kernel noise, while
// the injected drift defect (devMatrixDrift) lands three orders OUTSIDE it.
const devMatrixJitterScale = 1e-9

// devMatrixDrift is the injected out-of-tolerance logit error: large enough
// that no faithful jitter can mask it, small enough to be the classic "looks
// plausible, fails parity" numeric defect.
const devMatrixDrift = 5e-3

// devMatrixDefectBackend and devMatrixDefectStep pin where the injected
// defects land: the Vulkan backend, mid-sequence, so the passing prefix (and
// the four passing backends before it) prove the localization is doing work.
const (
	devMatrixDefectBackend = "vulkan"
	devMatrixDefectStep    = 2
)

// Injected defect classes for devMatrixEngine.
const (
	devMatrixDefectTokenFlip  = "token-flip"      // vulkan emits a wrong token at the defect step
	devMatrixDefectLogitDrift = "logit-drift"     // vulkan's logit drifts beyond tolerance at the defect step
	devMatrixDefectMissing    = "missing-backend" // vulkan is absent from the fan-in payload
)

// devMatrixVocab is the small fixed vocabulary the deterministic decode draws
// from. Eight entries keep the token space tiny while making an accidental
// collision between two different decodes vanishingly unlikely.
var devMatrixVocab = []string{"alloy", "breeze", "copper", "dune", "ember", "fjord", "grove", "hollow"}

// devMatrixSlots is the number of top-candidate logits captured per step.
const devMatrixSlots = 4

// devMatrixSeed anchors the golden decode. All backends share it — device
// parity means the DEVICE must not change the answer, so the answer is a pure
// function of the case, never of the backend.
const devMatrixSeed uint64 = 0x51ce950f2ca7d4f4

// devMatrixMix maps (state, x) to one pseudo-random draw via splitmix64: the
// golden-gamma constant advances the state and the finalizer mixes it. A pure
// function of its inputs — no carried state, no ambient entropy — so the
// golden decode and every backend's jitter replay identically.
func devMatrixMix(state, x uint64) uint64 {
	z := state + (x+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// devMatrixFold hashes a backend name to a jitter stream seed (FNV-1a), so
// each backend's numeric noise is distinct but deterministic.
func devMatrixFold(backend string) uint64 {
	const (
		fnvOffset = 0xcbf29ce484222325
		fnvPrime  = 0x100000001b3
	)
	h := uint64(fnvOffset)
	for i := 0; i < len(backend); i++ {
		h ^= uint64(backend[i])
		h *= fnvPrime
	}
	return h
}

// devMatrixGolden is the device-independent golden decode: steps tokens and
// per-step top-candidate logits, a pure function of the fixed seed. It is what
// the case's Reference carries and what every faithful backend must reproduce
// (tokens exactly, logits within tolerance).
func devMatrixGolden(steps int) ([]string, [][]float64) {
	toks := make([]string, 0, steps)
	logits := make([][]float64, 0, steps)
	for i := 0; i < steps; i++ {
		h := devMatrixMix(devMatrixSeed, uint64(i))
		toks = append(toks, devMatrixVocab[h%uint64(len(devMatrixVocab))])
		row := make([]float64, devMatrixSlots)
		for j := 0; j < devMatrixSlots; j++ {
			// A coarse fixed-point value (multiples of 1/64) exact in float64,
			// descending by slot like a real top-k logit list.
			row[j] = float64(devMatrixMix(h, uint64(j))%512)/64.0 - float64(j)
		}
		logits = append(logits, row)
	}
	return toks, logits
}

// devMatrixJitter is the deterministic per-(backend, step, slot) numeric noise
// a faithful backend adds to the golden logits: a value in
// (-devMatrixJitterScale, +devMatrixJitterScale], three orders of magnitude
// inside the tolerance. It models legitimate cross-device float reassociation
// — nonzero (so the oracle's tolerance path is exercised, not vacuous) yet
// never large enough to trip the gate.
func devMatrixJitter(backend string, step, slot int) float64 {
	h := devMatrixMix(devMatrixFold(backend), uint64(step*devMatrixSlots+slot))
	return (float64(h%2000)/1000.0 - 1.0) * devMatrixJitterScale
}

// devMatrixRotate returns the vocab entry after tok, wrapping. Rotation by one
// in a vocab of eight can never map a token to itself, so the token-flip
// mutant is GUARANTEED to diverge at exactly its step.
func devMatrixRotate(tok string) string {
	for i, v := range devMatrixVocab {
		if v == tok {
			return devMatrixVocab[(i+1)%len(devMatrixVocab)]
		}
	}
	return devMatrixVocab[0]
}

// devBackendTrace is one backend's labeled decode: which device produced it,
// the emitted tokens, and the captured per-step logits. The fan-in payload is
// a JSON array of these, carried in Trace.Text.
type devBackendTrace struct {
	Backend string      `json:"backend"`
	Tokens  []string    `json:"tokens"`
	Logits  [][]float64 `json:"logits,omitempty"`
}

// devMatrixDecode models one backend decoding the case: the golden tokens with
// that backend's deterministic numeric jitter on the logits, plus the injected
// defect iff this backend is the defect backend. A faithful backend differs
// from the golden logits bitwise (real devices do) but never beyond tolerance.
func devMatrixDecode(backend string, steps int, defect string) devBackendTrace {
	toks, logits := devMatrixGolden(steps)
	for i := range logits {
		for j := range logits[i] {
			logits[i][j] += devMatrixJitter(backend, i, j)
		}
	}
	if backend == devMatrixDefectBackend {
		switch defect {
		case devMatrixDefectTokenFlip:
			if devMatrixDefectStep < len(toks) {
				toks[devMatrixDefectStep] = devMatrixRotate(toks[devMatrixDefectStep])
			}
		case devMatrixDefectLogitDrift:
			if devMatrixDefectStep < len(logits) && len(logits[devMatrixDefectStep]) > 0 {
				logits[devMatrixDefectStep][0] += devMatrixDrift
			}
		}
	}
	return devBackendTrace{Backend: backend, Tokens: toks, Logits: logits}
}

// devMatrixPack serializes the labeled backend traces into one engine Trace:
// the JSON fan-in payload rides in Text (the additive seam — Judge's signature
// is untouched) and Tokens carries the first backend's stream as the primary
// differential surface for any generic oracle.
func devMatrixPack(runner string, traces []devBackendTrace) (Trace, error) {
	b, err := json.Marshal(traces)
	if err != nil {
		return Trace{}, fmt.Errorf("device matrix payload: %w", err)
	}
	t := Trace{Runner: runner, Text: string(b)}
	if len(traces) > 0 {
		t.Tokens = traces[0].Tokens
	}
	return t, nil
}

// devMatrixParse recovers the labeled backend traces from an engine Trace's
// Text payload.
func devMatrixParse(t Trace) ([]devBackendTrace, error) {
	var traces []devBackendTrace
	if err := json.Unmarshal([]byte(t.Text), &traces); err != nil {
		return nil, fmt.Errorf("device matrix payload: %w", err)
	}
	return traces, nil
}

// devMatrixRunner is the fan-in engine adapter: it decodes the case on every
// backend in the matrix and packs the labeled traces into one Trace. The zero
// value is a faithful matrix; defect (set via devMatrixEngine) injects a
// single-backend bug. A real harness fans out to actual device builds and
// packs their captures the same way, behind the same Runner seam.
type devMatrixRunner struct {
	label  string
	defect string
}

func (r devMatrixRunner) Name() string {
	if r.label != "" {
		return r.label
	}
	return "device-matrix-engine"
}

func (r devMatrixRunner) Run(c QualityCase) (Trace, error) {
	traces := make([]devBackendTrace, 0, len(devMatrixBackends))
	for _, backend := range devMatrixBackends {
		if r.defect == devMatrixDefectMissing && backend == devMatrixDefectBackend {
			continue
		}
		traces = append(traces, devMatrixDecode(backend, c.Params.MaxTokens, r.defect))
	}
	return devMatrixPack(r.Name(), traces)
}

// devMatrixEngine returns a fan-in engine runner with an optional injected
// single-backend defect: "" decodes faithfully on all five backends (parity
// holds within tolerance); "token-flip" makes Vulkan emit a wrong token at
// step 2; "logit-drift" pushes one Vulkan logit beyond tolerance at step 2;
// "missing-backend" omits Vulkan from the payload entirely. This is the
// deterministic mutant source the tests use to prove the matrix gate trips.
func devMatrixEngine(defect string) devMatrixRunner {
	switch defect {
	case devMatrixDefectTokenFlip:
		return devMatrixRunner{label: "engine-vulkan-token-flip", defect: defect}
	case devMatrixDefectLogitDrift:
		return devMatrixRunner{label: "engine-vulkan-logit-drift", defect: defect}
	case devMatrixDefectMissing:
		return devMatrixRunner{label: "engine-vulkan-missing", defect: defect}
	default:
		return devMatrixRunner{label: "engine-device-matrix-clean"}
	}
}

// devMatrixCase builds the deterministic device-matrix case: a temperature-zero
// decode budget and a reference trace produced by the golden decode itself.
// The backend list deliberately does NOT appear in the case — the matrix is
// the oracle's contract (devMatrixBackends), so a case cannot quietly opt a
// backend out of parity.
func devMatrixCase() QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 6}
	toks, logits := devMatrixGolden(params.MaxTokens)
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "device-matrix-parity-demo",
		Version:   1,
		Prompt:    "Decode the pinned greedy sequence identically on every supported backend.",
		Params:    params,
		Reference: Trace{Tokens: toks, Logits: logits, Text: strings.Join(toks, " ")},
		Oracles:   []string{"device-matrix-parity"},
	}
}

// devMatrixParity is the fan-in differential oracle for #4544: every backend
// in the device matrix must reproduce the reference — tokens exactly, logits
// within devMatrixTolerance — and every backend must be PRESENT. The first
// departure is localized to a named backend and token index, so "the matrix
// disagrees" becomes "vulkan diverged at token 2".
type devMatrixParity struct{}

func (devMatrixParity) Name() string { return "device-matrix-parity" }
func (devMatrixParity) Kind() string { return "differential" }

func init() { Register(devMatrixParity{}) }

func (devMatrixParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "device-matrix-parity", Kind: "differential", Pass: true}
	traces, err := devMatrixParse(eng)
	if err != nil {
		v.Pass = false
		v.Detail = fmt.Sprintf("engine trace carries no parseable device-matrix payload: %v", err)
		return v
	}
	byBackend := make(map[string]devBackendTrace, len(traces))
	for _, t := range traces {
		byBackend[t.Backend] = t
	}
	for _, backend := range devMatrixBackends {
		b, ok := byBackend[backend]
		if !ok {
			v.Pass = false
			v.Detail = fmt.Sprintf("backend %q missing from device matrix: an unchecked backend is not a green backend (present: %d of %d)",
				backend, len(byBackend), len(devMatrixBackends))
			return v
		}
		if bad := devMatrixJudgeBackend(&v, ref, b); bad {
			return v
		}
	}
	v.Detail = fmt.Sprintf("all %d backends (%s) matched the reference: %d tokens exact, logits within %g",
		len(devMatrixBackends), strings.Join(devMatrixBackends, ", "), len(ref.Tokens), devMatrixTolerance)
	return v
}

// devMatrixJudgeBackend compares one labeled backend trace against the
// reference, writing the failure (named backend + first divergent index) into
// v and reporting whether it failed. Tokens are compared exactly first — an
// argmax flip is the defect class that changes what a user reads — then logits
// within tolerance, so numeric drift is caught even before it flips a token.
func devMatrixJudgeBackend(v *Verdict, ref Trace, b devBackendTrace) bool {
	n := len(ref.Tokens)
	if len(b.Tokens) < n {
		n = len(b.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != b.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: b.Tokens[i]}
			v.Detail = fmt.Sprintf("backend %q diverged from reference at token %d: reference %q, %s %q",
				b.Backend, i, ref.Tokens[i], b.Backend, b.Tokens[i])
			return true
		}
	}
	if len(ref.Tokens) != len(b.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(b.Tokens, n)}
		v.Detail = fmt.Sprintf("backend %q token length diverged at %d: reference has %d tokens, %s has %d",
			b.Backend, n, len(ref.Tokens), b.Backend, len(b.Tokens))
		return true
	}
	steps := len(ref.Logits)
	if len(b.Logits) < steps {
		steps = len(b.Logits)
	}
	for i := 0; i < steps; i++ {
		refRow, engRow := ref.Logits[i], b.Logits[i]
		if len(refRow) != len(engRow) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: fmt.Sprintf("%d logit slots", len(refRow)), Engine: fmt.Sprintf("%d logit slots", len(engRow))}
			v.Detail = fmt.Sprintf("backend %q logit width diverged at step %d: reference has %d slots, %s has %d",
				b.Backend, i, len(refRow), b.Backend, len(engRow))
			return true
		}
		for j := range refRow {
			if delta := math.Abs(refRow[j] - engRow[j]); delta > devMatrixTolerance {
				v.Pass = false
				v.FirstDivergence = &Divergence{Index: i, Reference: fmt.Sprintf("%.9g", refRow[j]), Engine: fmt.Sprintf("%.9g", engRow[j])}
				v.Detail = fmt.Sprintf("backend %q logit diverged at step %d slot %d: |%.9g - %.9g| = %.3g exceeds tolerance %g",
					b.Backend, i, j, refRow[j], engRow[j], delta, devMatrixTolerance)
				return true
			}
		}
	}
	if len(ref.Logits) != len(b.Logits) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: steps, Reference: fmt.Sprintf("%d logit steps", len(ref.Logits)), Engine: fmt.Sprintf("%d logit steps", len(b.Logits))}
		v.Detail = fmt.Sprintf("backend %q logit step count diverged: reference has %d, %s has %d",
			b.Backend, len(ref.Logits), b.Backend, len(b.Logits))
		return true
	}
	return false
}
