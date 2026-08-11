package lightroteval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"runtime"
	"sort"
)

const (
	ContractVersion = "lightroteval/v1"
	RecipeVersion   = "lightrot-paper-2607.27704/v1"
	RuntimeVersion  = "fak-lightroteval-cpu-f64/v1"
	PaperID         = "arXiv:2607.27704"
	PaperSHA256     = "e9e6093c0b0025e0fa40b575c416d8e40cb287d97d434373d6878ec6f3762696"
)

type Outcome string

const (
	OutcomeSupported   Outcome = "supported"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeDelegate    Outcome = "delegate"
)

type ReasonCode string

const (
	ReasonEvaluated         ReasonCode = "LIGHTROT_EVALUATED_MODELED"
	ReasonUnknownContract   ReasonCode = "LIGHTROT_UNKNOWN_CONTRACT"
	ReasonUnknownRecipe     ReasonCode = "LIGHTROT_UNKNOWN_RECIPE"
	ReasonUnknownRuntime    ReasonCode = "LIGHTROT_UNKNOWN_RUNTIME"
	ReasonInvalidProvenance ReasonCode = "LIGHTROT_INVALID_PROVENANCE"
	ReasonInvalidInput      ReasonCode = "LIGHTROT_INVALID_INPUT"
)

type EvidenceKind string

const (
	EvidenceModeled  EvidenceKind = "modeled"
	EvidenceObserved EvidenceKind = "observed"
)

type ArtifactProvenance struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
	Source  string `json:"source"`
}
type ModelProvenance struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
	SHA256   string `json:"sha256"`
	License  string `json:"license"`
}
type RecipeProvenance struct {
	ID          string `json:"id"`
	Version     string `json:"version"`
	PaperID     string `json:"paper_id"`
	PaperSHA256 string `json:"paper_sha256"`
	Seed        uint64 `json:"seed"`
	BlockSize   int    `json:"block_size"`
}
type RuntimeProvenance struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	Backend string `json:"backend"`
}
type HardwareEnvelope struct {
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	Device      string `json:"device"`
	Accelerator string `json:"accelerator,omitempty"`
}
type Provenance struct {
	Artifact ArtifactProvenance `json:"artifact"`
	Model    ModelProvenance    `json:"model"`
	Recipe   RecipeProvenance   `json:"recipe"`
	Runtime  RuntimeProvenance  `json:"runtime"`
	Hardware HardwareEnvelope   `json:"hardware"`
}
type Request struct {
	ContractVersion string       `json:"contract_version"`
	Bits            int          `json:"bits"`
	Evidence        EvidenceKind `json:"evidence"`
	Samples         [][]float64  `json:"samples"`
	Provenance      Provenance   `json:"provenance"`
}
type Cost struct {
	PreprocessScalarOps uint64       `json:"preprocess_scalar_ops"`
	RuntimeScalarOps    uint64       `json:"runtime_scalar_ops"`
	WallNS              uint64       `json:"wall_ns,omitempty"`
	WallEvidence        EvidenceKind `json:"wall_evidence"`
}
type Metrics struct {
	ReconstructionAccuracy float64 `json:"reconstruction_accuracy"`
	MSE                    float64 `json:"mse"`
	MaxAbsError            float64 `json:"max_abs_error"`
}
type Candidate struct {
	ID      string  `json:"id"`
	Role    string  `json:"role"`
	Metrics Metrics `json:"metrics"`
	Cost    Cost    `json:"cost"`
}
type ClaimCheck struct {
	Verdict  string `json:"verdict"`
	Scope    string `json:"scope"`
	Baseline string `json:"baseline"`
}
type Result struct {
	ContractVersion string       `json:"contract_version"`
	Outcome         Outcome      `json:"outcome"`
	Reason          ReasonCode   `json:"reason"`
	Evidence        EvidenceKind `json:"evidence"`
	Provenance      Provenance   `json:"provenance"`
	Candidates      []Candidate  `json:"candidates,omitempty"`
	ClaimCheck      ClaimCheck   `json:"claim_check"`
	Delegate        string       `json:"delegate,omitempty"`
}

func Evaluate(r Request) Result {
	out := Result{ContractVersion: ContractVersion, Evidence: r.Evidence, Provenance: r.Provenance, ClaimCheck: ClaimCheck{Verdict: "not-yet", Scope: "bounded synthetic reconstruction; not downstream model quality or throughput", Baseline: "tuned_rotation and no_rotation"}}
	fail := func(o Outcome, reason ReasonCode) Result { out.Outcome = o; out.Reason = reason; return out }
	if r.ContractVersion != ContractVersion {
		return fail(OutcomeUnsupported, ReasonUnknownContract)
	}
	if r.Provenance.Recipe.ID != "lightrot" || r.Provenance.Recipe.Version != RecipeVersion || r.Provenance.Recipe.PaperID != PaperID || r.Provenance.Recipe.PaperSHA256 != PaperSHA256 {
		return fail(OutcomeUnsupported, ReasonUnknownRecipe)
	}
	if r.Provenance.Runtime.ID != "fak/lightroteval" || r.Provenance.Runtime.Version != RuntimeVersion || r.Provenance.Runtime.Backend != "cpu-reference-f64" {
		return fail(OutcomeDelegate, ReasonUnknownRuntime)
	}
	if !validProvenance(r.Provenance) {
		return fail(OutcomeUnsupported, ReasonInvalidProvenance)
	}
	if r.Evidence == EvidenceObserved {
		out.Delegate = "sanctioned lab runner must attach measured wall time and device read-back"
		return fail(OutcomeDelegate, ReasonUnknownRuntime)
	}
	if r.Evidence != EvidenceModeled {
		return fail(OutcomeUnsupported, ReasonInvalidProvenance)
	}
	if r.Bits < 2 || r.Bits > 8 {
		return fail(OutcomeUnsupported, ReasonInvalidInput)
	}
	rows, cols, ok := shape(r.Samples)
	if !ok || cols < 2 || r.Provenance.Recipe.BlockSize < 2 || cols%r.Provenance.Recipe.BlockSize != 0 {
		return fail(OutcomeUnsupported, ReasonInvalidInput)
	}
	if digestSamples(r.Samples) != r.Provenance.Artifact.SHA256 {
		return fail(OutcomeUnsupported, ReasonInvalidProvenance)
	}
	methods := []string{"lightrot", "tuned_rotation", "no_rotation"}
	for _, m := range methods {
		out.Candidates = append(out.Candidates, evaluate(m, r.Samples, r.Bits, r.Provenance.Recipe.Seed, r.Provenance.Recipe.BlockSize, rows, cols))
	}
	out.Outcome = OutcomeSupported
	out.Reason = ReasonEvaluated
	return out
}
func validProvenance(p Provenance) bool {
	return p.Artifact.ID != "" && p.Artifact.Version != "" && validDigest(p.Artifact.SHA256) && p.Artifact.Source != "" && p.Model.ID != "" && p.Model.Revision != "" && validDigest(p.Model.SHA256) && p.Model.License != "" && p.Hardware.OS != "" && p.Hardware.Arch != "" && p.Hardware.Device != ""
}
func validDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, e := hex.DecodeString(s)
	return e == nil
}
func shape(x [][]float64) (int, int, bool) {
	if len(x) == 0 || len(x[0]) == 0 {
		return 0, 0, false
	}
	n := len(x[0])
	for _, r := range x {
		if len(r) != n {
			return 0, 0, false
		}
		for _, v := range r {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, 0, false
			}
		}
	}
	return len(x), n, true
}
func digestSamples(x [][]float64) string {
	b, _ := json.Marshal(x)
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}
func evaluate(method string, x [][]float64, bits int, seed uint64, block, rows, cols int) Candidate {
	rot := identity(cols)
	pre := uint64(0)
	switch method {
	case "lightrot":
		rot = lightRotation(cols, block, seed)
		pre = uint64(cols*block + cols)
	case "tuned_rotation":
		rot = tunedRotation(cols)
		pre = uint64(cols * cols * 8)
	}
	y := matmul(x, rot)
	q := quantize(y, bits)
	restored := matmul(q, transpose(rot))
	mse, maxe, norm := errors(x, restored)
	acc := 1.0
	if norm > 0 {
		acc = 1 - mse/norm
	}
	return Candidate{ID: method, Role: map[string]string{"lightrot": "candidate", "tuned_rotation": "tuned_rotation_baseline", "no_rotation": "no_rotation_baseline"}[method], Metrics: Metrics{round(acc), round(mse), round(maxe)}, Cost: Cost{PreprocessScalarOps: pre, RuntimeScalarOps: uint64(rows*cols*cols*2 + rows*cols*3), WallEvidence: EvidenceModeled}}
}
func identity(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
		m[i][i] = 1
	}
	return m
}
func lightRotation(n, b int, seed uint64) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	inv := 1 / math.Sqrt(float64(b))
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	state := seed
	for i := n - 1; i > 0; i-- {
		state = state*6364136223846793005 + 1442695040888963407
		j := int(state % uint64(i+1))
		perm[i], perm[j] = perm[j], perm[i]
	}
	for base := 0; base < n; base += b {
		for i := 0; i < b; i++ {
			for j := 0; j < b; j++ {
				sign := 1.0
				if bitsOn(i&j)%2 == 1 {
					sign = -1
				}
				if ((seed >> uint((base+j)%63)) & 1) == 1 {
					sign = -sign
				}
				m[base+i][perm[base+j]] = sign * inv
			}
		}
	}
	return m
}
func bitsOn(x int) int {
	n := 0
	for x > 0 {
		x &= x - 1
		n++
	}
	return n
}
func tunedRotation(n int) [][]float64 {
	m := identity(n)
	for i := 0; i+1 < n; i += 2 {
		a := math.Pi/7 + float64(i)*0.013
		c, s := math.Cos(a), math.Sin(a)
		m[i][i] = c
		m[i][i+1] = -s
		m[i+1][i] = s
		m[i+1][i+1] = c
	}
	return m
}
func matmul(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(b[0]))
		for j := range out[i] {
			for k := range b {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}
func transpose(a [][]float64) [][]float64 {
	r := make([][]float64, len(a[0]))
	for i := range r {
		r[i] = make([]float64, len(a))
		for j := range a {
			r[i][j] = a[j][i]
		}
	}
	return r
}
func quantize(x [][]float64, bits int) [][]float64 {
	qmax := float64((int64(1) << uint(bits-1)) - 1)
	max := 0.0
	for _, r := range x {
		for _, v := range r {
			if math.Abs(v) > max {
				max = math.Abs(v)
			}
		}
	}
	out := make([][]float64, len(x))
	if max == 0 {
		return x
	}
	scale := max / qmax
	for i, r := range x {
		out[i] = make([]float64, len(r))
		for j, v := range r {
			z := math.Round(v / scale)
			if z > qmax {
				z = qmax
			}
			if z < -qmax-1 {
				z = -qmax - 1
			}
			out[i][j] = z * scale
		}
	}
	return out
}
func errors(a, b [][]float64) (float64, float64, float64) {
	var sum, maxe, norm float64
	n := 0
	for i := range a {
		for j := range a[i] {
			e := a[i][j] - b[i][j]
			sum += e * e
			norm += a[i][j] * a[i][j]
			if math.Abs(e) > maxe {
				maxe = math.Abs(e)
			}
			n++
		}
	}
	return sum / float64(n), maxe, norm / float64(n)
}
func round(v float64) float64 { return math.Round(v*1e12) / 1e12 }

func RuntimeEnvelope() HardwareEnvelope {
	return HardwareEnvelope{OS: runtime.GOOS, Arch: runtime.GOARCH, Device: "host-cpu"}
}
func CandidateIDs(r Result) []string {
	ids := make([]string, len(r.Candidates))
	for i, c := range r.Candidates {
		ids[i] = c.ID
	}
	sort.Strings(ids)
	return ids
}
