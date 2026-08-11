package cubicquanteval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sort"
)

const (
	SchemaV1                 = "fak.cubicquanteval.fixture/v1"
	ScopeReconstruction      = "scalar-reconstruction"
	ScopeModelQuality        = "model-quality"
	ScopeHardwarePerformance = "hardware-performance"
)

type Outcome string

const (
	Supported   Outcome = "supported"
	Unsupported Outcome = "unsupported"
	Delegate    Outcome = "delegate"
)

type ReasonCode string

const (
	ReasonEvaluated           ReasonCode = "CUBICQUANT_EVALUATED"
	ReasonUnknownSchema       ReasonCode = "UNKNOWN_SCHEMA_VERSION"
	ReasonCombinationRejected ReasonCode = "CUBICQUANT_COMBINATION_REJECTED"
	ReasonProvenanceMismatch  ReasonCode = "PROVENANCE_MISMATCH"
	ReasonQualityReroute      ReasonCode = "CUBICQUANT_QUALITY_REROUTE"
	ReasonAcceleratorReroute  ReasonCode = "CUBICQUANT_ACCELERATOR_REROUTE"
)

type Artifact struct{ ID, URI, SHA256 string }
type Recipe struct {
	ID, Revision   string
	Seed           uint64   `json:"seed"`
	SamplesPerCase int      `json:"samples_per_case"`
	GroupSize      int      `json:"group_size"`
	BitWidths      []int    `json:"bit_widths"`
	Distributions  []string `json:"distributions"`
	CubicShapeStep float64  `json:"cubic_shape_step"`
	ScaleSteps     int      `json:"scale_steps"`
}
type RuntimeProvenance struct{ ID, Revision, Delegate string }
type ModelProvenance struct {
	ID, Revision  string
	WeightsSHA256 string `json:"weights_sha256"`
}
type Fixture struct {
	SchemaVersion string            `json:"schema_version"`
	Artifact      Artifact          `json:"artifact"`
	Recipe        Recipe            `json:"recipe"`
	Runtime       RuntimeProvenance `json:"runtime"`
	Model         ModelProvenance   `json:"model"`
}
type Request struct {
	Scope       string
	FixtureJSON []byte
}
type Row struct {
	Distribution            string  `json:"distribution"`
	Bits                    int     `json:"bits"`
	Groups                  int     `json:"groups"`
	CubicRMSE               float64 `json:"cubic_rmse"`
	TunedUniformRMSE        float64 `json:"tuned_uniform_rmse"`
	TunedNonUniformRMSE     float64 `json:"tuned_nonuniform_rmse"`
	VersusUniformPercent    float64 `json:"versus_uniform_percent"`
	VersusNonUniformPercent float64 `json:"versus_nonuniform_percent"`
	Decision                string  `json:"decision"`
}
type Result struct {
	Outcome          Outcome           `json:"outcome"`
	Reason           ReasonCode        `json:"reason"`
	Detail           string            `json:"detail"`
	Artifact         Artifact          `json:"artifact"`
	Recipe           Recipe            `json:"recipe"`
	Runtime          RuntimeProvenance `json:"runtime"`
	Model            ModelProvenance   `json:"model"`
	FixtureSHA256    string            `json:"fixture_sha256"`
	Evidence         string            `json:"evidence"`
	InputBasis       string            `json:"input_basis"`
	MeasuredEnvelope string            `json:"measured_envelope"`
	Rows             []Row             `json:"rows,omitempty"`
}

func Evaluate(req Request) Result {
	if req.Scope == ScopeModelQuality {
		return Result{Outcome: Delegate, Reason: ReasonQualityReroute, Detail: "run a pinned model/task quality evaluation; scalar synthetic reconstruction is not model quality"}
	}
	if req.Scope == ScopeHardwarePerformance {
		return Result{Outcome: Delegate, Reason: ReasonAcceleratorReroute, Detail: "dispatch the pinned kernel and workload to sanctioned lab compute; this CPU evaluator reports no throughput"}
	}
	if req.Scope != ScopeReconstruction {
		return Result{Outcome: Unsupported, Reason: ReasonCombinationRejected, Detail: "supported scopes: scalar-reconstruction; model-quality and hardware-performance delegate"}
	}
	var f Fixture
	sum := sha256.Sum256(req.FixtureJSON)
	base := Result{FixtureSHA256: hex.EncodeToString(sum[:]), Evidence: "observed", InputBasis: "modeled-synthetic", MeasuredEnvelope: runtime.GOOS + "/" + runtime.GOARCH + " " + runtime.Version() + "; CPU scalar reconstruction only"}
	if err := json.Unmarshal(req.FixtureJSON, &f); err != nil {
		base.Outcome = Unsupported
		base.Reason = ReasonProvenanceMismatch
		base.Detail = "fixture JSON: " + err.Error()
		return base
	}
	base.Artifact, base.Recipe, base.Runtime, base.Model = f.Artifact, f.Recipe, f.Runtime, f.Model
	if f.SchemaVersion != SchemaV1 {
		base.Outcome = Unsupported
		base.Reason = ReasonUnknownSchema
		base.Detail = "unknown fixture schema " + f.SchemaVersion
		return base
	}
	if err := validateFixture(f); err != nil {
		base.Outcome = Unsupported
		base.Reason = ReasonProvenanceMismatch
		base.Detail = err.Error()
		return base
	}
	rows, err := evaluateFixture(f)
	if err != nil {
		base.Outcome = Unsupported
		base.Reason = ReasonCombinationRejected
		base.Detail = err.Error()
		return base
	}
	base.Outcome, base.Reason, base.Rows = Supported, ReasonEvaluated, rows
	base.Detail = "bounded ledger observed from the pinned synthetic recipe; decisions concern reconstruction integration only"
	return base
}

func validateFixture(f Fixture) error {
	const paperHash = "245a523fd1b06203c123e6b03c39ed8e3cef107dd7cc33917b280511e71c9df0"
	if f.Artifact.ID != "arxiv:2608.06763v1" || f.Artifact.URI != "https://arxiv.org/abs/2608.06763v1" || f.Artifact.SHA256 != paperHash {
		return errors.New("artifact identity, URI, and PDF SHA-256 must match arXiv:2608.06763v1")
	}
	if f.Recipe.ID != "cubicquant-bounded-reconstruction-v1" || f.Recipe.Revision != "1" || f.Recipe.Seed != 42 || f.Recipe.GroupSize != 128 || f.Recipe.SamplesPerCase < 128 || f.Recipe.SamplesPerCase%f.Recipe.GroupSize != 0 || f.Recipe.CubicShapeStep != 0.25 || f.Recipe.ScaleSteps != 17 {
		return errors.New("recipe does not match bounded-reconstruction-v1")
	}
	if fmt.Sprint(f.Recipe.BitWidths) != "[1 2 3 4 5 6 7 8]" || fmt.Sprint(f.Recipe.Distributions) != "[uniform gaussian laplace]" {
		return errors.New("recipe must evaluate uniform/gaussian/laplace across W1-W8")
	}
	if f.Runtime.ID != "fak/internal/cubicquanteval" || f.Runtime.Revision != "contract-v1" || f.Runtime.Delegate != "cpu-go-stdlib" {
		return errors.New("runtime provenance mismatch")
	}
	if f.Model.ID != "synthetic-reference-distributions" || f.Model.Revision != "splitmix64-boxmuller-inversecdf-v1" || f.Model.WeightsSHA256 != "none" {
		return errors.New("model provenance mismatch")
	}
	return nil
}

func evaluateFixture(f Fixture) ([]Row, error) {
	var rows []Row
	for di, dist := range f.Recipe.Distributions {
		xs := samples(dist, f.Recipe.SamplesPerCase, f.Recipe.Seed+uint64(di)*0x9e3779b97f4a7c15)
		for _, bits := range f.Recipe.BitWidths {
			var ce, ue, ne float64
			for off := 0; off < len(xs); off += f.Recipe.GroupSize {
				group := xs[off : off+f.Recipe.GroupSize]
				c, u, n := groupErrors(group, bits, f.Recipe.CubicShapeStep, f.Recipe.ScaleSteps)
				ce += c
				ue += u
				ne += n
			}
			count := float64(len(xs))
			cr, ur, nr := math.Sqrt(ce/count), math.Sqrt(ue/count), math.Sqrt(ne/count)
			decision := "abstain"
			if cr+1e-12 < ur && cr+1e-12 < nr {
				decision = "integrate"
			}
			rows = append(rows, Row{dist, bits, len(xs) / f.Recipe.GroupSize, cr, ur, nr, percent(ur, cr), percent(nr, cr), decision})
		}
	}
	return rows, nil
}
func percent(baseline, candidate float64) float64 {
	if baseline == 0 {
		return 0
	}
	return 100 * (baseline - candidate) / baseline
}

// groupErrors returns squared errors for cubic, tuned uniform, and a tuned
// symmetric Lloyd-Max codebook. Lloyd-Max is an adaptive non-uniform reference,
// not an implementation-format or throughput baseline.
func groupErrors(x []float64, bits int, shapeStep float64, scaleSteps int) (float64, float64, float64) {
	if bits == 1 {
		e := binaryError(x)
		return e, e, e
	}
	m := (1 << (bits - 1)) - 1
	uniform := fitParametric(x, m, shapeStep, scaleSteps, false)
	cubic := fitParametric(x, m, shapeStep, scaleSteps, true)
	nonuniform := lloydError(x, m)
	return cubic, uniform, nonuniform
}
func binaryError(x []float64) float64 {
	var a float64
	for _, v := range x {
		a += math.Abs(v)
	}
	a /= float64(len(x))
	var e float64
	for _, v := range x {
		q := a
		if v < 0 {
			q = -a
		}
		d := v - q
		e += d * d
	}
	return e
}
func fitParametric(x []float64, m int, step float64, scaleSteps int, cubic bool) float64 {
	max := 0.0
	for _, v := range x {
		if math.Abs(v) > max {
			max = math.Abs(v)
		}
	}
	best := math.Inf(1)
	eval := func(a, b float64) {
		if !monotone(a, b) {
			return
		}
		for si := 0; si < scaleSteps; si++ {
			s := max * (0.55 + 0.6*float64(si)/float64(scaleSteps-1))
			var e float64
			for _, v := range x {
				av := math.Abs(v)
				bi := 0
				bd := av
				for i := 1; i <= m; i++ {
					t := float64(i) / float64(m)
					q := s * (a*t + b*t*t + (1-a-b)*t*t*t)
					d := math.Abs(av - q)
					if d < bd {
						bd = d
						bi = i
					}
				}
				t := float64(bi) / float64(m)
				q := s * (a*t + b*t*t + (1-a-b)*t*t*t)
				d := av - q
				e += d * d
			}
			if e < best {
				best = e
			}
		}
	}
	if !cubic {
		eval(1, 0)
		return best
	}
	for a := 0.0; a <= 2.0001; a += step {
		for b := -1.0; b <= 1.5001; b += step {
			eval(a, b)
		}
	}
	return best
}
func monotone(a, b float64) bool {
	c := 1 - a - b
	vals := []float64{a, 3 - 2*a - b}
	if c > 0 && b < 0 {
		t := -b / (3 * c)
		if t > 0 && t < 1 {
			vals = append(vals, a-b*b/(3*c))
		}
	}
	for _, v := range vals {
		if v < -1e-12 {
			return false
		}
	}
	return true
}
func lloydError(x []float64, m int) float64 {
	mags := make([]float64, len(x))
	for i, v := range x {
		mags[i] = math.Abs(v)
	}
	sort.Float64s(mags)
	levels := make([]float64, m+1)
	for i := 1; i <= m; i++ {
		idx := int(float64(i) / float64(m) * float64(len(mags)-1))
		levels[i] = mags[idx]
	}
	for it := 0; it < 40; it++ {
		sums := make([]float64, len(levels))
		counts := make([]int, len(levels))
		for _, v := range mags {
			j := nearest(v, levels)
			sums[j] += v
			counts[j]++
		}
		for j := 1; j < len(levels); j++ {
			if counts[j] > 0 {
				levels[j] = sums[j] / float64(counts[j])
			}
		}
	}
	var e float64
	for _, v := range mags {
		q := levels[nearest(v, levels)]
		d := v - q
		e += d * d
	}
	return e
}
func nearest(v float64, levels []float64) int {
	j := sort.Search(len(levels), func(i int) bool { return levels[i] >= v })
	if j == 0 {
		return 0
	}
	if j == len(levels) {
		return j - 1
	}
	if v-levels[j-1] <= levels[j]-v {
		return j - 1
	}
	return j
}

type rng struct{ x uint64 }

func (r *rng) next() float64 {
	r.x += 0x9e3779b97f4a7c15
	z := r.x
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return (float64(z>>11) + 0.5) / 9007199254740992.0
}
func samples(dist string, n int, seed uint64) []float64 {
	r := rng{seed}
	out := make([]float64, 0, n)
	for len(out) < n {
		u := r.next()
		switch dist {
		case "uniform":
			out = append(out, 2*u-1)
		case "gaussian":
			v := r.next()
			out = append(out, math.Sqrt(-2*math.Log(u))*math.Cos(2*math.Pi*v))
		case "laplace":
			if u < 0.5 {
				out = append(out, math.Log(2*u)/math.Sqrt2)
			} else {
				out = append(out, -math.Log(2*(1-u))/math.Sqrt2)
			}
		}
	}
	return out
}
