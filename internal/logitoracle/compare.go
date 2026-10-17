// Package logitoracle dumps full next-token logit vectors from a native model
// session and compares two such dumps position by position. It is the
// whole-model numerical oracle for kernel default flips: two configurations of
// one artifact are judged by cosine, relative L2, KL, top-k agreement and a
// tie-aware greedy equivalence, never by greedy text identity across processes.
package logitoracle

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var (
	ErrEmptyLogits     = errors.New("logitoracle: empty logit vector")
	ErrLengthMismatch  = errors.New("logitoracle: logit vector lengths differ")
	ErrNonFinite       = errors.New("logitoracle: non-finite logit")
	ErrPromptMismatch  = errors.New("logitoracle: prompt token ids differ")
	ErrMissingPrompt   = errors.New("logitoracle: prompt absent from dump")
	ErrSchema          = errors.New("logitoracle: unsupported dump schema")
	ErrVocabMismatch   = errors.New("logitoracle: vocabulary sizes differ")
	ErrNoPrompts       = errors.New("logitoracle: no prompts")
	ErrBadThresholds   = errors.New("logitoracle: invalid thresholds")
	ErrTeacherMismatch = errors.New("logitoracle: teacher prompt differs from requested prompt")
	ErrInvalidHistory  = errors.New("logitoracle: invalid fed-token history")
	ErrLogitsChecksum  = errors.New("logitoracle: logits checksum missing, invalid or mismatched")
)

// Thresholds is the declared gate. A divergent greedy token counts as a tie only
// when both distributions rank the two tokens within TieEpsilon logits of each
// other.
type Thresholds struct {
	CosineMin  float64 `json:"cosine_min"`
	RelL2Max   float64 `json:"relative_l2_max"`
	TieEpsilon float64 `json:"tie_epsilon"`
	TopK       int     `json:"top_k"`
}

// StrictThresholds is the fak#12735 whole-model logit gate.
var StrictThresholds = Thresholds{CosineMin: 0.99999, RelL2Max: 1e-4, TieEpsilon: 0, TopK: 5}

// FlipThresholds is the gate declared for register-resident GDN default
// promotion before any whole-model observation (2026-10-17).
var FlipThresholds = Thresholds{CosineMin: 0.9999, RelL2Max: 0.01, TieEpsilon: 0.1, TopK: 5}

func (t Thresholds) validate() error {
	if t.TopK < 1 || t.CosineMin < -1 || t.CosineMin > 1 || t.RelL2Max < 0 || t.TieEpsilon < 0 ||
		math.IsNaN(t.CosineMin) || math.IsNaN(t.RelL2Max) || math.IsNaN(t.TieEpsilon) {
		return ErrBadThresholds
	}
	return nil
}

// PositionStats compares one candidate logit vector against the reference.
type PositionStats struct {
	Position      int     `json:"position"`
	Cosine        float64 `json:"cosine"`
	RelL2         float64 `json:"relative_l2"`
	MaxAbs        float64 `json:"max_abs"`
	KL            float64 `json:"kl_ref_cand"`
	RefTop1       int     `json:"ref_top1"`
	CandTop1      int     `json:"cand_top1"`
	RefTop2Margin float64 `json:"ref_top2_margin"`
	RefGap        float64 `json:"ref_gap"`
	CandGap       float64 `json:"cand_gap"`
	TopKOverlap   float64 `json:"topk_overlap"`
	Top1Agree     bool    `json:"top1_agree"`
	Tie           bool    `json:"tie"`
}

// ComparePosition computes the per-position statistics. RefGap is
// ref[refTop1]-ref[candTop1] and CandGap is cand[candTop1]-cand[refTop1]; a
// top-1 mismatch is a tie when both are at most eps.
func ComparePosition(ref, cand []float32, topK int, eps float64) (PositionStats, error) {
	var st PositionStats
	if len(ref) == 0 || len(cand) == 0 {
		return st, ErrEmptyLogits
	}
	if len(ref) != len(cand) {
		return st, fmt.Errorf("%w: ref=%d cand=%d", ErrLengthMismatch, len(ref), len(cand))
	}
	if topK < 1 {
		return st, ErrBadThresholds
	}
	var dot, refNorm, candNorm, errNorm, maxAbs float64
	refMax, candMax := math.Inf(-1), math.Inf(-1)
	for i := range ref {
		r, c := float64(ref[i]), float64(cand[i])
		if math.IsNaN(r) || math.IsInf(r, 0) || math.IsNaN(c) || math.IsInf(c, 0) {
			return st, fmt.Errorf("%w at index %d", ErrNonFinite, i)
		}
		dot += r * c
		refNorm += r * r
		candNorm += c * c
		d := c - r
		errNorm += d * d
		if a := math.Abs(d); a > maxAbs {
			maxAbs = a
		}
		refMax = math.Max(refMax, r)
		candMax = math.Max(candMax, c)
	}
	switch {
	case refNorm == 0 && candNorm == 0:
		st.Cosine = 1
	case refNorm == 0 || candNorm == 0:
		st.Cosine = 0
	default:
		st.Cosine = dot / math.Sqrt(refNorm*candNorm)
	}
	switch {
	case refNorm > 0:
		st.RelL2 = math.Sqrt(errNorm / refNorm)
	case errNorm == 0:
		st.RelL2 = 0
	default:
		st.RelL2 = math.Inf(1)
	}
	st.MaxAbs = maxAbs
	st.KL = klDivergence(ref, cand, refMax, candMax)
	k := min(topK, len(ref))
	refTop, candTop := TopK(ref, k), TopK(cand, k)
	st.RefTop1, st.CandTop1 = refTop[0], candTop[0]
	if len(ref) > 1 {
		second := TopK(ref, 2)[1]
		st.RefTop2Margin = float64(ref[st.RefTop1]) - float64(ref[second])
	}
	st.RefGap = float64(ref[st.RefTop1]) - float64(ref[st.CandTop1])
	st.CandGap = float64(cand[st.CandTop1]) - float64(cand[st.RefTop1])
	st.Top1Agree = st.RefTop1 == st.CandTop1
	st.Tie = !st.Top1Agree && st.RefGap <= eps && st.CandGap <= eps
	inRef := make(map[int]struct{}, k)
	for _, id := range refTop {
		inRef[id] = struct{}{}
	}
	shared := 0
	for _, id := range candTop {
		if _, ok := inRef[id]; ok {
			shared++
		}
	}
	st.TopKOverlap = float64(shared) / float64(k)
	return st, nil
}

// klDivergence is KL(softmax(ref) || softmax(cand)) in nats.
func klDivergence(ref, cand []float32, refMax, candMax float64) float64 {
	var refSum, candSum float64
	for i := range ref {
		refSum += math.Exp(float64(ref[i]) - refMax)
		candSum += math.Exp(float64(cand[i]) - candMax)
	}
	refLogZ, candLogZ := refMax+math.Log(refSum), candMax+math.Log(candSum)
	var kl float64
	for i := range ref {
		lr := float64(ref[i]) - refLogZ
		p := math.Exp(lr)
		if p == 0 {
			continue
		}
		kl += p * (lr - (float64(cand[i]) - candLogZ))
	}
	if kl < 0 {
		return 0
	}
	return kl
}

// TopK returns the indices of the k largest values, ties broken by lower index.
func TopK(v []float32, k int) []int {
	k = min(k, len(v))
	if k <= 0 {
		return nil
	}
	idx := make([]int, 0, k)
	for i, x := range v {
		if len(idx) < k {
			idx = append(idx, i)
			sort.SliceStable(idx, func(a, b int) bool { return before(v, idx[a], idx[b]) })
			continue
		}
		last := idx[k-1]
		if x > v[last] {
			idx[k-1] = i
			for j := k - 1; j > 0 && before(v, idx[j], idx[j-1]); j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
	}
	return idx
}

func before(v []float32, a, b int) bool {
	if v[a] != v[b] {
		return v[a] > v[b]
	}
	return a < b
}

// Greedy equivalence verdicts.
const (
	GreedyIdentical     = "IDENTICAL"
	GreedyTieEquivalent = "TIE_EQUIVALENT"
	GreedyNotEquivalent = "NOT_EQUIVALENT"
)

// Gate verdicts.
const (
	VerdictPass       = "PASS"
	VerdictFail       = "FAIL"
	VerdictIncomplete = "INCOMPLETE"
)

// PromptReport folds the compared positions of one prompt.
type PromptReport struct {
	Name            string          `json:"name"`
	Positions       int             `json:"positions"`
	Compared        int             `json:"compared"`
	Excluded        int             `json:"excluded"`
	FirstFedDiverge *int            `json:"first_fed_divergence,omitempty"`
	CosineMin       float64         `json:"cosine_min"`
	RelL2Max        float64         `json:"relative_l2_max"`
	RelL2Median     float64         `json:"relative_l2_median"`
	KLMax           float64         `json:"kl_max"`
	KLMean          float64         `json:"kl_mean"`
	MaxAbs          float64         `json:"max_abs"`
	Top1Agree       int             `json:"top1_agree"`
	TopKOverlapMean float64         `json:"topk_overlap_mean"`
	TopKOverlapMin  float64         `json:"topk_overlap_min"`
	Mismatches      []PositionStats `json:"mismatches,omitempty"`
	Worst           PositionStats   `json:"worst"`
	Greedy          string          `json:"greedy"`
	StrictPass      bool            `json:"strict_pass"`
	GatePass        bool            `json:"gate_pass"`
}

// Accumulator folds PositionStats for one prompt.
type Accumulator struct {
	name  string
	stats []PositionStats
}

func NewAccumulator(name string) *Accumulator { return &Accumulator{name: name} }

func (a *Accumulator) Add(st PositionStats) { a.stats = append(a.stats, st) }

// Fold produces the prompt report under the declared and strict gates.
func (a *Accumulator) Fold(positions, excluded int, th Thresholds) PromptReport {
	r := PromptReport{Name: a.name, Positions: positions, Compared: len(a.stats), Excluded: excluded,
		CosineMin: 1, TopKOverlapMin: 1, Greedy: GreedyIdentical}
	if len(a.stats) == 0 {
		r.Greedy = GreedyNotEquivalent
		return r
	}
	rel := make([]float64, 0, len(a.stats))
	var klSum, overlapSum float64
	strict, gate := true, true
	for i, st := range a.stats {
		if i == 0 || st.Cosine < r.Worst.Cosine {
			r.Worst = st
		}
		r.CosineMin = math.Min(r.CosineMin, st.Cosine)
		r.RelL2Max = math.Max(r.RelL2Max, st.RelL2)
		r.KLMax = math.Max(r.KLMax, st.KL)
		r.MaxAbs = math.Max(r.MaxAbs, st.MaxAbs)
		r.TopKOverlapMin = math.Min(r.TopKOverlapMin, st.TopKOverlap)
		klSum += st.KL
		overlapSum += st.TopKOverlap
		rel = append(rel, st.RelL2)
		if st.Top1Agree {
			r.Top1Agree++
		} else {
			r.Mismatches = append(r.Mismatches, st)
			if st.Tie {
				if r.Greedy == GreedyIdentical {
					r.Greedy = GreedyTieEquivalent
				}
			} else {
				r.Greedy = GreedyNotEquivalent
			}
		}
		strict = strict && st.Cosine >= StrictThresholds.CosineMin && st.RelL2 <= StrictThresholds.RelL2Max
		gate = gate && st.Cosine >= th.CosineMin && st.RelL2 <= th.RelL2Max
	}
	sort.Float64s(rel)
	r.RelL2Median = rel[len(rel)/2]
	if len(rel)%2 == 0 {
		r.RelL2Median = (rel[len(rel)/2-1] + rel[len(rel)/2]) / 2
	}
	r.KLMean = klSum / float64(len(a.stats))
	r.TopKOverlapMean = overlapSum / float64(len(a.stats))
	r.StrictPass = strict && r.Greedy == GreedyIdentical
	r.GatePass = gate && r.Greedy != GreedyNotEquivalent
	return r
}

// Report is the whole comparison receipt.
type Report struct {
	Schema        string         `json:"schema"`
	Reference     string         `json:"reference"`
	Candidate     string         `json:"candidate"`
	Thresholds    Thresholds     `json:"thresholds"`
	Strict        Thresholds     `json:"strict_thresholds"`
	Prompts       []PromptReport `json:"prompts"`
	Greedy        string         `json:"greedy"`
	Verdict       string         `json:"verdict"`
	StrictVerdict string         `json:"strict_verdict"`
}

// ReportSchema names the comparison receipt format.
const ReportSchema = "fak.logit-oracle.compare/v1"

// FoldReport combines prompt reports. Excluded positions (a free-running
// candidate whose fed tokens diverged) make an otherwise passing gate
// INCOMPLETE, because those positions were never compared.
func FoldReport(prompts []PromptReport, th Thresholds) Report {
	r := Report{Schema: ReportSchema, Thresholds: th, Strict: StrictThresholds, Prompts: prompts,
		Greedy: GreedyIdentical, Verdict: VerdictPass, StrictVerdict: VerdictPass}
	if len(prompts) == 0 {
		r.Greedy, r.Verdict, r.StrictVerdict = GreedyNotEquivalent, VerdictFail, VerdictFail
		return r
	}
	incomplete := false
	for _, p := range prompts {
		switch p.Greedy {
		case GreedyNotEquivalent:
			r.Greedy = GreedyNotEquivalent
		case GreedyTieEquivalent:
			if r.Greedy == GreedyIdentical {
				r.Greedy = GreedyTieEquivalent
			}
		}
		if !p.GatePass {
			r.Verdict = VerdictFail
		}
		if !p.StrictPass {
			r.StrictVerdict = VerdictFail
		}
		incomplete = incomplete || p.Excluded > 0
	}
	if incomplete && r.Verdict == VerdictPass {
		r.Verdict = VerdictIncomplete
	}
	if incomplete && r.StrictVerdict == VerdictPass {
		r.StrictVerdict = VerdictIncomplete
	}
	return r
}
