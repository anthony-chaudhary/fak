// Package computetune turns replayable workload profiles into compatibility-bound
// kernel selections. Tuning is deliberately offline; runtime code only performs
// immutable manifest lookup and falls back when no exact compatible entry exists.
package computetune

// The paired sign test in this file is adapted from magnitudedev/magnitude,
// inference/seismic/measure/src/statistics.rs at commit
// 8d7dbcc1baf4eb3ae7f1c3449dd463f3b375f576 (Apache-2.0). Changes were made: it
// is re-expressed in Go and bound to Tune's challenger-versus-fallback admission.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"time"
)

// Schema is the version tag embedded in tuning manifests and reports.
const Schema = "fak.compute-tuning/1"

// Operation is the bounded operation family supported by this first tuning spine.
type Operation string

// OpMatMul is the sole operation family this first tuning spine supports.
const OpMatMul Operation = "matmul"

// Compatibility binds a selection to the execution environment that produced it.
type Compatibility struct {
	Backend          string `json:"backend"`
	Device           string `json:"device"`
	SoftwareRevision string `json:"software_revision"`
	KernelRevision   string `json:"kernel_revision"`
}

// Profile is one replayable workload shape. M, N, and K use conventional GEMM dimensions.
type Profile struct {
	Operation Operation     `json:"operation"`
	M         int           `json:"m"`
	N         int           `json:"n"`
	K         int           `json:"k"`
	DType     string        `json:"dtype"`
	Compat    Compatibility `json:"compatibility"`
	Frequency uint64        `json:"frequency"`
	Weight    float64       `json:"weight"`
}

func (p Profile) validate() error {
	if p.Operation != OpMatMul || p.M <= 0 || p.N <= 0 || p.K <= 0 || p.DType == "" {
		return errors.New("computetune: invalid matmul profile")
	}
	if p.Compat.Backend == "" || p.Compat.Device == "" || p.Compat.SoftwareRevision == "" || p.Compat.KernelRevision == "" {
		return errors.New("computetune: incomplete compatibility tuple")
	}
	if math.IsNaN(p.Weight) || math.IsInf(p.Weight, 0) || p.Weight < 0 {
		return errors.New("computetune: invalid profile weight")
	}
	return nil
}

func (p Profile) key() string {
	return fmt.Sprintf("%s/%d/%d/%d/%s/%s/%s/%s/%s", p.Operation, p.M, p.N, p.K, p.DType, p.Compat.Backend, p.Compat.Device, p.Compat.SoftwareRevision, p.Compat.KernelRevision)
}

// Candidate is an offline-enumerated implementation of MatMul for one profile.
type Candidate interface {
	ID() string
	Run(context.Context, Profile) ([]float32, error)
}

// Correct checks a candidate result against the reference result.
type Correct func(got, reference []float32) error

// DefaultRoundingAlpha is the rounding-step error-ratio threshold (alpha = 8)
// adapted from magnitudedev/magnitude (comparison.rs:6, precision.rs:31).
const DefaultRoundingAlpha = 8.0

// RoundingCellWidth returns the rounding-cell width w_T(d) for the activation
// dtype at value d, and reports whether dtype requires bit-exact equality.
func RoundingCellWidth(dtype string, d float64) (width float64, bitExact bool) {
	switch dtype {
	case "bool", "packed", "i8", "i16", "i32", "i64", "int8", "int16", "int32", "int64",
		"u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64":
		return 0, true
	case "f64", "fp64", "float64":
		return floatMantissaStep(d, 52, -1022), false
	case "f32", "fp32", "float32", "":
		return floatMantissaStep(d, 23, -126), false
	case "f16", "fp16", "float16":
		return floatMantissaStep(d, 10, -14), false
	case "bf16", "bfloat16":
		return floatMantissaStep(d, 7, -126), false
	case "f8_e4m3", "fp8_e4m3", "e4m3":
		return floatMantissaStep(d, 3, -6), false
	case "f8_e5m2", "fp8_e5m2", "e5m2":
		return floatMantissaStep(d, 2, -14), false
	default:
		return 0, true
	}
}

func floatMantissaStep(v float64, mantissaBits, minNormalExp int) float64 {
	av := math.Abs(v)
	minNormal := math.Ldexp(1, minNormalExp)
	if av < minNormal {
		return math.Ldexp(1, minNormalExp-mantissaBits)
	}
	_, exp := math.Frexp(av) // av in [0.5, 1) * 2^exp, so unbiased exp is exp-1
	return math.Ldexp(1, (exp-1)-mantissaBits)
}

// RoundingStepCorrect returns a Correct check enforcing
// sqrt(sum((c-d)^2) / sum(w_T(d)^2/12)) <= alpha for floating activation
// dtypes, and bit-exact equality for integer, bool, packed, and non-finite values.
func RoundingStepCorrect(dtype string, alpha float64) Correct {
	if alpha <= 0 {
		alpha = DefaultRoundingAlpha
	}
	return func(got, reference []float32) error {
		if len(got) != len(reference) {
			return fmt.Errorf("length mismatch: got %d, want %d", len(got), len(reference))
		}
		var errPower, stepPower float64
		for i := range reference {
			c := float64(got[i])
			d := float64(reference[i])
			if math.IsNaN(c) || math.IsInf(c, 0) || math.IsNaN(d) || math.IsInf(d, 0) {
				if math.Float32bits(got[i]) != math.Float32bits(reference[i]) {
					return fmt.Errorf("non-finite bit mismatch at %d: got %g, want %g", i, c, d)
				}
				continue
			}
			w, exact := RoundingCellWidth(dtype, d)
			if exact {
				if math.Float32bits(got[i]) != math.Float32bits(reference[i]) {
					return fmt.Errorf("bit-exact mismatch at %d (%s): got %g, want %g", i, dtype, c, d)
				}
				continue
			}
			diff := c - d
			errPower += diff * diff
			stepPower += (w * w) / 12.0
		}
		if stepPower == 0 {
			if errPower != 0 {
				return fmt.Errorf("exact error power %g > 0", errPower)
			}
			return nil
		}
		ratio := math.Sqrt(errPower / stepPower)
		if ratio > alpha {
			return fmt.Errorf("rounding-step error ratio %.4g > alpha %.4g", ratio, alpha)
		}
		return nil
	}
}

// Measure times one candidate invocation in the declared timer domain. Hardware
// implementations may use device events rather than host wall time.
type Measure func(context.Context, Candidate, Profile) (time.Duration, error)

// AuditedMeasure is a Measure that also returns the output of the timed run, so
// Tune can check every timed sample against the reference.
type AuditedMeasure func(context.Context, Candidate, Profile) (time.Duration, []float32, error)

// Policy makes the amortization and timing contract explicit.
type Policy struct {
	Warmup            int           `json:"warmup"`
	Repeats           int           `json:"repeats"`
	Statistic         string        `json:"statistic"`
	TimerDomain       string        `json:"timer_domain"`
	FallbackCandidate string        `json:"fallback_candidate"`
	SelectionOverhead time.Duration `json:"selection_overhead"`
	// SignTestAlpha, when positive, makes a challenger earn its place over the
	// fallback through a paired sign test instead of a lower median alone. It is
	// the error rate of the whole selection for one profile: it is split evenly
	// over the challengers compared. Zero keeps median-only selection.
	SignTestAlpha float64 `json:"sign_test_alpha,omitempty"`
	// InputScales lists activation multipliers (e.g. 1, 0.01, -1) every candidate
	// must match the reference at before timing. Empty keeps today's behavior.
	InputScales []float64 `json:"input_scales,omitempty"`
	// RaggedVariants also checks M-1, N-1, K-1 and a clamp(1,3) shape before timing.
	RaggedVariants bool `json:"ragged_variants,omitempty"`
}

// SignTestResult is the receipt of one paired comparison against the fallback.
// Ratios holds each round's challenger cost over the fallback cost measured in
// that same round; a round is ahead when its ratio is below one. Alpha is the
// share of the policy's error rate this one comparison was run at.
type SignTestResult struct {
	Alpha  float64   `json:"alpha"`
	Rounds int       `json:"rounds"`
	Ahead  int       `json:"ahead"`
	Ratios []float64 `json:"ratios,omitempty"`
	Wealth float64   `json:"wealth"`
	Leads  bool      `json:"leads"`
}

// CandidateResult preserves failures instead of silently removing them.
type CandidateResult struct {
	CandidateID   string `json:"candidate_id"`
	Correct       bool   `json:"correct"`
	RefusalReason string `json:"refusal_reason,omitempty"`
	// VariantsUnchecked marks a candidate that could not run the requested scaled variants.
	VariantsUnchecked bool `json:"variants_unchecked,omitempty"`
	// TimingAudit is "audited" when every timed output was checked, else "unaudited".
	TimingAudit string `json:"timing_audit,omitempty"`
	// PostTimingCheck is "passed" or "failed" for the re-run after timing.
	PostTimingCheck string          `json:"post_timing_check,omitempty"`
	Samples         []time.Duration `json:"samples,omitempty"`
	Statistic       time.Duration   `json:"statistic,omitempty"`
	SignTest        *SignTestResult `json:"sign_test,omitempty"`
}

// ProfileResult is the inspectable tuning receipt for one profile.
type ProfileResult struct {
	Profile        Profile           `json:"profile"`
	Candidates     []CandidateResult `json:"candidates"`
	SelectedID     string            `json:"selected_id"`
	BreakEvenReuse uint64            `json:"break_even_reuse_count"`
}

// Report captures policy, all candidate outcomes, and manifest provenance.
type Report struct {
	Schema         string          `json:"schema"`
	Policy         Policy          `json:"policy"`
	Profiles       []ProfileResult `json:"profiles"`
	ManifestDigest string          `json:"manifest_digest"`
}

// Entry is a compatibility-bound immutable selection record.
type Entry struct {
	Profile        Profile `json:"profile"`
	CandidateID    string  `json:"candidate_id"`
	BreakEvenReuse uint64  `json:"break_even_reuse_count"`
}

// Manifest has no mutator; NewManifest and Entries defensively copy its entries.
type Manifest struct{ entries []Entry }

// NewManifest validates entries and returns an immutable Manifest sorted by
// profile key. It rejects empty candidate IDs, duplicate profiles, and invalid
// profiles so Lookup can binary-search the sorted slice.
func NewManifest(entries []Entry) (Manifest, error) {
	seen := make(map[string]bool, len(entries))
	copyEntries := slices.Clone(entries)
	for _, e := range copyEntries {
		if err := e.Profile.validate(); err != nil {
			return Manifest{}, err
		}
		if e.CandidateID == "" || seen[e.Profile.key()] {
			return Manifest{}, errors.New("computetune: empty candidate or duplicate profile")
		}
		seen[e.Profile.key()] = true
	}
	sort.Slice(copyEntries, func(i, j int) bool { return copyEntries[i].Profile.key() < copyEntries[j].Profile.key() })
	return Manifest{entries: copyEntries}, nil
}

// Entries returns a defensive copy of the manifest entries.
func (m Manifest) Entries() []Entry { return slices.Clone(m.entries) }

// Lookup returns the candidate ID bound to exactly this profile. It fails
// closed: invalid or unknown profiles yield ok=false so callers fall back.
func (m Manifest) Lookup(p Profile) (string, bool) {
	if p.validate() != nil {
		return "", false
	}
	i, ok := slices.BinarySearchFunc(m.entries, p.key(), func(e Entry, key string) int {
		if e.Profile.key() < key {
			return -1
		}
		if e.Profile.key() > key {
			return 1
		}
		return 0
	})
	if !ok {
		return "", false
	}
	return m.entries[i].CandidateID, true
}

// MarshalJSON encodes the manifest with its schema tag, giving Digest a
// stable canonical form independent of entry order.
func (m Manifest) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Schema  string  `json:"schema"`
		Entries []Entry `json:"entries"`
	}{Schema: Schema, Entries: m.Entries()})
}

// Digest returns the sha256 hex of the canonical JSON encoding, prefixed
// with "sha256:", for binding reports to their manifest contents.
func (m Manifest) Digest() (string, error) {
	b, err := m.MarshalJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// Candidate validation precedes benchmark timing, and lookup fails closed.
// Tune deterministically enumerates candidates, validates output before any
// timing, then selects the lowest median. Failed candidates remain in the report.
// With Policy.SignTestAlpha set, a lower median only nominates a challenger: it
// replaces the fallback when it also leads a paired sign test against it.
func Tune(ctx context.Context, profiles []Profile, candidates []Candidate, reference Candidate, correct Correct, measure Measure, policy Policy) (Manifest, Report, error) {
	return tune(ctx, profiles, candidates, reference, correct, measure, nil, policy)
}

// TuneAudited is Tune with every timed sample's output checked against the
// reference: a candidate whose timed output mismatches is refused with a
// "timed output mismatch" reason and cannot win.
func TuneAudited(ctx context.Context, profiles []Profile, candidates []Candidate, reference Candidate, correct Correct, measure AuditedMeasure, policy Policy) (Manifest, Report, error) {
	if measure == nil {
		return Manifest{}, Report{}, errors.New("computetune: incomplete tuning inputs")
	}
	return tune(ctx, profiles, candidates, reference, correct, nil, measure, policy)
}

// timedMismatchError marks an audited timed sample whose output failed correct.
type timedMismatchError struct{ err error }

func (e timedMismatchError) Error() string { return e.err.Error() }

func timedReason(stage string, err error) string {
	var m timedMismatchError
	if errors.As(err, &m) {
		return "timed output mismatch: " + err.Error()
	}
	return stage + ": " + err.Error()
}

func tune(ctx context.Context, profiles []Profile, candidates []Candidate, reference Candidate, correct Correct, plain Measure, audited AuditedMeasure, policy Policy) (Manifest, Report, error) {
	if len(profiles) == 0 || len(candidates) == 0 || reference == nil || (plain == nil && audited == nil) {
		return Manifest{}, Report{}, errors.New("computetune: incomplete tuning inputs")
	}
	if policy.Warmup < 0 || policy.Repeats <= 0 || policy.Statistic != "median" || policy.TimerDomain == "" || policy.FallbackCandidate == "" || policy.SelectionOverhead < 0 || !(policy.SignTestAlpha >= 0 && policy.SignTestAlpha < 1) {
		return Manifest{}, Report{}, errors.New("computetune: invalid policy")
	}
	ids := make(map[string]bool, len(candidates))
	byID := make(map[string]Candidate, len(candidates))
	for _, c := range candidates {
		if c == nil || c.ID() == "" || ids[c.ID()] {
			return Manifest{}, Report{}, errors.New("computetune: invalid candidate set")
		}
		ids[c.ID()] = true
		byID[c.ID()] = c
	}
	order := newRoundOrder()
	if !ids[policy.FallbackCandidate] {
		return Manifest{}, Report{}, errors.New("computetune: fallback is not a candidate")
	}

	report := Report{Schema: Schema, Policy: policy}
	entries := make([]Entry, 0, len(profiles))
	for _, profile := range profiles {
		if err := profile.validate(); err != nil {
			return Manifest{}, Report{}, err
		}
		profileCorrect := correct
		if profileCorrect == nil {
			profileCorrect = RoundingStepCorrect(profile.DType, DefaultRoundingAlpha)
		}
		want, err := reference.Run(ctx, profile)
		if err != nil {
			return Manifest{}, Report{}, fmt.Errorf("computetune: reference: %w", err)
		}
		measure := plain
		if audited != nil {
			measure = func(ctx context.Context, c Candidate, p Profile) (time.Duration, error) {
				d, out, err := audited(ctx, c, p)
				if err != nil {
					return 0, err
				}
				if cerr := profileCorrect(out, want); cerr != nil {
					return 0, timedMismatchError{cerr}
				}
				return d, nil
			}
		}
		pr := ProfileResult{Profile: profile}
		for _, candidate := range candidates {
			cr := CandidateResult{CandidateID: candidate.ID()}
			got, err := candidate.Run(ctx, profile)
			if err != nil {
				cr.RefusalReason = "correctness execution: " + err.Error()
				pr.Candidates = append(pr.Candidates, cr)
				continue
			}
			if err := profileCorrect(got, want); err != nil {
				cr.RefusalReason = "incorrect result: " + err.Error()
				pr.Candidates = append(pr.Candidates, cr)
				continue
			}
			if reason, unchecked := checkVariants(ctx, profile, candidate, reference, profileCorrect, policy); reason != "" {
				cr.RefusalReason = reason
				pr.Candidates = append(pr.Candidates, cr)
				continue
			} else {
				cr.VariantsUnchecked = unchecked
			}
			cr.Correct = true
			cr.TimingAudit = "unaudited"
			if audited != nil {
				cr.TimingAudit = "audited"
			}
			failed := false
			for i := 0; i < policy.Warmup; i++ {
				if _, err := measure(ctx, candidate, profile); err != nil {
					cr.RefusalReason = timedReason("warmup", err)
					cr.Correct = false
					failed = true
					break
				}
			}
			for i := 0; !failed && i < policy.Repeats; i++ {
				d, err := measure(ctx, candidate, profile)
				if err != nil || d <= 0 {
					if err == nil {
						err = errors.New("non-positive duration")
					}
					cr.RefusalReason = timedReason("timing", err)
					cr.Correct = false
					failed = true
					break
				}
				cr.Samples = append(cr.Samples, d)
			}
			if !failed {
				cr.PostTimingCheck = "passed"
				post, err := candidate.Run(ctx, profile)
				if err == nil {
					err = profileCorrect(post, want)
				}
				if err != nil {
					cr.PostTimingCheck = "failed"
					cr.RefusalReason = "post-timing parity: " + err.Error()
					cr.Correct = false
					failed = true
				}
			}
			if !failed {
				cr.Statistic = median(cr.Samples)
			}
			pr.Candidates = append(pr.Candidates, cr)
		}
		fallback := resultByID(pr.Candidates, policy.FallbackCandidate)
		winner := fallback
		for i := range pr.Candidates {
			r := &pr.Candidates[i]
			if r.Correct && r.Statistic > 0 && (winner == nil || !winner.Correct || r.Statistic < winner.Statistic || (r.Statistic == winner.Statistic && r.CandidateID < winner.CandidateID)) {
				winner = r
			}
		}
		if fallback == nil || !fallback.Correct {
			return Manifest{}, Report{}, fmt.Errorf("computetune: fallback %q failed for %s", policy.FallbackCandidate, profile.key())
		}
		if winner == nil {
			winner = fallback
		}
		if policy.SignTestAlpha > 0 && winner != fallback {
			winner, err = admitBySignTest(ctx, measure, byID, pr.Candidates, fallback, profile, policy.SignTestAlpha, &order)
			if err != nil {
				return Manifest{}, Report{}, err
			}
		}
		pr.SelectedID = winner.CandidateID
		if winner.Statistic < fallback.Statistic {
			pr.BreakEvenReuse = uint64(math.Ceil(float64(policy.SelectionOverhead) / float64(fallback.Statistic-winner.Statistic)))
		}
		entries = append(entries, Entry{Profile: profile, CandidateID: pr.SelectedID, BreakEvenReuse: pr.BreakEvenReuse})
		report.Profiles = append(report.Profiles, pr)
	}
	manifest, err := NewManifest(entries)
	if err != nil {
		return Manifest{}, Report{}, err
	}
	report.ManifestDigest, err = manifest.Digest()
	return manifest, report, err
}

// signTestWealth is M_n(s), the uniform mixture over p in (1/2, 1) of the
// likelihood ratio of "a round is ahead with probability p" against one half:
// the sum over j <= s of C(n+1, j), divided by (n+1)*C(n, s). While the median
// ratio is not below one, a round is ahead with probability at most one half
// whatever the timing distribution, so M_n is a nonnegative supermartingale
// starting at one and by Ville's inequality ever reaches 1/alpha with
// probability at most alpha. That bound covers every look together, so the
// comparison may be read after each round. Each term is derived from the one
// above it, so no binomial coefficient is formed along the way.
func signTestWealth(rounds, favourable int) float64 {
	n, s := float64(rounds), float64(favourable)
	term := (n + 1) / (n + 1 - s)
	sum := 0.0
	for j := favourable; j >= 0; j-- {
		sum += term
		term *= float64(j) / (n + 2 - float64(j))
	}
	return sum / (n + 1)
}

// roundOrder picks which side of a paired round is timed first (xorshift64*).
// A device whose speed is still drifting gives the later slot of a round a
// different clock; a fixed order would favour one side every round, while a
// random one favours each side equally often.
type roundOrder uint64

func newRoundOrder() roundOrder { return 0x9e3779b97f4a7c15 }

func (o *roundOrder) challengerFirst() bool {
	x := uint64(*o)
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	*o = roundOrder(x)
	return (x*0x2545f4914f6cdd1d)>>33&1 == 0
}

func measurePositive(ctx context.Context, measure Measure, c Candidate, p Profile) (time.Duration, error) {
	d, err := measure(ctx, c, p)
	if err != nil {
		return 0, err
	}
	if d <= 0 {
		return 0, errors.New("non-positive duration")
	}
	return d, nil
}

// pairedSignTest runs rounds that each time the challenger and the incumbent
// together, and reads the test after every round. It ends when the challenger
// leads at error rate alpha or at the first round it is not ahead: a lead in the
// fewest rounds needs every round ahead, so a miss is taken as no lead rather
// than paid for with more rounds. Timing errors are returned per side.
func pairedSignTest(ctx context.Context, measure Measure, challenger, incumbent Candidate, p Profile, alpha float64, order *roundOrder) (res SignTestResult, challengerErr, incumbentErr error) {
	res = SignTestResult{Alpha: alpha, Wealth: 1}
	for {
		var c, i time.Duration
		if order.challengerFirst() {
			if c, challengerErr = measurePositive(ctx, measure, challenger, p); challengerErr == nil {
				i, incumbentErr = measurePositive(ctx, measure, incumbent, p)
			}
		} else {
			if i, incumbentErr = measurePositive(ctx, measure, incumbent, p); incumbentErr == nil {
				c, challengerErr = measurePositive(ctx, measure, challenger, p)
			}
		}
		if challengerErr != nil || incumbentErr != nil {
			return res, challengerErr, incumbentErr
		}
		ratio := float64(c) / float64(i)
		res.Rounds++
		if ratio < 1 {
			res.Ahead++
		}
		res.Ratios = append(res.Ratios, ratio)
		res.Wealth = signTestWealth(res.Rounds, res.Ahead)
		if res.Wealth*alpha >= 1 {
			res.Leads = true
			return res, nil, nil
		}
		if ratio >= 1 {
			return res, nil, nil
		}
	}
}

// admitBySignTest tries the challengers whose median beats the fallback, fastest
// first, and returns the first that leads its paired sign test, or the fallback
// when none does. Each comparison runs at alpha over the number of challengers,
// so the chance that any of them wrongly replaces the fallback stays within
// alpha. A challenger whose paired timing fails is refused in place; a fallback
// that cannot be timed fails the tuning, as it does everywhere else.
func admitBySignTest(ctx context.Context, measure Measure, byID map[string]Candidate, results []CandidateResult, fallback *CandidateResult, p Profile, alpha float64, order *roundOrder) (*CandidateResult, error) {
	var challengers []*CandidateResult
	for i := range results {
		r := &results[i]
		if r != fallback && r.Correct && r.Statistic > 0 && r.Statistic < fallback.Statistic {
			challengers = append(challengers, r)
		}
	}
	sort.Slice(challengers, func(a, b int) bool {
		if challengers[a].Statistic != challengers[b].Statistic {
			return challengers[a].Statistic < challengers[b].Statistic
		}
		return challengers[a].CandidateID < challengers[b].CandidateID
	})
	share := alpha / float64(len(challengers))
	for _, r := range challengers {
		res, challengerErr, fallbackErr := pairedSignTest(ctx, measure, byID[r.CandidateID], byID[fallback.CandidateID], p, share, order)
		if fallbackErr != nil {
			return nil, fmt.Errorf("computetune: fallback %q failed a paired round for %s: %w", fallback.CandidateID, p.key(), fallbackErr)
		}
		r.SignTest = &res
		if challengerErr != nil {
			r.RefusalReason = timedReason("sign test timing", challengerErr)
			r.Correct = false
			r.Statistic = 0
			continue
		}
		if res.Leads {
			return r, nil
		}
	}
	return fallback, nil
}

func resultByID(results []CandidateResult, id string) *CandidateResult {
	for i := range results {
		if results[i].CandidateID == id {
			return &results[i]
		}
	}
	return nil
}
func median(samples []time.Duration) time.Duration {
	s := slices.Clone(samples)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// DispatchMatMul is the latency-path consumer: exact immutable lookup only, no
// exploration. Missing/incompatible manifests or unavailable winners use fallback.
func DispatchMatMul(ctx context.Context, profile Profile, manifest Manifest, candidates map[string]Candidate, fallback Candidate) ([]float32, string, error) {
	if fallback == nil {
		return nil, "", errors.New("computetune: nil fallback")
	}
	chosen := fallback
	if id, ok := manifest.Lookup(profile); ok {
		if c := candidates[id]; c != nil {
			chosen = c
		}
	}
	out, err := chosen.Run(ctx, profile)
	return out, chosen.ID(), err
}
