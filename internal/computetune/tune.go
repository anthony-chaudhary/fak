// Package computetune turns replayable workload profiles into compatibility-bound
// kernel selections. Tuning is deliberately offline; runtime code only performs
// immutable manifest lookup and falls back when no exact compatible entry exists.
package computetune

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

const Schema = "fak.compute-tuning/1"

// Operation is the bounded operation family supported by this first tuning spine.
type Operation string

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

// Measure times one candidate invocation in the declared timer domain. Hardware
// implementations may use device events rather than host wall time.
type Measure func(context.Context, Candidate, Profile) (time.Duration, error)

// Policy makes the amortization and timing contract explicit.
type Policy struct {
	Warmup            int           `json:"warmup"`
	Repeats           int           `json:"repeats"`
	Statistic         string        `json:"statistic"`
	TimerDomain       string        `json:"timer_domain"`
	FallbackCandidate string        `json:"fallback_candidate"`
	SelectionOverhead time.Duration `json:"selection_overhead"`
}

// CandidateResult preserves failures instead of silently removing them.
type CandidateResult struct {
	CandidateID   string          `json:"candidate_id"`
	Correct       bool            `json:"correct"`
	RefusalReason string          `json:"refusal_reason,omitempty"`
	Samples       []time.Duration `json:"samples,omitempty"`
	Statistic     time.Duration   `json:"statistic,omitempty"`
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

func (m Manifest) Entries() []Entry { return slices.Clone(m.entries) }

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

func (m Manifest) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Schema  string  `json:"schema"`
		Entries []Entry `json:"entries"`
	}{Schema: Schema, Entries: m.Entries()})
}

func (m Manifest) Digest() (string, error) {
	b, err := m.MarshalJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// Invariant: candidate validation precedes benchmark timing, and lookup is fail-closed.
// Tune deterministically enumerates candidates, validates output before any
// timing, then selects the lowest median. Failed candidates remain in the report.
func Tune(ctx context.Context, profiles []Profile, candidates []Candidate, reference Candidate, correct Correct, measure Measure, policy Policy) (Manifest, Report, error) {
	if len(profiles) == 0 || len(candidates) == 0 || reference == nil || correct == nil || measure == nil {
		return Manifest{}, Report{}, errors.New("computetune: incomplete tuning inputs")
	}
	if policy.Warmup < 0 || policy.Repeats <= 0 || policy.Statistic != "median" || policy.TimerDomain == "" || policy.FallbackCandidate == "" || policy.SelectionOverhead < 0 {
		return Manifest{}, Report{}, errors.New("computetune: invalid policy")
	}
	ids := make(map[string]bool, len(candidates))
	for _, c := range candidates {
		if c == nil || c.ID() == "" || ids[c.ID()] {
			return Manifest{}, Report{}, errors.New("computetune: invalid candidate set")
		}
		ids[c.ID()] = true
	}
	if !ids[policy.FallbackCandidate] {
		return Manifest{}, Report{}, errors.New("computetune: fallback is not a candidate")
	}

	report := Report{Schema: Schema, Policy: policy}
	entries := make([]Entry, 0, len(profiles))
	for _, profile := range profiles {
		if err := profile.validate(); err != nil {
			return Manifest{}, Report{}, err
		}
		want, err := reference.Run(ctx, profile)
		if err != nil {
			return Manifest{}, Report{}, fmt.Errorf("computetune: reference: %w", err)
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
			if err := correct(got, want); err != nil {
				cr.RefusalReason = "incorrect result: " + err.Error()
				pr.Candidates = append(pr.Candidates, cr)
				continue
			}
			cr.Correct = true
			failed := false
			for i := 0; i < policy.Warmup; i++ {
				if _, err := measure(ctx, candidate, profile); err != nil {
					cr.RefusalReason = "warmup: " + err.Error()
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
					cr.RefusalReason = "timing: " + err.Error()
					cr.Correct = false
					failed = true
					break
				}
				cr.Samples = append(cr.Samples, d)
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
