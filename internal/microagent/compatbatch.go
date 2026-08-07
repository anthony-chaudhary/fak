package microagent

import (
	"errors"
	"sort"
	"strings"
	"time"
)

// CompatibilityKey names model-turn properties that must match before work can
// share a physical prefill/decode batch. Empty required fields fail open to a
// singleton instead of guessing compatibility.
type CompatibilityKey struct {
	Model, Sampling, Tools, Prefix, Phase string
	SequenceBucket                        int
}
type CompatibleWork struct {
	ID                 string
	Key                CompatibilityKey
	Tokens             int
	Priority           int
	Enqueued, Deadline time.Time
	Cancelled          bool
}
type CompatibilityConfig struct {
	MaxBatch, MaxQueuePerClass int
	MaxPadding                 float64
	StarvationAfter            time.Duration
	Now                        time.Time
}
type CompatibilityBatch struct {
	Key             CompatibilityKey
	IDs             []string
	Tokens          []int
	SingletonReason string
}
type CompatibilityStats struct {
	Submitted, Cancelled, Rejected, Scheduled, Batches, SingletonFallbacks int
	MaxQueueAge                                                            time.Duration
	PaddingTax                                                             float64
	BatchFill                                                              float64
	NominalUse                                                             float64
}

// ComposeCompatible deterministically forms bounded compatible batches. Priority is
// increased by queue aging, deadlines break ties, and each class contributes at
// most one batch per pass so a deep class cannot starve another.
func ComposeCompatible(in []CompatibleWork, c CompatibilityConfig) ([]CompatibilityBatch, CompatibilityStats, error) {
	if c.MaxBatch <= 0 || c.MaxQueuePerClass <= 0 || c.MaxPadding < 0 {
		return nil, CompatibilityStats{}, errors.New("microagent: invalid compatibility config; set max_batch and max_queue_per_class > 0 and max_padding >= 0")
	}
	if c.Now.IsZero() {
		c.Now = time.Now()
	}
	s := CompatibilityStats{Submitted: len(in)}
	groups := map[string][]CompatibleWork{}
	var singles []CompatibleWork
	for _, w := range in {
		if w.Cancelled {
			s.Cancelled++
			continue
		}
		if w.ID == "" || w.Tokens <= 0 {
			return nil, s, errors.New("microagent: work requires id and positive tokens; assign a nonempty id and tokens > 0 before scheduling")
		}
		if incompleteKey(w.Key) {
			singles = append(singles, w)
			continue
		}
		k := compatKey(w.Key)
		q := groups[k]
		if len(q) >= c.MaxQueuePerClass {
			s.Rejected++
			continue
		}
		groups[k] = append(q, w)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	rank := func(q []CompatibleWork) {
		sort.SliceStable(q, func(i, j int) bool {
			ai, aj := agedPriority(q[i], c), agedPriority(q[j], c)
			if ai != aj {
				return ai > aj
			}
			di, dj := q[i].Deadline, q[j].Deadline
			if !di.Equal(dj) {
				if di.IsZero() {
					return false
				}
				if dj.IsZero() {
					return true
				}
				return di.Before(dj)
			}
			return q[i].Enqueued.Before(q[j].Enqueued)
		})
	}
	var out []CompatibilityBatch
	for _, w := range singles {
		out = append(out, CompatibilityBatch{Key: w.Key, IDs: []string{w.ID}, Tokens: []int{w.Tokens}, SingletonReason: "classification-unavailable"})
		s.SingletonFallbacks++
		observeAge(&s, w, c.Now)
	}
	for _, k := range keys {
		q := groups[k]
		rank(q)
		for len(q) > 0 {
			n := min(c.MaxBatch, len(q))
			for n > 1 && padding(q[:n]) > c.MaxPadding {
				n--
			}
			b := CompatibilityBatch{Key: q[0].Key}
			for _, w := range q[:n] {
				b.IDs = append(b.IDs, w.ID)
				b.Tokens = append(b.Tokens, w.Tokens)
				observeAge(&s, w, c.Now)
			}
			out = append(out, b)
			q = q[n:]
		}
	}
	s.Batches = len(out)
	for _, b := range out {
		s.Scheduled += len(b.IDs)
		s.PaddingTax += paddingTokens(b.Tokens)
	}
	if s.Batches > 0 {
		s.BatchFill = float64(s.Scheduled) / float64(s.Batches*c.MaxBatch)
		s.NominalUse = s.BatchFill
	}
	den := 0
	for _, b := range out {
		m := maxInt(b.Tokens)
		den += m * len(b.Tokens)
	}
	if den > 0 {
		s.PaddingTax /= float64(den)
	}
	return out, s, nil
}
func incompleteKey(k CompatibilityKey) bool {
	return strings.TrimSpace(k.Model) == "" || strings.TrimSpace(k.Sampling) == "" || strings.TrimSpace(k.Tools) == "" || strings.TrimSpace(k.Phase) == "" || k.SequenceBucket <= 0
}
func compatKey(k CompatibilityKey) string {
	return k.Model + "\x00" + k.Sampling + "\x00" + k.Tools + "\x00" + k.Prefix + "\x00" + k.Phase + "\x00" + string(rune(k.SequenceBucket))
}
func agedPriority(w CompatibleWork, c CompatibilityConfig) int {
	p := w.Priority
	if c.StarvationAfter > 0 && !w.Enqueued.IsZero() {
		p += int(c.Now.Sub(w.Enqueued) / c.StarvationAfter)
	}
	return p
}
func observeAge(s *CompatibilityStats, w CompatibleWork, now time.Time) {
	if !w.Enqueued.IsZero() && now.After(w.Enqueued) {
		a := now.Sub(w.Enqueued)
		if a > s.MaxQueueAge {
			s.MaxQueueAge = a
		}
	}
}
func padding(v []CompatibleWork) float64 {
	xs := make([]int, len(v))
	for i := range v {
		xs[i] = v[i].Tokens
	}
	d := 0
	for _, x := range xs {
		d += x
	}
	m := maxInt(xs)
	if m == 0 {
		return 0
	}
	return float64(m*len(xs)-d) / float64(m*len(xs))
}
func paddingTokens(xs []int) float64 {
	m := maxInt(xs)
	d := 0
	for _, x := range xs {
		d += x
	}
	return float64(m*len(xs) - d)
}
func maxInt(xs []int) int {
	m := 0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}
