// Adapted from ai-dynamo/aiperf
// modified from the original): bounded-memory merging t-digest quantile sketch
// paired with Welford/Chan online parallel mean/variance, exact count/sum/min/max,
// and an optional per-record-replay capability gate for list-metric aggregation.
package perfledger

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// DefaultTDigestCompression matches aiperf's calibrated default compression
// factor of 500 for sub-0.05% tail quantile error at large sample counts.
const DefaultTDigestCompression = 500.0

// Centroid is one weighted mean cluster inside a TDigest.
type Centroid struct {
	Mean   float64 `json:"mean"`
	Weight float64 `json:"weight"`
}

// TDigest is a bounded-memory quantile sketch with Welford online variance and
// exact count, sum, min, and max tracking. A TDigest is not safe for
// concurrent mutation without external synchronization.
type TDigest struct {
	compression float64
	maxUnmerged int
	centroids   []Centroid
	unmerged    []Centroid
	count       int64
	sum         float64
	min         float64
	max         float64
	mean        float64
	m2          float64
	totalWeight float64
}

// NewTDigest creates a TDigest with the given compression parameter. A zero
// compression selects DefaultTDigestCompression; negative or non-finite values
// are refused.
func NewTDigest(compression float64) (*TDigest, error) {
	if compression == 0 {
		compression = DefaultTDigestCompression
	}
	if compression < 0 || math.IsNaN(compression) || math.IsInf(compression, 0) {
		return nil, fmt.Errorf("perfledger: TDigest compression must be positive and finite, got %v", compression)
	}
	maxUnmerged := int(math.Ceil(compression * 2))
	if maxUnmerged < 128 {
		maxUnmerged = 128
	}
	return &TDigest{
		compression: compression,
		maxUnmerged: maxUnmerged,
		min:         math.Inf(1),
		max:         math.Inf(-1),
	}, nil
}

// Add inserts one finite sample into the digest, compressing when the unmerged
// buffer reaches capacity so memory stays O(compression).
func (d *TDigest) Add(x float64) error {
	if d == nil {
		return errors.New("perfledger: TDigest.Add: nil digest")
	}
	if !isFiniteFloat(x) {
		return fmt.Errorf("perfledger: TDigest.Add: non-finite value %v", x)
	}
	d.count++
	d.sum += x
	d.totalWeight += 1
	if x < d.min {
		d.min = x
	}
	if x > d.max {
		d.max = x
	}
	delta := x - d.mean
	d.mean += delta / float64(d.count)
	delta2 := x - d.mean
	d.m2 += delta * delta2

	d.unmerged = append(d.unmerged, Centroid{Mean: x, Weight: 1})
	if len(d.unmerged) >= d.maxUnmerged {
		d.compress()
	}
	return nil
}

// AddAll validates and inserts all finite values in xs. If any element is
// non-finite, an error is returned without mutating the digest.
func (d *TDigest) AddAll(xs []float64) error {
	if d == nil {
		return errors.New("perfledger: TDigest.AddAll: nil digest")
	}
	for i, x := range xs {
		if !isFiniteFloat(x) {
			return fmt.Errorf("perfledger: TDigest.AddAll: non-finite value at index %d: %v", i, x)
		}
	}
	for _, x := range xs {
		if err := d.Add(x); err != nil {
			return err
		}
	}
	return nil
}

// Merge folds other into d, combining centroids and applying Chan's parallel
// Welford update for exact pooled mean and variance.
func (d *TDigest) Merge(other *TDigest) error {
	if d == nil {
		return errors.New("perfledger: TDigest.Merge: nil receiver")
	}
	if other == nil || other.count == 0 {
		return nil
	}
	if d.compression == 0 {
		d.compression = DefaultTDigestCompression
		d.maxUnmerged = 1000
		d.min = math.Inf(1)
		d.max = math.Inf(-1)
	}
	if d.count == 0 {
		d.count = other.count
		d.sum = other.sum
		d.min = other.min
		d.max = other.max
		d.mean = other.mean
		d.m2 = other.m2
		d.totalWeight = other.totalWeight
		d.centroids = append([]Centroid(nil), other.centroids...)
		d.unmerged = append([]Centroid(nil), other.unmerged...)
		if len(d.unmerged) >= d.maxUnmerged {
			d.compress()
		}
		return nil
	}

	totalCount := d.count + other.count
	delta := other.mean - d.mean
	nA := float64(d.count)
	nB := float64(other.count)
	nTotal := float64(totalCount)
	d.mean = (d.mean*nA + other.mean*nB) / nTotal
	d.m2 = d.m2 + other.m2 + delta*delta*nA*nB/nTotal
	d.count = totalCount
	d.sum += other.sum
	d.totalWeight += other.totalWeight
	if other.min < d.min {
		d.min = other.min
	}
	if other.max > d.max {
		d.max = other.max
	}

	d.unmerged = append(d.unmerged, other.centroids...)
	d.unmerged = append(d.unmerged, other.unmerged...)
	if len(d.unmerged) >= d.maxUnmerged {
		d.compress()
	}
	return nil
}

// Count returns the exact number of samples added or merged.
func (d *TDigest) Count() int64 {
	if d == nil {
		return 0
	}
	return d.count
}

// Sum returns the exact running sum of samples.
func (d *TDigest) Sum() float64 {
	if d == nil {
		return 0
	}
	return d.sum
}

// Mean returns the online arithmetic mean, or 0 when empty.
func (d *TDigest) Mean() float64 {
	if d == nil || d.count == 0 {
		return 0
	}
	return d.mean
}

// Min returns the exact minimum sample, or 0 when empty.
func (d *TDigest) Min() float64 {
	if d == nil || d.count == 0 {
		return 0
	}
	return d.min
}

// Max returns the exact maximum sample, or 0 when empty.
func (d *TDigest) Max() float64 {
	if d == nil || d.count == 0 {
		return 0
	}
	return d.max
}

// StdDev returns the population standard deviation (ddof=0).
func (d *TDigest) StdDev() float64 {
	if d == nil || d.count == 0 {
		return 0
	}
	v := d.m2 / float64(d.count)
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v)
}

// SampleStdDev returns the sample standard deviation (ddof=1), or 0 for < 2 samples.
func (d *TDigest) SampleStdDev() float64 {
	if d == nil || d.count < 2 {
		return 0
	}
	v := d.m2 / float64(d.count-1)
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v)
}

// CentroidCount compresses pending unmerged samples and returns the retained
// centroid count.
func (d *TDigest) CentroidCount() int {
	if d == nil {
		return 0
	}
	d.compress()
	return len(d.centroids)
}

// Quantile estimates the q-th quantile (q in [0, 1]) from the compressed centroids.
func (d *TDigest) Quantile(q float64) (float64, error) {
	if d == nil || d.count == 0 {
		return 0, errors.New("perfledger: TDigest.Quantile: empty digest")
	}
	if math.IsNaN(q) || q < 0 || q > 1 {
		return 0, fmt.Errorf("perfledger: TDigest.Quantile: quantile must be in [0, 1], got %v", q)
	}
	if len(d.unmerged) > 0 {
		d.compress()
	}
	if q == 0 {
		return d.min, nil
	}
	if q == 1 {
		return d.max, nil
	}
	if len(d.centroids) == 1 {
		return d.centroids[0].Mean, nil
	}

	n := d.totalWeight
	index := q * n
	first := d.centroids[0]
	if index < first.Weight/2 {
		if first.Weight == 1 {
			return d.min, nil
		}
		return d.min + (index/(first.Weight/2))*(first.Mean-d.min), nil
	}

	prefix := first.Weight / 2
	for i := 0; i < len(d.centroids)-1; i++ {
		left := d.centroids[i]
		right := d.centroids[i+1]
		step := (left.Weight + right.Weight) / 2
		if index <= prefix+step {
			frac := (index - prefix) / step
			return left.Mean + frac*(right.Mean-left.Mean), nil
		}
		prefix += step
	}

	last := d.centroids[len(d.centroids)-1]
	if last.Weight == 1 {
		return d.max, nil
	}
	tailHalf := last.Weight / 2
	frac := (index - (n - tailHalf)) / tailHalf
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	return last.Mean + frac*(d.max-last.Mean), nil
}

// PercentileBand returns the standard 9-point percentile band estimated by the digest.
func (d *TDigest) PercentileBand() (PercentileBand, error) {
	if d == nil || d.count == 0 {
		return PercentileBand{}, errors.New("perfledger: TDigest.PercentileBand: empty digest")
	}
	qs := []float64{0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99}
	vals := make([]float64, len(qs))
	for i, q := range qs {
		v, err := d.Quantile(q)
		if err != nil {
			return PercentileBand{}, err
		}
		vals[i] = v
	}
	return PercentileBand{
		Count: int(d.count),
		P1:    vals[0],
		P5:    vals[1],
		P10:   vals[2],
		P25:   vals[3],
		P50:   vals[4],
		P75:   vals[5],
		P90:   vals[6],
		P95:   vals[7],
		P99:   vals[8],
	}, nil
}

func (d *TDigest) kScale(q float64) float64 {
	if q <= 0 {
		return -d.compression / 4
	}
	if q >= 1 {
		return d.compression / 4
	}
	return (d.compression / (2 * math.Pi)) * math.Asin(2*q-1)
}

func (d *TDigest) compress() {
	if len(d.unmerged) == 0 {
		return
	}
	all := make([]Centroid, 0, len(d.centroids)+len(d.unmerged))
	all = append(all, d.centroids...)
	all = append(all, d.unmerged...)
	d.unmerged = d.unmerged[:0]
	if len(all) == 0 || d.totalWeight <= 0 {
		d.centroids = nil
		return
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].Mean < all[j].Mean
	})

	compressed := make([]Centroid, 0, min(len(all), int(math.Ceil(d.compression))+4))
	cur := all[0]
	wSoFar := 0.0
	kLimit := d.kScale(0) + 1.0
	for _, next := range all[1:] {
		proposed := cur.Weight + next.Weight
		qNext := (wSoFar + proposed) / d.totalWeight
		if next.Mean == cur.Mean || d.kScale(qNext) <= kLimit {
			cur.Mean = (cur.Mean*cur.Weight + next.Mean*next.Weight) / proposed
			cur.Weight = proposed
			continue
		}
		compressed = append(compressed, cur)
		wSoFar += cur.Weight
		kLimit = d.kScale(wSoFar/d.totalWeight) + 1.0
		cur = next
	}
	compressed = append(compressed, cur)
	d.centroids = compressed
}

// TDigestSummary holds exact aggregate scalars alongside sketch quantiles.
type TDigestSummary struct {
	Count         int64          `json:"count"`
	Sum           float64        `json:"sum"`
	Avg           float64        `json:"avg"`
	StdDev        float64        `json:"std_dev"`
	Min           float64        `json:"min"`
	Max           float64        `json:"max"`
	Band          PercentileBand `json:"band"`
	CentroidCount int            `json:"centroid_count"`
}

// ListMetricAggregator aggregates list-valued per-record metrics (such as ITL
// chunk lists) in bounded memory via TDigest, optionally retaining flat values
// only when SupportsPerRecordReplay is explicitly enabled.
type ListMetricAggregator struct {
	SupportsPerRecordReplay bool
	digest                  *TDigest
	replayed                []float64
}

// NewListMetricAggregator constructs a list-metric aggregator with bounded
// sketch memory (compression <= 0 defaults to DefaultTDigestCompression).
func NewListMetricAggregator(compression float64, supportsPerRecordReplay bool) (*ListMetricAggregator, error) {
	td, err := NewTDigest(compression)
	if err != nil {
		return nil, err
	}
	return &ListMetricAggregator{
		SupportsPerRecordReplay: supportsPerRecordReplay,
		digest:                  td,
	}, nil
}

// ObserveList ingests a per-record list of finite float values.
func (a *ListMetricAggregator) ObserveList(values []float64) error {
	if a == nil {
		return errors.New("perfledger: ListMetricAggregator: nil receiver")
	}
	if err := a.digest.AddAll(values); err != nil {
		return err
	}
	if a.SupportsPerRecordReplay {
		a.replayed = append(a.replayed, values...)
	}
	return nil
}

// ReplayedValues returns the retained raw list values when
// SupportsPerRecordReplay is true, and nil otherwise.
func (a *ListMetricAggregator) ReplayedValues() []float64 {
	if a == nil || !a.SupportsPerRecordReplay {
		return nil
	}
	return append([]float64(nil), a.replayed...)
}

// Summary computes exact scalars and t-digest quantiles across all observed lists.
func (a *ListMetricAggregator) Summary() (TDigestSummary, error) {
	if a == nil || a.digest == nil {
		return TDigestSummary{}, errors.New("perfledger: ListMetricAggregator: nil receiver")
	}
	band, err := a.digest.PercentileBand()
	if err != nil {
		return TDigestSummary{}, err
	}
	return TDigestSummary{
		Count:         a.digest.Count(),
		Sum:           a.digest.Sum(),
		Avg:           a.digest.Mean(),
		StdDev:        a.digest.StdDev(),
		Min:           a.digest.Min(),
		Max:           a.digest.Max(),
		Band:          band,
		CentroidCount: a.digest.CentroidCount(),
	}, nil
}
