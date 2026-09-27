package runtimeobs

import (
	"bytes"
	"encoding/json"
	"math"
	"strconv"
)

var jsonNull = []byte("null")

// U64 is an observed uint64 or an explicit unavailable marker. Unavailable
// marshals as JSON null so a consumer can never read it as an observed zero.
type U64 struct {
	V  uint64
	OK bool
}

func (u U64) MarshalJSON() ([]byte, error) {
	if !u.OK {
		return jsonNull, nil
	}
	return strconv.AppendUint(nil, u.V, 10), nil
}

func (u *U64) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, jsonNull) {
		*u = U64{}
		return nil
	}
	v, err := strconv.ParseUint(string(b), 10, 64)
	if err != nil {
		return err
	}
	*u = U64{V: v, OK: true}
	return nil
}

// F64 is an observed finite float64 or an explicit unavailable marker.
type F64 struct {
	V  float64
	OK bool
}

func (f F64) MarshalJSON() ([]byte, error) {
	if !f.OK || math.IsNaN(f.V) || math.IsInf(f.V, 0) {
		return jsonNull, nil
	}
	return strconv.AppendFloat(nil, f.V, 'g', -1, 64), nil
}

func (f *F64) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, jsonNull) {
		*f = F64{}
		return nil
	}
	v, err := strconv.ParseFloat(string(b), 64)
	if err != nil {
		return err
	}
	*f = F64{V: v, OK: true}
	return nil
}

// Bucket is one non-empty histogram bucket [Lo, Hi). Lo may be -Inf and Hi
// +Inf; JSON has no infinities, so an infinite bound marshals as null.
type Bucket struct {
	Lo float64
	Hi float64
	N  uint64
}

type bucketJSON struct {
	Lo *float64 `json:"lo"`
	Hi *float64 `json:"hi"`
	N  uint64   `json:"n"`
}

func finitePtr(f float64) *float64 {
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return nil
	}
	return &f
}

func (b Bucket) MarshalJSON() ([]byte, error) {
	return json.Marshal(bucketJSON{Lo: finitePtr(b.Lo), Hi: finitePtr(b.Hi), N: b.N})
}

func (b *Bucket) UnmarshalJSON(p []byte) error {
	var j bucketJSON
	if err := json.Unmarshal(p, &j); err != nil {
		return err
	}
	*b = Bucket{Lo: math.Inf(-1), Hi: math.Inf(1), N: j.N}
	if j.Lo != nil {
		b.Lo = *j.Lo
	}
	if j.Hi != nil {
		b.Hi = *j.Hi
	}
	return nil
}

// Hist is a sparse histogram: only non-empty buckets, ascending. Count is the
// sum of bucket counts. Receipt histograms are cumulative since process start;
// Delta histograms cover one interval.
type Hist struct {
	OK      bool
	Count   uint64
	Buckets []Bucket
}

type histJSON struct {
	Count   uint64   `json:"count"`
	Buckets []Bucket `json:"buckets"`
}

func (h Hist) MarshalJSON() ([]byte, error) {
	if !h.OK {
		return jsonNull, nil
	}
	b := h.Buckets
	if b == nil {
		b = []Bucket{}
	}
	return json.Marshal(histJSON{Count: h.Count, Buckets: b})
}

func (h *Hist) UnmarshalJSON(p []byte) error {
	if bytes.Equal(p, jsonNull) {
		*h = Hist{}
		return nil
	}
	var j histJSON
	if err := json.Unmarshal(p, &j); err != nil {
		return err
	}
	*h = Hist{OK: true, Count: j.Count, Buckets: j.Buckets}
	if len(h.Buckets) == 0 {
		h.Buckets = nil
	}
	return nil
}

// Quantile returns a conservative estimate of the q-quantile: the upper bound
// of the bucket holding it (the lower bound when that bucket is open-ended).
// ok is false for an unavailable or empty histogram.
func (h Hist) Quantile(q float64) (v float64, ok bool) {
	if !h.OK || h.Count == 0 || len(h.Buckets) == 0 {
		return 0, false
	}
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	target := uint64(math.Ceil(q * float64(h.Count)))
	if target == 0 {
		target = 1
	}
	var cum uint64
	for _, b := range h.Buckets {
		cum += b.N
		if cum >= target {
			if math.IsInf(b.Hi, 1) {
				return b.Lo, true
			}
			return b.Hi, true
		}
	}
	last := h.Buckets[len(h.Buckets)-1]
	if math.IsInf(last.Hi, 1) {
		return last.Lo, true
	}
	return last.Hi, true
}

// Since returns h minus an earlier cumulative histogram of the same series.
// Buckets are matched by bounds; a bucket that shrank or vanished means the
// series reset, which is ErrCounterReset rather than a negative count.
func (h Hist) Since(prev Hist) (Hist, error) {
	if !h.OK || !prev.OK {
		return Hist{}, nil
	}
	out := Hist{OK: true}
	j := 0
	for _, b := range h.Buckets {
		n := b.N
		for j < len(prev.Buckets) && less(prev.Buckets[j], b) {
			// An earlier bucket present before but absent now shrank to zero.
			return Hist{}, ErrCounterReset
		}
		if j < len(prev.Buckets) && prev.Buckets[j].Lo == b.Lo && prev.Buckets[j].Hi == b.Hi {
			if prev.Buckets[j].N > n {
				return Hist{}, ErrCounterReset
			}
			n -= prev.Buckets[j].N
			j++
		}
		if n == 0 {
			continue
		}
		out.Count += n
		out.Buckets = append(out.Buckets, Bucket{Lo: b.Lo, Hi: b.Hi, N: n})
	}
	if j != len(prev.Buckets) {
		return Hist{}, ErrCounterReset
	}
	return out, nil
}

// less orders buckets by bounds.
func less(a, b Bucket) bool {
	if a.Lo != b.Lo {
		return a.Lo < b.Lo
	}
	return a.Hi < b.Hi
}

// Summary is a fixed-size digest of a Hist for consumers that cannot carry
// buckets: the sample count and conservative quantiles (see Hist.Quantile).
// Max is the bound of the highest non-empty bucket. Every field is null for
// an unavailable histogram; the quantiles are also null for an empty one.
type Summary struct {
	Count U64 `json:"count"`
	P50   F64 `json:"p50"`
	P90   F64 `json:"p90"`
	P99   F64 `json:"p99"`
	Max   F64 `json:"max"`
}

// Summarize digests h. A quantile that lands in the fully unbounded bucket
// has no finite bound and stays unavailable rather than becoming an infinity.
func Summarize(h Hist) Summary {
	if !h.OK {
		return Summary{}
	}
	s := Summary{Count: U64{V: h.Count, OK: true}}
	q := func(p float64) F64 {
		v, ok := h.Quantile(p)
		if !ok || math.IsInf(v, 0) || math.IsNaN(v) {
			return F64{}
		}
		return F64{V: v, OK: true}
	}
	s.P50, s.P90, s.P99, s.Max = q(0.5), q(0.9), q(0.99), q(1)
	return s
}
