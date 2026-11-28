// Ported: the geometric 1us..60s ratio-1.2 boundaries, boundary normalization,
// the 32-shard relaxed-atomic histogram, the bucket-derived snapshot, and the
// cumulative Prometheus text export (_bucket{le}, +Inf, _sum, _count).
// fak adds: a per-call runtime-random shard pick in place of the per-thread
// hash, and a time.Duration Observe convenience over the upstream microsecond
// value.
//
// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package modelengine

import (
	"math"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// LatencyHistogramShardCount is the fixed shard fan-out of LatencyHistogram.
const LatencyHistogramShardCount = 32

const (
	latencyHistogramRatio = 1.2
	latencyHistogramMaxUS = 60.0 * 1000.0 * 1000.0
)

// LatencyHistogramDefaultBoundaries returns microsecond bounds from 1us to
// 60s with ratio 1.2. A histogram_quantile estimate can land on the far edge
// of the bucket holding the rank, so relative error is bounded by about 20%.
func LatencyHistogramDefaultBoundaries() []float64 {
	var buckets []float64
	boundary := 1.0
	for boundary < latencyHistogramMaxUS {
		buckets = append(buckets, boundary)
		next := boundary * latencyHistogramRatio
		if !(next > boundary) {
			break
		}
		boundary = next
	}
	if len(buckets) == 0 || buckets[len(buckets)-1] < latencyHistogramMaxUS {
		buckets = append(buckets, latencyHistogramMaxUS)
	}
	return buckets
}

func latencyHistogramNormalizeBoundaries(in []float64) []float64 {
	out := make([]float64, 0, len(in))
	for _, v := range in {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			out = append(out, v)
		}
	}
	sort.Float64s(out)
	uniq := out[:0]
	for i, v := range out {
		if i == 0 || v != out[i-1] {
			uniq = append(uniq, v)
		}
	}
	if len(uniq) == 0 {
		return LatencyHistogramDefaultBoundaries()
	}
	return uniq
}

type latencyHistogramShard struct {
	sum     atomic.Uint64
	buckets []atomic.Uint64
	_       [32]byte // 64-byte shards so neighbouring sums do not false-share
}

// LatencyHistogram is a cumulative Prometheus histogram updated without a
// mutex: each Observe adds into one shard with plain atomics and a scrape
// sums the shards.
type LatencyHistogram struct {
	name       string
	help       string
	labels     map[string]string
	boundaries []float64
	shards     [LatencyHistogramShardCount]latencyHistogramShard
}

// LatencyHistogramSnapshot is a merged view; Cumulative[i] counts
// observations <= Boundaries[i].
type LatencyHistogramSnapshot struct {
	Boundaries []float64
	Cumulative []uint64
	Count      uint64
	SumUS      uint64
}

// NewLatencyHistogram builds a histogram over microsecond boundaries; nil,
// empty or all non-finite boundaries fall back to the defaults.
func NewLatencyHistogram(name, help string, labels map[string]string, boundaries []float64) *LatencyHistogram {
	h := &LatencyHistogram{
		name:       name,
		help:       help,
		labels:     labels,
		boundaries: latencyHistogramNormalizeBoundaries(boundaries),
	}
	for i := range h.shards {
		h.shards[i].buckets = make([]atomic.Uint64, len(h.boundaries)+1)
	}
	return h
}

// Go has no thread identity to pin on; the runtime's per-M random source
// spreads writers without the shared counter a round-robin pick would contend.
func (h *LatencyHistogram) shard() *latencyHistogramShard {
	return &h.shards[rand.Uint32()%LatencyHistogramShardCount]
}

func (h *LatencyHistogram) bucketIndex(valueUS uint64) int {
	return sort.SearchFloat64s(h.boundaries, float64(valueUS))
}

// ObserveUS records one latency in microseconds.
func (h *LatencyHistogram) ObserveUS(valueUS uint64) {
	if h == nil {
		return
	}
	s := h.shard()
	s.buckets[h.bucketIndex(valueUS)].Add(1)
	s.sum.Add(valueUS)
}

// Observe records d truncated to whole microseconds; negatives count as 0.
func (h *LatencyHistogram) Observe(d time.Duration) {
	if d < 0 {
		d = 0
	}
	h.ObserveUS(uint64(d / time.Microsecond))
}

// Collect merges the shards. Count is derived from the same bucket loads so a
// concurrent Observe can never make a finite bucket exceed +Inf.
func (h *LatencyHistogram) Collect() LatencyHistogramSnapshot {
	snap := LatencyHistogramSnapshot{Boundaries: append([]float64(nil), h.boundaries...)}
	raw := make([]uint64, len(h.boundaries)+1)
	for i := range h.shards {
		snap.SumUS += h.shards[i].sum.Load()
		for b := range raw {
			raw[b] += h.shards[i].buckets[b].Load()
		}
	}
	for _, n := range raw {
		snap.Count += n
	}
	snap.Cumulative = make([]uint64, len(h.boundaries))
	var running uint64
	for b := range h.boundaries {
		running += raw[b]
		snap.Cumulative[b] = running
	}
	return snap
}

// Name returns the metric family name.
func (h *LatencyHistogram) Name() string { return h.name }

// Help returns the metric family help text.
func (h *LatencyHistogram) Help() string { return h.help }

func latencyHistogramEscapeLabel(v string) string {
	var b strings.Builder
	for i := 0; i < len(v); i++ {
		c := v[i]
		switch c {
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '"':
			b.WriteString(`\"`)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

func latencyHistogramFormatLe(boundary float64) string {
	if !math.IsInf(boundary, 0) && !math.IsNaN(boundary) && boundary >= 0 &&
		boundary == math.Floor(boundary) && boundary < float64(math.MaxUint64) {
		return strconv.FormatUint(uint64(boundary), 10)
	}
	return strconv.FormatFloat(boundary, 'g', 17, 64)
}

// AppendSamples writes sample lines only; the caller owns HELP and TYPE.
func (h *LatencyHistogram) AppendSamples(out *strings.Builder) {
	if h == nil {
		return
	}
	snap := h.Collect()
	keys := make([]string, 0, len(h.labels))
	for k := range h.labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var base strings.Builder
	for _, k := range keys {
		if base.Len() > 0 {
			base.WriteByte(',')
		}
		base.WriteString(k)
		base.WriteString(`="`)
		base.WriteString(latencyHistogramEscapeLabel(h.labels[k]))
		base.WriteByte('"')
	}
	baseLabels := base.String()
	write := func(suffix, extra string, value uint64) {
		out.WriteString(h.name)
		out.WriteString(suffix)
		if baseLabels == "" && extra == "" {
			out.WriteByte(' ')
		} else {
			out.WriteByte('{')
			out.WriteString(baseLabels)
			if extra != "" {
				if baseLabels != "" {
					out.WriteByte(',')
				}
				out.WriteString(extra)
			}
			out.WriteString("} ")
		}
		out.WriteString(strconv.FormatUint(value, 10))
		out.WriteByte('\n')
	}
	for i, bound := range snap.Boundaries {
		write("_bucket", `le="`+latencyHistogramFormatLe(bound)+`"`, snap.Cumulative[i])
	}
	write("_bucket", `le="+Inf"`, snap.Count)
	write("_sum", "", snap.SumUS)
	write("_count", "", snap.Count)
}

// AppendPrometheusText writes HELP, TYPE and the samples.
func (h *LatencyHistogram) AppendPrometheusText(out *strings.Builder) {
	if h == nil {
		return
	}
	out.WriteString("# HELP " + h.name + " " + h.help + "\n")
	out.WriteString("# TYPE " + h.name + " histogram\n")
	h.AppendSamples(out)
}
