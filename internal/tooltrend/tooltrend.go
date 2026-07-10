package tooltrend

import (
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/toolrollup"
)

// Schema is the stable identifier stamped on an emitted Trend so a consumer can
// tell which fold produced a report and refuse a shape it does not understand.
const Schema = "fak.tooltrend.v1"

// DefaultTopK caps how many movers each trend list returns. It is the default a
// bare Fold uses; FoldTopK takes an explicit cap.
const DefaultTopK = 10

// eps is the share-delta below which a move is treated as flat (no change) and
// dropped from the movers. Shares are exact ratios of small integer counts, so a
// tiny epsilon only absorbs float rounding, never a real one-call shift.
const eps = 1e-9

// Bucket is one labeled group of tool calls fed to the trend — typically the
// calls of a single agent session / trajectory. Label names the bucket (a
// session id, a date, an ordinal); Calls is its tool-call corpus. A bucket with
// no calls is valid: it folds to a Point with empty mixes and a zero error rate.
type Bucket struct {
	Label string
	Calls []toolrollup.ToolCall
}

// Point is one bucket folded to its mixes. ToolMix maps each tool TYPE to its
// share of the bucket's calls, in [0,1]; ShapeMix maps each output size-class
// (see SizeClass) to its share of the bucket's calls, in [0,1]. ErrorRate is the
// fraction of the bucket's calls that did not succeed. Both maps are non-nil
// (possibly empty) so a consumer never dereferences a nil map.
type Point struct {
	Label     string             `json:"label"`
	Calls     int                `json:"calls"`
	ToolMix   map[string]float64 `json:"tool_mix"`
	ShapeMix  map[string]float64 `json:"shape_mix"`
	ErrorRate float64            `json:"error_rate"`
}

// Move is one key's net change in share from the first bucket to the last. Delta
// is To-From (signed); AbsChange is its magnitude (the ranking key); Direction is
// the closed vocabulary "up" | "down". A key present in only one endpoint has an
// implicit 0 share at the other, so an appearing tool reads as a rise from 0 and a
// vanishing tool as a fall to 0.
type Move struct {
	Key       string  `json:"key"`
	From      float64 `json:"from"`
	To        float64 `json:"to"`
	Delta     float64 `json:"delta"`
	AbsChange float64 `json:"abs_change"`
	Direction string  `json:"direction"`
}

// Trend is the folded report: one Point per input bucket in input order, plus the
// biggest tool-mix and output-shape movers between the first and last bucket.
// ToolMovers and ShapeMovers are non-nil (possibly empty) and each capped at the
// requested top-K.
type Trend struct {
	Schema      string  `json:"schema"`
	Buckets     int     `json:"buckets"`
	Points      []Point `json:"points"`
	ToolMovers  []Move  `json:"tool_movers"`
	ShapeMovers []Move  `json:"shape_movers"`
}

// Fold folds an ordered slice of buckets into a Trend, keeping DefaultTopK movers
// per list. See FoldTopK.
func Fold(buckets []Bucket) Trend {
	return FoldTopK(buckets, DefaultTopK)
}

// FoldTopK folds an ordered slice of buckets into a Trend, keeping at most topK
// movers per list (topK <= 0 means "no limit"). The fold is pure and
// deterministic: points preserve input order; movers rank by absolute change
// descending, then key ascending. Fewer than two buckets yields points with no
// movers (there is no first-to-last delta to report). A nil/empty input yields an
// empty (non-nil) Trend.
func FoldTopK(buckets []Bucket, topK int) Trend {
	points := make([]Point, 0, len(buckets))
	for _, b := range buckets {
		points = append(points, foldBucket(b))
	}

	tr := Trend{
		Schema:      Schema,
		Buckets:     len(buckets),
		Points:      points,
		ToolMovers:  []Move{},
		ShapeMovers: []Move{},
	}
	if len(points) >= 2 {
		first, last := points[0], points[len(points)-1]
		tr.ToolMovers = movers(first.ToolMix, last.ToolMix, topK)
		tr.ShapeMovers = movers(first.ShapeMix, last.ShapeMix, topK)
	}
	return tr
}

// foldBucket reduces one bucket to its Point. Tool-mix shares and per-tool error
// counts come from reusing toolrollup.Rollup, so a tool's share here is exactly
// the share the per-tool rollup report shows.
func foldBucket(b Bucket) Point {
	n := len(b.Calls)
	pt := Point{
		Label:    b.Label,
		Calls:    n,
		ToolMix:  map[string]float64{},
		ShapeMix: map[string]float64{},
	}
	if n == 0 {
		return pt
	}

	stats := toolrollup.Rollup(b.Calls)
	var errors int
	for _, s := range stats {
		pt.ToolMix[s.Tool] = s.Share
		errors += s.Errors
	}
	pt.ErrorRate = float64(errors) / float64(n)

	shapeCount := map[string]int{}
	for _, c := range b.Calls {
		shapeCount[SizeClass(c.TokensOut)]++
	}
	for cls, c := range shapeCount {
		pt.ShapeMix[cls] = float64(c) / float64(n)
	}
	return pt
}

// movers computes the per-key share change from first to last over the union of
// both maps, drops flat keys, and ranks by absolute change descending then key
// ascending. topK <= 0 keeps every mover.
func movers(first, last map[string]float64, topK int) []Move {
	keys := map[string]struct{}{}
	for k := range first {
		keys[k] = struct{}{}
	}
	for k := range last {
		keys[k] = struct{}{}
	}

	out := make([]Move, 0, len(keys))
	for k := range keys {
		f, l := first[k], last[k]
		delta := l - f
		abs := math.Abs(delta)
		if abs < eps {
			continue // flat — no reportable movement
		}
		out = append(out, Move{
			Key:       k,
			From:      f,
			To:        l,
			Delta:     delta,
			AbsChange: abs,
			Direction: direction(delta),
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].AbsChange != out[j].AbsChange {
			return out[i].AbsChange > out[j].AbsChange
		}
		return out[i].Key < out[j].Key
	})
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out
}

// direction maps a signed delta to the closed direction vocabulary. A flat delta
// never reaches here (movers drops it), so only "up"/"down" are produced.
func direction(delta float64) string {
	if delta > 0 {
		return "up"
	}
	return "down"
}

// SizeClass buckets an output token count into a coarse, ordered response-shape
// class. The thresholds are decade-scaled so a class shift marks an order-of-
// magnitude change in response size, not noise. A non-positive count is "empty".
func SizeClass(tokensOut int) string {
	switch {
	case tokensOut <= 0:
		return "empty"
	case tokensOut < 100:
		return "small"
	case tokensOut < 1000:
		return "medium"
	case tokensOut < 10000:
		return "large"
	default:
		return "xlarge"
	}
}
