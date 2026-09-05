package tooltrend

import (
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/toolrollup"
)

// Schema identifies emitted Trend payloads for consumer compatibility validation.
const Schema = "fak.tooltrend.v1"

// DefaultTopK caps how many movers each trend report returns by default.
const DefaultTopK = 10

const eps = 1e-9

// Bucket is one labeled group of tool calls from an agent session.
type Bucket struct {
	Label string
	Calls []toolrollup.ToolCall
}

// Point records folded tool, shape, and error rate mixes for one bucket.
type Point struct {
	Label     string             `json:"label"`
	Calls     int                `json:"calls"`
	ToolMix   map[string]float64 `json:"tool_mix"`
	ShapeMix  map[string]float64 `json:"shape_mix"`
	ErrorRate float64            `json:"error_rate"`
}

// Move records one key's net change in share from the first to last bucket.
type Move struct {
	Key       string  `json:"key"`
	From      float64 `json:"from"`
	To        float64 `json:"to"`
	Delta     float64 `json:"delta"`
	AbsChange float64 `json:"abs_change"`
	Direction string  `json:"direction"`
}

// Trend is the folded report containing ordered points and top movers.
type Trend struct {
	Schema      string  `json:"schema"`
	Buckets     int     `json:"buckets"`
	Points      []Point `json:"points"`
	ToolMovers  []Move  `json:"tool_movers"`
	ShapeMovers []Move  `json:"shape_movers"`
}

// Fold folds an ordered slice of buckets into a Trend using DefaultTopK.
func Fold(buckets []Bucket) Trend {
	return FoldTopK(buckets, DefaultTopK)
}

// FoldTopK folds an ordered slice of buckets into a Trend with a movers cap.
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
			continue
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

func direction(delta float64) string {
	if delta > 0 {
		return "up"
	}
	return "down"
}

// SizeClass maps an output token count to a coarse response-size category.
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
