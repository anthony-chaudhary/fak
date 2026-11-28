package compute

// Adapted from NVIDIA/TensorRT-LLM (Apache-2.0) at commit
// tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py and
// tensorrt_llm/_torch/moe/fused_moe/cutedsl/grouped_gemm/nvfp4.py.
// Changes: the MoE decode front-end ordering contract is re-expressed as a
// host-only Go reference; no upstream code, kernel, or stream logic is copied.

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrMoeFrontGeometry refuses shapes the front end cannot route or run.
	ErrMoeFrontGeometry = errors.New("compute: moe front-fusion geometry refused")
	// ErrMoeFrontRoute refuses non-finite router logits and unroutable ids.
	ErrMoeFrontRoute = errors.New("compute: moe front-fusion routing refused")
)

// MoeFrontScoring selects how router logits become expert scores.
type MoeFrontScoring int

const (
	MoeFrontSoftmax MoeFrontScoring = iota
	MoeFrontSigmoid
)

// MoeFrontEpilogue names where the router scale meets the latent_down (FC2)
// output.
type MoeFrontEpilogue int

const (
	// MoeFrontFusedFinalize scales and scatter-adds each FC2 row into the
	// token's output inside the grouped walk, so no permuted FC2 buffer exists.
	MoeFrontFusedFinalize MoeFrontEpilogue = iota
	// MoeFrontUnpermute materializes the permuted FC2 output and gathers it back
	// per token in a separate pass.
	MoeFrontUnpermute
)

// MoeFrontConfig fixes the decode front-end geometry. FC1 weights are
// row-major [NumExperts][Inter][Hidden]; latent_down (FC2) weights are
// row-major [NumExperts][Hidden][Inter].
type MoeFrontConfig struct {
	NumExperts  int
	TopK        int
	Hidden      int
	Inter       int
	Scoring     MoeFrontScoring
	Renormalize bool
	// TileSize is the grouped-GEMM tile the runner is tuned for. Sort tables
	// built at another tile are re-sorted, never reinterpreted.
	TileSize int
	// QuantizeFC1 rounds each activated FC1 row through NVFP4 in the FC1
	// epilogue, so latent_down consumes the quantized operand. Requires Inter to
	// be a multiple of NVFP4BlockSize.
	QuantizeFC1 bool
	Epilogue    MoeFrontEpilogue
	// Activation is applied elementwise to FC1 output; nil means identity.
	Activation func(float32) float32
}

// MoeFrontSortTables are the align/sort outputs a routing pass produced for a
// specific id panel. They stay valid only for exactly those ids at exactly
// that tile size.
type MoeFrontSortTables struct {
	TileSize int
	Align    *MoeAlignResult
	ids      []int32
}

// MoeFrontRoute is a routed decode batch: ExpertIDs and Scales are row-major
// [tokens][TopK]. Tables is nil when routing did not sort.
type MoeFrontRoute struct {
	Tokens    int
	TopK      int
	ExpertIDs []int32
	Scales    []float32
	Tables    *MoeFrontSortTables
}

// MoeFrontWorkItem is one (token, k) row in grouped-GEMM dispatch order.
type MoeFrontWorkItem struct {
	Expert   int
	Token    int
	Slot     int
	Expanded int // Token*TopK + Slot
	Permuted int // position in the padded sorted panel
}

// MoeFrontInput carries exactly one routing source: fixed Logits
// [tokens][NumExperts], a RouterWeight [NumExperts][Hidden] whose logits are
// computed inside the front end, or a Prerouted batch from host plumbing.
type MoeFrontInput struct {
	X            []float32 // [tokens][Hidden]
	Tokens       int
	Logits       []float32
	RouterWeight []float32
	Prerouted    *MoeFrontRoute
	FC1          []float32
	FC2          []float32
}

// MoeFrontResult is the combined output plus the plan that produced it.
type MoeFrontResult struct {
	Out []float32 // [tokens][Hidden]
	// Route is what reached the grouped GEMM.
	Route *MoeFrontRoute
	// ReusedSortTables reports that routing's tables fed the GEMM without a
	// second sort.
	ReusedSortTables bool
	Order            []MoeFrontWorkItem
}

func (c MoeFrontConfig) validate() error {
	if c.NumExperts <= 0 || c.NumExperts > MoeAlignMaxExperts {
		return fmt.Errorf("%w: numExperts=%d", ErrMoeFrontGeometry, c.NumExperts)
	}
	if c.TopK <= 0 || c.TopK > c.NumExperts {
		return fmt.Errorf("%w: topK=%d numExperts=%d", ErrMoeFrontGeometry, c.TopK, c.NumExperts)
	}
	if c.Hidden <= 0 || c.Inter <= 0 || c.TileSize <= 0 {
		return fmt.Errorf("%w: hidden/inter/tile=%d/%d/%d", ErrMoeFrontGeometry, c.Hidden, c.Inter, c.TileSize)
	}
	if c.QuantizeFC1 && c.Inter%NVFP4BlockSize != 0 {
		return fmt.Errorf("%w: inter=%d is not a multiple of the NVFP4 block", ErrMoeFrontGeometry, c.Inter)
	}
	if c.Scoring != MoeFrontSoftmax && c.Scoring != MoeFrontSigmoid {
		return fmt.Errorf("%w: scoring=%d", ErrMoeFrontGeometry, c.Scoring)
	}
	if c.Epilogue != MoeFrontFusedFinalize && c.Epilogue != MoeFrontUnpermute {
		return fmt.Errorf("%w: epilogue=%d", ErrMoeFrontGeometry, c.Epilogue)
	}
	return nil
}

// MoeFrontRouteWithSort scores every token's logits, selects its top-k
// experts, and builds the sort tables for those ids in the same pass, so the
// grouped GEMM need not sort again. Ties in score go to the lower expert id.
func MoeFrontRouteWithSort(logits []float32, tokens int, cfg MoeFrontConfig) (*MoeFrontRoute, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if tokens < 0 || len(logits) != tokens*cfg.NumExperts {
		return nil, fmt.Errorf("%w: logits length %d want %d*%d", ErrMoeFrontGeometry, len(logits), tokens, cfg.NumExperts)
	}
	ids := make([]int32, tokens*cfg.TopK)
	scales := make([]float32, tokens*cfg.TopK)
	scores := make([]float64, cfg.NumExperts)
	taken := make([]bool, cfg.NumExperts)
	for t := 0; t < tokens; t++ {
		row := logits[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
		if err := moeFrontScore(row, cfg.Scoring, scores); err != nil {
			return nil, fmt.Errorf("%w: token %d: %v", ErrMoeFrontRoute, t, err)
		}
		for e := range taken {
			taken[e] = false
		}
		var picked float64
		for k := 0; k < cfg.TopK; k++ {
			best := -1
			for e, s := range scores {
				if !taken[e] && (best < 0 || s > scores[best]) {
					best = e
				}
			}
			taken[best] = true
			ids[t*cfg.TopK+k] = int32(best)
			picked += scores[best]
		}
		for k := 0; k < cfg.TopK; k++ {
			s := scores[ids[t*cfg.TopK+k]]
			if cfg.Renormalize && picked > 0 {
				s /= picked
			}
			scales[t*cfg.TopK+k] = float32(s)
		}
	}
	tables, err := moeFrontSort(ids, cfg)
	if err != nil {
		return nil, err
	}
	return &MoeFrontRoute{Tokens: tokens, TopK: cfg.TopK, ExpertIDs: ids, Scales: scales, Tables: tables}, nil
}

func moeFrontScore(row []float32, scoring MoeFrontScoring, out []float64) error {
	maxLogit := math.Inf(-1)
	for e, v := range row {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("logit %d is non-finite", e)
		}
		maxLogit = math.Max(maxLogit, f)
	}
	if scoring == MoeFrontSigmoid {
		for e, v := range row {
			out[e] = 1 / (1 + math.Exp(-float64(v)))
		}
		return nil
	}
	// Subtracting the row max keeps exp finite for any finite logit.
	var sum float64
	for e, v := range row {
		out[e] = math.Exp(float64(v) - maxLogit)
		sum += out[e]
	}
	for e := range out {
		out[e] /= sum
	}
	return nil
}

func moeFrontSort(ids []int32, cfg MoeFrontConfig) (*MoeFrontSortTables, error) {
	align, err := MoeAlignBlockSizes(ids, cfg.NumExperts, cfg.TileSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMoeFrontGeometry, err)
	}
	return &MoeFrontSortTables{TileSize: cfg.TileSize, Align: align, ids: append([]int32(nil), ids...)}, nil
}

// Describes reports whether the tables were built for exactly these ids at
// this tile size. Ids replaced after routing (a forced router, a replayed
// calibration) make tables describe other ids, so they must be dropped.
func (s *MoeFrontSortTables) Describes(ids []int32, tileSize int) bool {
	if s == nil || s.Align == nil || s.TileSize != tileSize || len(s.ids) != len(ids) {
		return false
	}
	for i, id := range ids {
		if s.ids[i] != id {
			return false
		}
	}
	return true
}

// MoeFrontResolveTables returns the align tables the grouped GEMM consumes:
// routing's own tables when they still describe the ids at the tuned tile,
// otherwise a fresh sort.
func MoeFrontResolveTables(route *MoeFrontRoute, cfg MoeFrontConfig) (*MoeAlignResult, bool, error) {
	if route.Tables.Describes(route.ExpertIDs, cfg.TileSize) {
		return route.Tables.Align, true, nil
	}
	tables, err := moeFrontSort(route.ExpertIDs, cfg)
	if err != nil {
		return nil, false, err
	}
	return tables.Align, false, nil
}

// MoeFrontDispatchOrder walks the sorted panel tile by tile, giving the
// router-first order: every expert's rows are contiguous and experts appear in
// ascending id order, so each grouped-GEMM tile touches one expert's weights.
// Pad slots are skipped.
func MoeFrontDispatchOrder(align *MoeAlignResult, topK, tileSize int) []MoeFrontWorkItem {
	order := make([]MoeFrontWorkItem, 0, align.Numel)
	for p := 0; p < int(align.NumTokensPostPad); p++ {
		expanded := align.SortedTokenIDs[p]
		if expanded == align.Numel {
			continue
		}
		e := int(align.ExpertIDs[p/tileSize])
		order = append(order, MoeFrontWorkItem{
			Expert:   e,
			Token:    int(expanded) / topK,
			Slot:     int(expanded) % topK,
			Expanded: int(expanded),
			Permuted: p,
		})
	}
	return order
}

// MoeFrontForward runs the decode front end: route (or accept a pre-routed
// batch), resolve sort tables, dispatch router-first through FC1, the FC1
// epilogue (activation, optional NVFP4), latent_down, and the configured
// router-scale epilogue.
func MoeFrontForward(in MoeFrontInput, cfg MoeFrontConfig) (*MoeFrontResult, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if in.Tokens < 0 || len(in.X) != in.Tokens*cfg.Hidden {
		return nil, fmt.Errorf("%w: x length %d want %d*%d", ErrMoeFrontGeometry, len(in.X), in.Tokens, cfg.Hidden)
	}
	if len(in.FC1) != cfg.NumExperts*cfg.Inter*cfg.Hidden || len(in.FC2) != cfg.NumExperts*cfg.Hidden*cfg.Inter {
		return nil, fmt.Errorf("%w: fc1/fc2 length %d/%d", ErrMoeFrontGeometry, len(in.FC1), len(in.FC2))
	}
	route, err := moeFrontRouteInput(in, cfg)
	if err != nil {
		return nil, err
	}
	out := make([]float32, in.Tokens*cfg.Hidden)
	if in.Tokens == 0 {
		return &MoeFrontResult{Out: out, Route: route}, nil
	}
	align, reused, err := MoeFrontResolveTables(route, cfg)
	if err != nil {
		return nil, err
	}
	order := MoeFrontDispatchOrder(align, cfg.TopK, cfg.TileSize)

	var permuted []float32
	if cfg.Epilogue == MoeFrontUnpermute {
		permuted = make([]float32, int(align.NumTokensPostPad)*cfg.Hidden)
	}
	h := make([]float32, cfg.Inter)
	y := make([]float32, cfg.Hidden)
	for _, w := range order {
		if err := moeFrontFC1(in, cfg, w, h); err != nil {
			return nil, err
		}
		fc2 := in.FC2[w.Expert*cfg.Hidden*cfg.Inter:]
		for j := 0; j < cfg.Hidden; j++ {
			var acc float32
			for i := 0; i < cfg.Inter; i++ {
				acc += fc2[j*cfg.Inter+i] * h[i]
			}
			y[j] = acc
		}
		if cfg.Epilogue == MoeFrontFusedFinalize {
			scale := route.Scales[w.Expanded]
			dst := out[w.Token*cfg.Hidden : (w.Token+1)*cfg.Hidden]
			for j, v := range y {
				dst[j] += scale * v
			}
			continue
		}
		copy(permuted[w.Permuted*cfg.Hidden:], y)
	}
	if cfg.Epilogue == MoeFrontUnpermute {
		expandedToPermuted := make([]int, len(route.ExpertIDs))
		for _, w := range order {
			expandedToPermuted[w.Expanded] = w.Permuted
		}
		for t := 0; t < in.Tokens; t++ {
			dst := out[t*cfg.Hidden : (t+1)*cfg.Hidden]
			for k := 0; k < cfg.TopK; k++ {
				x := t*cfg.TopK + k
				src := permuted[expandedToPermuted[x]*cfg.Hidden:]
				for j := range dst {
					dst[j] += route.Scales[x] * src[j]
				}
			}
		}
	}
	return &MoeFrontResult{Out: out, Route: route, ReusedSortTables: reused, Order: order}, nil
}

func moeFrontRouteInput(in MoeFrontInput, cfg MoeFrontConfig) (*MoeFrontRoute, error) {
	sources := 0
	for _, set := range []bool{in.Logits != nil, in.RouterWeight != nil, in.Prerouted != nil} {
		if set {
			sources++
		}
	}
	if sources != 1 {
		return nil, fmt.Errorf("%w: want exactly one routing source, got %d", ErrMoeFrontRoute, sources)
	}
	if in.Prerouted != nil {
		return moeFrontCheckPrerouted(in.Prerouted, in.Tokens, cfg)
	}
	logits := in.Logits
	if in.RouterWeight != nil {
		if in.Tokens == 0 {
			return MoeFrontRouteWithSort(nil, 0, cfg)
		}
		var err error
		logits, err = MoeRouterGemmFp32(in.RouterWeight, in.X, in.Tokens, cfg.Hidden, cfg.NumExperts)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrMoeFrontGeometry, err)
		}
	}
	return MoeFrontRouteWithSort(logits, in.Tokens, cfg)
}

func moeFrontCheckPrerouted(r *MoeFrontRoute, tokens int, cfg MoeFrontConfig) (*MoeFrontRoute, error) {
	n := tokens * cfg.TopK
	if r.Tokens != tokens || r.TopK != cfg.TopK || len(r.ExpertIDs) != n || len(r.Scales) != n {
		return nil, fmt.Errorf("%w: prerouted shape tokens=%d topK=%d ids=%d scales=%d", ErrMoeFrontGeometry, r.Tokens, r.TopK, len(r.ExpertIDs), len(r.Scales))
	}
	for t := 0; t < tokens; t++ {
		row := r.ExpertIDs[t*cfg.TopK : (t+1)*cfg.TopK]
		for k, id := range row {
			if id < 0 || int(id) >= cfg.NumExperts {
				return nil, fmt.Errorf("%w: prerouted id %d at token %d slot %d", ErrMoeFrontRoute, id, t, k)
			}
			// A duplicate would route one token through one expert twice.
			for _, prev := range row[:k] {
				if prev == id {
					return nil, fmt.Errorf("%w: prerouted id %d repeated at token %d", ErrMoeFrontRoute, id, t)
				}
			}
		}
	}
	for i, s := range r.Scales {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return nil, fmt.Errorf("%w: prerouted scale %d is non-finite", ErrMoeFrontRoute, i)
		}
	}
	return r, nil
}

func moeFrontFC1(in MoeFrontInput, cfg MoeFrontConfig, w MoeFrontWorkItem, h []float32) error {
	x := in.X[w.Token*cfg.Hidden : (w.Token+1)*cfg.Hidden]
	fc1 := in.FC1[w.Expert*cfg.Inter*cfg.Hidden:]
	for i := 0; i < cfg.Inter; i++ {
		var acc float32
		for j, v := range x {
			acc += fc1[i*cfg.Hidden+j] * v
		}
		if cfg.Activation != nil {
			acc = cfg.Activation(acc)
		}
		h[i] = acc
	}
	if !cfg.QuantizeFC1 {
		return nil
	}
	// Per-row quantization keeps the rounded operand independent of dispatch
	// order, so router-first permutation cannot change latent_down's input.
	packed, scales, global, err := QuantizeNVFP4Reference(1, cfg.Inter, h)
	if err != nil {
		return fmt.Errorf("%w: fc1 epilogue: %v", ErrMoeFrontGeometry, err)
	}
	q, err := DecodeNVFP4Reference(1, cfg.Inter, packed, scales, global)
	if err != nil {
		return fmt.Errorf("%w: fc1 epilogue: %v", ErrMoeFrontGeometry, err)
	}
	copy(h, q)
	return nil
}
