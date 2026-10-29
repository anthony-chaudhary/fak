package model

import "os"

// v41CrossLayerPrefetchEnv set to "1" opts a session's V4.1 batched routed layer
// into next-layer predicted read-ahead (ExpertPrefetchCrossLayer has the same
// effect per session). Unset or any other value keeps same-layer read-ahead only.
// An ExpertPrefetchOnDemand session stays off either way.
const v41CrossLayerPrefetchEnv = "FAK_V41_XLAYER_PREFETCH"

// v41CrossLayerPrefetchEnabled reports whether this session's routed layer runs
// the cross-layer predictor. It is read once when the routed layer binds.
func (s *Session) v41CrossLayerPrefetchEnabled() bool {
	if s == nil || s.ExpertPrefetch == ExpertPrefetchOnDemand {
		return false
	}
	return s.ExpertPrefetch == ExpertPrefetchCrossLayer || os.Getenv(v41CrossLayerPrefetchEnv) == "1"
}

// v41CrossLayerPrefetch runs after layer's demand set is paged in and before its
// expert GEMMs: it reconciles the prediction made for layer against the realized
// picks, then predicts layer+1 from layer's expert input and issues the
// predicted set as checkpoint read-ahead hints. It never changes a pick, a
// weight or a demand read, so a mispredict costs advice bandwidth only.
func (s *Session) v41CrossLayerPrefetch(layer int, picks []routePick, x []float32) {
	if s == nil || s.M == nil {
		return
	}
	s.recordNextLayerActual(layer, picks)
	next := layer + 1
	if next >= s.M.Cfg.NumLayers {
		return
	}
	if s.M.v41HashRoutedLayer(next) {
		// Token-id routing is exact, so no gate GEMV runs. This path has no
		// token id, so it records the layer and leaves it to demand page-in.
		addCrossLayerPrefetchStats(s, CrossLayerPrefetchStats{HashRouted: 1})
		return
	}
	s.v41HintPredictedPicks(next, s.M.v41PredictLayerPicks(next, x))
}

// v41HashRoutedLayer reports whether layer routes by a token-to-expert table
// rather than a gate.
func (m *Model) v41HashRoutedLayer(layer int) bool {
	_, ok := m.manifest[v4HashTensorName(layer)]
	return ok
}

// v41PredictLayerPicks applies layer's router to x on the host: the gate GEMV and
// the same noaux score-plus-bias top-k the real router uses. x is the previous
// layer's expert input, so the picks are a prediction. Any refusal returns nil.
func (m *Model) v41PredictLayerPicks(layer int, x []float32) []routePick {
	cfg := m.Cfg
	E, H := cfg.NumExperts, cfg.HiddenSize
	if E <= 0 || H <= 0 || len(x) != H {
		return nil
	}
	name := layerName(layer, "ffn.gate.weight")
	if out, in, ok := m.residentShape(name); !ok || out != E || in != H {
		return nil
	}
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return nil
	}
	var bias []float32
	if m.has(m.v41LeafName(layer, v41LeafGateBias)) {
		bias = m.v41LeafTensor(layer, v41LeafGateBias)
	}
	picks, err := v41RouteForGeometry(m.residentMatRows(name, x, E, H), bias, routeCfg, true)
	if err != nil {
		return nil
	}
	return picks
}

// v41HintPredictedPicks records predicted as the pending prediction for target
// and hints the longest pick-order prefix whose projections the ring could pin
// together. Hints go through readAheadV41RoutedPicks, so they obey its rules:
// OS advice only, no ring staging, no heat, no resident fallback, resident
// projections skipped.
func (s *Session) v41HintPredictedPicks(target int, predicted []routePick) {
	if s == nil || s.M == nil || len(predicted) == 0 {
		return
	}
	recordCrossLayerPrediction(s, target, predicted)
	if s.ExpertRingBytes <= 0 {
		return
	}
	names := make([][3]string, len(predicted))
	for slot, pick := range predicted {
		if routed := s.M.v41RoutedNames(target, pick.expert); routed != nil {
			names[slot] = routed.weights
		} else {
			stem := "ffn.experts." + itoa(pick.expert)
			names[slot] = [3]string{layerName(target, stem+".w1.weight"), layerName(target, stem+".w3.weight"), layerName(target, stem+".w2.weight")}
		}
	}
	r := s.routedExpertRing(names[0][0])
	if r == nil || s.M.expertCheckpoint == nil {
		return
	}
	done := s.ringEnter(r)
	defer done()
	n := len(names)
	for n > 0 && !s.v41RoutedLayerRingFits(r, names[:n]) {
		n--
	}
	if n == 0 {
		return
	}
	before := s.M.expertCheckpoint.Stats().Readahead.Bytes
	s.readAheadV41RoutedPicks(r, names[:n])
	if hinted := s.M.expertCheckpoint.Stats().Readahead.Bytes - before; hinted > 0 {
		addCrossLayerPrefetchStats(s, CrossLayerPrefetchStats{HintedBytes: hinted})
	}
}
