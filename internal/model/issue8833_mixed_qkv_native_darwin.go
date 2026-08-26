//go:build darwin && arm64 && cgo

package model

import "github.com/anthony-chaudhary/fak/internal/metalgemm"

func (s *Session) initMixedQKVNative() {
	if s == nil || s.M == nil || s.mixedQKV.mode == mixedQKVOff {
		return
	}
	s.mixedQKV.exec = func(req mixedQKVRequest) mixedQKVResult {
		qw := s.M.metalQ8Weight(req.Q, s.M.q8w[req.Q])
		kw := s.M.metalQ8Weight(req.K, s.M.q8w[req.K])
		vw := s.M.metalQ4KWeight(req.V, s.M.q4kw[req.V])
		if qw == nil || kw == nil || vw == nil {
			return mixedQKVResult{Err: metalgemm.MixedQKVUnavailable("weight upload unavailable")}
		}
		x := s.quantizeVecQ8(req.Input)
		selector := metalgemm.MixedQKVControl
		if req.Mode == mixedQKVCandidate {
			selector = metalgemm.MixedQKVCandidate
		}
		got, err := metalgemm.ExecuteMixedQKV(selector, metalgemm.MixedQKVInput{
			Q: qw, K: kw, V: vw, XQ: x.q, XD: x.d, XF: req.Input, Hidden: req.Hidden,
		})
		return mixedQKVResult{Q: got.Q, K: got.K, V: got.V, Submitted: got.Submitted, Err: err}
	}
}
