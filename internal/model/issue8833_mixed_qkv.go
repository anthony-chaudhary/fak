package model

import (
	"errors"
	"fmt"
	"os"
	"strings"
)

// mixedQKVMode is deliberately unexported: #8833 is an off-by-default experiment, not a
// compatibility promise. The environment is read when a Session is constructed, never globally.
type mixedQKVMode uint8

const (
	mixedQKVOff mixedQKVMode = iota
	mixedQKVControl
	mixedQKVCandidate
)

const mixedQKVEnv = "FAK_QWEN35_MIXED_QKV"

func parseMixedQKVMode(value string) (mixedQKVMode, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "":
		return mixedQKVOff, nil
	case "control":
		return mixedQKVControl, nil
	case "candidate":
		return mixedQKVCandidate, nil
	default:
		return mixedQKVOff, fmt.Errorf("model: %s must be control or candidate, got %q", mixedQKVEnv, value)
	}
}

type mixedQKVRequest struct {
	Mode                    mixedQKVMode
	Q, K, V                 string
	Input                   []float32
	Hidden, Query, Key, Val int
}

type mixedQKVResult struct {
	Q, K, V   []float32
	Submitted bool
	Err       error
}

type mixedQKVExecutor func(mixedQKVRequest) mixedQKVResult

type mixedQKVSession struct {
	mode mixedQKVMode
	exec mixedQKVExecutor
}

func (s *Session) initMixedQKV() {
	if s == nil {
		return
	}
	s.mixedQKV = newMixedQKVSessionFromEnv()
	s.initMixedQKVNative()
}

func newMixedQKVSessionFromEnv() mixedQKVSession {
	mode, err := parseMixedQKVMode(os.Getenv(mixedQKVEnv))
	if err != nil {
		// Invalid opt-in is fail-closed rather than silently selecting experimental GPU work.
		return mixedQKVSession{}
	}
	return mixedQKVSession{mode: mode}
}

// mixedQKVGeometry is intentionally exact. The mixed family is Q8_0 query/key plus Q4_K value
// from the Qwen3.5 9B full-attention projection: hidden 4096, 16 query heads, 4 KV heads,
// head-dim 256, and the packed query gate doubles query output to 8192.
func mixedQKVGeometry(cfg Config, q, k, v int) bool {
	return cfg.IsQwen35Hybrid() && cfg.AttnOutputGate && cfg.HiddenSize == 4096 &&
		cfg.NumHeads == 16 && cfg.NumKVHeads == 4 && cfg.HeadDim == 256 &&
		q == 8192 && k == 1024 && v == 1024
}

func (s *Session) tryMixedQKV(mat matKernel, qn, kn, vn string, xp any, xf []float32, q, k, v int) ([][]float32, bool, error) {
	if s == nil || s.mixedQKV.mode == mixedQKVOff || s.mixedQKV.exec == nil ||
		!mixedQKVGeometry(s.M.Cfg, q, k, v) {
		return nil, false, nil
	}
	// Family gating is exact: Q/K must be Q8 minority tensors and V must be resident Q4_K.
	if s.M.q8w[qn] == nil || s.M.q8w[kn] == nil || s.M.q4kw[vn] == nil ||
		s.M.q4kw[qn] != nil || s.M.q4kw[kn] != nil || s.M.q8w[vn] != nil {
		return nil, false, nil
	}
	res := s.mixedQKV.exec(mixedQKVRequest{Mode: s.mixedQKV.mode, Q: qn, K: kn, V: vn,
		Input: xf, Hidden: s.M.Cfg.HiddenSize, Query: q, Key: k, Val: v})
	if res.Err != nil {
		if res.Submitted {
			return nil, true, res.Err // Never retry after ownership has submitted GPU work.
		}
		return nil, false, nil // Typed/pre-submit decline: preserve the established mat.mul fallback.
	}
	if !res.Submitted || len(res.Q) != q || len(res.K) != k || len(res.V) != v {
		return nil, false, nil
	}
	return [][]float32{res.Q, res.K, res.V}, true, nil
}

var errMixedQKVPostSubmit = errors.New("model: mixed QKV failed after Metal submission")
