package qwen4exp

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

var ErrUnsupportedKernelPair = errors.New("qwen4exp: unsupported QSA/GDN kernel pairing")

// DequantizeFP8E4M3FNPLE expands one PLE row using the row's exact scale.
// E4M3FN has no infinities; exponent 15 encodes finite values except the two
// NaN bit patterns. Rejecting NaN keeps malformed checkpoint rows fail-closed.
func DequantizeFP8E4M3FNPLE(row []byte, scale float32) ([]float32, error) {
	if len(row) == 0 || scale == 0 || float32(math.Abs(float64(scale))) != scale || math.IsInf(float64(scale), 0) {
		return nil, errors.New("qwen4exp: invalid FP8 PLE row or scale")
	}
	out := make([]float32, len(row))
	for i, bits := range row {
		sign := float32(1)
		if bits&0x80 != 0 {
			sign = -1
		}
		exp, frac := (bits>>3)&0xf, bits&7
		if exp == 0xf && frac == 7 {
			return nil, fmt.Errorf("qwen4exp: FP8 PLE row contains NaN at %d", i)
		}
		var value float32
		if exp == 0 {
			value = float32(frac) * (1.0 / 512.0)
		} else {
			value = float32(math.Ldexp(float64(8+frac), int(exp)-10))
		}
		out[i] = sign * value * scale
	}
	return out, nil
}

// TextModelTensor reports whether a checkpoint tensor belongs to the text
// model. Upstream multimodal exclusions are prefix boundaries, not substring
// matches: a text tensor merely containing "visual" remains text-owned.
func TextModelTensor(name string, excludedPrefixes []string) bool {
	for _, prefix := range excludedPrefixes {
		prefix = strings.TrimSuffix(prefix, ".")
		if name == prefix || strings.HasPrefix(name, prefix+".") {
			return false
		}
	}
	return true
}

// NormalizeTPPlan treats an absent upstream tensor-parallel plan as an empty
// plan. Callers can execute the single-rank path rather than dereference nil.
func NormalizeTPPlan(plan map[string]string) map[string]string {
	if plan == nil {
		return map[string]string{}
	}
	out := make(map[string]string, len(plan))
	for k, v := range plan {
		out[k] = v
	}
	return out
}

type ToolTokenGuard struct{ zeroRun int }

// Observe rejects a second consecutive token zero while thinking/tool parsing
// is active. One zero can be a legitimate tokenizer id; repetition without
// parser progress is the upstream loop signature and must halt deterministically.
func (g *ToolTokenGuard) Observe(token int, parserProgress bool, active bool) error {
	if !active || parserProgress || token != 0 {
		g.zeroRun = 0
		return nil
	}
	g.zeroRun++
	if g.zeroRun > 1 {
		return errors.New("qwen4exp: thinking/tool parser made no progress at token 0")
	}
	return nil
}

func ValidateQSAAndGDNKernels(qsa, gdn string) error {
	compatible := map[string]string{"cuda-exact": "cuda-exact", "metal-exact": "metal-exact", "cpu-oracle": "cpu-oracle"}
	if qsa == "" || gdn == "" || compatible[qsa] != gdn {
		return fmt.Errorf("%w: qsa=%q gdn=%q", ErrUnsupportedKernelPair, qsa, gdn)
	}
	return nil
}
