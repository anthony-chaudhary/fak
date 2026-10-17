package model

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// Independent scalar witness for the model.py 4dd75a8b / kernel.py 517ef625
// equations. It decodes the fixture bytes itself and does not call production
// row/linear/FP8/RMSNorm/RoPE/mHC/router/attention/head helpers. This oracle
// binds gate temperature to 1, the reference default; FAK has no such field.
// This is reduced-fixture CPU math evidence, not checkpoint or GPU parity.
type dsparkOracle struct {
	cfg             Config
	weights         map[string]tinySTTensor
	embedding, head []float32
	context         [3][][]float32
}

func dsparkOBF(x float32) float32 {
	b := math.Float32bits(x)
	return math.Float32frombits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}

func dsparkORound(x []float32) []float32 {
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = dsparkOBF(v)
	}
	return y
}

func dsparkOE4(code byte) float32 {
	sign := float32(1)
	if code&128 != 0 {
		sign = -1
	}
	code &= 127
	exp, mant := int(code>>3), int(code&7)
	if exp == 0 {
		return sign * float32(math.Ldexp(float64(mant), -9))
	}
	return sign * float32(math.Ldexp(1+float64(mant)/8, exp-7))
}

// Exhaustive nearest-code search, independent of the production binary search.
func dsparkOQuant(x []float32) []float32 {
	out := make([]float32, len(x))
	for start := 0; start < len(x); start += 32 {
		amax := float32(1e-4)
		for _, v := range x[start : start+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		ratio := float32(amax * float32(1.0/448.0))
		scale := float32(math.Exp2(math.Ceil(math.Log2(float64(ratio)))))
		for j, v := range x[start : start+32] {
			z := float32(math.Abs(float64(v / scale)))
			best, distance := byte(0), float32(math.Inf(1))
			for code := 0; code <= 126; code++ {
				delta := float32(math.Abs(float64(z - dsparkOE4(byte(code)))))
				if delta < distance || delta == distance && code%2 == 0 {
					best, distance = byte(code), delta
				}
			}
			value := dsparkOE4(best) * scale
			if math.Signbit(float64(v)) {
				value = -value
			}
			out[start+j] = dsparkOBF(value)
		}
	}
	return out
}

func (o *dsparkOracle) row(name string, row int) []float32 {
	w := o.weights[name]
	width := w.shape[0]
	if len(w.shape) == 2 {
		width = w.shape[1]
	}
	if w.dtype == "I8" {
		width *= 2
	}
	out := make([]float32, width)
	scale := o.weights[strings.TrimSuffix(name, ".weight")+".scale"]
	fp4 := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
	for k := range out {
		offset := row*width + k
		switch w.dtype {
		case "F32":
			out[k] = math.Float32frombits(binary.LittleEndian.Uint32(w.data[4*offset:]))
		case "BF16":
			out[k] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(w.data[2*offset:])) << 16)
		case "F8_E4M3":
			s := scale.data[(row/32)*scale.shape[1]+k/32]
			out[k] = dsparkOE4(w.data[offset]) * float32(math.Exp2(float64(int(s)-127)))
		case "I8":
			code := w.data[offset/2]
			if k%2 == 0 {
				code &= 15
			} else {
				code >>= 4
			}
			s := scale.data[row*scale.shape[1]+k/32]
			out[k] = fp4[code] * float32(math.Exp2(float64(int(s)-127)))
		default:
			panic("unexpected oracle storage")
		}
	}
	return out
}

func dsparkODot(a, b []float32) float32 {
	var x float32
	for i := range a {
		x += float32(a[i] * b[i])
	}
	return x
}

func (o *dsparkOracle) linear(name string, x []float32, bf bool) []float32 {
	w := o.weights[name]
	if w.dtype == "I8" || w.dtype == "F8_E4M3" {
		x = dsparkOQuant(x)
	}
	out := make([]float32, w.shape[0])
	for i := range out {
		out[i] = dsparkODot(o.row(name, i), x)
		if bf {
			out[i] = dsparkOBF(out[i])
		}
	}
	return out
}

func (o *dsparkOracle) norm(name string, x []float32) []float32 {
	w := o.row(name, 0)
	var sum float32
	for _, v := range x {
		sum += float32(v * v)
	}
	inv := float32(1 / math.Sqrt(float64(sum/float32(len(x))+float32(o.cfg.RMSNormEps))))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = dsparkOBF(float32(v*inv) * w[i])
	}
	return out
}

func (o *dsparkOracle) rope(x []float32, position int, inverse bool) []float32 {
	x = append([]float32(nil), x...)
	rd := o.cfg.QKRopeHeadDim
	for j := 0; j < rd/2; j++ {
		angle := float64(position) / math.Pow(o.cfg.RopeTheta, float64(2*j)/float64(rd))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		if inverse {
			s = -s
		}
		k := len(x) - rd + 2*j
		a, b := x[k], x[k+1]
		x[k], x[k+1] = float32(a*c)-float32(b*s), float32(b*c)+float32(a*s)
	}
	return dsparkORound(x)
}

func (o *dsparkOracle) kv(stage, pos int, x []float32) []float32 {
	p := fmt.Sprintf("mtp.%d.attn.", stage)
	return dsparkOQuant(o.rope(o.norm(p+"kv_norm.weight", o.linear(p+"wkv.weight", x, true)), pos, false))
}

func (o *dsparkOracle) seed(rows []v41DSparkContextRow) {
	for _, row := range rows {
		var x []float32
		for _, values := range row.Hidden {
			x = append(x, values...)
		}
		main := o.norm("mtp.0.main_norm.weight", o.linear("mtp.0.main_proj.weight", x, true))
		for stage := 0; stage < 3; stage++ {
			o.context[stage] = append(o.context[stage], o.kv(stage, row.Position, main))
		}
	}
}

type dsparkOMix struct{ pre, post, comb []float32 }

func dsparkOSigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

func (o *dsparkOracle) mix(stage int, sub string, streams [][]float32) dsparkOMix {
	var flat []float32
	for _, x := range streams {
		flat = append(flat, x...)
	}
	p := fmt.Sprintf("mtp.%d.hc_%s_", stage, sub)
	z := o.linear(p+"fn", flat, false)
	inv := float32(1 / math.Sqrt(float64(dsparkODot(flat, flat)/float32(len(flat))+float32(o.cfg.RMSNormEps))))
	scale, base := o.row(p+"scale", 0), o.row(p+"base", 0)
	eps := float32(o.cfg.DeepSeekV41.HCEps)
	if eps == 0 {
		eps = 1e-6
	}
	m := dsparkOMix{make([]float32, 4), make([]float32, 4), make([]float32, 16)}
	for i := 0; i < 4; i++ {
		m.pre[i] = dsparkOSigmoid(float32(z[i]*inv)*scale[0]+base[i]) + eps
		m.post[i] = 2 * dsparkOSigmoid(float32(z[4+i]*inv)*scale[1]+base[4+i])
	}
	for r := 0; r < 4; r++ {
		maximum := float32(math.Inf(-1))
		for c := 0; c < 4; c++ {
			k := 4*r + c
			m.comb[k] = float32(z[8+k]*inv)*scale[2] + base[8+k]
			maximum = max(maximum, m.comb[k])
		}
		var total float32
		for c := 0; c < 4; c++ {
			k := 4*r + c
			m.comb[k] = float32(math.Exp(float64(m.comb[k] - maximum)))
			total += m.comb[k]
		}
		for c := 0; c < 4; c++ {
			k := 4*r + c
			m.comb[k] = m.comb[k]/total + eps
		}
	}
	normalize := func(columns bool) {
		for outer := 0; outer < 4; outer++ {
			var sum float32
			for inner := 0; inner < 4; inner++ {
				k := outer*4 + inner
				if columns {
					k = inner*4 + outer
				}
				sum += m.comb[k]
			}
			for inner := 0; inner < 4; inner++ {
				k := outer*4 + inner
				if columns {
					k = inner*4 + outer
				}
				m.comb[k] /= sum + eps
			}
		}
	}
	normalize(true)
	iters := o.cfg.DeepSeekV41.HCSinkhornIters
	if iters == 0 {
		iters = 20
	}
	for n := 1; n < iters; n++ {
		normalize(false)
		normalize(true)
	}
	return m
}

func dsparkOPre(streams [][]float32, pre []float32) []float32 {
	out := make([]float32, len(streams[0]))
	for s, x := range streams {
		for j, v := range x {
			out[j] += float32(v * pre[s])
		}
	}
	return dsparkORound(out)
}
func dsparkOPost(streams [][]float32, x []float32, m dsparkOMix) [][]float32 {
	out := make([][]float32, 4)
	for dst := range out {
		out[dst] = make([]float32, len(x))
		for src, v := range streams {
			for j, y := range v {
				out[dst][j] += float32(y * m.comb[src*4+dst])
			}
		}
		for j, y := range x {
			out[dst][j] = dsparkOBF(out[dst][j] + float32(y*m.post[dst]))
		}
	}
	return out
}

func (o *dsparkOracle) attention(stage, position int, input [5][]float32) [5][]float32 {
	p := fmt.Sprintf("mtp.%d.attn.", stage)
	c := o.cfg
	keys := append([][]float32(nil), o.context[stage]...)
	var query [5][]float32
	for i, x := range input {
		query[i] = o.linear(p+"wq_b.weight", o.norm(p+"q_norm.weight", o.linear(p+"wq_a.weight", x, true)), true)
		for h := 0; h < c.NumHeads; h++ {
			copy(query[i][h*c.HeadDim:], o.rope(query[i][h*c.HeadDim:(h+1)*c.HeadDim], position+i, false))
		}
		keys = append(keys, o.kv(stage, position+i, x))
	}
	sink := o.row(p+"attn_sink", 0)
	var result [5][]float32
	for i := range input {
		values := make([]float32, c.NumHeads*c.HeadDim)
		for h := 0; h < c.NumHeads; h++ {
			q := query[i][h*c.HeadDim : (h+1)*c.HeadDim]
			scores := make([]float32, len(keys))
			maximum := sink[h]
			for k, key := range keys {
				scores[k] = dsparkODot(q, key) * float32(1/math.Sqrt(float64(c.HeadDim)))
				maximum = max(maximum, scores[k])
			}
			denom := float32(math.Exp(float64(sink[h] - maximum)))
			for k := range scores {
				scores[k] = float32(math.Exp(float64(scores[k] - maximum)))
				denom += scores[k]
			}
			v := values[h*c.HeadDim : (h+1)*c.HeadDim]
			for k, key := range keys {
				weight := scores[k] / denom
				for j, x := range key {
					v[j] += float32(weight * x)
				}
			}
			copy(v, o.rope(dsparkORound(v), position+i, true))
		}
		values = dsparkOQuant(values)
		joined := make([]float32, c.OGroups*c.OLoraRank)
		groupWidth := len(values) / c.OGroups
		for j := range joined {
			w := dsparkORound(o.row(p+"wo_a.weight", j))
			group := j / c.OLoraRank
			joined[j] = dsparkOBF(dsparkODot(w, values[group*groupWidth:(group+1)*groupWidth]))
		}
		result[i] = o.linear(p+"wo_b.weight", joined, true)
	}
	return result
}

func (o *dsparkOracle) routing(stage int, x []float32) ([]int, []float32) {
	p := fmt.Sprintf("mtp.%d.ffn.gate.", stage)
	z, bias := o.linear(p+"weight", x, false), o.row(p+"e_score_correction_bias", 0)
	raw, ids := make([]float32, 128), make([]int, 128)
	for i, v := range z {
		raw[i] = float32(math.Sqrt(float64(float32(math.Max(float64(v), 0) + math.Log1p(math.Exp(-math.Abs(float64(v))))))))
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool { return raw[ids[i]]+bias[ids[i]] > raw[ids[j]]+bias[ids[j]] })
	ids = ids[:3]
	var total float32
	for _, id := range ids {
		total += raw[id]
	}
	weights := make([]float32, 128)
	for _, id := range ids {
		weights[id] = float32(raw[id]/float32(total+float32(1e-20))) * float32(o.cfg.RoutedScalingFactor)
	}
	sort.Ints(ids)
	return ids, weights
}
func (o *dsparkOracle) expert(stem string, x []float32, weight float32) []float32 {
	g, u := o.linear(stem+".w1.weight", x, true), o.linear(stem+".w3.weight", x, true)
	limit := float32(o.cfg.SwigluLimit)
	for i := range g {
		if limit > 0 {
			g[i] = min(g[i], limit)
			u[i] = max(-limit, min(u[i], limit))
		}
		g[i] = dsparkOBF(float32(float32(g[i]*dsparkOSigmoid(g[i]))*u[i]) * weight)
	}
	return o.linear(stem+".w2.weight", g, true)
}
func (o *dsparkOracle) moe(stage int, x []float32, wrongRoute bool) []float32 {
	ids, weights := o.routing(stage, x)
	p := fmt.Sprintf("mtp.%d.ffn.", stage)
	sum := make([]float32, o.cfg.HiddenSize)
	for _, id := range ids {
		selected := id
		if wrongRoute && id == 127 {
			selected = 126
		}
		y := o.expert(p+"experts."+strconv.Itoa(selected), x, weights[id])
		for j, v := range y {
			sum[j] += v
		}
	}
	for j, v := range o.expert(p+"shared_experts", x, 1) {
		sum[j] += v
	}
	return dsparkORound(sum)
}

type dsparkOStage struct{ attentionInput, attention, ffnInput, ffn [5][]float32 }
type dsparkOResult struct {
	hidden, logits [5][]float32
	confidence     [5]float32
	tokens         [5]int
	stages         [3]dsparkOStage
}

func (o *dsparkOracle) forward(anchor, skip int, wrongRoute, zeroHead bool) dsparkOResult {
	c := o.cfg
	position := len(o.context[0])
	var streams [5][][]float32
	var pre [5][]float32
	for i := range streams {
		token := c.DeepSeekV41.DSpark.NoiseTokenID
		if i == 0 {
			token = anchor
		}
		x := dsparkORound(o.embedding[token*c.HiddenSize : (token+1)*c.HiddenSize])
		for s := 0; s < 4; s++ {
			streams[i] = append(streams[i], append([]float32(nil), x...))
		}
		pre[i] = []float32{1, 0, 0, 0}
	}
	var result dsparkOResult
	for stage := 0; stage < 3; stage++ {
		if stage == skip {
			continue
		}
		p := fmt.Sprintf("mtp.%d.", stage)
		var attnMix [5]dsparkOMix
		trace := &result.stages[stage]
		for i := range streams {
			attnMix[i] = o.mix(stage, "attn", streams[i])
			trace.attentionInput[i] = o.norm(p+"attn_norm.weight", dsparkOPre(streams[i], pre[i]))
		}
		trace.attention = o.attention(stage, position, trace.attentionInput)
		for i := range streams {
			updated := dsparkOPost(streams[i], trace.attention[i], attnMix[i])
			ffnMix := o.mix(stage, "ffn", updated)
			trace.ffnInput[i] = o.norm(p+"ffn_norm.weight", dsparkOPre(updated, attnMix[i].pre))
			trace.ffn[i] = o.moe(stage, trace.ffnInput[i], wrongRoute)
			streams[i], pre[i] = dsparkOPost(updated, trace.ffn[i], ffnMix), ffnMix.pre
		}
	}
	previous := anchor
	for i := range result.hidden {
		result.hidden[i] = dsparkOPre(streams[i], pre[i])
		x := o.norm("mtp.2.norm.weight", result.hidden[i])
		markov := o.row("mtp.2.markov_head.embed.weight", previous)
		logits := o.linear("mtp.2.markov_head.head.weight", markov, false)
		best := 0
		for token := range logits {
			if !zeroHead {
				logits[token] += dsparkODot(o.head[token*c.HiddenSize:(token+1)*c.HiddenSize], x)
			}
			if logits[token] > logits[best] {
				best = token
			}
		}
		result.logits[i], result.tokens[i] = logits, best
		confInput := append(append([]float32(nil), result.hidden[i]...), markov...)
		result.confidence[i] = o.linear("mtp.2.confidence_head.proj.weight", confInput, false)[0]
		previous = best
	}
	return result
}

func dsparkOCheck(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths %d/%d", name, len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsNaN(float64(want[i])) || math.IsInf(float64(got[i]), 0) || math.IsInf(float64(want[i]), 0) || math.Abs(float64(got[i]-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d]: got %.9g want %.9g", name, i, got[i], want[i])
		}
	}
}
func dsparkODistance(a, b [5][]float32) float64 {
	var d float64
	for i := range a {
		for j, v := range a[i] {
			d = math.Max(d, math.Abs(float64(v-b[i][j])))
		}
	}
	return d
}

// fak-test:runtime medium est=5s lane=default
func TestV41DSparkIndependentScalarOracle(t *testing.T) {
	ctx := context.Background()
	o := &dsparkOracle{weights: map[string]tinySTTensor{}}
	f, _, rows := v41DSparkReferenceFixture(t, func(ts map[string]tinySTTensor) {
		for name, w := range ts {
			stage := int(name[4] - '0')
			if w.dtype == "I8" {
				expert := 0
				parts := strings.Split(name, ".")
				if len(parts) > 3 && parts[3] == "experts" {
					expert, _ = strconv.Atoi(parts[4])
				}
				for i := range w.data {
					lo := byte(1 + (expert+stage+i)%6)
					hi := byte(1 + (2*expert+stage+3*i)%6)
					if (expert+i)%3 == 0 {
						hi |= 8
					}
					w.data[i] = lo | (hi << 4)
				}
			} else if w.dtype == "F8_E4M3" {
				codes := [...]byte{0x20, 0xa8, 0x30, 0x18, 0xb0, 0x28, 0xa0}
				for i := range w.data {
					w.data[i] = codes[(i*(2*stage+1)+stage+len(name))%len(codes)]
				}
			} else if w.dtype == "F8_E8M0" {
				for i := range w.data {
					w.data[i] = byte(121 + stage + i%2)
				}
			} else if strings.HasSuffix(name, "gate.weight") {
				for i := 0; i < len(w.data)/2; i++ {
					v := dsparkOBF(float32((i/64+stage)%7-3) / 64)
					binary.LittleEndian.PutUint16(w.data[2*i:], uint16(math.Float32bits(v)>>16))
				}
			} else if strings.HasSuffix(name, "e_score_correction_bias") {
				for i := 0; i < 128; i++ {
					v := float32(-16)
					switch i {
					case 2:
						v = 4
					case 11:
						v = 6
					case 127:
						v = 8
					}
					binary.LittleEndian.PutUint32(w.data[4*i:], math.Float32bits(v))
				}
			} else if strings.HasSuffix(name, "norm.weight") {
				for i := 0; i < len(w.data)/2; i++ {
					v := dsparkOBF(1 + float32(stage)/16 + float32(i%3)/32)
					binary.LittleEndian.PutUint16(w.data[2*i:], uint16(math.Float32bits(v)>>16))
				}
			} else if strings.Contains(name, "markov_head.head") {
				for i := 0; i < len(w.data)/2; i++ {
					v := dsparkOBF(float32(i%7-3) / 16)
					binary.LittleEndian.PutUint16(w.data[2*i:], uint16(math.Float32bits(v)>>16))
				}
			} else if strings.HasSuffix(name, "_fn") {
				for i := 0; i < len(w.data)/4; i++ {
					binary.LittleEndian.PutUint32(w.data[4*i:], math.Float32bits(float32((i*(stage+1))%13-6)/2048))
				}
			}
			ts[name] = w
			o.weights[name] = tinySTTensor{dtype: w.dtype, shape: append([]int(nil), w.shape...), data: append([]byte(nil), w.data...)}
		}
	})
	o.cfg = f.cfg
	// Read-only scalar input copies; production outputs never supply the oracle.
	embedding := f.model.tensor("model.embed_tokens.weight")
	for i := range embedding {
		embedding[i] = dsparkOBF(float32((i*7+i/64)%23-11) / 16)
	}
	head := f.model.tensor("lm_head.weight")
	for i := range head {
		head[i] = dsparkOBF(float32((i*3+i/64*5)%19-9) / 64)
	}
	o.embedding, o.head = append([]float32(nil), embedding...), append([]float32(nil), head...)
	o.seed(rows)
	if err := f.AppendCommitted(ctx, rows); err != nil {
		t.Fatal(err)
	}
	for stage := 0; stage < 3; stage++ {
		for i := range rows {
			dsparkOCheck(t, fmt.Sprintf("stage%d context%d", stage, i), f.ring[stage][i], o.context[stage][i])
		}
	}
	want := o.forward(1, -1, false, false)
	for stage, trace := range want.stages {
		got, err := f.attention(ctx, stage, trace.attentionInput)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			dsparkOCheck(t, fmt.Sprintf("stage%d attention%d", stage, i), got[i], trace.attention[i])
			ffn, err := f.moe(ctx, stage, trace.ffnInput[i])
			if err != nil {
				t.Fatal(err)
			}
			dsparkOCheck(t, fmt.Sprintf("stage%d routed/shared FFN%d", stage, i), ffn, trace.ffn[i])
		}
		ids, weights := o.routing(stage, trace.ffnInput[0])
		picks, err := f.route(ctx, stage, trace.ffnInput[0])
		if err != nil {
			t.Fatal(err)
		}
		if len(picks) != 3 || fmt.Sprint(ids) != "[2 11 127]" {
			t.Fatalf("bad oracle routing %v / %v", ids, picks)
		}
		for i, pick := range picks {
			if pick.expert != ids[i] {
				t.Fatalf("stage%d routed expert %d, want %d", stage, pick.expert, ids[i])
			}
			dsparkOCheck(t, "route weight", []float32{pick.weight}, []float32{weights[ids[i]]})
		}
	}
	got, err := f.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.hidden {
		dsparkOCheck(t, fmt.Sprintf("full hidden%d", i), got.hidden[i], want.hidden[i])
		dsparkOCheck(t, fmt.Sprintf("full logits%d", i), got.Logits[i], want.logits[i])
		dsparkOCheck(t, fmt.Sprintf("confidence%d", i), []float32{got.ConfidenceLogits[i]}, []float32{want.confidence[i]})
		if got.Proposed[i] != want.tokens[i] {
			t.Fatalf("token%d got%d want%d", i, got.Proposed[i], want.tokens[i])
		}
	}
	// Mutation controls must be numerically visible, not merely different code.
	for stage := 0; stage < 3; stage++ {
		bad := o.forward(1, stage, false, false)
		if dsparkODistance(want.hidden, bad.hidden) < 1e-3 || dsparkODistance(want.logits, bad.logits) < 1e-4 {
			t.Fatalf("skipping stage%d is invisible", stage)
		}
	}
	wrong := o.forward(1, -1, true, false)
	if dsparkODistance(want.hidden, wrong.hidden) < 1e-3 || dsparkODistance(want.logits, wrong.logits) < 1e-4 {
		t.Fatal("wrong selected expert is invisible")
	}
	noHead := o.forward(1, -1, false, true)
	if dsparkODistance(want.logits, noHead.logits) < 1e-3 {
		t.Fatal("nonzero shared head is invisible")
	}
	// Same lineage alone cannot admit an incomplete context ring.
	for stage := 0; stage < 3; stage++ {
		saved := f.ring[stage][0]
		f.ring[stage][0] = nil
		if proposal, err := f.Propose(ctx, 1); proposal != nil || err == nil {
			t.Fatalf("stage%d incomplete context admitted", stage)
		}
		f.ring[stage][0] = saved
	}
}
