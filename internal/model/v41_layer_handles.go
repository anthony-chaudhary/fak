package model

import (
	"sync/atomic"
	"unsafe"
)

// v41Leaf indexes a per-layer V4.1 tensor name the decode step resolves every token.
type v41Leaf uint8

const (
	v41LeafAttnNorm v41Leaf = iota
	v41LeafFFNNorm
	v41LeafAttnSink
	v41LeafWQA
	v41LeafWQB
	v41LeafWQANorm
	v41LeafKVNorm
	v41LeafWKV
	v41LeafWOA
	v41LeafWOB
	v41LeafGate
	v41LeafGateBias
	v41LeafSharedW1
	v41LeafSharedW3
	v41LeafSharedW2
	v41LeafMHCMix
	v41LeafMHCBase
	v41LeafMHCScale
	v41LeafMHCFFNMix
	v41LeafMHCFFNBase
	v41LeafMHCFFNScale
	v41LeafCompressorNorm
	v41LeafCount
)

var v41LeafSuffix = [v41LeafCount]string{
	v41LeafAttnNorm:       "attn_norm.weight",
	v41LeafFFNNorm:        "ffn_norm.weight",
	v41LeafAttnSink:       "attn.sink",
	v41LeafWQA:            "attn.wq_a.weight",
	v41LeafWQB:            "attn.wq_b.weight",
	v41LeafWQANorm:        "attn.wq_a_norm.weight",
	v41LeafKVNorm:         "attn.kv_norm.weight",
	v41LeafWKV:            "attn.wkv.weight",
	v41LeafWOA:            "attn.wo_a.weight",
	v41LeafWOB:            "attn.wo_b.weight",
	v41LeafGate:           "ffn.gate.weight",
	v41LeafGateBias:       "ffn.gate.e_score_correction_bias",
	v41LeafSharedW1:       "ffn.shared_experts.w1.weight",
	v41LeafSharedW3:       "ffn.shared_experts.w3.weight",
	v41LeafSharedW2:       "ffn.shared_experts.w2.weight",
	v41LeafMHCMix:         "mhc.mixes.weight",
	v41LeafMHCBase:        "mhc.base",
	v41LeafMHCScale:       "mhc.scale",
	v41LeafMHCFFNMix:      "mhc.ffn_mixes.weight",
	v41LeafMHCFFNBase:     "mhc.ffn_base",
	v41LeafMHCFFNScale:    "mhc.ffn_scale",
	v41LeafCompressorNorm: "attn.compressor.norm.weight",
}

// v41MHCLeafOf maps an mHC trio suffix to its leaf.
func v41MHCLeafOf(suffix string) (v41Leaf, bool) {
	switch suffix {
	case "mhc.mixes.weight":
		return v41LeafMHCMix, true
	case "mhc.base":
		return v41LeafMHCBase, true
	case "mhc.scale":
		return v41LeafMHCScale, true
	case "mhc.ffn_mixes.weight":
		return v41LeafMHCFFNMix, true
	case "mhc.ffn_base":
		return v41LeafMHCFFNBase, true
	case "mhc.ffn_scale":
		return v41LeafMHCFFNScale, true
	}
	return 0, false
}

// v41ExpertNames are one routed expert's projection weight and bias names.
type v41ExpertNames struct {
	weights, biases [3]string // w1, w3, w2
}

const v41MaxNamedExperts = 1 << 12

// v41NameTable holds a model's per-layer V4.1 tensor names. Names are pure
// functions of layer and leaf, so the table never goes stale; every store
// lookup still runs at its site and sees in-place store changes.
type v41NameTable struct {
	layers, experts int
	leaves          [][v41LeafCount]string
	mhcParams       [][2]string
	routed          [][]unsafe.Pointer // *v41ExpertNames, filled on first route
}

// v41NameTableBuilds counts name-table builds; a warm step must not add one.
var v41NameTableBuilds atomic.Uint64

func (m *Model) v41Names() *v41NameTable {
	if m.v41NamesOff {
		return nil
	}
	layers, experts := m.Cfg.NumLayers, m.Cfg.NumExperts
	if t := (*v41NameTable)(atomic.LoadPointer(&m.v41NameTab)); t != nil && t.layers == layers && t.experts == experts {
		return t
	}
	if layers <= 0 || layers > layerNameCacheLayers || experts < 0 || experts > v41MaxNamedExperts {
		return nil
	}
	t := &v41NameTable{
		layers: layers, experts: experts,
		leaves: make([][v41LeafCount]string, layers), mhcParams: make([][2]string, layers), routed: make([][]unsafe.Pointer, layers),
	}
	for l := range layers {
		for leaf, suffix := range v41LeafSuffix {
			t.leaves[l][leaf] = layerName(l, suffix)
		}
		t.mhcParams[l] = [2]string{
			v41MHCParamsKeyNamed(t.leaves[l][v41LeafMHCScale], t.leaves[l][v41LeafMHCBase]),
			v41MHCParamsKeyNamed(t.leaves[l][v41LeafMHCFFNScale], t.leaves[l][v41LeafMHCFFNBase]),
		}
		t.routed[l] = make([]unsafe.Pointer, experts)
	}
	v41NameTableBuilds.Add(1)
	atomic.StorePointer(&m.v41NameTab, unsafe.Pointer(t))
	return t
}

// v41LeafName is layerName(l, v41LeafSuffix[leaf]).
func (m *Model) v41LeafName(l int, leaf v41Leaf) string {
	if t := m.v41Names(); t != nil && l >= 0 && l < t.layers {
		return t.leaves[l][leaf]
	}
	return layerName(l, v41LeafSuffix[leaf])
}

// v41LeafTensor is m.tensor(layerName(l, v41LeafSuffix[leaf])).
func (m *Model) v41LeafTensor(l int, leaf v41Leaf) []float32 {
	return m.tensor(m.v41LeafName(l, leaf))
}

// v41MHCName is layerName(l, suffix) for an mHC trio suffix chosen by the caller.
func (m *Model) v41MHCName(l int, suffix string) string {
	if leaf, ok := v41MHCLeafOf(suffix); ok {
		return m.v41LeafName(l, leaf)
	}
	return layerName(l, suffix)
}

func v41MHCParamsKeyNamed(scale, base string) string {
	return "v41-mhc-split-params:" + scale + ":" + base
}

// v41MHCParamsKey is the device cache key of one layer's mHC split parameters.
func (m *Model) v41MHCParamsKey(l int, scaleLeaf, baseLeaf string) string {
	if t := m.v41Names(); t != nil && l >= 0 && l < t.layers {
		switch {
		case scaleLeaf == v41LeafSuffix[v41LeafMHCScale] && baseLeaf == v41LeafSuffix[v41LeafMHCBase]:
			return t.mhcParams[l][0]
		case scaleLeaf == v41LeafSuffix[v41LeafMHCFFNScale] && baseLeaf == v41LeafSuffix[v41LeafMHCFFNBase]:
			return t.mhcParams[l][1]
		}
	}
	return v41MHCParamsKeyNamed(layerName(l, scaleLeaf), layerName(l, baseLeaf))
}

// v41RoutedNames returns routed expert e's names on layer l, or nil when the
// table does not cover them and the caller keeps its own name build.
func (m *Model) v41RoutedNames(l, e int) *v41ExpertNames {
	t := m.v41Names()
	if t == nil || l < 0 || l >= t.layers || e < 0 || e >= t.experts {
		return nil
	}
	slot := &t.routed[l][e]
	if p := atomic.LoadPointer(slot); p != nil {
		return (*v41ExpertNames)(p)
	}
	stem := "ffn.experts." + itoa(e)
	x := &v41ExpertNames{}
	for i, proj := range [3]string{".w1.weight", ".w3.weight", ".w2.weight"} {
		x.weights[i] = layerName(l, stem+proj)
		x.biases[i] = v41BiasName(x.weights[i])
	}
	atomic.StorePointer(slot, unsafe.Pointer(x))
	return x
}

// v41BiasName is the bias tensor beside a ".weight" tensor.
func v41BiasName(weight string) string {
	return weight[:len(weight)-len("weight")] + "bias"
}

// v41ExpertStemNames is the three projection names under a routed stem
// "ffn.experts.<e>".
func (m *Model) v41ExpertStemNames(l int, stem string) [3]string {
	if e, ok := v41ExpertOfStem(stem); ok {
		if x := m.v41RoutedNames(l, e); x != nil {
			return x.weights
		}
	}
	return [3]string{layerName(l, stem+".w1.weight"), layerName(l, stem+".w3.weight"), layerName(l, stem+".w2.weight")}
}

// v41ExpertOfStem inverts "ffn.experts." + itoa(e) for e >= 0.
func v41ExpertOfStem(stem string) (int, bool) {
	const prefix = "ffn.experts."
	if len(stem) <= len(prefix) || stem[:len(prefix)] != prefix {
		return 0, false
	}
	digits := stem[len(prefix):]
	if len(digits) > 9 || (len(digits) > 1 && digits[0] == '0') {
		return 0, false
	}
	e := 0
	for i := 0; i < len(digits); i++ {
		c := digits[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		e = e*10 + int(c-'0')
	}
	return e, true
}

// v41OrBiasName is bias, or the bias name derived from weight when bias is empty.
func v41OrBiasName(bias, weight string) string {
	if bias != "" {
		return bias
	}
	return v41BiasName(weight)
}
