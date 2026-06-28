package compute

// tpu_arch.go — the always-compiled, hardware-independent half of the TPU / Neural-Engine
// backend (issue #261 / C-004). It is the device taxonomy a compiler build needs BEFORE any
// kernel runs: which neural accelerators fak targets, which COMPILER LANE each one reaches the
// HAL through, the lane's native compute dtype, and the canonical compile-target token. None of
// this needs a TPU or a Neural Engine to be correct, so it is pure Go with no build tag and is
// unit-witnessed on any host — the same split ROCM-C002-NOTES.md uses (ship the exact,
// host-tractable part; defer the device run). The cgo backends themselves (the `//go:build xla`
// TPU twin that records the op-list and lowers it through XLA/StableHLO, and the `//go:build
// coreml` Neural-Engine twin, each registering an Approx backend) land on a node with the silicon
// where they can actually compile and be witnessed; see TPU-C004-NOTES.md for that hand-off.
//
// Why a taxonomy and not a free-form string: the issue title lumps two very different
// accelerators ("TPU/Neural Engine") under one backend, but they do not share a code path. A
// Google TPU is a whole-graph, ahead-of-time-compiled DATAFLOW part reached through XLA/PJRT
// (Caps.GraphCompile); Apple's Neural Engine is a fixed-vendor-op-menu EDGE NPU reached through
// CoreML (Caps.FusedFFN), and is distinct from the Apple Metal GPU backend already shipped
// (metal.go). The XLA-vs-CoreML lane is the load-bearing split — it picks the whole backend
// strategy and the native numeric tier (a TPU MXU computes in bf16; the ANE computes in fp16),
// the same way the CDNA/RDNA split is load-bearing for ROCm. This table is the single place that
// mapping lives, so an unrecognized accelerator fails closed instead of silently lowering through
// the wrong compiler.

// AccelLane is the compiler/lowering lane a neural accelerator reaches the HAL through. It is the
// load-bearing classification: it selects the backend strategy (whole-graph compile vs fixed op
// menu) and therefore the primary optional capability (Caps) the backend advertises.
type AccelLane uint8

const (
	// LaneUnknown is the zero value: an accelerator fak has no lowering lane for.
	LaneUnknown AccelLane = iota
	// LaneXLA is the whole-graph, ahead-of-time path: the backend runs the Backend methods in
	// record-only mode to capture the in-process op-list, lowers it to StableHLO, and hands it to
	// XLA (through PJRT) which compiles and places it on the device. This is the Google-TPU lane;
	// the corresponding capability is Caps.GraphCompile. (See the Dataflow lens in
	// docs/explainers/hardware-portability.md — the same recorded-op-list mechanism.)
	LaneXLA
	// LaneCoreML is the fixed-vendor-op-menu path: the backend maps coarse blocks (a whole MLP) to
	// CoreML MLProgram ops and lets CoreML place them on the Apple Neural Engine, staging weights
	// in a device-native packed layout. This is the Apple-Neural-Engine lane; the corresponding
	// capabilities are Caps.FusedFFN + WeightSource (the Edge-NPU lens in the portability doc).
	LaneCoreML
)

// String returns the short lane label.
func (l AccelLane) String() string {
	switch l {
	case LaneXLA:
		return "xla"
	case LaneCoreML:
		return "coreml"
	default:
		return "unknown"
	}
}

// GraphCompiled reports whether the lane lowers a whole recorded op-list ahead of time (the XLA
// dataflow path) rather than dispatching a fixed menu of vendor ops per block (CoreML). This is
// the predicate a build/runtime path keys on to decide which compiler to invoke.
func (l AccelLane) GraphCompiled() bool { return l == LaneXLA }

// PrimaryCap names the single optional Backend capability (a field of Caps) a backend in this
// lane keys on, so the taxonomy and the HAL contract stay in sync: XLA -> "GraphCompile",
// CoreML -> "FusedFFN". It is the documented bridge from this host-tractable table to the
// device backend the next agent writes; it returns "" for an unknown lane.
func (l AccelLane) PrimaryCap() string {
	switch l {
	case LaneXLA:
		return "GraphCompile"
	case LaneCoreML:
		return "FusedFFN"
	default:
		return ""
	}
}

// AccelFamily is a neural-accelerator generation fak can target.
type AccelFamily uint8

const (
	// AccelUnknown is the zero value: a generation fak has no target for.
	AccelUnknown AccelFamily = iota
	// Google Cloud TPU generations (XLA lane, bf16-native MXU).
	TPUv2  // Cloud TPU v2
	TPUv3  // Cloud TPU v3
	TPUv4  // Cloud TPU v4 (Pufferfish)
	TPUv5e // Cloud TPU v5e (efficiency/Lite)
	TPUv5p // Cloud TPU v5p (performance/Pod)
	TPUv6e // Cloud TPU v6e (Trillium)
	// Apple Neural Engine generations (CoreML lane, fp16-native).
	ANEA17 // A17 Pro / M3-family 16-core ANE
	ANEM4  // M4 / A18-family ANE
)

// String returns the short generation label.
func (f AccelFamily) String() string {
	switch f {
	case TPUv2:
		return "TPUv2"
	case TPUv3:
		return "TPUv3"
	case TPUv4:
		return "TPUv4"
	case TPUv5e:
		return "TPUv5e"
	case TPUv5p:
		return "TPUv5p"
	case TPUv6e:
		return "TPUv6e"
	case ANEA17:
		return "ANEA17"
	case ANEM4:
		return "ANEM4"
	default:
		return "unknown"
	}
}

// IsTPU reports whether the family is a Google Cloud TPU (the XLA lane). ANE families are not.
func (f AccelFamily) IsTPU() bool {
	return f == TPUv2 || f == TPUv3 || f == TPUv4 || f == TPUv5e || f == TPUv5p || f == TPUv6e
}

// IsNeuralEngine reports whether the family is an Apple Neural Engine (the CoreML lane).
func (f AccelFamily) IsNeuralEngine() bool { return f == ANEA17 || f == ANEM4 }

// AccelArch is one supported accelerator target: its canonical fak target token, its family, the
// compiler lane it lowers through, the lane's native compute dtype, and a representative product.
type AccelArch struct {
	Target      string      // canonical target token, e.g. "tpu-v5e" (selects this accelerator)
	Family      AccelFamily // generation
	Lane        AccelLane   // XLA (TPU/GraphCompile) or CoreML (Neural Engine/FusedFFN)
	NativeDtype Dtype       // the lane's native compute tier: BF16 on a TPU MXU, F16 on the ANE
	Examples    string      // representative product(s)
	aliases     []string    // device-reported spellings (already normalized) that resolve here
}

// accelArches is the supported-target table, declared once: TPU generations (XLA, bf16) first,
// then Apple Neural Engine generations (CoreML, fp16). The aliases are the noisy spellings a
// runtime probe actually reports (XLA device-kind strings like "TPU v5 lite"; Apple chip names).
var accelArches = []AccelArch{
	{Target: "tpu-v2", Family: TPUv2, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v2", aliases: []string{"tpuv2", "v2"}},
	{Target: "tpu-v3", Family: TPUv3, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v3", aliases: []string{"tpuv3", "v3"}},
	{Target: "tpu-v4", Family: TPUv4, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v4", aliases: []string{"tpuv4", "v4"}},
	{Target: "tpu-v5e", Family: TPUv5e, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v5e (Lite)", aliases: []string{"tpuv5e", "tpuv5lite", "v5e", "v5litepod"}},
	{Target: "tpu-v5p", Family: TPUv5p, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v5p (Pod)", aliases: []string{"tpuv5p", "tpuv5", "v5p", "v5pod"}},
	{Target: "tpu-v6e", Family: TPUv6e, Lane: LaneXLA, NativeDtype: BF16, Examples: "Cloud TPU v6e (Trillium)", aliases: []string{"tpuv6e", "tpuv6lite", "trillium", "v6e"}},
	{Target: "ane-a17", Family: ANEA17, Lane: LaneCoreML, NativeDtype: F16, Examples: "A17 Pro / M3 16-core Neural Engine", aliases: []string{"anea17", "a17", "a17pro", "m3", "m3pro", "m3max"}},
	{Target: "ane-m4", Family: ANEM4, Lane: LaneCoreML, NativeDtype: F16, Examples: "M4 / A18 Neural Engine", aliases: []string{"anem4", "m4", "m4pro", "m4max", "a18", "a18pro"}},
}

// accelByKey indexes the table by the canonical target token AND every alias (all normalized) for
// O(1) lookup of a noisy device-reported string.
var accelByKey = func() map[string]AccelArch {
	m := make(map[string]AccelArch, len(accelArches)*3)
	for _, a := range accelArches {
		m[normalizeAccel(a.Target)] = a
		for _, al := range a.aliases {
			m[normalizeAccel(al)] = a
		}
	}
	return m
}()

// normalizeAccel canonicalizes a device-reported accelerator string to a compact key. Runtimes
// report these targets with case noise, separators, vendor prefixes, and parenthetical detail —
// e.g. "TPU v5e", "tpu-v5e", "TPUv4", "Apple M3 Pro (ANE)" — that all denote one compile target.
// It lowercases, drops a parenthetical suffix, strips spaces / hyphens / underscores / dots, and
// drops a leading "apple " or trailing " neural engine" descriptor. It does not invent a target:
// an unrecognized key is returned for Lookup to reject (fail closed).
func normalizeAccel(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '(' { // drop a parenthetical descriptor like "(ANE)"
			break
		}
		switch c {
		case ' ', '\t', '\n', '\r', '-', '_', '.':
			continue // strip separators and whitespace
		}
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out = append(out, c)
	}
	k := string(out)
	// Drop common vendor/role descriptors that do not change the target.
	for _, pre := range []string{"apple", "google", "cloud"} {
		if len(k) > len(pre) && k[:len(pre)] == pre {
			k = k[len(pre):]
		}
	}
	for _, suf := range []string{"neuralengine", "ane"} {
		if len(k) > len(suf) && k[len(k)-len(suf):] == suf {
			k = k[:len(k)-len(suf)]
		}
	}
	return k
}

// LookupAccelArch resolves a device-reported accelerator string (any case, separators, vendor
// prefix, or parenthetical) to its supported target, or (zero, false) if fak has no lane for it.
// This is the fail-closed admission a build/runtime path uses so an unknown accelerator is never
// silently lowered through the wrong compiler.
func LookupAccelArch(name string) (AccelArch, bool) {
	a, ok := accelByKey[normalizeAccel(name)]
	return a, ok
}

// AccelTarget returns the canonical target token for a device-reported accelerator string, or
// ("", false) if unsupported. The deferred build path (the XLA / CoreML twins) selects its
// lowering lane from the resolved AccelArch; this is the token a `--backend` selector accepts.
func AccelTarget(name string) (string, bool) {
	a, ok := LookupAccelArch(name)
	if !ok {
		return "", false
	}
	return a.Target, true
}

// KnownAccelArches returns the supported-target table in declared order. A `fak` diagnostic or a
// compiler build script enumerates it to print exactly which neural accelerators this build of fak
// targets — the honest answer to "does fak run on my TPU / Neural Engine?".
func KnownAccelArches() []AccelArch {
	out := make([]AccelArch, len(accelArches))
	copy(out, accelArches)
	return out
}
