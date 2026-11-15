package kvbudget

// This file adds a pure, GPU-free FORWARD-ADMISSION gate for the KV-cache
// layout axes a Hopper (sm_90) MLA serve must pin before it can be admitted
// (MBL-021: every declared config axis needs a forward-admission gate). Instead
// of a serve failing at RUNTIME on an unsupported KV dtype / block-size /
// window / beam / compress combination, the gate rejects the combination at
// PLAN time from config headers only, and — for a supported combination —
// recomputes the per-block byte footprint from the declared layout so the fit
// math (DeriveMaxBlocks) is charged the right block size.
//
// The supported set mirrors the TensorRT-LLM Hopper KV requirements
// (examples/deepseek_v4/README.md, Apache-2.0): KV dtype fp8_ds_mla,
// tokens_per_block 128 or 256, window_size 128, max_beam_width 1, and compress
// ratios 1/4/128. The Dynamo H200 recipe pins the same kv_cache_dtype.
//
// Global KV is layout-dependent: the DSV41 recipe's ~890 B/token figure holds
// only for its own layout, so the per-block bytes are RECOMPUTED from the
// declared layout axes here — never a hardcoded constant.
//
// Everything is a deterministic check over config values: no hardware, no wall
// clock, no network.

// KVLayoutConfig is the declared KV-cache layout: the config axes a Hopper MLA
// serve must pin. The gate validates each axis against the supported set and,
// for a supported combination, recomputes the per-block byte footprint.
type KVLayoutConfig struct {
	// DType is the KV element precision (e.g. "fp8_ds_mla").
	DType string
	// TokensPerBlock is the number of tokens one KV block holds.
	TokensPerBlock int
	// WindowSize is the sliding-window cap in tokens.
	WindowSize int
	// MaxBeamWidth is the maximum beam width the layout supports.
	MaxBeamWidth int
	// CompressRatio is the KV compression ratio.
	CompressRatio int
	// Layout is the KV layout identifier: the per-token element geometry the
	// per-block bytes derive from.
	Layout string
}

// KVLayoutRefusal is a closed, typed refusal vocabulary for a KV-layout
// forward-admission verdict. The empty value means admitted; a non-empty value
// means the combination was refused at plan time and no bytes were recomputed.
type KVLayoutRefusal string

const (
	// KVLayoutAdmitted is the empty refusal carried by an admitted verdict.
	KVLayoutAdmitted KVLayoutRefusal = ""
	// KVLayoutUnsupportedDType refuses an unsupported KV element precision.
	KVLayoutUnsupportedDType KVLayoutRefusal = "unsupported_kv_dtype"
	// KVLayoutUnsupportedBlockSize refuses an unsupported tokens_per_block.
	KVLayoutUnsupportedBlockSize KVLayoutRefusal = "unsupported_tokens_per_block"
	// KVLayoutUnsupportedWindow refuses an unsupported window_size.
	KVLayoutUnsupportedWindow KVLayoutRefusal = "unsupported_window_size"
	// KVLayoutUnsupportedBeam refuses an unsupported max_beam_width.
	KVLayoutUnsupportedBeam KVLayoutRefusal = "unsupported_max_beam_width"
	// KVLayoutUnsupportedCompress refuses an unsupported compress ratio.
	KVLayoutUnsupportedCompress KVLayoutRefusal = "unsupported_compress_ratio"
	// KVLayoutUnsupportedLayout refuses an unknown KV layout.
	KVLayoutUnsupportedLayout KVLayoutRefusal = "unsupported_kv_layout"
)

// KVLayoutAdmission is the outcome of a forward-admission check. On success
// Admitted is true and PerBlockBytes carries the recomputed per-block byte
// footprint — feed it to DeriveMaxBlocks as bytesPerBlock so the fit math is
// charged the layout's true block size. On refusal Admitted is false,
// PerBlockBytes is zero, and Reason carries the typed cause.
type KVLayoutAdmission struct {
	Admitted      bool
	Reason        KVLayoutRefusal
	PerBlockBytes uint64
}

// The Hopper MLA KV layout axes the gate admits, mirroring the TensorRT-LLM
// Hopper KV requirements (examples/deepseek_v4/README.md, Apache-2.0).
const (
	// hopperKVDType is the KV element precision the Hopper MLA path supports.
	hopperKVDType = "fp8_ds_mla"
	// hopperWindowSize is the sliding-window cap the Hopper MLA path supports.
	hopperWindowSize = 128
	// hopperMaxBeamWidth is the maximum beam width the Hopper MLA path supports.
	hopperMaxBeamWidth = 1
)

// hopperTokensPerBlock are the KV block sizes (tokens per block) the Hopper
// MLA path supports.
var hopperTokensPerBlock = map[int]bool{128: true, 256: true}

// hopperCompressRatios are the KV compression ratios the Hopper MLA path
// supports.
var hopperCompressRatios = map[int]bool{1: true, 4: true, 128: true}

// hopperDTypeBytesPerElem is the effective bytes-per-element of each supported
// Hopper MLA KV dtype. fp8_ds_mla is an 8-bit (1 B/elem) DeepSeek-MLA
// precision.
var hopperDTypeBytesPerElem = map[string]int{
	"fp8_ds_mla": 1,
}

// kvLayoutGeometry is the per-token element geometry of a supported Hopper MLA
// KV layout: the MLA latent + decoupled rope key (+ DSA indexer key) one token
// caches, summed over layers. The per-block byte footprint is recomputed from
// this geometry and the declared dtype — the DSV41 recipe's ~890 B/token figure
// holds only for its own layout, so it is never a hardcoded constant.
type kvLayoutGeometry struct {
	layers        int
	kvLoraRank    int
	qkRopeHeadDim int
	indexLayers   int
	indexHeadDim  int
}

// elemsPerToken is the KV elements one token caches at this layout:
// layers × (kvLoraRank + qkRopeHeadDim) + indexLayers × indexHeadDim — the same
// MLA + DSA formula Shape.KVElemsPerToken uses.
func (g kvLayoutGeometry) elemsPerToken() int {
	return g.layers*(g.kvLoraRank+g.qkRopeHeadDim) + g.indexLayers*g.indexHeadDim
}

// hopperKVLayouts is the per-token element geometry of each supported Hopper
// MLA KV layout. The dsv41_mla entry is the DeepSeek-V4.1 MLA + DSA geometry
// the Hopper recipe serves.
var hopperKVLayouts = map[string]kvLayoutGeometry{
	"dsv41_mla": {
		layers:        61,
		kvLoraRank:    512,
		qkRopeHeadDim: 64,
		indexLayers:   61,
		indexHeadDim:  128,
	},
}

// AdmitKVLayout is the forward-admission gate for a declared Hopper MLA KV
// layout. It validates each config axis against the supported set and, for a
// supported combination, recomputes the per-block byte footprint from the
// declared layout and dtype. It rejects an unsupported combination at plan time
// with a typed refusal, instead of letting a serve fail at runtime. Deterministic;
// no hardware, no clock.
func AdmitKVLayout(cfg KVLayoutConfig) KVLayoutAdmission {
	if cfg.DType != hopperKVDType {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedDType}
	}
	if !hopperTokensPerBlock[cfg.TokensPerBlock] {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedBlockSize}
	}
	if cfg.WindowSize != hopperWindowSize {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedWindow}
	}
	if cfg.MaxBeamWidth != hopperMaxBeamWidth {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedBeam}
	}
	if !hopperCompressRatios[cfg.CompressRatio] {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedCompress}
	}
	geo, ok := hopperKVLayouts[cfg.Layout]
	if !ok {
		return KVLayoutAdmission{Reason: KVLayoutUnsupportedLayout}
	}
	bytesPerElem := hopperDTypeBytesPerElem[cfg.DType]
	perTokenBytes := int64(geo.elemsPerToken()) * int64(bytesPerElem)
	perBlockBytes := perTokenBytes * int64(cfg.TokensPerBlock)
	return KVLayoutAdmission{Admitted: true, PerBlockBytes: uint64(perBlockBytes)}
}
