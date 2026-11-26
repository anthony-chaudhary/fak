//go:build darwin && arm64 && cgo

package metalgemm

// ink2TinyFuseLegacy ("tinyfuse" in FAK_INK2_LEGACY) keeps the general loops
// of the decode attend and qkv post instead of their fixed-step builds below.
var ink2TinyFuseLegacy = Ink2Legacy("tinyfuse")

// The attend source with its score and value loops taking several keys per
// step; FAK_INK2_LEGACY=tinyfuse keeps the one-key loops.
var ink2AttendNamedSource = "#define INK2_ATT_NAMED 1\n" + ink2AttendSource

// The conv source with ink2_qkv_post's loops at fixed trip counts, for the
// shapes it names; FAK_INK2_LEGACY=tinyfuse keeps the general loops.
var ink2QKVNamedSource = "#define INK2_QKV_NAMED 1\n" + ink2ConvSource

const (
	ink2QKVNamedHeadDim = 128
	ink2QKVNamedRelLen  = 16
	ink2QKVNamedLanes   = 32
	ink2QKVNamedTaps    = 4
)
