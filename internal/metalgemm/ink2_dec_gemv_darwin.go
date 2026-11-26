//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"strings"
)

//go:embed ink2_dec_gemv.metal
var ink2DecGEMV string

// The packed decoders plus the scale-folding routed kernels, one string so the
// pipeline cache compiles it once.
var ink2DecSource = func() string { return packedCodebooks + packedDequant + packedGEMV + ink2DecGEMV }()

// The routed-product names in FAK_INK2_LEGACY: "routed" (the routed_ kernels and
// the separate scale pass), "scale" (only the separate scale pass), "nvfp4w"
// (NVFP4 weights decoded from the table) and "routed2" (runtime column counts
// instead of the compile-time-shaped kernels). Names of other families are
// ignored here.
type ink2DecLegacySet struct{ routed, scale, nvfp4w, routed2 bool }

var ink2DecLegacy = parseInk2DecLegacy(os.Getenv(Ink2LegacyEnv))

func parseInk2DecLegacy(s string) (set ink2DecLegacySet) {
	for _, name := range strings.Split(s, ",") {
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "routed":
			set.routed = true
		case "scale":
			set.scale = true
		case "nvfp4w":
			set.nvfp4w = true
		case "routed2":
			set.routed2 = true
		}
	}
	return set
}

// ink2DecPlan is a routed kernel, its threadgroup of 1<<rowBits SIMD groups
// and the rows each SIMD group computes (0 means 1); a compile-time-shaped
// kernel also names the only column count it computes.
type ink2DecPlan struct {
	kernel  string
	rowBits int
	rows    int
	cols    int
}

func (pl ink2DecPlan) rowsPerSIMD() int { return max(pl.rows, 1) }

// ink2DecRoutedNVFP4 is the NVFP4 routed kernel that the microbenchmark timed
// fastest at the resident forward's gate_up and down shapes.
var ink2DecRoutedNVFP4 = ink2DecPlan{kernel: "ink2_dec_routed_nvfp4w", rowBits: 1}

type ink2DecShapeKey struct {
	typ  uint32
	cols int
}

// ink2DecShaped holds the compile-time-shaped kernels (ink2_dec_routed_<type>c<cols>r<rows>)
// the microbenchmark timed fastest at the resident forward's routed shapes.
var ink2DecShaped = map[ink2DecShapeKey]ink2DecPlan{
	{17, 2304}: {"ink2_dec_routed_iq2_xsc2304r2", 1, 2, 2304},
	{18, 2304}: {"ink2_dec_routed_iq3_xxsc2304r2", 2, 2, 2304},
	{21, 2304}: {"ink2_dec_routed_iq3_sc2304r2", 0, 2, 2304},
	{20, 1152}: {"ink2_dec_routed_iq4_nlc1152r2", 1, 2, 1152},
}

// ink2DecRoutedPick picks the routed kernel of a weight type, or none for the
// routed_ kernel and its separate scale pass.
func ink2DecRoutedPick(typ uint32, cols int) ink2DecPlan {
	legacy := ink2DecLegacy
	if legacy.routed || int(typ) >= len(ink2RoutedKernel) || ink2RoutedKernel[typ] == "" {
		return ink2DecPlan{}
	}
	if pl, ok := ink2DecShaped[ink2DecShapeKey{typ, cols}]; ok && !legacy.routed2 {
		return pl
	}
	if typ == 40 && !legacy.nvfp4w {
		return ink2DecRoutedNVFP4
	}
	return ink2DecPlan{kernel: "ink2_dec_routed_" + ink2RoutedKernel[typ][len("routed_"):], rowBits: 2}
}

type ink2DecShape struct{ rows, cols, byteOffset, xStride, scaled, rowBits uint32 }

// ink2DecXMax is the activation length ink2_dec_routed_nvfp4t stages.
const ink2DecXMax = 4096

// routedDec encodes y[t] = W[ids[t]] x[t] for k slots with the kernel of pl,
// times scale[ids[t]] when scale is present.
func (d *Ink2Device) routedDec(pl ink2DecPlan, w Ink2Weight, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref) error {
	bytes, err := d.packedShape(w, x, xStride)
	if err != nil {
		return err
	}
	if !ids.Valid() || !x.Valid() || !y.Valid() || k < 1 || xStride < 0 || xStride > math.MaxUint32 ||
		pl.rowBits < 0 || pl.rowBits > 5 || pl.rows < 0 || (pl.cols != 0 && pl.cols != w.Cols) || (pl.kernel == "ink2_dec_routed_nvfp4t" && w.Cols > ink2DecXMax) {
		return fmt.Errorf("%w: routed matvec of %d slots, input stride %d, plan %+v", ErrInk2Shape, k, xStride, pl)
	}
	p, err := d.Pipeline(ink2DecSource, pl.kernel)
	if err != nil {
		return err
	}
	group := 32 << pl.rowBits
	if p.width != 32 || int(p.maxThreads) < group {
		return fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", pl.kernel, group)
	}
	shape := ink2DecShape{rows: uint32(w.Rows), cols: uint32(w.Cols), byteOffset: uint32(w.ByteOff & 3),
		xStride: uint32(xStride), rowBits: uint32(pl.rowBits)}
	scaleArg := Ink2Ref{}.Span(0)
	if scale.Valid() {
		// The scale table's length is the expert count, which only the ids know.
		shape.scaled, scaleArg = 1, scale.Span(0)
	}
	// A Metal binding offset is 4-byte aligned; the kernel adds the remainder.
	perGroup := pl.rowsPerSIMD() << pl.rowBits
	return d.Dispatch(p, [3]int{(w.Rows + perGroup - 1) / perGroup, k, 1}, [3]int{group, 1, 1},
		Ink2BufferAt(w.Buf, w.ByteOff&^3, w.Experts*bytes+w.ByteOff&3),
		x.Span((k-1)*xStride+w.Cols), y.Span(k*w.Rows), ids.Span(k), scaleArg, Ink2Const(shape))
}
