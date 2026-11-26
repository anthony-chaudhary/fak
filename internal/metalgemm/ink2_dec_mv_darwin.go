//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
)

//go:embed ink2_dec_mv.metal
var ink2DecMVText string

var ink2DecMVSource = func() string { return ink2DecSource + ink2DecMVText }()

// ink2DecMVLegacy is FAK_INK2_LEGACY's "mvlat": the mv_ kernels for every matvec.
var ink2DecMVLegacy = Ink2Legacy("mvlat")

// ink2MVChoice is an ink2_dec_mv_ kernel over threadgroups of 1<<rowBits rows.
type ink2MVChoice struct {
	kernel  string
	rowBits int
}

// The scalar weight types. The block types stay on mv_: at the decode shapes
// they already stream near the memory bandwidth, and naming their loads at
// compile time timed no faster.
var ink2MVTypeName = map[uint32]string{0: "f32", 1: "f16", 30: "bf16"}

// ink2MVPick picks the decode matvec kernel of a weight, or none for mv_.
func ink2MVPick(typ uint32, cols int) ink2MVChoice {
	name, ok := ink2MVTypeName[typ]
	if ink2DecMVLegacy || !ok || (cols != 2304 && cols != 3072) {
		return ink2MVChoice{}
	}
	return ink2MVChoice{fmt.Sprintf("ink2_dec_mv_%s_%d", name, cols), 1}
}

type ink2MVShape struct{ rows, byteOffset, rowBits uint32 }

// matVecDec encodes y = W x with the kernel of pl.
func (d *Ink2Device) matVecDec(pl ink2MVChoice, w Ink2Weight, x, y Ink2Ref) error {
	bytes, err := d.packedShape(w, x, 0)
	if err != nil {
		return err
	}
	if !x.Valid() || !y.Valid() || pl.rowBits < 0 || pl.rowBits > 5 {
		return fmt.Errorf("%w: matvec plan %+v", ErrInk2Shape, pl)
	}
	p, err := d.Pipeline(ink2DecMVSource, pl.kernel)
	if err != nil {
		return err
	}
	group := 32 << pl.rowBits
	if p.width != 32 || int(p.maxThreads) < group {
		return fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", pl.kernel, group)
	}
	// A Metal binding offset is 4-byte aligned; the kernel adds the remainder.
	return d.Dispatch(p, [3]int{(w.Rows + 1<<pl.rowBits - 1) >> pl.rowBits, 1, 1}, [3]int{group, 1, 1},
		Ink2BufferAt(w.Buf, w.ByteOff&^3, bytes+w.ByteOff&3), x.Span(w.Cols), y.Span(w.Rows),
		Ink2Const(ink2MVShape{uint32(w.Rows), uint32(w.ByteOff & 3), uint32(pl.rowBits)}))
}
