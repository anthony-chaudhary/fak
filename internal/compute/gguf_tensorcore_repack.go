// offline tensor-core fragment weight+scale repack with idempotent repack guard, lossless unpack, and GGUF preservation.
package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

const (
	// TensorCoreFragTileOut matches MMA 16xN tile rows along output channels.
	TensorCoreFragTileOut = 16
	// TensorCoreWordNibbles packs 8 4-bit weights per uint32 MMA operand register.
	TensorCoreWordNibbles = 8

	repackMagicHeader = uint32(0x464B5443) // "FKTC"
)

// marlinFragPerm defines the offline nibble interleave order matching tensor-core
// uint32 lane unpacking: [0, 2, 4, 6, 1, 3, 5, 7].
var marlinFragPerm = [TensorCoreWordNibbles]int{0, 2, 4, 6, 1, 3, 5, 7}
var marlinFragInvPerm = [TensorCoreWordNibbles]int{0, 4, 1, 5, 2, 6, 3, 7}

// RepackedQ4KTensor holds an offline tensor-core-repacked copy of GGUF Q4_K weights.
// The original GGUF slice passed to RepackQ4KOnce is never modified.
type RepackedQ4KTensor struct {
	Magic        uint32
	OutDim       int
	InDim        int
	TileOut      int
	BlocksPerRow int
	NumRowTiles  int
	Repacked     bool
	RepackCount  int
	// Tiled block storage: [tileO, blkIdx, rowInTile, blockBytes(144)] with
	// each 128-byte nibble region permuted into tensor-core word-interleaved order
	// and d/dmin packed into interleaved tile scale tables alongside block headers.
	TiledBlocks []byte
	TileScales  []float32 // Pre-decoded per-tile [tileO, blkIdx, rowInTile, 2] (d, dmin) for coalesced fragment loads.
}

// RepackedAWQTensor holds an offline tensor-core-repacked copy of AWQ uint4 weights + scales.
type RepackedAWQTensor struct {
	Magic       uint32
	OutDim      int
	InDim       int
	TileOut     int
	Repacked    bool
	RepackCount int
	// Packed u32 words in [tileO, wordIdx, rowInTile] with Marlin fragment nibble interleave.
	TiledWords  []uint32
	TiledScales []float32
}

func interleave4BitBlockBytes(dst, src []byte) {
	// Permute each 4-byte (8-nibble) group using marlinFragPerm [0,2,4,6,1,3,5,7].
	for base := 0; base < len(src); base += 4 {
		var nibs [TensorCoreWordNibbles]byte
		for b := 0; b < 4; b++ {
			v := src[base+b]
			nibs[2*b] = v & 0x0f
			nibs[2*b+1] = (v >> 4) & 0x0f
		}
		var outWord uint32
		for dstSlot, srcIdx := range marlinFragPerm {
			outWord |= uint32(nibs[srcIdx]&0x0f) << (dstSlot * 4)
		}
		binary.LittleEndian.PutUint32(dst[base:base+4], outWord)
	}
}

func unInterleave4BitBlockBytes(dst, src []byte) {
	for base := 0; base < len(src); base += 4 {
		w := binary.LittleEndian.Uint32(src[base : base+4])
		var orig [TensorCoreWordNibbles]byte
		for origIdx, srcSlot := range marlinFragInvPerm {
			orig[origIdx] = byte((w >> (srcSlot * 4)) & 0x0f)
		}
		for b := 0; b < 4; b++ {
			dst[base+b] = (orig[2*b] & 0x0f) | ((orig[2*b+1] & 0x0f) << 4)
		}
	}
}

// RepackQ4KTensorOffline repacks verbatim GGUF Q4_K bytes into tiled tensor-core fragment
// layout. The caller's raw slice is read-only and never altered.
func RepackQ4KTensorOffline(raw []byte, outDim, inDim int) (*RepackedQ4KTensor, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("tensorcore repack Q4_K: invalid shape [%d, %d]", outDim, inDim)
	}
	if inDim%q4kSuper != 0 {
		return nil, fmt.Errorf("tensorcore repack Q4_K: inDim=%d must be multiple of %d", inDim, q4kSuper)
	}
	blocksPerRow := inDim / q4kSuper
	wantLen := outDim * blocksPerRow * q4kSuperBlock
	if len(raw) != wantLen {
		return nil, fmt.Errorf("tensorcore repack Q4_K: raw len %d != %d", len(raw), wantLen)
	}

	tileOut := TensorCoreFragTileOut
	numTiles := (outDim + tileOut - 1) / tileOut
	tiledBlocks := make([]byte, numTiles*blocksPerRow*tileOut*q4kSuperBlock)
	tileScales := make([]float32, numTiles*blocksPerRow*tileOut*2)

	for r := 0; r < outDim; r++ {
		tO := r / tileOut
		inT := r % tileOut
		for b := 0; b < blocksPerRow; b++ {
			srcIdx := (r*blocksPerRow + b) * q4kSuperBlock
			srcBlk := raw[srcIdx : srcIdx+q4kSuperBlock]

			dstSlot := ((tO*blocksPerRow)+b)*tileOut + inT
			dstOff := dstSlot * q4kSuperBlock
			// Preserve header bytes (d, dmin, 12-byte scales) verbatim; repack 128-byte quant payload.
			copy(tiledBlocks[dstOff:dstOff+16], srcBlk[:16])
			interleave4BitBlockBytes(tiledBlocks[dstOff+16:dstOff+q4kSuperBlock], srcBlk[16:q4kSuperBlock])

			d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(srcBlk[0:2])))
			dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(srcBlk[2:4])))
			tileScales[dstSlot*2] = d
			tileScales[dstSlot*2+1] = dmin
		}
	}

	return &RepackedQ4KTensor{
		Magic:        repackMagicHeader,
		OutDim:       outDim,
		InDim:        inDim,
		TileOut:      tileOut,
		BlocksPerRow: blocksPerRow,
		NumRowTiles:  numTiles,
		Repacked:     true,
		RepackCount:  1,
		TiledBlocks:  tiledBlocks,
		TileScales:   tileScales,
	}, nil
}

// RepackOnce applies an idempotent guard to an existing RepackedQ4KTensor: calling it
// repeatedly never mutates or double-permutes the repacked tensor-core buffers.
func (rt *RepackedQ4KTensor) RepackOnce() error {
	if rt == nil {
		return errors.New("tensorcore repack Q4_K: nil tensor")
	}
	if rt.Magic != repackMagicHeader {
		return errors.New("tensorcore repack Q4_K: invalid magic header")
	}
	if rt.Repacked {
		// Idempotent no-op guard: already in tensor-core fragment layout.
		return nil
	}
	return errors.New("tensorcore repack Q4_K: inconsistent un-repacked state")
}

// UnpackRaw recovers exact byte-for-byte GGUF Q4_K super-block bytes from the repacked tensor.
func (rt *RepackedQ4KTensor) UnpackRaw() ([]byte, error) {
	if rt == nil || rt.Magic != repackMagicHeader || !rt.Repacked {
		return nil, errors.New("tensorcore unpack Q4_K: tensor is not validly repacked")
	}
	raw := make([]byte, rt.OutDim*rt.BlocksPerRow*q4kSuperBlock)
	for r := 0; r < rt.OutDim; r++ {
		tO := r / rt.TileOut
		inT := r % rt.TileOut
		for b := 0; b < rt.BlocksPerRow; b++ {
			srcSlot := ((tO*rt.BlocksPerRow)+b)*rt.TileOut + inT
			srcOff := srcSlot * q4kSuperBlock
			dstOff := (r*rt.BlocksPerRow + b) * q4kSuperBlock
			copy(raw[dstOff:dstOff+16], rt.TiledBlocks[srcOff:srcOff+16])
			unInterleave4BitBlockBytes(raw[dstOff+16:dstOff+q4kSuperBlock], rt.TiledBlocks[srcOff+16:srcOff+q4kSuperBlock])
		}
	}
	return raw, nil
}

// RepackedQ4KMatMul executes matmul directly against the repacked tensor-core layout
// without unpacking back to GGUF order.
func (rt *RepackedQ4KTensor) MatMul(x []float32, rows int) ([]float32, error) {
	if rt == nil || rt.Magic != repackMagicHeader || !rt.Repacked {
		return nil, errors.New("tensorcore matmul Q4_K: tensor not repacked")
	}
	if rows <= 0 || len(x) != rows*rt.InDim {
		return nil, fmt.Errorf("tensorcore matmul Q4_K: invalid x len %d for rows=%d in=%d", len(x), rows, rt.InDim)
	}

	out := make([]float32, rows*rt.OutDim)
	buf := make([]float32, q4kSuper)
	var unblk [128]byte

	for r := 0; r < rt.OutDim; r++ {
		tO := r / rt.TileOut
		inT := r % rt.TileOut
		for row := 0; row < rows; row++ {
			xsRow := x[row*rt.InDim : (row+1)*rt.InDim]
			var acc float32
			for b := 0; b < rt.BlocksPerRow; b++ {
				slot := ((tO*rt.BlocksPerRow)+b)*rt.TileOut + inT
				blk := rt.TiledBlocks[slot*q4kSuperBlock : (slot+1)*q4kSuperBlock]
				d := rt.TileScales[slot*2]
				dmin := rt.TileScales[slot*2+1]
				scales := blk[4:16]
				// Read interleaved nibble words directly into super-block weights in fragment order.
				qPacked := blk[16:q4kSuperBlock]
				for base := 0; base < 128; base += 4 {
					w := binary.LittleEndian.Uint32(qPacked[base : base+4])
					for p := 0; p < 4; p++ {
						lo := byte((w >> (marlinFragInvPerm[2*p] * 4)) & 0x0f)
						hi := byte((w >> (marlinFragInvPerm[2*p+1] * 4)) & 0x0f)
						unblk[base+p] = lo | (hi << 4)
					}
				}
				qi, is := 0, 0
				for j := 0; j < q4kSuper; j += 64 {
					sc, m := kquantbits.ScaleMinK4(is, scales)
					d1, m1 := d*float32(sc), dmin*float32(m)
					sc, m = kquantbits.ScaleMinK4(is+1, scales)
					d2, m2 := d*float32(sc), dmin*float32(m)
					for l := 0; l < 32; l++ {
						buf[j+l] = d1*float32(unblk[qi+l]&0x0f) - m1
					}
					for l := 0; l < 32; l++ {
						buf[j+32+l] = d2*float32(unblk[qi+l]>>4) - m2
					}
					qi += 32
					is += 2
				}
				xs := xsRow[b*q4kSuper:]
				var s0, s1, s2, s3 float32
				for i := 0; i < q4kSuper; i += 4 {
					s0 += buf[i] * xs[i]
					s1 += buf[i+1] * xs[i+1]
					s2 += buf[i+2] * xs[i+2]
					s3 += buf[i+3] * xs[i+3]
				}
				acc += (s0 + s1) + (s2 + s3)
			}
			out[row*rt.OutDim+r] = acc
		}
	}
	return out, nil
}

// RepackAWQTensorOffline repacks packed AWQ [outDim, inDim/2] weights + [outDim] scales
// into tensor-core uint32 fragment words without mutating packed or scales.
func RepackAWQTensorOffline(packed []byte, scales []float32, outDim, inDim int) (*RepackedAWQTensor, error) {
	if outDim <= 0 || inDim <= 0 || inDim%TensorCoreWordNibbles != 0 {
		return nil, fmt.Errorf("tensorcore repack AWQ: outDim=%d inDim=%d (inDim must be multiple of %d)", outDim, inDim, TensorCoreWordNibbles)
	}
	if len(packed) != outDim*(inDim/2) {
		return nil, fmt.Errorf("tensorcore repack AWQ: packed len %d != %d", len(packed), outDim*(inDim/2))
	}
	if len(scales) != outDim {
		return nil, fmt.Errorf("tensorcore repack AWQ: scales len %d != %d", len(scales), outDim)
	}

	tileOut := TensorCoreFragTileOut
	numTiles := (outDim + tileOut - 1) / tileOut
	wordsPerRow := inDim / TensorCoreWordNibbles
	tiledWords := make([]uint32, numTiles*wordsPerRow*tileOut)
	tiledScales := make([]float32, numTiles*tileOut)

	var scratch [4]byte
	for r := 0; r < outDim; r++ {
		tO := r / tileOut
		inT := r % tileOut
		tiledScales[tO*tileOut+inT] = scales[r]
		rowBytes := packed[r*(inDim/2) : (r+1)*(inDim/2)]
		for w := 0; w < wordsPerRow; w++ {
			interleave4BitBlockBytes(scratch[:], rowBytes[w*4:(w+1)*4])
			tiledWords[((tO*wordsPerRow)+w)*tileOut+inT] = binary.LittleEndian.Uint32(scratch[:])
		}
	}

	return &RepackedAWQTensor{
		Magic:       repackMagicHeader,
		OutDim:      outDim,
		InDim:       inDim,
		TileOut:     tileOut,
		Repacked:    true,
		RepackCount: 1,
		TiledWords:  tiledWords,
		TiledScales: tiledScales,
	}, nil
}

// RepackOnce enforces idempotent repacking on RepackedAWQTensor.
func (ra *RepackedAWQTensor) RepackOnce() error {
	if ra == nil || ra.Magic != repackMagicHeader {
		return errors.New("tensorcore repack AWQ: invalid or nil repacked tensor")
	}
	if ra.Repacked {
		return nil
	}
	return errors.New("tensorcore repack AWQ: inconsistent un-repacked state")
}

// UnpackRaw recovers exact original AWQ [outDim, inDim/2] packed bytes and [outDim] scales.
func (ra *RepackedAWQTensor) UnpackRaw() ([]byte, []float32, error) {
	if ra == nil || ra.Magic != repackMagicHeader || !ra.Repacked {
		return nil, nil, errors.New("tensorcore unpack AWQ: tensor is not repacked")
	}
	wordsPerRow := ra.InDim / TensorCoreWordNibbles
	packed := make([]byte, ra.OutDim*(ra.InDim/2))
	scales := make([]float32, ra.OutDim)
	var src4, dst4 [4]byte

	for r := 0; r < ra.OutDim; r++ {
		tO := r / ra.TileOut
		inT := r % ra.TileOut
		scales[r] = ra.TiledScales[tO*ra.TileOut+inT]
		rowBytes := packed[r*(ra.InDim/2) : (r+1)*(ra.InDim/2)]
		for w := 0; w < wordsPerRow; w++ {
			word := ra.TiledWords[((tO*wordsPerRow)+w)*ra.TileOut+inT]
			binary.LittleEndian.PutUint32(src4[:], word)
			unInterleave4BitBlockBytes(dst4[:], src4[:])
			copy(rowBytes[w*4:(w+1)*4], dst4[:])
		}
	}
	return packed, scales, nil
}

// MatMul computes GEMV/GEMM directly from the interleaved tensor-core word layout.
func (ra *RepackedAWQTensor) MatMul(x []float32, rows int) ([]float32, error) {
	if ra == nil || ra.Magic != repackMagicHeader || !ra.Repacked {
		return nil, errors.New("tensorcore matmul AWQ: tensor not repacked")
	}
	if rows <= 0 || len(x) != rows*ra.InDim {
		return nil, fmt.Errorf("tensorcore matmul AWQ: invalid x len %d for rows=%d in=%d", len(x), rows, ra.InDim)
	}
	wordsPerRow := ra.InDim / TensorCoreWordNibbles
	out := make([]float32, rows*ra.OutDim)
	for r := 0; r < ra.OutDim; r++ {
		tO := r / ra.TileOut
		inT := r % ra.TileOut
		sc := ra.TiledScales[tO*ra.TileOut+inT]
		for row := 0; row < rows; row++ {
			xRow := x[row*ra.InDim : (row+1)*ra.InDim]
			var acc float32
			for w := 0; w < wordsPerRow; w++ {
				word := ra.TiledWords[((tO*wordsPerRow)+w)*ra.TileOut+inT]
				colBase := w * TensorCoreWordNibbles
				for k, slot := range marlinFragInvPerm {
					q := int((word>>(slot*4))&0x0f) - 8
					acc += (float32(q) * sc) * xRow[colBase+k]
				}
			}
			out[row*ra.OutDim+r] = acc
		}
	}
	return out, nil
}
