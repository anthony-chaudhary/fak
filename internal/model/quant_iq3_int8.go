package model

// quant_iq3_int8.go is the portable default-off decode spine for issue #4628.
// The source GGUF must already contain IQ3_XXS blocks for the complete dense MLP
// gate/up/down band. There is intentionally no encoder and no Q4->IQ3 conversion.

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

var w3MLPProjections = [...]string{"gate", "up", "down"}

// W3MLPRequested reports the opt-in load selection. Only the documented truthy
// spellings enable it; every other value preserves the historical default.
func W3MLPRequested() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_W3_MLP"))) {
	case "1", "on", "true":
		return true
	default:
		return false
	}
}

// ResidentW3MLPEligible admits only exact dense Qwen3.5-family hybrid MLP
// projection names. It rejects attention, experts/shared experts/router, vision,
// MTP, biases, lookalikes, out-of-range layers, non-hybrid architectures, and MoE.
func ResidentW3MLPEligible(cfg Config, canon string) bool {
	if !cfg.IsQwen35Hybrid() || cfg.IsMoE() || cfg.NumLayers <= 0 {
		return false
	}
	const prefix = "model.layers."
	rest, ok := strings.CutPrefix(canon, prefix)
	if !ok {
		return false
	}
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return false
	}
	layerText := rest[:dot]
	layer, err := strconv.Atoi(layerText)
	if err != nil || layer < 0 || layer >= cfg.NumLayers || strconv.Itoa(layer) != layerText {
		return false
	}
	suffix := rest[dot+1:]
	for _, projection := range w3MLPProjections {
		if suffix == "mlp."+projection+"_proj.weight" {
			return true
		}
	}
	return false
}

func residentW3MLPName(layer int, projection string) string {
	return fmt.Sprintf("model.layers.%d.mlp.%s_proj.weight", layer, projection)
}

// ValidateResidentW3MLP fails closed unless every layer has exactly one tagged
// IQ3_XXS gate/up/down tensor and no tensor outside that band carries the tag.
func (m *Model) ValidateResidentW3MLP() error {
	expected := 3 * m.Cfg.NumLayers
	observed := m.ResidentW3MLPCount()
	if !m.Cfg.IsQwen35Hybrid() || m.Cfg.IsMoE() || m.Cfg.NumLayers <= 0 {
		return fmt.Errorf("model: FAK_W3_MLP requires a dense Qwen3.5-family hybrid model (observed %d/%d tagged tensors)", observed, expected)
	}

	expectedNames := make(map[string]struct{}, expected)
	for layer := 0; layer < m.Cfg.NumLayers; layer++ {
		for _, projection := range w3MLPProjections {
			name := residentW3MLPName(layer, projection)
			expectedNames[name] = struct{}{}
			qt := m.kqw[name]
			switch {
			case qt == nil:
				return fmt.Errorf("model: FAK_W3_MLP missing %s (observed %d/%d tagged tensors)", name, observed, expected)
			case !qt.w3MLP:
				return fmt.Errorf("model: FAK_W3_MLP tensor %s is not tagged for W3 decode (observed %d/%d tagged tensors)", name, observed, expected)
			case qt.kind != kindIQ3XXS:
				return fmt.Errorf("model: FAK_W3_MLP tensor %s has kind %s, want IQ3_XXS (observed %d/%d tagged tensors)", name, qt.kind, observed, expected)
			}
		}
	}

	var unexpected []string
	for name, qt := range m.kqw {
		if qt == nil || !qt.w3MLP {
			continue
		}
		if _, ok := expectedNames[name]; !ok {
			unexpected = append(unexpected, name)
		}
	}
	if len(unexpected) > 0 {
		sort.Strings(unexpected)
		return fmt.Errorf("model: FAK_W3_MLP unexpected tagged tensor %s (observed %d/%d tagged tensors)", unexpected[0], observed, expected)
	}
	if observed != expected {
		return fmt.Errorf("model: FAK_W3_MLP incomplete tagged band (observed %d/%d tagged tensors)", observed, expected)
	}
	return nil
}

// ResidentW3MLPCount is a load/read-back diagnostic; W3 tensors are a tagged
// subset of the existing resident k-quant store, not a second weight copy.
func (m *Model) ResidentW3MLPCount() int {
	count := 0
	for _, qt := range m.kqw {
		if qt != nil && qt.w3MLP {
			count++
		}
	}
	return count
}

// HasResidentW3MLP reports whether name is a tagged IQ3_XXS dense-MLP tensor.
func (m *Model) HasResidentW3MLP(name string) bool {
	qt := m.kqw[name]
	return qt != nil && qt.w3MLP && qt.kind == kindIQ3XXS
}

// iq3xxsReduceRowScalar writes one exact integer dot per 32-wide IQ3_XXS
// sub-block. IQ3_XXS is symmetric, so unlike Q4_K it needs no activation-sum
// correction. Integer bounds fit comfortably in int32.
func iq3xxsReduceRowScalar(row []byte, nblk int, qx []int8, IS []int32) {
	for b := 0; b < nblk; b++ {
		blk := row[b*iq3xxsBlockBytes : (b+1)*iq3xxsBlockBytes]
		qs := blk[2 : 2+qkK/4]
		sas := blk[2+qkK/4:]
		for sub := 0; sub < 8; sub++ {
			aux := binary.LittleEndian.Uint32(sas[4*sub:])
			gridBase := sub * 8
			activationBase := b*qkK + sub*32
			var dot int32
			for group := 0; group < 4; group++ {
				signs := ksignsIQ2XS[(aux>>(7*uint(group)))&127]
				g1 := iq3xxsGrid[qs[gridBase+2*group]]
				g2 := iq3xxsGrid[qs[gridBase+2*group+1]]
				for j := 0; j < 4; j++ {
					w1 := int32(byte(g1 >> (8 * uint(j))))
					if signs&(1<<uint(j)) != 0 {
						w1 = -w1
					}
					w2 := int32(byte(g2 >> (8 * uint(j))))
					if signs&(1<<uint(j+4)) != 0 {
						w2 = -w2
					}
					dot += w1 * int32(qx[activationBase+group*8+j])
					dot += w2 * int32(qx[activationBase+group*8+j+4])
				}
			}
			IS[b*8+sub] = dot
		}
	}
}

// iq3xxsReduceRow is the portable spine's dispatch point. Architecture-specific
// acceleration can replace this body only with an exact reducer witness.
func iq3xxsReduceRow(row []byte, nblk int, qx []int8, IS []int32) {
	iq3xxsReduceRowScalar(row, nblk, qx, IS)
}

// iq3xxsCombineRow applies the IQ3 per-sub-block scales to the exact integer
// reductions in a fixed block/sub-block order shared by every future arch tier.
func iq3xxsCombineRow(row []byte, nblk int, dx []float32, IS []int32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := row[b*iq3xxsBlockBytes : (b+1)*iq3xxsBlockBytes]
		d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk)))
		sas := blk[2+qkK/4:]
		base := b * 8
		for sub := 0; sub < 8; sub++ {
			aux := binary.LittleEndian.Uint32(sas[4*sub:])
			db := d * (0.5 + float32(aux>>28)) * 0.5
			acc += (float32(IS[base+sub]) * db) * dx[base+sub]
		}
	}
	return acc
}

// iq3xxsMatRowsRangeInt8 is the row-range decode GEMV body. Scratch is
// allocated once per worker range and reused for every row.
func iq3xxsMatRowsRangeInt8(qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	iq3xxsMatRowsRangeInt8Raw(qt.raw, qt, qv, y, lo, hi)
}

func iq3xxsMatRowsRangeInt8Raw(raw []byte, qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	if len(raw) == 0 {
		raw = qt.raw
	}
	nblk := qt.nblk
	IS := make([]int32, nblk*8)
	rowBytes := qt.rowBytes()
	for rowIndex := lo; rowIndex < hi; rowIndex++ {
		row := raw[rowIndex*rowBytes : (rowIndex+1)*rowBytes]
		iq3xxsReduceRow(row, nblk, qv.q, IS)
		y[rowIndex] = iq3xxsCombineRow(row, nblk, qv.d, IS)
	}
}
