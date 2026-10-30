package model

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=50ms lane=default
func TestExpertCheckpointIQuantBoundedFault(t *testing.T) {
	for _, tc := range []struct {
		quant      ExpertCheckpointQuant
		name       string
		kind       kQuantKind
		dtype      compute.Dtype
		blockBytes int
	}{
		// Append-only wire values keep the parent revision compilable, so
		// its refusal is an admission failure rather than a missing-symbol error.
		{ExpertCheckpointQ3K + 1, "IQ2_XXS", kindIQ2XXS, compute.IQ2_XXS, 66},
		{ExpertCheckpointQ3K + 2, "IQ2_XS", kindIQ2XS, compute.IQ2_XS, 74},
		{ExpertCheckpointQ3K + 3, "IQ1_M", kindIQ1M, compute.IQ1_M, 56},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const rows, cols, experts = 2, 256, 3
			stride := rows * tc.blockBytes
			raw := make([]byte, experts*stride)
			for i := range raw {
				raw[i] = byte(i*31 + i/stride)
			}
			for off := 0; off < len(raw); off += tc.blockBytes {
				if tc.kind == kindIQ1M {
					// IQ1_M distributes the F16 global scale across four nibbles.
					raw[off+49] &= 15
					raw[off+51] &= 15
					raw[off+53] = raw[off+53]&15 | 0xc0
					raw[off+55] = raw[off+55]&15 | 0x30
				} else {
					binary.LittleEndian.PutUint16(raw[off:], 0x3c00)
				}
			}
			source := &iquantsReadProbe{Reader: bytes.NewReader(raw)}
			tier := NewExpertCheckpointTier(0)
			if err := tier.AddShard(source, int64(len(raw)), []FusedExpertTensor{{Name: "blk.0.ffn_gate_exps.weight", Arch: "deepseek41", Layer: 0, Proj: "gate_proj", Quant: tc.quant, Experts: experts, Rows: rows, Cols: cols}}); err != nil {
				t.Fatal(err)
			}
			if tc.quant.String() != tc.name {
				t.Fatal("packed checkpoint quant wire identity changed")
			}
			name := "model.layers.0.ffn.experts.1.w1.weight"
			stage, staged := tier.staging(name)
			for _, probe := range []struct {
				backend compute.Backend
				want    bool
			}{
				{iquantsMarkerProbe{compute.Default()}, false},
				{iquantsDtypeProbe{Backend: compute.Default(), dtype: compute.F32}, false},
				{iquantsDtypeProbe{Backend: compute.Default(), dtype: tc.dtype}, true},
			} {
				got, admitted := tier.stagingForBackend(name, probe.backend)
				if admitted != probe.want || (got != nil) != probe.want {
					t.Fatal("new IQ staging ignored the backend's explicit dtype promise")
				}
			}
			if source.bytes != 0 {
				t.Fatal("descriptor admission read expert payload")
			}
			if !staged || stage.dt != tc.dtype || stage.bytes != int64(stride) {
				t.Fatal("i-quant staged with the wrong encoding or byte budget")
			}
			tensor := stage.mk()
			host, ok := tensor.Buf().(compute.HostBuffer)
			if !ok || tensor.Dtype != tc.dtype || len(host.I8()) != stride {
				t.Fatal("i-quant staging changed the packed representation")
			}
			for i, value := range host.I8() {
				if byte(value) != raw[stride+i] {
					t.Fatal("staged the wrong expert range")
				}
			}
			m := &Model{expertCheckpoint: tier}
			weight, err := m.v41CPUMissResolve(0, name)
			if err != nil {
				t.Fatal(err)
			}
			if weight.expert.kq == nil || weight.expert.kq.kind != tc.kind || !bytes.Equal(weight.expert.kq.raw, raw[stride:2*stride]) {
				t.Fatal("faulted expert is not its original packed host representation")
			}
			x := make([]float32, cols)
			for i := range x {
				x[i] = 1.0 / cols
			}
			got, ok := v41CPUMissGEMV(weight, x, rows, cols)
			if !ok {
				t.Fatal("packed host consumer declined its faulted expert")
			}
			want := make([]float32, rows)
			kQuantMatRowsInto(&kQuantTensor{out: rows, in: cols, nblk: 1, kind: tc.kind, raw: raw[stride : 2*stride]}, x, want)
			for i := range want {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatal("streamed packed CPU consumer changed the resident result")
				}
			}
			if tc.kind == kindIQ1M {
				// The default routed triple uses this selected-projection door;
				// it must work without enabling the optional CPU-miss controller.
				decoded, err := m.v41ExpertF32IntoNamed(0, name, nil)
				if err != nil || len(decoded) != rows*cols {
					t.Fatalf("default IQ1_M host materialization: len=%d err=%v", len(decoded), err)
				}
				reference := make([]float32, rows*cols)
				DequantIQ1M(reference, raw[stride:2*stride])
				for i := range reference {
					if math.Float32bits(decoded[i]) != math.Float32bits(reference[i]) {
						t.Fatal("default IQ1_M triple door changed its packed source values")
					}
				}
			}
			wantReads := 2
			if tc.kind == kindIQ1M {
				wantReads++ // The default selected-projection decode is a third stream-through fault.
			}
			if source.bytes != wantReads*stride || source.last != int64(stride) || tier.Stats().ResidentBytes != 0 {
				t.Fatal("one-expert fault escaped the stream-through range/budget")
			}
		})
	}
}

type iquantsReadProbe struct {
	*bytes.Reader
	bytes int
	last  int64
}

func (r *iquantsReadProbe) ReadAt(p []byte, off int64) (int, error) {
	r.bytes += len(p)
	r.last = off
	return r.Reader.ReadAt(p, off)
}

// These CPU-backed probes exercise only descriptor admission, never a GPU.
type iquantsMarkerProbe struct{ compute.Backend }

func (iquantsMarkerProbe) SupportsRoutedExpertKQuant() bool { return true }

type iquantsDtypeProbe struct {
	compute.Backend
	dtype compute.Dtype
}

func (b iquantsDtypeProbe) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == b.dtype
}
