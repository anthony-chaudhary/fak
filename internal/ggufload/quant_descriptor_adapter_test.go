package ggufload

// quant_descriptor_adapter_test.go is the witness suite for the W4AFP8
// foreign-descriptor adapter (ported from vllm-project/vllm,
// vllm/model_executor/layers/quantization/w4afp8.py@
// load-time translation of a fixed foreign descriptor onto the canonical W4A8
// FP8 MoE scheme, the routed-expert tensor-name remap, the int4->uint4b8 byte
// repack, and the fail-closed refusal of unknown spellings. No model weights
// are read or written.

import (
	"bytes"
	"errors"
	"testing"
)

// knownForeignW4AFP8 is the fixed foreign descriptor the adapter translates.
// It is the exact SGLang-style W4AFP8 spelling: INT4 group-128 symmetric
// weights, FP8 per-token dynamic activations.
func knownForeignW4AFP8() ForeignQuantDescriptor {
	return ForeignQuantDescriptor{
		Scheme:      "w4afp8",
		WeightBits:  4,
		WeightType:  "int",
		Symmetric:   true,
		Strategy:    "group",
		GroupSize:   128,
		ActBits:     8,
		ActType:     "float",
		ActStrategy: "token",
		ActDynamic:  true,
	}
}

// TestQuantDescriptorAdapter pins the descriptor-adapter contract: a fixed
// foreign W4AFP8 descriptor translates onto the canonical W4A8 FP8 MoE scheme
// with the numeric precision preserved, and every unknown spelling is refused
// with the ErrUnknownQuantSpelling sentinel.
func TestQuantDescriptorAdapter(t *testing.T) {
	t.Run("translate fixed foreign descriptor to canonical", func(t *testing.T) {
		got, err := TranslateQuantDescriptor(knownForeignW4AFP8())
		if err != nil {
			t.Fatalf("TranslateQuantDescriptor(known) = %v, want nil", err)
		}
		want := CanonicalQuantDescriptor{
			Scheme:      "w4a8-fp8-moe",
			WeightBits:  4,
			WeightType:  "int",
			Symmetric:   true,
			Strategy:    "group",
			GroupSize:   128,
			ActBits:     8,
			ActType:     "float",
			ActStrategy: "token",
			ActDynamic:  true,
		}
		if got != want {
			t.Fatalf("TranslateQuantDescriptor = %+v, want %+v", got, want)
		}
		// The scheme identity must actually change; the precision must not.
		if got.Scheme == knownForeignW4AFP8().Scheme {
			t.Fatalf("canonical scheme %q still equals the foreign scheme", got.Scheme)
		}
		if got.WeightBits != 4 || got.ActBits != 8 || got.GroupSize != 128 {
			t.Fatalf("canonical precision drifted: weight=%d act=%d group=%d",
				got.WeightBits, got.ActBits, got.GroupSize)
		}
	})

	t.Run("refuse unknown scheme identity", func(t *testing.T) {
		d := knownForeignW4AFP8()
		d.Scheme = "w4afp8-legacy"
		if _, err := TranslateQuantDescriptor(d); !errors.Is(err, ErrUnknownQuantSpelling) {
			t.Fatalf("TranslateQuantDescriptor(drifted scheme) = %v, want ErrUnknownQuantSpelling", err)
		}
	})

	t.Run("refuse wrong group size", func(t *testing.T) {
		d := knownForeignW4AFP8()
		d.GroupSize = 64
		if _, err := TranslateQuantDescriptor(d); !errors.Is(err, ErrUnknownQuantSpelling) {
			t.Fatalf("TranslateQuantDescriptor(group=64) = %v, want ErrUnknownQuantSpelling", err)
		}
	})

	t.Run("refuse asymmetric weight", func(t *testing.T) {
		d := knownForeignW4AFP8()
		d.Symmetric = false
		if _, err := TranslateQuantDescriptor(d); !errors.Is(err, ErrUnknownQuantSpelling) {
			t.Fatalf("TranslateQuantDescriptor(asymmetric) = %v, want ErrUnknownQuantSpelling", err)
		}
	})

	t.Run("refuse static activation", func(t *testing.T) {
		d := knownForeignW4AFP8()
		d.ActDynamic = false
		if _, err := TranslateQuantDescriptor(d); !errors.Is(err, ErrUnknownQuantSpelling) {
			t.Fatalf("TranslateQuantDescriptor(static act) = %v, want ErrUnknownQuantSpelling", err)
		}
	})

	t.Run("refuse any single drifted field", func(t *testing.T) {
		drifts := map[string]func(*ForeignQuantDescriptor){
			"weight type":  func(d *ForeignQuantDescriptor) { d.WeightType = "float" },
			"strategy":     func(d *ForeignQuantDescriptor) { d.Strategy = "channel" },
			"act bits":     func(d *ForeignQuantDescriptor) { d.ActBits = 16 },
			"act type":     func(d *ForeignQuantDescriptor) { d.ActType = "int" },
			"act strategy": func(d *ForeignQuantDescriptor) { d.ActStrategy = "tensor" },
		}
		for name, drift := range drifts {
			d := knownForeignW4AFP8()
			drift(&d)
			got, err := TranslateQuantDescriptor(d)
			if !errors.Is(err, ErrUnknownQuantSpelling) {
				t.Fatalf("TranslateQuantDescriptor(drifted %s) = %v, want ErrUnknownQuantSpelling", name, err)
			}
			if got != (CanonicalQuantDescriptor{}) {
				t.Fatalf("TranslateQuantDescriptor(drifted %s) returned %+v on refusal, want zero", name, got)
			}
		}
	})

	t.Run("refuse wrong weight bits", func(t *testing.T) {
		d := knownForeignW4AFP8()
		d.WeightBits = 8
		if _, err := TranslateQuantDescriptor(d); !errors.Is(err, ErrUnknownQuantSpelling) {
			t.Fatalf("TranslateQuantDescriptor(8-bit weight) = %v, want ErrUnknownQuantSpelling", err)
		}
	})
}

// TestQuantDescriptorAdapterTensorNameRemap pins the routed-expert tensor-name
// half of the adapter: weight -> weight_packed, weight_scale_inv ->
// weight_scale, input_scale dropped, and any other name refused.
func TestQuantDescriptorAdapterTensorNameRemap(t *testing.T) {
	cases := []struct {
		name    string
		want    string
		wantOK  bool
		wantErr bool
	}{
		{name: "blk.0.experts.3.w1.weight", want: "blk.0.experts.3.w1.weight_packed", wantOK: true},
		{name: "blk.0.experts.3.w1.weight_scale_inv", want: "blk.0.experts.3.w1.weight_scale", wantOK: true},
		{name: "blk.0.experts.3.w1.input_scale", want: "", wantOK: false},
		{name: "model.layers.2.mlp.experts.17.down_proj.weight", want: "model.layers.2.mlp.experts.17.down_proj.weight_packed", wantOK: true},
		{name: "model.layers.2.mlp.experts.17.gate_proj.input_scale", want: "", wantOK: false},
		// Dense tensors keep the native FP8 serialization: never renamed, and
		// their input scale is never dropped.
		{name: "model.layers.0.self_attn.q_proj.weight", wantErr: true},
		{name: "model.layers.0.self_attn.q_proj.weight_scale_inv", wantErr: true},
		{name: "model.layers.0.self_attn.q_proj.input_scale", wantErr: true},
		// The expert projection list is closed.
		{name: "blk.0.experts.3.gate.weight", wantErr: true},
		{name: "blk.0.sharedexperts.3.w1.weight", wantErr: true},
		{name: "blk.0.experts.x.w1.weight", wantErr: true},
		{name: "blk.0.experts.3.w1.weight.extra", wantErr: true},
		{name: "blk.0.experts.3.w1.bias", wantErr: true},
		{name: "blk.0.experts.3.w1", wantErr: true},
		{name: "", wantErr: true},
	}
	for _, tc := range cases {
		got, ok, err := MapQuantTensorName(tc.name)
		if tc.wantErr {
			if !errors.Is(err, ErrUnknownQuantSpelling) {
				t.Fatalf("MapQuantTensorName(%q) err = %v, want ErrUnknownQuantSpelling", tc.name, err)
			}
			if got != "" || ok {
				t.Fatalf("MapQuantTensorName(%q) = (%q,%v), want empty/false on refusal", tc.name, got, ok)
			}
			continue
		}
		if err != nil {
			t.Fatalf("MapQuantTensorName(%q) = %v, want nil", tc.name, err)
		}
		if got != tc.want || ok != tc.wantOK {
			t.Fatalf("MapQuantTensorName(%q) = (%q,%v), want (%q,%v)", tc.name, got, ok, tc.want, tc.wantOK)
		}
	}
	for _, tc := range cases {
		if got := IsRoutedExpertQuantTensor(tc.name); got == tc.wantErr {
			t.Fatalf("IsRoutedExpertQuantTensor(%q) = %v, want %v", tc.name, got, !tc.wantErr)
		}
	}
}

// TestQuantDescriptorAdapterRepack pins the int4->uint4b8 byte repack: each
// byte is XORed with 0x88, the payload length is preserved, the input is not
// mutated, and a mis-sized payload is refused.
func TestQuantDescriptorAdapterRepack(t *testing.T) {
	// A deterministic 8-byte payload (two int32 words) of signed int4 bytes.
	in := []byte{0x00, 0x01, 0x7f, 0x80, 0xff, 0x10, 0x20, 0x30}
	want := []byte{0x88, 0x89, 0xf7, 0x08, 0x77, 0x98, 0xa8, 0xb8}

	got, err := RepackSignedInt4ToUint4B8(in)
	if err != nil {
		t.Fatalf("RepackSignedInt4ToUint4B8 = %v, want nil", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("RepackSignedInt4ToUint4B8 = % x, want % x", got, want)
	}
	// The input must be untouched.
	if !bytes.Equal(in, []byte{0x00, 0x01, 0x7f, 0x80, 0xff, 0x10, 0x20, 0x30}) {
		t.Fatalf("RepackSignedInt4ToUint4B8 mutated its input: % x", in)
	}

	// A payload that is not a multiple of 4 is refused.
	if _, err := RepackSignedInt4ToUint4B8([]byte{0x00, 0x01, 0x02}); err == nil {
		t.Fatal("RepackSignedInt4ToUint4B8(mis-sized) = nil, want error")
	}
}
