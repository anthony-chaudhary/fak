package compute

import (
	"errors"
	"testing"
)

// Embedding Backend supplies irrelevant method signatures only. Any accidental
// allocation/upload/execute/name probe through it panics, so these witnesses
// remain metadata-only.
type packedPolicyMarker struct {
	Backend
	caps                     Caps
	decl                     Qwen35GDNPackedInputCapability
	decodePath, sequencePath string
	q4, q6                   bool
}

func (b *packedPolicyMarker) Caps() Caps { return b.caps }
func (b *packedPolicyMarker) Qwen35GDNPackedInputCapability() Qwen35GDNPackedInputCapability {
	return b.decl
}
func (b *packedPolicyMarker) Qwen35GDNPath() string             { return b.decodePath }
func (b *packedPolicyMarker) Qwen35SequencePrefillPath() string { return b.sequencePath }
func (b *packedPolicyMarker) SupportsDeviceWeightDtype(dt Dtype) bool {
	return (dt == Q4_K && b.q4) || (dt == Q6_K && b.q6)
}

type packedPolicyDecode struct{ *packedPolicyMarker }

func (*packedPolicyDecode) Qwen35GDNDecode(
	normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj, convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (Tensor, Tensor, Tensor, error) {
	panic("policy must not execute GDN")
}

type packedPolicyFull struct{ *packedPolicyDecode }

func (*packedPolicyFull) Qwen35SequencePrefill(Qwen35SequencePrefillRequest) (Qwen35SequencePrefillResult, error) {
	panic("policy must not execute sequence")
}

type packedPolicyQ6Decline struct{ *packedPolicyFull }

func (*packedPolicyQ6Decline) SupportsQ6KMatMul() bool { return false }

type packedPolicyWholeOnly struct {
	Backend
	qwen35GDNPackedImplementation
}
type packedPolicyNoDtype struct {
	Backend
	qwen35GDNPackedImplementation
	Qwen35GDNPackedInputBackend
}

func (*packedPolicyNoDtype) Caps() Caps { return Caps{DeviceMemory: true, UploadDtype: true} }

type packedPolicyNoCapability struct {
	Backend
}

func (*packedPolicyNoCapability) Caps() Caps                           { return Caps{DeviceMemory: true, UploadDtype: true} }
func (*packedPolicyNoCapability) SupportsDeviceWeightDtype(Dtype) bool { return true }

func packedPolicyFixture() *packedPolicyFull {
	return &packedPolicyFull{&packedPolicyDecode{&packedPolicyMarker{
		caps:       Caps{DeviceMemory: true, UploadDtype: true},
		decl:       Qwen35GDNPackedInputCapability{Contract: Qwen35GDNPackedInputContractV1, Q4K: true, Q6K: true},
		decodePath: Qwen35GDNCUDAPath, sequencePath: Qwen35SequencePrefillPath, q4: true, q6: true,
	}}}
}

// fak-test:runtime fast est=5ms lane=default
func TestQwen35GDNPackedPolicyRequiresWholeOperations(t *testing.T) {
	request := Qwen35GDNPackedInputRequest{Enable: true}
	var nilFull *packedPolicyFull
	good := packedPolicyFixture()
	for name, backend := range map[string]Backend{
		"nil": nil, "typed nil": nilFull, "generic matmul": &packedPolicyNoCapability{},
		"markers only": good.packedPolicyMarker, "decode without sequence": good.packedPolicyDecode,
		"whole operations without declaration": &packedPolicyWholeOnly{qwen35GDNPackedImplementation: good},
		"declaration without dtype probe":      &packedPolicyNoDtype{qwen35GDNPackedImplementation: good, Qwen35GDNPackedInputBackend: good},
	} {
		t.Run(name, func(t *testing.T) {
			p, err := ResolveQwen35GDNPackedInputPolicy(backend, request)
			if !errors.Is(err, ErrQwen35GDNPackedPolicy) || p.Enabled() {
				t.Fatalf("admitted incomplete backend: %+v %v", p, err)
			}
		})
	}
	for name, change := range map[string]func(*packedPolicyFull){
		"decode identity":   func(b *packedPolicyFull) { b.decodePath = "cuda" },
		"sequence identity": func(b *packedPolicyFull) { b.sequencePath = "unknown" },
		"contract version":  func(b *packedPolicyFull) { b.decl.Contract += "2" },
		"host memory":       func(b *packedPolicyFull) { b.caps.DeviceMemory = false },
		"no quant upload":   func(b *packedPolicyFull) { b.caps.UploadDtype = false },
		"no declared dtype": func(b *packedPolicyFull) { b.decl.Q4K = false; b.decl.Q6K = false },
		"no native dtype":   func(b *packedPolicyFull) { b.q4 = false; b.q6 = false },
	} {
		t.Run(name, func(t *testing.T) {
			b := packedPolicyFixture()
			change(b)
			p, err := ResolveQwen35GDNPackedInputPolicy(b, request)
			if !errors.Is(err, ErrQwen35GDNPackedPolicy) || p.Enabled() {
				t.Fatalf("admitted incompatible backend: %+v %v", p, err)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestQwen35GDNPackedPolicyControlsAndDtypes(t *testing.T) {
	for _, request := range []Qwen35GDNPackedInputRequest{
		{}, {Enable: true, NativeRowsOff: true}, {Enable: true, GDNRowsOff: true},
		{Enable: true, NativeRowsOff: true, GDNRowsOff: true},
	} {
		p, err := ResolveQwen35GDNPackedInputPolicy(nil, request)
		if err != nil || p.Enabled() || p.CaptureLegal() {
			t.Fatal("disable control lost precedence", p, err)
		}
	}
	for _, q4 := range []bool{false, true} {
		for _, q6 := range []bool{false, true} {
			b := packedPolicyFixture()
			b.decl.Q4K = q4
			b.decl.Q6K = q6
			p, err := ResolveQwen35GDNPackedInputPolicy(b, Qwen35GDNPackedInputRequest{Enable: true})
			if !q4 && !q6 {
				if !errors.Is(err, ErrQwen35GDNPackedPolicy) {
					t.Fatal("empty contract admitted")
				}
				continue
			}
			if err != nil || p.Allows(Q4_K) != q4 || p.Allows(Q6_K) != q6 || p.CaptureLegal() {
				t.Fatal("independent dtype gates", p, err)
			}
			for _, dt := range []Dtype{F32, Q8_0, Q2_K, Q3_K, Q5_K, IQ4_XS} {
				if p.Allows(dt) {
					t.Fatal("unrelated dtype leaked", dt)
				}
			}
			b.decl.Q4K = false
			b.decl.Q6K = false
			b.q4 = false
			b.q6 = false
			if p.Allows(Q4_K) != q4 || p.Allows(Q6_K) != q6 {
				t.Fatal("resolved decision is not immutable")
			}
		}
	}
	b := packedPolicyFixture()
	b.q4 = false
	p, err := ResolveQwen35GDNPackedInputPolicy(b, Qwen35GDNPackedInputRequest{Enable: true})
	if err != nil || p.Allows(Q4_K) || !p.Allows(Q6_K) {
		t.Fatal("actual dtype probe ignored", p, err)
	}
	p, err = ResolveQwen35GDNPackedInputPolicy(&packedPolicyQ6Decline{packedPolicyFixture()}, Qwen35GDNPackedInputRequest{Enable: true})
	if err != nil || !p.Allows(Q4_K) || p.Allows(Q6_K) {
		t.Fatal("Q6-specific decline ignored", p, err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestQwen35GDNPackedInputPlanBytesAndGeometry(t *testing.T) {
	p, err := ResolveQwen35GDNPackedInputPolicy(packedPolicyFixture(), Qwen35GDNPackedInputRequest{Enable: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                   string
		g                      Qwen35GDNPackedInputGeometry
		qkv, z                 int
		q4qkv, q6qkv, q4z, q6z int
	}{
		{"small", Qwen35GDNPackedInputGeometry{768, 2, 6, 128, 128}, 1280, 768, 552960, 806400, 331776, 483840},
		{"qwen38 metadata", Qwen35GDNPackedInputGeometry{5120, 16, 48, 128, 128}, 10240, 6144, 29491200, 43008000, 17694720, 25804800},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, v := range []struct {
				role        Qwen35GDNPackedInputRole
				dt          Dtype
				rows, bytes int
			}{
				{Qwen35GDNPackedQKV, Q4_K, tc.qkv, tc.q4qkv}, {Qwen35GDNPackedQKV, Q6_K, tc.qkv, tc.q6qkv},
				{Qwen35GDNPackedZ, Q4_K, tc.z, tc.q4z}, {Qwen35GDNPackedZ, Q6_K, tc.z, tc.q6z},
			} {
				shape := []int{v.rows, tc.g.Hidden}
				plan, err := p.Plan(v.role, v.dt, shape, tc.g)
				if err != nil || plan.Bytes != v.bytes || plan.Rows != v.rows || plan.Columns != tc.g.Hidden || plan.Block != 256 {
					t.Fatal("packed extent", plan, err)
				}
				shape[0]++
				if plan.Rows != v.rows {
					t.Fatal("plan aliases caller shape")
				}
				if err := plan.ValidatePayload(v.bytes); err != nil {
					t.Fatal(err)
				}
				for _, n := range []int{-1, 0, v.bytes - 1, v.bytes + 1} {
					if !errors.Is(plan.ValidatePayload(n), ErrQwen35GDNPackedPolicy) {
						t.Fatal("bad payload admitted", n)
					}
				}
			}
		})
	}
	g := Qwen35GDNPackedInputGeometry{768, 2, 6, 128, 128}
	for _, role := range []Qwen35GDNPackedInputRole{0, 3, 255} {
		if _, err := p.Plan(role, Q4_K, []int{1280, 768}, g); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
			t.Fatal("unknown role")
		}
	}
	for _, shape := range [][]int{nil, {1280}, {1280, 768, 1}, {1281, 768}, {1280, 512}, {-1, 768}} {
		if _, err := p.Plan(Qwen35GDNPackedQKV, Q4_K, shape, g); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
			t.Fatal("bad shape", shape)
		}
	}
	for _, bad := range []Qwen35GDNPackedInputGeometry{
		{0, 2, 6, 128, 128}, {767, 2, 6, 128, 128}, {768, 0, 6, 128, 128}, {768, 4, 6, 128, 128},
		{768, 2, 6, 0, 128}, {768, 2, 6, 1025, 128}, {768, 2, 6, 128, 1025},
		{768, int(^uint32(0) >> 1), int(^uint32(0) >> 1), 128, 128},
	} {
		if _, err := p.Plan(Qwen35GDNPackedQKV, Q4_K, []int{1280, 768}, bad); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
			t.Fatal("bad geometry", bad)
		}
	}
	// Geometry matches shape, but its weight element product exceeds CUDA int.
	big := Qwen35GDNPackedInputGeometry{1 << 20, 16, 48, 128, 128}
	if _, err := p.Plan(Qwen35GDNPackedQKV, Q4_K, []int{10240, big.Hidden}, big); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
		t.Fatal("element overflow")
	}
	// Weight elements fit; the backend's minimum eight-lane panel does not.
	panel := Qwen35GDNPackedInputGeometry{1 << 28, 1, 1, 1, 1}
	if _, err := p.Plan(Qwen35GDNPackedZ, Q4_K, []int{1, panel.Hidden}, panel); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
		t.Fatal("panel overflow")
	}
	if _, err := (Qwen35GDNPackedInputPolicy{}).Plan(Qwen35GDNPackedQKV, Q4_K, []int{1280, 768}, g); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
		t.Fatal("zero policy")
	}
	if err := (Qwen35GDNPackedInputPlan{}).ValidatePayload(0); !errors.Is(err, ErrQwen35GDNPackedPolicy) {
		t.Fatal("zero plan")
	}
}
