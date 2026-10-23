package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The numerical parity fixture explicitly selects historical Halo memory facts;
// its recording backend is not evidence of a physical CUDA memory profile.
type qsaProfileRecordingBackend struct{ *qkNormRecordingBackend }

func (*qsaProfileRecordingBackend) DeviceMemoryProfile() (compute.DeviceMemoryProfile, bool) {
	return compute.LookupDeviceMemoryProfile("gfx1151")
}

type qsaProfileTestBackend struct {
	compute.Backend
	profile                      compute.DeviceMemoryProfile
	known                        bool
	reads, uploads, profileCalls int
}

func (*qsaProfileTestBackend) Name() string { return "vulkan" }
func (*qsaProfileTestBackend) Tier() string { return "gfx1151" }
func (b *qsaProfileTestBackend) DeviceMemoryProfile() (compute.DeviceMemoryProfile, bool) {
	b.profileCalls++
	return b.profile, b.known
}
func (b *qsaProfileTestBackend) Read(t compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(t)
}
func (b *qsaProfileTestBackend) Upload(t compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(t, dtype)
}

type qsaTierTestBackend struct {
	compute.Backend
	name, tier string
}

func (b qsaTierTestBackend) Name() string { return b.name }
func (b qsaTierTestBackend) Tier() string { return b.tier }

type qsaProfileTestKV struct {
	compute.KVStore
	tokens, views int
	keys, values  compute.Tensor
}

func (k *qsaProfileTestKV) Len() int { return k.tokens }
func (k *qsaProfileTestKV) KeysView(int) compute.Tensor {
	k.views++
	return k.keys
}
func (k *qsaProfileTestKV) ValuesView(int) compute.Tensor {
	k.views++
	return k.values
}

// fak-test:runtime fast est=10ms lane=default
func TestQwen35QSAScratchUsesDeviceProfile(t *testing.T) {
	t.Parallel()
	halo, _ := compute.LookupDeviceMemoryProfile("gfx1151")
	profile := func(capacity, alignment int64) compute.DeviceMemoryProfile {
		p := halo
		p.Architecture = "selected-test-device"
		p.CacheCapacityBytes, p.StorageAlignmentBytes = capacity, alignment
		return p
	}
	partial := halo
	partial.CacheCapacityKnown = false
	missingAlignment := halo
	missingAlignment.StorageAlignmentKnown = false
	missingSubgroup := halo
	missingSubgroup.SubgroupWidthKnown = false
	invalidSubgroup := halo
	invalidSubgroup.SubgroupWidth = 0
	maxInt := int(^uint(0) >> 1)
	type scratchCase struct {
		name       string
		profile    compute.DeviceMemoryProfile
		known      bool
		tokens, hd int
		nH, nKV    int
		grp        int
		wantBytes  int64
	}
	cases := []scratchCase{
		{"Halo last admitted width", halo, true, 16384, 1820, 1, 1, 1, 33546240},
		{"Halo next width exceeds cap", halo, true, 16384, 1821, 1, 1, 1, 0},
		{"explicit smaller exact cap", profile(18432, 128), true, 16384, 1, 1, 1, 1, 18432},
		{"explicit cap one byte short", profile(18431, 128), true, 16384, 1, 1, 1, 1, 0},
		{"larger profile admits former overflow", profile(64<<20, 128), true, 16384, 1821, 1, 1, 1, 33564672},
		{"each buffer aligned separately", profile(20000, 10000), true, 16384, 1, 1, 1, 1, 20000},
		{"alignment pushes over cap", profile(19999, 10000), true, 16384, 1, 1, 1, 1, 0},
		{"provider refusal overrides Halo tier", halo, false, 16384, 1, 1, 1, 1, 0},
		{"unknown provider", compute.DeviceMemoryProfile{}, false, 16384, 1, 1, 1, 1, 0},
		{"partial capacity", partial, true, 16384, 1, 1, 1, 1, 0},
		{"partial alignment", missingAlignment, true, 16384, 1, 1, 1, 1, 0},
		{"partial subgroup", missingSubgroup, true, 16384, 1, 1, 1, 1, 0},
		{"nonpositive subgroup", invalidSubgroup, true, 16384, 1, 1, 1, 1, 0},
		{"nonpositive capacity", profile(0, 128), true, 16384, 1, 1, 1, 1, 0},
		{"nonpositive alignment", profile(1<<20, 0), true, 16384, 1, 1, 1, 1, 0},
		{"alignment addition overflow", profile(math.MaxInt64, math.MaxInt64), true, 16384, 1, 1, 1, 1, 0},
		{"aligned pair overflow", profile(math.MaxInt64, math.MaxInt64/2+1), true, 16384, 1, 1, 1, 1, 0},
		{"below dynamic threshold", halo, true, 16383, 1, 1, 1, 1, 0},
		{"negative context", halo, true, -1, 1, 1, 1, 1, 0},
		{"negative head dimension", halo, true, 16384, -1, 1, 1, 1, 0},
		{"negative query heads", halo, true, 16384, 1, -1, 1, 1, 0},
		{"negative KV heads", halo, true, 16384, 1, 1, -1, 1, 0},
		{"negative group", halo, true, 16384, 1, 1, 1, -1, 0},
		{"invalid head dimension", halo, true, 16384, 0, 1, 1, 1, 0},
		{"invalid group relation", halo, true, 16384, 1, 3, 2, 1, 0},
		{"head product overflow", halo, true, 16384, maxInt, 2, 2, 1, 0},
		{"KV product overflow", halo, true, maxInt, 2, 1, 1, 1, 0},
		{"score rows and outer slice overflow", halo, true, 16384, 1, maxInt / 4, 1, maxInt / 4, 0},
	}
	if maxInt > math.MaxInt32 {
		tokens := int64(math.MaxInt32)*compute.QSABlockSize + 1
		cases = append(cases, scratchCase{"block index overflow", halo, true, int(tokens), 1, 1, 1, 1, 0})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			be := &qsaProfileTestBackend{profile: tc.profile, known: tc.known}
			plan, err := planQwen35QSAScratch(be, tc.tokens, tc.hd, tc.nH, tc.nKV, tc.grp)
			if tc.wantBytes != 0 {
				if err != nil || plan.scratchBytes != tc.wantBytes {
					t.Fatalf("scratch plan = %+v, %v; want %d bytes", plan, err, tc.wantBytes)
				}
				if int64(plan.gatherLen)*8 > plan.scratchBytes {
					t.Fatal("gathered K/V payload exceeds admitted bound")
				}
				return
			}
			if err == nil {
				t.Fatalf("invalid or unavailable plan admitted: %+v", plan)
			}
			kv := &qsaProfileTestKV{tokens: tc.tokens}
			s := &Session{Backend: be, M: &Model{Cfg: Config{HeadDim: tc.hd, NumHeads: tc.nH, NumKVHeads: tc.nKV}}, halKV: kv}
			if _, err := s.qwen35QSASparseAttentionHAL(0, 0, compute.Tensor{}, tc.grp, 1); err == nil {
				t.Fatal("production QSA call accepted unavailable plan")
			}
			if be.reads != 0 || be.uploads != 0 || kv.views != 0 || s.qwen35HAL != nil {
				t.Fatalf("decline touched QSA state or tensors: reads=%d uploads=%d views=%d state=%v", be.reads, be.uploads, kv.views, s.qwen35HAL)
			}
		})
	}
	for _, tc := range []struct {
		name, tier string
		want       bool
	}{
		{"vulkan", "gfx1151", true},
		{"vulkan", "AMD Radeon 8060S", true},
		{"vulkan", "gfx942", false},
		{"vulkan", "gfx11510 Radeon 8060S", false},
		{"cuda", "gfx1151", false},
	} {
		t.Run(tc.name+" "+tc.tier, func(t *testing.T) {
			_, err := planQwen35QSAScratch(qsaTierTestBackend{name: tc.name, tier: tc.tier}, 16384, 1, 1, 1, 1)
			if (err == nil) != tc.want {
				t.Fatalf("tier admission err=%v; want admitted=%t", err, tc.want)
			}
		})
	}
	t.Run("host gather retains exact capacities", func(t *testing.T) {
		be := &qsaProfileTestBackend{Backend: compute.Default(), profile: profile(20000, 10000), known: true}
		kv := &qsaProfileTestKV{
			tokens: 16384,
			keys:   compute.NewF32(be.Backend, []int{16384}, make([]float32, 16384)),
			values: compute.NewF32(be.Backend, []int{16384}, make([]float32, 16384)),
		}
		cfg := Config{HeadDim: 1, NumHeads: 1, NumKVHeads: 1}
		s := &Session{Backend: be, M: &Model{Cfg: cfg}, halKV: kv, qwen35HAL: &qwen35HALState{
			qsaGatheredK: make([]float32, compute.QSAMaxGatherTokens+1024),
			qsaGatheredV: make([]float32, compute.QSAMaxGatherTokens+1024),
		}}
		q := compute.NewF32(be.Backend, []int{1}, []float32{1})
		if _, err := s.qwen35QSASparseAttentionHAL(0, 0, q, 1, 1); err != nil {
			t.Fatal(err)
		}
		if be.reads != 3 || be.uploads != 1 || kv.views != 2 {
			t.Fatalf("host execution reads=%d uploads=%d views=%d", be.reads, be.uploads, kv.views)
		}
		if cap(s.qwen35HAL.qsaGatheredK) != compute.QSAMaxGatherTokens || cap(s.qwen35HAL.qsaGatheredV) != compute.QSAMaxGatherTokens {
			t.Fatal("gather retained excess backing capacity")
		}
		if bytes, ok := s.LastQSAScratchBytes(); !ok || bytes != 20000 {
			t.Fatalf("scratch receipt = %d, %t; want aligned bound 20000", bytes, ok)
		}
	})
	t.Run("device-only refuses before profile or transfer", func(t *testing.T) {
		be := &qsaProfileTestBackend{Backend: compute.Default(), profile: halo, known: true}
		s := &Session{Backend: be, M: &Model{}}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := deviceOnlyRefusal(t, func() {
			_, _ = s.qwen35QSASparseAttentionHAL(0, 0, compute.Tensor{}, 1, 1)
		})
		if !errors.Is(err, ErrDeviceOnlyHostStage) || !s.BackendSessionClosed() {
			t.Fatalf("device-only refusal = %v; closed=%t", err, s.BackendSessionClosed())
		}
		if be.reads != 0 || be.uploads != 0 || be.profileCalls != 0 {
			t.Fatal("device-only QSA consulted profile or transferred tensors")
		}
	})
}
