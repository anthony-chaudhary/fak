//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41BF16PublishVulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41BF16PublishDeviceOnlyFixture is the full-profile routed-expert fixture with
// compression and the indexer off: its routed rows all run the fused device
// seam, and it reaches neither compressed publication nor indexer weights, the
// two inventory entries that stay host operations.
func v41BF16PublishDeviceOnlyFixture(t *testing.T) *Model {
	t.Helper()
	m := v41IncrementalExpertFixture(t, true, false)
	m.Cfg.DeepSeekV41.CompressRatios = []int{0}
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	return m
}

func v41BF16PublishVulkanTrim(be compute.Backend) {
	if trimmer, ok := be.(interface{ Trim() }); ok {
		trimmer.Trim()
	}
}

func v41BF16PublishVulkanSnapshot(t *testing.T, be compute.Backend) compute.BackendExecutionSnapshot {
	t.Helper()
	v41BF16PublishVulkanTrim(be)
	snap, ok, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !ok || !snap.TransferCountersObserved || !snap.DeviceAllocationObserved {
		t.Fatalf("execution snapshot unavailable: ok=%t err=%v", ok, err)
	}
	return snap
}

func v41BF16PublishBitIdentical(t *testing.T, label string, got, want v41DeviceParityRun) {
	t.Helper()
	for i := range want.logits {
		if len(got.logits[i]) != len(want.logits[i]) {
			t.Fatalf("%s step %d width %d want %d", label, i, len(got.logits[i]), len(want.logits[i]))
		}
		for j := range want.logits[i] {
			if math.Float32bits(got.logits[i][j]) != math.Float32bits(want.logits[i][j]) {
				t.Fatalf("%s step %d logit %d device-published %g host-arm %g", label, i, j, got.logits[i][j], want.logits[i][j])
			}
		}
	}
}

// The ordinary BF16 publication (#13813) replaces every reachable host
// v41RoundBF16 in a full-profile session. /seam checks each site bit-exactly
// against the host RNE with its native transfer contract; /default-session runs
// a device-only session with zero host rounds and host-arm-identical logits.
// fak-test:justify why=integration when=changed:internal/model/v41_bf16_publish*
// fak-test:runtime slow est=60s lane=optin
func TestV41BF16PublishVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	if _, ok := be.(compute.V41BF16PublishBackend); !ok {
		t.Fatal("ordinary Vulkan backend lacks the BF16 publication")
	}
	fixture := func() *Model {
		m, _ := v41IncrementalEngramFixture(t, false)
		return m
	}
	t.Run("seam", func(t *testing.T) {
		m := fixture()
		t.Cleanup(func() {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		publish := s.v41BF16PublishFunc()
		if publish == nil || s.v41State().gatherOps.bf16Publish == nil || s.v41State().bf16Boundary == nil {
			t.Fatal("ordinary publication not bound on the Vulkan session")
		}
		type site struct {
			layer int
			name  string
		}
		sites := []site{{-1, "model.norm.weight input"}, {-1, "model.norm.weight output"}}
		var names []string
		for name := range v41BF16PublishSites {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			sites = append(sites, site{m.Cfg.NumLayers - 1, name})
		}
		width := m.Cfg.HiddenSize
		lanes := 0
		for i, st := range sites {
			row := v41BF16PublishRow(width)
			row[i%width] = math.Float32frombits(0x7f7f7fff)
			want := make([]float32, width)
			for j, v := range row {
				want[j] = v41RoundBF16(v)
			}
			original := append([]float32(nil), row...)
			before := v41BF16PublishVulkanSnapshot(t, be)
			stop := enableV41BF16Witness()
			got, err := publish(st.layer, st.name, row)
			rounds := v41HostBF16Rounds.Load()
			stop()
			if err != nil {
				t.Fatalf("%s: %v", st.name, err)
			}
			after := v41BF16PublishVulkanSnapshot(t, be)
			d, err := compute.BackendExecutionDelta(before, after)
			if err != nil {
				t.Fatal(err)
			}
			c := d.Counters
			if rounds != 0 || c.ComputeDispatches != 1 || c.H2DCount != 1 || c.H2DBytes != uint64(4*width) || c.D2HCount != 2 || c.D2HBytes != uint64(4*width+16) || c.Fallbacks != 0 || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
				t.Fatalf("%s: host rounds=%d native contract=%+v live %d->%d", st.name, rounds, c, before.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
			}
			for j := range want {
				if math.Float32bits(got[j]) != math.Float32bits(want[j]) || math.Float32bits(row[j]) != math.Float32bits(original[j]) {
					t.Fatalf("%s lane %d device %08x host %08x", st.name, j, math.Float32bits(got[j]), math.Float32bits(want[j]))
				}
				lanes++
			}
		}
		// A resident result published before its one readback.
		resident := v41BF16PublishRow(width)
		x := uploadHostF32Class(be, []int{2, width / 2}, resident, compute.MemoryActivation, "V4.1 BF16 resident witness")
		defer be.Free(x)
		published, ok, err := s.v41PublishResidentBF16(x, compute.V41BF16PublishRoutedDown)
		if err != nil || !ok {
			t.Fatalf("resident publication: ok=%t err=%v", ok, err)
		}
		for j, v := range be.Read(published) {
			if math.Float32bits(v) != math.Float32bits(v41RoundBF16(resident[j])) {
				t.Fatalf("resident lane %d", j)
			}
		}
		be.Free(published)
		t.Logf("V41_BF16_PUBLISH_SEAM sites=%d lanes=%d bit_exact=%d", len(sites), lanes, lanes)
	})
	t.Run("device-fault", func(t *testing.T) {
		m := fixture()
		t.Cleanup(func() {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		row := v41BF16PublishRow(m.Cfg.HiddenSize)
		row[5] = math.Float32frombits(0x7f7f8000)
		row[9] = float32(math.NaN())
		_, err = s.v41BF16PublishFunc()(0, "attn.kv_norm.weight projection", row)
		var numeric *compute.V41BF16CheckedArithmeticError
		if !errors.Is(err, ErrV41BF16BoundaryOperation) || !errors.As(err, &numeric) || numeric.Index != 5 || numeric.Site != compute.V41BF16PublishKVNormInput || !s.BackendSessionClosed() {
			t.Fatalf("device fault not attributed to the lowest index or session left open: %v", err)
		}
	})
	t.Run("engram-session", func(t *testing.T) {
		// Default policy on the three-layer Engram fixture: the device-published
		// session must match the host-arm session bit for bit. Its routed rows can
		// decline to host arms, so host rounds are only logged here.
		prompt := []int{1, 2, 3}
		devModel, hostArmModel := fixture(), fixture()
		defer func() {
			for _, m := range []*Model{devModel, hostArmModel} {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			}
		}()
		dev, err := devModel.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer dev.Close()
		if dev.v41State().gatherOps.bf16Publish == nil {
			t.Fatal("publication not bound")
		}
		stop := enableV41BF16Witness()
		got := v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps)
		rounds, publishes, resident := v41HostBF16Rounds.Load(), v41BF16PublishCalls.Load(), v41BF16ResidentCalls.Load()
		stop()
		hostArm, err := hostArmModel.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer hostArm.Close()
		hs := hostArm.v41State()
		hs.gatherOps.bf16Publish, hs.bf16Boundary = nil, nil
		hs.setAttentionBF16(nil)
		alt := v41DeviceParitySession(t, hostArm, prompt, v41DeviceParitySteps)
		v41BF16PublishBitIdentical(t, "engram", got, alt)
		t.Logf("V41_BF16_PUBLISH_ENGRAM_SESSION tokens=%v host_arm=%v ab_bit_identical=true publications=%d resident=%d host_rounds_default_policy=%d", got.tokens, alt.tokens, publishes, resident, rounds)
	})
	t.Run("deferred-ab", func(t *testing.T) {
		// The deferred status read is the synchronous session bit for bit: the
		// same rows and status records cross to the host, behind fewer fences.
		prompt := []int{1, 2, 3}
		type arm struct {
			run                   v41DeviceParityRun
			publications, defers  int64
			d2h, d2hBytes, submit uint64
		}
		measure := func(env string) arm {
			t.Setenv(v41DeferredBF16PublishEnv, env)
			m := fixture()
			defer func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			}()
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			before := v41BF16PublishVulkanSnapshot(t, be)
			stop := enableV41BF16Witness()
			got := v41DeviceParitySession(t, s, prompt, v41DeviceParitySteps)
			publications, defers := v41BF16PublishCalls.Load()+v41BF16ResidentCalls.Load(), v41BF16DeferredCalls.Load()
			stop()
			d, err := compute.BackendExecutionDelta(before, v41BF16PublishVulkanSnapshot(t, be))
			if err != nil {
				t.Fatal(err)
			}
			if s.BackendSessionClosed() {
				t.Fatalf("%s=%s session retired", v41DeferredBF16PublishEnv, env)
			}
			return arm{got, publications, defers, d.Counters.D2HCount, d.Counters.D2HBytes, d.Counters.DispatchSubmits}
		}
		sync, deferred := measure("0"), measure("1")
		v41BF16PublishBitIdentical(t, "deferred status", deferred.run, sync.run)
		if sync.defers != 0 || deferred.defers == 0 || deferred.defers != deferred.publications || deferred.publications != sync.publications {
			t.Fatalf("publications sync=%d deferred=%d (deferred-status %d/%d)", sync.publications, deferred.publications, sync.defers, deferred.defers)
		}
		if deferred.d2h != sync.d2h || deferred.d2hBytes != sync.d2hBytes {
			t.Fatalf("D2H regions %d/%d bytes %d/%d, want the same reads", deferred.d2h, sync.d2h, deferred.d2hBytes, sync.d2hBytes)
		}
		if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" && deferred.submit >= sync.submit {
			t.Fatalf("dispatch submits deferred=%d synchronous=%d, want fewer", deferred.submit, sync.submit)
		}
		t.Logf("V41_BF16_PUBLISH_DEFERRED_AB tokens=%v ab_bit_identical=true publications=%d deferred_status=%d d2h=%d submits_sync=%d submits_deferred=%d",
			deferred.run.tokens, deferred.publications, deferred.defers, deferred.d2h, sync.submit, deferred.submit)
	})
	t.Run("default-session", func(t *testing.T) {
		prompt := []int{1, 2, 3}
		ref, devModel, hostArmModel := v41BF16PublishDeviceOnlyFixture(t), v41BF16PublishDeviceOnlyFixture(t), v41BF16PublishDeviceOnlyFixture(t)
		defer func() {
			for _, m := range []*Model{ref, devModel, hostArmModel} {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			}
		}()
		want := v41DeviceParityReference(t, ref, prompt, v41DeviceParitySteps)

		dev, err := devModel.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer dev.Close()
		dev.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		if unbound := dev.v41DeviceOnlyUnboundOps(); len(unbound) != 0 {
			t.Fatalf("device-only fixture leaves ops unbound: %v", v41HostOpNames(unbound))
		}
		refusal := dev.v41DeviceOnlyAdmissionRefusal()
		var typed *v41DeviceOnlyHostOpError
		if refusal != nil {
			if !errors.As(refusal, &typed) {
				t.Fatalf("untyped admission refusal: %v", refusal)
			}
			for _, op := range typed.ops {
				if op.name == "bf16-boundary" {
					t.Fatalf("admission still names bf16-boundary: %v", refusal)
				}
			}
		}
		t.Logf("device-only admission with the static inventory: %v", refusal)
		// The remaining static entries are not BF16 boundaries and this fixture
		// reaches none of them. Clearing them lets the device-only forward run so
		// the host-round witness observes it.
		saved := v41DeviceOnlyHostOps
		v41DeviceOnlyHostOps = nil
		defer func() { v41DeviceOnlyHostOps = saved }()
		sites := map[string]int{}
		st := dev.v41State()
		inner, final := st.gatherOps.bf16PublishFunc(), st.bf16Boundary
		if inner == nil || final == nil || st.attentionBF16Func() == nil {
			t.Fatal("publication seams not bound on the device-only session")
		}
		count := func(next v41BF16BoundaryFunc) v41BF16BoundaryFunc {
			return func(layer int, site string, values []float32) ([]float32, error) {
				sites[fmt.Sprintf("%s@%d", site, layer)]++
				return next(layer, site, values)
			}
		}
		st.gatherOps.setBF16Publication(count(inner))
		st.bf16Boundary = count(final)
		stop := enableV41BF16Witness()
		var got v41DeviceParityRun
		dispatches := v41SeamDispatches(t, be, func() { got = v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps) })
		rounds, publishes, resident := v41HostBF16Rounds.Load(), v41BF16PublishCalls.Load(), v41BF16ResidentCalls.Load()
		stop()
		if dev.HostFallbackObserved() || dev.BackendSessionClosed() {
			t.Fatal("device-only session observed a host fallback or closed")
		}
		if rounds != 0 {
			t.Fatalf("device-only session rounded %d values to BF16 on the host", rounds)
		}
		if publishes == 0 || resident == 0 {
			t.Fatalf("device-only session issued publications=%d resident=%d", publishes, resident)
		}
		tokens := len(prompt) + v41DeviceParitySteps - 1
		for l := 0; l < devModel.Cfg.NumLayers; l++ {
			// Rotation is the identity at position 0, so only pre-rotation rows are
			// guaranteed to need rounding at every token.
			for _, site := range []string{"attn.wq_a_norm.weight normalization", v41TailRoPEInputSite} {
				if n := sites[fmt.Sprintf("%s@%d", site, l)]; n < tokens {
					t.Errorf("layer %d %s published %d times, want >= %d", l, site, n, tokens)
				}
			}
		}

		hostArm, err := hostArmModel.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer hostArm.Close()
		hs := hostArm.v41State()
		hs.gatherOps.bf16Publish, hs.bf16Boundary = nil, nil
		hs.setAttentionBF16(nil)
		alt := v41DeviceParitySession(t, hostArm, prompt, v41DeviceParitySteps)
		v41BF16PublishBitIdentical(t, "device-only", got, alt)
		// The host-arm session is bit-identical, so any logit drift against the
		// host reference belongs to the fixture's other device seams; only the
		// greedy tokens are required to match it.
		refWorst := 0.0
		for i := range want.tokens {
			if got.tokens[i] != want.tokens[i] {
				t.Errorf("greedy token[%d]=%d want reference %d", i, got.tokens[i], want.tokens[i])
			}
			var scale, delta float64
			for j, w := range want.logits[i] {
				scale = max(scale, math.Abs(float64(w)))
				delta = max(delta, math.Abs(float64(got.logits[i][j]-w)))
			}
			refWorst = max(refWorst, delta/max(1, scale))
		}
		keys := make([]string, 0, len(sites))
		for k := range sites {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		t.Logf("V41_BF16_PUBLISH_SESSION tokens=%v host_arm=%v reference=%v host_rounds=%d publications=%d resident=%d compute_dispatches=%d ab_bit_identical=true reference_worst_rel=%.4g", got.tokens, alt.tokens, want.tokens, rounds, publishes, resident, dispatches, refWorst)
		for _, k := range keys {
			t.Logf("site %s calls=%d", k, sites[k])
		}
	})
}
