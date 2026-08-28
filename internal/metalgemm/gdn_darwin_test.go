//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"math"
	"testing"
	"time"
)

const (
	gdnCosineFloor = 0.999999
	gdnMaxAbsLimit = 0.0001
)

func nativeGDNGeometry(g oracleGDNGeometry) GDNGeometry {
	return GDNGeometry{NumKeyHeads: g.nK, NumValueHeads: g.nV, KeyHeadDim: g.kHd, ValueHeadDim: g.vHd, ConvKernel: g.kernel}
}

func nativeGDNPanel(p oracleGDNPanel) GDNPanel {
	return GDNPanel{Tokens: p.tokens, Mixed: p.mixed, Z: p.z, B: p.b, A: p.a, Conv1D: p.conv, ALog: p.aLog, DTBias: p.dtBias, Norm: p.norm, RMSNormEpsilon: p.eps}
}

func gdnCosineMaxAbs(want, got []float32) (float64, float64) {
	if len(want) != len(got) || len(want) == 0 {
		return 0, math.Inf(1)
	}
	dot, wn, gn, maxAbs := float64(0), float64(0), float64(0), float64(0)
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		dot += w * g
		wn += w * w
		gn += g * g
		if delta := math.Abs(w - g); delta > maxAbs {
			maxAbs = delta
		}
	}
	if wn == 0 && gn == 0 {
		return 1, maxAbs
	}
	return dot / math.Sqrt(wn*gn), maxAbs
}

func requireGDNParity(t *testing.T, name string, want, got []float32) {
	t.Helper()
	cosine, maxAbs := gdnCosineMaxAbs(want, got)
	if cosine < gdnCosineFloor || maxAbs > gdnMaxAbsLimit {
		t.Fatalf("%s parity cosine=%.9f maxAbs=%g, want cosine>=%.6f maxAbs<=%g", name, cosine, maxAbs, gdnCosineFloor, gdnMaxAbsLimit)
	}
	t.Logf("%s parity: elements=%d cosine=%.9f maxAbs=%g", name, len(got), cosine, maxAbs)
}

func requireGDNAccounting(t *testing.T, accounting GDNAccounting, outputReadback int) {
	t.Helper()
	if accounting.CommandBufferID == 0 || !accounting.Committed || !accounting.CompletedWait || accounting.Encoders != 1 {
		t.Fatalf("native lifecycle=%+v, want one committed/waited command buffer and encoder", accounting)
	}
	if accounting.StateH2DTransfers != 0 || accounting.StateD2HTransfers != 0 || accounting.HostRecurrenceSteps != 0 || accounting.OwnedBuffers != 2 || accounting.PrivateStateBuffers != 2 {
		t.Fatalf("state/accounting=%+v, want zero state transfers/host recurrence and two owned buffers", accounting)
	}
	if accounting.PanelH2DTransfers != 8 || accounting.OutputD2HTransfers != outputReadback || accounting.StateBytes == 0 {
		t.Fatalf("staging/accounting=%+v, want eight input stages, output readback=%d, and observed state bytes", accounting, outputReadback)
	}
	t.Logf("native accounting: command_buffers=1 encoders=%d state_bytes=%d state_h2d=%d state_d2h=%d host_steps=%d panel_h2d=%d output_d2h=%d",
		accounting.Encoders, accounting.StateBytes, accounting.StateH2DTransfers, accounting.StateD2HTransfers,
		accounting.HostRecurrenceSteps, accounting.PanelH2DTransfers, accounting.OutputD2HTransfers)
}

func TestGDNSequenceMetalParityStateIdentityAndReset(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	start := GDNLiveBufferCount()
	g := oracleGDNGeometry{nK: 2, nV: 4, kHd: 4, vHd: 8, kernel: 3}
	native, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		t.Fatal(err)
	}
	defer native.Close()
	convHandle, recurrentHandle := native.Handles()
	if convHandle == 0 || recurrentHandle == 0 || convHandle == recurrentHandle {
		t.Fatalf("state handles=%d/%d, want distinct non-zero", convHandle, recurrentHandle)
	}
	if got := GDNLiveBufferCount(); got != start+2 {
		t.Fatalf("live buffers after allocation=%d, want %d", got, start+2)
	}

	oracleState := newOracleGDNState(g)
	for step, panel := range []oracleGDNPanel{oracleGDNFixture(g, 5, .1), oracleGDNFixture(g, 3, .9)} {
		want := oracleGDNRun(g, panel, oracleState)
		got, accounting, accepted, runErr := native.Run(nativeGDNPanel(panel))
		if !accepted || runErr != nil {
			t.Fatalf("step %d = accepted %v err %v", step, accepted, runErr)
		}
		requireGDNAccounting(t, accounting, 1)
		requireGDNParity(t, "output", want, got)
		conv, recurrent, snapshotErr := native.Snapshot()
		if snapshotErr != nil {
			t.Fatal(snapshotErr)
		}
		requireGDNParity(t, "convolution state", oracleState.conv, conv)
		requireGDNParity(t, "recurrent state", oracleState.recurrent, recurrent)
		if gotConv, gotRecurrent := native.Handles(); gotConv != convHandle || gotRecurrent != recurrentHandle {
			t.Fatalf("step %d replaced handles %d/%d with %d/%d", step, convHandle, recurrentHandle, gotConv, gotRecurrent)
		}
	}

	if err := native.Reset(); err != nil {
		t.Fatal(err)
	}
	if gotConv, gotRecurrent := native.Handles(); gotConv != convHandle || gotRecurrent != recurrentHandle {
		t.Fatalf("reset changed handles to %d/%d", gotConv, gotRecurrent)
	}
	conv, recurrent, err := native.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range map[string][]float32{"conv": conv, "recurrent": recurrent} {
		for i, value := range values {
			if value != 0 {
				t.Fatalf("reset %s[%d]=%g, want zero", name, i, value)
			}
		}
	}
	native.Close()
	native.Close()
	if got := GDNLiveBufferCount(); got != start {
		t.Fatalf("live buffers after exact-once close=%d, want %d", got, start)
	}
}

func TestGDNSequenceMetalSeedMatchesCPUOracleAndOwnsState(t *testing.T) {
	oracleGeometry := oracleGDNGeometry{nK: 2, nV: 4, kHd: 4, vHd: 8, kernel: 3}
	panels := []oracleGDNPanel{
		oracleGDNFixture(oracleGeometry, 2, .1),
		oracleGDNFixture(oracleGeometry, 1, .7),
	}
	g := nativeGDNGeometry(oracleGeometry)
	seedSource, err := NewGDNState(g)
	if err != nil {
		t.Fatalf("NewGDNState(seed): %v", err)
	}
	first := nativeGDNPanel(panels[0])
	if _, _, accepted, err := seedSource.Run(first); err != nil || !accepted {
		t.Fatalf("seed source run: accepted=%v err=%v", accepted, err)
	}
	seedConv, seedRecurrent, err := seedSource.Snapshot()
	if err != nil {
		t.Fatalf("seed snapshot: %v", err)
	}
	seedSource.Close()

	want := newOracleGDNState(oracleGeometry)
	want.conv = append(want.conv[:0], seedConv...)
	want.recurrent = append(want.recurrent[:0], seedRecurrent...)
	wantOut := oracleGDNRun(oracleGeometry, panels[1], want)

	state, err := NewGDNState(g)
	if err != nil {
		t.Fatalf("NewGDNState: %v", err)
	}
	defer state.Close()
	if state.owner != 0 {
		t.Fatalf("seed regression requires the first valid native owner, got %d", state.owner)
	}
	convHandle, recurrentHandle := state.Handles()
	beforeConv, beforeRecurrent, _ := state.Snapshot()
	if err := state.Seed(seedConv[:len(seedConv)-1], seedRecurrent); err == nil {
		t.Fatal("short seed accepted")
	}
	afterConv, afterRecurrent, _ := state.Snapshot()
	requireGDNParity(t, "declined seed conv no-mutation", beforeConv, afterConv)
	requireGDNParity(t, "declined seed recurrent no-mutation", beforeRecurrent, afterRecurrent)
	if err := state.Seed(seedConv, seedRecurrent); err != nil {
		t.Fatalf("Seed: %v", err)
	}
	seededConv, seededRecurrent, err := state.Snapshot()
	if err != nil {
		t.Fatalf("seeded snapshot before run: %v", err)
	}
	requireGDNParity(t, "seed install convolution", seedConv, seededConv)
	requireGDNParity(t, "seed install recurrent", seedRecurrent, seededRecurrent)
	if gotConv, gotRecurrent := state.Handles(); gotConv != convHandle || gotRecurrent != recurrentHandle {
		t.Fatalf("seed changed handles from %d/%d to %d/%d", convHandle, recurrentHandle, gotConv, gotRecurrent)
	}
	if err := state.Reset(); err != nil {
		t.Fatalf("Reset after seed: %v", err)
	}
	resetConv, resetRecurrent, err := state.Snapshot()
	if err != nil {
		t.Fatalf("reset snapshot: %v", err)
	}
	for name, values := range map[string][]float32{"conv": resetConv, "recurrent": resetRecurrent} {
		for i, value := range values {
			if value != 0 {
				t.Fatalf("reset %s[%d]=%g, want zero", name, i, value)
			}
		}
	}
	if gotConv, gotRecurrent := state.Handles(); gotConv != convHandle || gotRecurrent != recurrentHandle {
		t.Fatalf("reset changed handles from %d/%d to %d/%d", convHandle, recurrentHandle, gotConv, gotRecurrent)
	}
	if err := state.Seed(seedConv, seedRecurrent); err != nil {
		t.Fatalf("reseed after reset: %v", err)
	}
	gotOut, accounting, accepted, err := state.Run(nativeGDNPanel(panels[1]))
	if err != nil || !accepted {
		t.Fatalf("seeded run: accepted=%v err=%v", accepted, err)
	}
	requireGDNAccounting(t, accounting, 1)
	requireGDNParity(t, "seeded output", wantOut, gotOut)
	gotConv, gotRecurrent, err := state.Snapshot()
	if err != nil {
		t.Fatalf("seeded snapshot: %v", err)
	}
	requireGDNParity(t, "seeded convolution", want.conv, gotConv)
	requireGDNParity(t, "seeded recurrent", want.recurrent, gotRecurrent)
}

func TestGDNSequenceMetalSeedSupportsEmptyConvolutionWindow(t *testing.T) {
	g := oracleGDNGeometry{nK: 1, nV: 1, kHd: 2, vHd: 2, kernel: 1}
	state, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	conv, recurrent, err := state.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(conv) != 0 {
		t.Fatalf("kernel-1 convolution window has %d elements, want zero", len(conv))
	}
	for i := range recurrent {
		recurrent[i] = float32(i+1) / 16
	}
	if err := state.Seed(conv, recurrent); err != nil {
		t.Fatalf("seed empty convolution window: %v", err)
	}
	gotConv, gotRecurrent, err := state.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(gotConv) != 0 {
		t.Fatalf("seeded kernel-1 convolution window has %d elements", len(gotConv))
	}
	requireGDNParity(t, "kernel-1 seeded recurrent", recurrent, gotRecurrent)
}

func TestGDNSequenceMetalValidationBeforeMutationAndOwnerIsolation(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	start := GDNLiveBufferCount()
	g := oracleGDNGeometry{nK: 2, nV: 4, kHd: 4, vHd: 8, kernel: 3}
	a, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		a.Close()
		t.Fatal(err)
	}
	defer a.Close()
	defer b.Close()
	aConv, aRec := a.Handles()
	bConv, bRec := b.Handles()
	if aConv == bConv || aConv == bRec || aRec == bConv || aRec == bRec {
		t.Fatalf("owners share handles A=%d/%d B=%d/%d", aConv, aRec, bConv, bRec)
	}

	bad := oracleGDNFixture(g, 2, .3)
	bad.mixed = bad.mixed[:len(bad.mixed)-1]
	beforeConv, beforeRecurrent, err := a.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	_, accounting, accepted, err := a.Run(nativeGDNPanel(bad))
	var declined *GDNDeclinedError
	if accepted || !errors.As(err, &declined) || accounting.Committed {
		t.Fatalf("invalid panel = accepted %v accounting=%+v err=%T %v", accepted, accounting, err, err)
	}
	afterConv, afterRecurrent, err := a.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for i := range beforeConv {
		if beforeConv[i] != afterConv[i] {
			t.Fatalf("validation mutated conv[%d]", i)
		}
	}
	for i := range beforeRecurrent {
		if beforeRecurrent[i] != afterRecurrent[i] {
			t.Fatalf("validation mutated recurrent[%d]", i)
		}
	}

	oracleA, oracleB := newOracleGDNState(g), newOracleGDNState(g)
	for step, run := range []struct {
		owner  *GDNState
		oracle *oracleGDNState
		panel  oracleGDNPanel
	}{{a, oracleA, oracleGDNFixture(g, 2, .1)}, {b, oracleB, oracleGDNFixture(g, 3, .8)}, {a, oracleA, oracleGDNFixture(g, 2, 1.4)}} {
		want := oracleGDNRun(g, run.panel, run.oracle)
		got, observed, accepted, runErr := run.owner.Run(nativeGDNPanel(run.panel))
		if !accepted || runErr != nil {
			t.Fatalf("interleave step %d = accepted %v err %v", step, accepted, runErr)
		}
		requireGDNAccounting(t, observed, 1)
		requireGDNParity(t, "isolated output", want, got)
	}
	a.Close()
	b.Close()
	if got := GDNLiveBufferCount(); got != start {
		t.Fatalf("isolated owners leaked buffers=%d, start=%d", got, start)
	}
}

func TestGDNSequenceMetalInjectedPostSubmitFailureCleansOwner(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	start := GDNLiveBufferCount()
	g := oracleGDNGeometry{nK: 2, nV: 4, kHd: 4, vHd: 8, kernel: 3}
	state, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		t.Fatal(err)
	}
	output, accounting, accepted, err := state.run(nativeGDNPanel(oracleGDNFixture(g, 2, .2)), true)
	if !accepted || !IsGDNPostSubmit(err) || output != nil {
		t.Fatalf("injected failure = accepted %v output=%v err=%T %v", accepted, output, err, err)
	}
	requireGDNAccounting(t, accounting, 0)
	if conv, recurrent := state.Handles(); conv != 0 || recurrent != 0 {
		t.Fatalf("failed owner retained handles %d/%d", conv, recurrent)
	}
	state.Close()
	state.Close()
	if got := GDNLiveBufferCount(); got != start {
		t.Fatalf("injected failure leaked buffers=%d, want %d", got, start)
	}
}

func TestGDNExactQwenOwnerRegistrySupportsB8Capacity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const (
		requiredOwners   = 48 * 8
		reviewedCapacity = 512
	)
	geometry := GDNGeometry{
		NumKeyHeads: 16, NumValueHeads: 48,
		KeyHeadDim: 128, ValueHeadDim: 128, ConvKernel: 4,
	}
	if got := gdnOwnerCapacity(); got != reviewedCapacity {
		t.Fatalf("native GDN owner capacity=%d, want reviewed bound %d", got, reviewedCapacity)
	}
	baselineBuffers := GDNLiveBufferCount()
	if baselineBuffers != 0 {
		t.Fatalf("capacity witness requires an empty registry, live buffers=%d", baselineBuffers)
	}
	baselineAllocated := gdnCurrentAllocatedBytes()
	started := time.Now()
	owners := make([]*GDNState, 0, reviewedCapacity)
	defer func() {
		for _, owner := range owners {
			owner.Close()
		}
	}()
	seenHandles := make(map[GDNStateHandle]struct{}, 2*reviewedCapacity)
	openOwner := func(index int) *GDNState {
		t.Helper()
		owner, err := NewGDNState(geometry)
		if err != nil {
			t.Fatalf("allocate exact-Qwen owner %d: %v", index, err)
		}
		convHandle, recurrentHandle := owner.Handles()
		for _, handle := range []GDNStateHandle{convHandle, recurrentHandle} {
			if handle == 0 {
				t.Fatalf("owner %d published a zero handle", index)
			}
			if _, exists := seenHandles[handle]; exists {
				t.Fatalf("owner %d reused live handle %d", index, handle)
			}
			seenHandles[handle] = struct{}{}
		}
		owners = append(owners, owner)
		return owner
	}

	for index := 0; index < requiredOwners; index++ {
		openOwner(index)
	}
	if got, want := GDNLiveBufferCount(), baselineBuffers+2*requiredOwners; got != want {
		t.Fatalf("B8 live buffers=%d, want %d", got, want)
	}
	peakB8Allocated := gdnCurrentAllocatedBytes()

	convSeed := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	recurrentSeed := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	seedLocation := func(index, length, stride int) int { return (index * stride) % length }
	seedValue := func(index, offset int) float32 { return float32(index*2+offset+1) / 2048 }
	for index, owner := range owners[:requiredOwners] {
		convIndex := seedLocation(index, len(convSeed), 73)
		recurrentIndex := seedLocation(index, len(recurrentSeed), 7919)
		convSeed[convIndex] = seedValue(index, 0)
		recurrentSeed[recurrentIndex] = -seedValue(index, 1)
		if err := owner.Seed(convSeed, recurrentSeed); err != nil {
			t.Fatalf("seed exact-Qwen owner %d: %v", index, err)
		}
		convSeed[convIndex] = 0
		recurrentSeed[recurrentIndex] = 0
	}
	assertSeededOwners := func(stage string) {
		t.Helper()
		for index, owner := range owners[:requiredOwners] {
			conv, recurrent, err := owner.Snapshot()
			if err != nil {
				t.Fatalf("%s snapshot owner %d: %v", stage, index, err)
			}
			convIndex := seedLocation(index, len(conv), 73)
			recurrentIndex := seedLocation(index, len(recurrent), 7919)
			for element, got := range conv {
				want := float32(0)
				if element == convIndex {
					want = seedValue(index, 0)
				}
				if got != want {
					t.Fatalf("%s owner %d convolution[%d]=%g, want %g", stage, index, element, got, want)
				}
			}
			for element, got := range recurrent {
				want := float32(0)
				if element == recurrentIndex {
					want = -seedValue(index, 1)
				}
				if got != want {
					t.Fatalf("%s owner %d recurrent[%d]=%g, want %g", stage, index, element, got, want)
				}
			}
		}
	}
	assertSeededOwners("B8 seeded isolation")

	for index := requiredOwners; index < reviewedCapacity; index++ {
		openOwner(index)
	}
	if got, want := GDNLiveBufferCount(), baselineBuffers+2*reviewedCapacity; got != want {
		t.Fatalf("full registry live buffers=%d, want %d", got, want)
	}
	peakCapacityAllocated := gdnCurrentAllocatedBytes()
	failedOwner, err := NewGDNState(geometry)
	var declined *GDNDeclinedError
	if failedOwner != nil || !errors.As(err, &declined) {
		if failedOwner != nil {
			failedOwner.Close()
		}
		t.Fatalf("capacity+1 owner=%v err=%T %v, want clean decline", failedOwner, err, err)
	}
	if got, want := GDNLiveBufferCount(), baselineBuffers+2*reviewedCapacity; got != want {
		t.Fatalf("capacity+1 changed published buffers=%d, want %d", got, want)
	}
	assertSeededOwners("after capacity+1 decline")

	for _, owner := range owners {
		owner.Close()
	}
	owners = owners[:0]
	if got := GDNLiveBufferCount(); got != baselineBuffers {
		t.Fatalf("release-to-baseline live buffers=%d, want %d", got, baselineBuffers)
	}
	replacement, err := NewGDNState(geometry)
	if err != nil {
		t.Fatalf("allocate after full release: %v", err)
	}
	for _, handle := range func() []GDNStateHandle {
		conv, recurrent := replacement.Handles()
		return []GDNStateHandle{conv, recurrent}
	}() {
		if _, reused := seenHandles[handle]; reused {
			t.Fatalf("post-release allocation reused handle %d", handle)
		}
	}
	replacement.Close()
	if got := GDNLiveBufferCount(); got != baselineBuffers {
		t.Fatalf("replacement release live buffers=%d, want %d", got, baselineBuffers)
	}
	t.Logf("exact-Qwen GDN owner receipt: required=%d capacity=%d live_peak=%d state_bytes_each=%d metal_allocated_baseline=%d metal_allocated_B8=%d metal_allocated_capacity=%d elapsed=%s capacity_plus_one=declined cleanup=baseline",
		requiredOwners, reviewedCapacity, 2*reviewedCapacity,
		uint64((geometry.ConvKernel-1)*geometry.convDim()+geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)*4,
		baselineAllocated, peakB8Allocated, peakCapacityAllocated, time.Since(started).Round(time.Millisecond))
}

func gdnPanelTransientBytes(g GDNGeometry, tokens int) uint64 {
	keyDim, valueDim, convDim := g.keyDim(), g.valueDim(), g.convDim()
	f32 := func(elements int) uint64 { return uint64(elements) * 4 }
	return f32(tokens*convDim) + // mixed input
		f32(tokens*valueDim) + // z input
		2*f32(tokens*g.NumValueHeads) + // b and a inputs
		f32(convDim*g.ConvKernel) + // convolution weights
		2*f32(g.NumValueHeads) + // A_log and dt_bias
		f32(g.ValueHeadDim) + // normalization weights
		f32(tokens*convDim) + // private convolution output
		2*f32(tokens*keyDim) + // private q/k normalized panels
		f32(tokens*valueDim) // shared core output
}

func TestGDNPanelResourcesDrainPerCall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const (
		stateCount    = 48
		panelsPer512  = 8
		productionRun = 4
	)
	g := oracleGDNGeometry{nK: 16, nV: 48, kHd: 128, vHd: 128, kernel: 4}
	geometry := nativeGDNGeometry(g)
	perPanel := gdnPanelTransientBytes(geometry, GDNMaxPanelTokens)
	if perPanel != 9_626_496 {
		t.Fatalf("derived panel transient bytes=%d, want 9626496", perPanel)
	}
	perChunk := perPanel * panelsPer512 * stateCount
	fourChunks := perChunk * productionRun
	if perChunk != 3_696_574_464 || fourChunks != 14_786_297_856 {
		t.Fatalf("production derivation panel=%d chunk=%d four_chunks=%d", perPanel, perChunk, fourChunks)
	}

	startBuffers := GDNLiveBufferCount()
	startAllocated := gdnCurrentAllocatedBytes()
	states := make([]*GDNState, 0, stateCount)
	for i := 0; i < stateCount; i++ {
		state, err := NewGDNState(geometry)
		if err != nil {
			for _, opened := range states {
				opened.Close()
			}
			t.Fatalf("allocate state %d: %v", i, err)
		}
		states = append(states, state)
	}
	defer func() {
		for _, state := range states {
			state.Close()
		}
		if got := GDNLiveBufferCount(); got != startBuffers {
			t.Errorf("live buffers after close=%d, want %d", got, startBuffers)
		}
	}()

	panel := oracleGDNFixture(g, GDNMaxPanelTokens, .125)
	oracleState := newOracleGDNState(g)
	var lastOutput []float32
	runWave := func(wave int) uint64 {
		t.Helper()
		want := oracleGDNRun(g, panel, oracleState)
		for i, state := range states {
			got, accounting, accepted, err := state.Run(nativeGDNPanel(panel))
			if !accepted || err != nil {
				t.Fatalf("wave %d state %d: accepted=%v err=%v", wave, i, accepted, err)
			}
			if i == 0 {
				requireGDNAccounting(t, accounting, 1)
				requireGDNParity(t, "exact-geometry output", want, got)
				lastOutput = got
			}
		}
		allocated := gdnCurrentAllocatedBytes()
		t.Logf("allocation wave=%d metal_allocated_bytes=%d per_panel_bytes=%d state_count=%d output_elements=%d", wave, allocated, perPanel, stateCount, len(lastOutput))
		return allocated
	}

	// The first wave warms pipelines and driver caches. Two subsequent identical
	// waves must return to the same completed-call allocation band.
	warm := runWave(0)
	second := runWave(1)
	third := runWave(2)
	limit := 2 * perPanel
	delta := func(after, before uint64) uint64 {
		if after <= before {
			return 0
		}
		return after - before
	}
	secondGrowth, thirdGrowth := delta(second, warm), delta(third, second)
	t.Logf("allocation plateau warm=%d second=%d third=%d growth=%d/%d limit=%d production_four_chunk_bytes=%d", warm, second, third, secondGrowth, thirdGrowth, limit, fourChunks)
	if secondGrowth > limit || thirdGrowth > limit {
		t.Fatalf("completed GDN panels retained Metal allocation: growth=%d/%d, want each <=%d", secondGrowth, thirdGrowth, limit)
	}

	conv, recurrent, err := states[0].Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	requireGDNParity(t, "exact-geometry convolution state", oracleState.conv, conv)
	requireGDNParity(t, "exact-geometry recurrent state", oracleState.recurrent, recurrent)

	for _, state := range states {
		state.Close()
	}
	states = nil
	if got := GDNLiveBufferCount(); got != startBuffers {
		t.Fatalf("live buffers after close=%d, want %d", got, startBuffers)
	}
	t.Logf("allocation cleanup before=%d after=%d live_buffers=%d", startAllocated, gdnCurrentAllocatedBytes(), GDNLiveBufferCount())
}
