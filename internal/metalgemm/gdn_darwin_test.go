//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"math"
	"testing"
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
