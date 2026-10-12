package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A backend without the device row decode would decode Engram rows on the
// host, so a device-only admission names that decode for every model that
// declares Engram layers, even when the gate and pool seams bind; a backend
// with it binds the row. The host decode stays visible to the BF16 witness.
// fak-test:runtime fast est=2s lane=default
func TestV41DeviceOnlyNamesHostEngramRowDecode(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, true)
	hostDecode := v41EngramPoolBackend(true, true)
	hostDecode.rowDecode = false
	s := v41DenseTestSession(t, m, hostDecode)
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	unbound := v41HostOpNames(s.v41DeviceOnlyUnboundOps())
	if !slices.Contains(unbound, "engram-row-decode") || slices.Contains(unbound, "engram-stream-injection") {
		t.Fatalf("Engram session unbound=%v, want engram-row-decode without engram-stream-injection", unbound)
	}
	refusal := s.v41DeviceOnlyAdmissionRefusal()
	var typed *v41DeviceOnlyHostOpError
	if !errors.Is(refusal, errV41DeviceOnlyHostOp) || !errors.As(refusal, &typed) ||
		!slices.Contains(typed.ops, v41HostOp{name: "engram-row-decode", site: "SupportsV41EngramRowDecode"}) {
		t.Fatalf("Engram device-only admission = %v, want the typed engram-row-decode refusal", refusal)
	}
	if got := v41HostOpNames(v41DenseTestSession(t, m, v41EngramPoolBackend(true, true)).v41DeviceOnlyUnboundOps()); slices.Contains(got, "engram-row-decode") {
		t.Fatalf("a backend with the device row decode still names it: %v", got)
	}

	plain, _ := v41IncrementalEngramFixture(t, false)
	plain.Cfg.DeepSeekV41.EngramLayerIDs = nil
	if got := v41HostOpNames(v41DenseTestSession(t, plain, hostDecode).v41DeviceOnlyUnboundOps()); slices.Contains(got, "engram-row-decode") {
		t.Fatalf("model without Engram layers names the row decode: %v", got)
	}

	// Every element of a host-decoded packed row is one witnessed BF16 round.
	const dim = compute.V41EngramRowDecodeMaxDim
	row := make([]byte, V41EngramPackedRowBytes)
	for i := range dim {
		row[i] = 0x38
	}
	for i := dim; i < V41EngramPackedRowBytes; i++ {
		row[i] = 127
	}
	var decoded []float32
	var err error
	rounds := v41BF16PublishHostRounds(t, func() { decoded, err = decodeV41EngramRow(row, dim, V41EngramPackedRowBytes) })
	if err != nil || len(decoded) != dim || decoded[0] != 1 || rounds != int64(len(decoded)) {
		t.Fatalf("packed decode err=%v len=%d first=%v host_rounds=%d, want one witnessed round per element", err, len(decoded), decoded, rounds)
	}
}

// The resident mHC entry refuses a non-finite embedding row with the host's
// BF16 refusal without rounding anything on the host or dispatching.
// fak-test:runtime fast est=1s lane=default
func TestV41MHCResidentNonfiniteRowRefusesWithoutHostRound(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	b := newV41MHCTestBackend(true)
	s := v41DenseTestSession(t, m, b)
	d := s.v41State().mhcDevice
	if d == nil {
		t.Fatal("mHC seam not bound")
	}
	row := make([]float32, m.Cfg.HiddenSize)
	row[1] = float32(math.NaN())
	_, hostErr := v41FullInitialStreams(row)
	var err error
	rounds := v41BF16PublishHostRounds(t, func() { _, err = d.newResident([][]float32{row}) })
	var hostStage, stage *V41ForwardError
	if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &stage) || !errors.As(hostErr, &hostStage) ||
		stage.Stage != hostStage.Stage || stage.Layer != hostStage.Layer {
		t.Fatalf("resident non-finite row = %v, want the host BF16 refusal %v", err, hostErr)
	}
	if rounds != 0 || len(b.calls) != 0 {
		t.Fatalf("refusal made %d host BF16 rounds and dispatched %v", rounds, b.calls)
	}
}

// A grouped-output backend whose BF16 publication admits the hidden and input
// widths but not the rank width.
type v41GroupedPublishGapBackend struct {
	*v41GroupedBackend
	rank      int
	publishes int
}

func (b *v41GroupedPublishGapBackend) SupportsV41BF16Publish(n int) bool { return n > 0 && n != b.rank }
func (b *v41GroupedPublishGapBackend) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	b.publishes++
	values := append([]float32(nil), b.v41DenseTestBackend.Backend.Read(x)...)
	for i, v := range values {
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(v)))
	}
	return compute.NewF32(b, []int{len(values)}, values), nil
}
func (b *v41GroupedPublishGapBackend) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 BF16 publication input" {
		return b.v41GroupedBackend.Upload(x, dt)
	}
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.v41DenseTestBackend.Backend.Read(x)...))
}

// Under device-only a full grouped projection whose publication declines its
// width refuses typed instead of rounding the readback on the host.
// fak-test:runtime fast est=1s lane=default
func TestV41FullGroupedDeviceOnlyRefusesUnpublishedWidth(t *testing.T) {
	m := v41GroupedBF16Fixture(t)
	c := m.Cfg
	totalIn := c.NumHeads * c.HeadDim
	if c.OLoraRank == totalIn || c.OLoraRank == c.HiddenSize {
		t.Fatalf("fixture rank %d equals a published width (input %d, hidden %d); the witness would be vacuous", c.OLoraRank, totalIn, c.HiddenSize)
	}
	b := &v41GroupedPublishGapBackend{v41GroupedBackend: newV41GroupedBackend(t, m), rank: c.OLoraRank}
	s := v41DenseTestSession(t, m, b)
	if s.v41BF16PublishFunc() == nil {
		t.Fatal("publication not bound")
	}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	project := m.v41GroupedOutputProjector(0, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize, &v41ProjScratch{groupedOutput: s.v41GroupedOutputFunc(), gatherOps: s.v41State().gatherOps})
	input := make([]float32, totalIn)
	// A BF16 tie, so the input publication runs on the device.
	input[0], input[1] = 1, math.Float32frombits(0x3f808000)
	var got []float32
	var err error
	rounds := v41BF16PublishHostRounds(t, func() { got, err = project(input) })
	if got != nil || !errors.Is(err, errV41DeviceOnlyHostOp) || !s.BackendSessionClosed() {
		t.Fatalf("device-only grouped projection = %v closed=%t, want the typed bf16-boundary refusal", err, s.BackendSessionClosed())
	}
	if rounds != 0 || b.publishes != 1 || b.matmulAttempts != 1 {
		t.Fatalf("host_rounds=%d publications=%d matmuls=%d, want 0 rounds after the input publication and first group", rounds, b.publishes, b.matmulAttempts)
	}
}
