package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DeviceHopTransport is the StageTransport for bands that run on different devices of
// one process (a DeviceBandPlan). At each boundary it hands the hidden state from the
// previous band's device to the receiving band's device through a compute.HiddenStateHop:
// the CUDA backend's peer or pinned-bounce copy on hardware, compute.HostHop on a host.
// The resume layer stays on the host; only the F32 rows travel, so the pipeline's
// bit-exact gates hold for any hop that copies bytes faithfully.
type DeviceHopTransport struct {
	Hop compute.HiddenStateHop
	// Devices[i] is stage i's device ordinal (DeviceBandPlan.Bands[i].Device).
	Devices []int
	// Receipts records every crossing in order.
	Receipts []compute.DeviceHandoffReceipt
}

// NewDeviceHopTransport builds a transport whose stage devices follow plan.
func NewDeviceHopTransport(hop compute.HiddenStateHop, plan DeviceBandPlan) *DeviceHopTransport {
	devs := make([]int, len(plan.Bands))
	for i, b := range plan.Bands {
		devs[i] = b.Device
	}
	return &DeviceHopTransport{Hop: hop, Devices: devs}
}

// Send flattens the rows, hops them from stage dstStage-1's device to stage dstStage's
// device, and rebuilds the rows.
func (t *DeviceHopTransport) Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error) {
	if t == nil || t.Hop == nil {
		return nil, 0, fmt.Errorf("model: DeviceHopTransport into stage %d has no hop", dstStage)
	}
	if dstStage <= 0 || dstStage >= len(t.Devices) {
		return nil, 0, fmt.Errorf("model: DeviceHopTransport stage %d outside %d device stages", dstStage, len(t.Devices))
	}
	seq := len(hidden)
	width := 0
	if seq > 0 {
		width = len(hidden[0])
	}
	flat := make([]float32, 0, seq*width)
	for r, row := range hidden {
		if len(row) != width {
			return nil, 0, fmt.Errorf("model: DeviceHopTransport row %d len = %d, want uniform %d", r, len(row), width)
		}
		flat = append(flat, row...)
	}
	got, rec, err := t.Hop.HopF32(flat, t.Devices[dstStage-1], t.Devices[dstStage])
	if err != nil {
		return nil, 0, fmt.Errorf("model: DeviceHopTransport into stage %d: %w", dstStage, err)
	}
	if len(got) != len(flat) {
		return nil, 0, fmt.Errorf("model: DeviceHopTransport into stage %d returned %d floats, sent %d", dstStage, len(got), len(flat))
	}
	t.Receipts = append(t.Receipts, rec)
	out := make([][]float32, seq)
	for r := range out {
		out[r] = got[r*width : (r+1)*width : (r+1)*width]
	}
	return out, nextLo, nil
}
