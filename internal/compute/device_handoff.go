package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
)

// device_handoff.go — the cross-device hidden-state handoff between two layer bands
// that run on different CUDA devices in one process. Only the residual hidden state
// crosses a band boundary (model.PlanDeviceBands keeps KV and recurrent state on the
// band's own device), so the handoff is a plain F32 byte copy: it must be bit-exact.
//
// Route selection is portable and host-tested here; the CUDA backend executes it
// (cuda_device_handoff.go) with cudaMemcpyPeer for a peer route and a pinned host
// staging buffer for the bounce route. FAK_CUDA_HANDOFF=bounce forces the bounce route
// so an operator can A/B the two on hardware where peer access is reported.

// HandoffRoute is how a hidden state moves from the source band's device to the next.
type HandoffRoute uint8

const (
	HandoffRouteUnknown HandoffRoute = iota
	// HandoffSameDevice: both bands share a device; the copy is device-local.
	HandoffSameDevice
	// HandoffPeer: direct device-to-device copy (cudaMemcpyPeer) with peer access on.
	HandoffPeer
	// HandoffHostBounce: device-to-host into pinned staging, then host-to-device.
	HandoffHostBounce
)

func (r HandoffRoute) String() string {
	switch r {
	case HandoffSameDevice:
		return "same-device"
	case HandoffPeer:
		return "peer"
	case HandoffHostBounce:
		return "host-bounce"
	default:
		return "unknown"
	}
}

// Sentinel errors for handoff callers (errors.Is).
var (
	ErrHandoffInvalid = errors.New("compute: invalid device handoff")
	ErrHandoffFailed  = errors.New("compute: device handoff failed")
)

// DeviceHandoffReceipt records one boundary crossing.
type DeviceHandoffReceipt struct {
	Src    int          `json:"src"`
	Dst    int          `json:"dst"`
	Route  HandoffRoute `json:"route"`
	Bytes  int64        `json:"bytes"`
	Chunks int          `json:"chunks"` // staging chunks for the bounce route, else 1
}

// PeerAccessProbe reports whether device src can copy straight into device dst.
type PeerAccessProbe func(src, dst int) (bool, error)

// HandoffForceBounceFromEnv reads FAK_CUDA_HANDOFF: "bounce" forces the host route.
func HandoffForceBounceFromEnv() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("FAK_CUDA_HANDOFF")), "bounce")
}

// SelectHandoffRoute picks the route for one src->dst crossing: same-device when the
// ordinals match, peer when the probe admits it (and bounce is not forced), else bounce.
// A probe error does not fall back silently; it is returned.
func SelectHandoffRoute(src, dst int, probe PeerAccessProbe, forceBounce bool) (HandoffRoute, error) {
	if src < 0 || dst < 0 {
		return HandoffRouteUnknown, fmt.Errorf("%w: device ordinals src=%d dst=%d", ErrHandoffInvalid, src, dst)
	}
	if src == dst {
		return HandoffSameDevice, nil
	}
	if forceBounce || probe == nil {
		return HandoffHostBounce, nil
	}
	ok, err := probe(src, dst)
	if err != nil {
		return HandoffRouteUnknown, fmt.Errorf("%w: peer probe %d->%d: %v", ErrHandoffFailed, src, dst, err)
	}
	if ok {
		return HandoffPeer, nil
	}
	return HandoffHostBounce, nil
}

// ByteRange is a half-open [Off, Off+Len) span of a buffer.
type ByteRange struct {
	Off, Len int
}

// HostBounceChunks splits an n-byte copy into spans no larger than the staging buffer.
// n == 0 yields no chunks; a non-positive staging size is invalid.
func HostBounceChunks(n, staging int) ([]ByteRange, error) {
	if n < 0 || staging <= 0 {
		return nil, fmt.Errorf("%w: bounce of %d bytes through %d-byte staging", ErrHandoffInvalid, n, staging)
	}
	out := make([]ByteRange, 0, (n+staging-1)/staging)
	for off := 0; off < n; off += staging {
		out = append(out, ByteRange{Off: off, Len: min(staging, n-off)})
	}
	return out, nil
}

// HiddenStateHop moves one band's F32 hidden state from device src to device dst and
// returns what the receiving band reads. It must be bit-exact.
type HiddenStateHop interface {
	HopF32(src []float32, srcDev, dstDev int) ([]float32, DeviceHandoffReceipt, error)
}

// HostHop is the portable HiddenStateHop: it runs the same route selection and staging
// chunk plan the CUDA hop runs, with host slices standing in for device buffers. It is
// the host witness that the route and chunk logic preserve every bit.
type HostHop struct {
	Probe        PeerAccessProbe
	ForceBounce  bool
	StagingBytes int // bounce staging size; 0 selects DefaultHandoffStagingBytes
}

// DefaultHandoffStagingBytes is the pinned staging size per device pair: 4 MiB holds a
// 200-token Qwen3.8 prompt band (5120 x 4 B x 200) in one chunk.
const DefaultHandoffStagingBytes = 4 << 20

// HopF32 copies src through the selected route. The bounce route copies through one
// staging buffer chunk by chunk, as the device path does.
func (h HostHop) HopF32(src []float32, srcDev, dstDev int) ([]float32, DeviceHandoffReceipt, error) {
	route, err := SelectHandoffRoute(srcDev, dstDev, h.Probe, h.ForceBounce)
	if err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}
	rec := DeviceHandoffReceipt{Src: srcDev, Dst: dstDev, Route: route, Bytes: int64(len(src)) * 4, Chunks: 1}
	out := make([]float32, len(src))
	if route != HandoffHostBounce {
		copy(out, src)
		return out, rec, nil
	}
	staging := h.StagingBytes
	if staging == 0 {
		staging = DefaultHandoffStagingBytes
	}
	staging -= staging % 4 // whole words per chunk
	chunks, err := HostBounceChunks(len(src)*4, staging)
	if err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}
	stage := make([]uint32, staging/4)
	for _, c := range chunks {
		lo, n := c.Off/4, c.Len/4
		for i := 0; i < n; i++ { // D2H into staging
			stage[i] = math.Float32bits(src[lo+i])
		}
		for i := 0; i < n; i++ { // H2D out of staging
			out[lo+i] = math.Float32frombits(stage[i])
		}
	}
	rec.Chunks = len(chunks)
	return out, rec, nil
}
