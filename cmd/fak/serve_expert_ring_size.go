package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_expert_ring_size.go — the operator knob that sizes the activated-expert device ring
// (--expert-ring-bytes / FAK_EXPERT_RING_BYTES).
//
//	token    the default: one token's activated experts, exactly the historical ring.
//	auto     the smaller of a device bound and a host bound, after the dense base, the transient
//	         scratch and the KV for the target context are reserved, and never more than
//	         FAK_EXPERT_RING_POOL_FRACTION of host MemTotal. The default is 0.50 when the ring
//	         pages live in mapped GTT (integrated, known carve-out <= 4 GiB) and 0.10 otherwise,
//	         where DEVICE_LOCAL pages can fill a small carve-out and hit the driver's BO_VA
//	         ceiling mid-request. The routed band beyond the ring stays SSD-backed.
//	<bytes>  an explicit ring, admitted or refused at load.
//
// On a UMA box (shared host RAM, carve-out <= serveExpertRingUMACarveoutMax) the device ring,
// the page cache and every other plan row draw on one pool, so auto and explicit sizes are
// judged against MemAvailable as that pool. A size that does not fit refuses at load with a
// typed *compute.FitError that also matches ErrServeExpertRingTooBig, never mid-request.

// FakExpertRingBytesEnv is the environment twin of --expert-ring-bytes.
const FakExpertRingBytesEnv = "FAK_EXPERT_RING_BYTES"

// FakExpertRingPoolFractionEnv caps an auto ring at this fraction of host MemTotal (0 < f <= 0.5).
const FakExpertRingPoolFractionEnv = "FAK_EXPERT_RING_POOL_FRACTION"

// ErrServeExpertRingTooBig marks a ring size the load-time memory bound refused.
var ErrServeExpertRingTooBig = errors.New("fak serve: activated-expert device ring does not fit")

// ErrServeExpertRingSize marks an --expert-ring-bytes value that cannot name a ring.
var ErrServeExpertRingSize = errors.New("fak serve: invalid --expert-ring-bytes")

const (
	serveExpertRingHostMarginFloor    int64   = 6 << 30
	serveExpertRingHostMarginFraction float64 = 0.05
	serveExpertRingUMACarveoutMax     int64   = 4 << 30
	serveExpertRingPoolFractionDef    float64 = 0.10
	serveExpertRingPoolFractionGTT    float64 = 0.5
	serveExpertRingPoolFractionMax    float64 = 0.5
)

type serveExpertRingMode string

const (
	serveExpertRingToken    serveExpertRingMode = "token"
	serveExpertRingAuto     serveExpertRingMode = "auto"
	serveExpertRingExplicit serveExpertRingMode = "explicit"
)

type serveExpertRingSize struct {
	Mode  serveExpertRingMode
	Bytes int64
	// PoolFraction caps an auto ring at this fraction of host MemTotal; 0 means the default.
	PoolFraction float64
}

func (s serveExpertRingSize) String() string {
	if s.Mode == serveExpertRingExplicit {
		return strconv.FormatInt(s.Bytes, 10)
	}
	if s.Mode == "" {
		return string(serveExpertRingToken)
	}
	return string(s.Mode)
}

// parseServeExpertRingSize accepts token, auto, or a positive byte count with an optional
// binary suffix (K, M, G, T; KiB/KB style spellings are the same powers of 1024).
func parseServeExpertRingSize(raw string) (serveExpertRingSize, error) {
	v := strings.ToLower(strings.TrimSpace(raw))
	switch v {
	case "", string(serveExpertRingToken):
		return serveExpertRingSize{Mode: serveExpertRingToken}, nil
	case string(serveExpertRingAuto):
		return serveExpertRingSize{Mode: serveExpertRingAuto}, nil
	}
	shift := 0
	for _, unit := range []struct {
		suffix string
		shift  int
	}{{"tib", 40}, {"gib", 30}, {"mib", 20}, {"kib", 10}, {"tb", 40}, {"gb", 30}, {"mb", 20}, {"kb", 10}, {"t", 40}, {"g", 30}, {"m", 20}, {"k", 10}, {"b", 0}} {
		if strings.HasSuffix(v, unit.suffix) {
			v, shift = strings.TrimSpace(strings.TrimSuffix(v, unit.suffix)), unit.shift
			break
		}
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil || n <= 0 || n > math.MaxInt64>>shift {
		return serveExpertRingSize{}, fmt.Errorf("%w %q: want token, auto, or a positive byte count", ErrServeExpertRingSize, raw)
	}
	return serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: n << shift}, nil
}

// resolveServeExpertRingSize lets a non-empty flag select the mode over the environment;
// auto uses the pool fraction regardless of how it was selected.
func resolveServeExpertRingSize(flagValue string, getenv func(string) string) (serveExpertRingSize, error) {
	if strings.TrimSpace(flagValue) == "" && getenv != nil {
		flagValue = getenv(FakExpertRingBytesEnv)
	}
	size, err := parseServeExpertRingSize(flagValue)
	if err != nil || size.Mode != serveExpertRingAuto || getenv == nil {
		return size, err
	}
	size.PoolFraction, err = parseServeExpertRingPoolFraction(getenv(FakExpertRingPoolFractionEnv))
	return size, err
}

// parseServeExpertRingPoolFraction returns 0 (the host tier's default) for an empty value and
// refuses anything outside (0, 0.5] with ErrServeExpertRingSize, matching a malformed
// FAK_EXPERT_RING_BYTES.
func parseServeExpertRingPoolFraction(raw string) (float64, error) {
	v := strings.TrimSpace(raw)
	if v == "" {
		return 0, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || !(f > 0 && f <= serveExpertRingPoolFractionMax) {
		return 0, fmt.Errorf("%w: %s=%q: want a fraction in (0, %g]", ErrServeExpertRingSize, FakExpertRingPoolFractionEnv, raw, serveExpertRingPoolFractionMax)
	}
	return f, nil
}

// serveExpertRingHost is the host-side snapshot a non-token size is judged against.
type serveExpertRingHost struct {
	Total, Avail int64
	Known        bool
	// RSSAnonPeak is the estimated anonymous host peak of the load route (ggufload.HostLoadPeak).
	RSSAnonPeak int64
	// SharedPool: device allocations draw on MemAvailable (integrated tier, carve-out <= 4 GiB or
	// unreadable).
	SharedPool    bool
	Carveout      int64
	CarveoutKnown bool
}

// serveExpertRingDefaultPoolFraction mirrors compute's host-visible ring-page policy: there the
// ring never fills the carve-out, and a 64.2 GiB auto ring served on strix3 (512 MiB carve-out).
func serveExpertRingDefaultPoolFraction(host serveExpertRingHost) float64 {
	if host.SharedPool && host.CarveoutKnown && host.Carveout > 0 && host.Carveout <= serveExpertRingUMACarveoutMax {
		return serveExpertRingPoolFractionGTT
	}
	return serveExpertRingPoolFractionDef
}

func serveExpertRingHostMargin(total int64) int64 {
	return max(serveExpertRingHostMarginFloor, int64(float64(max(total, 0))*serveExpertRingHostMarginFraction))
}

// serveExpertRingBinding names the bound that set an auto ring, so a startup log shows which
// live reading (device ceiling, host MemAvailable, pool cap, routed band, floor or the load-time
// guard's live budget) chose the size instead of leaving it to be inferred from the inputs.
type serveExpertRingBinding string

const (
	serveExpertRingBindingNone   serveExpertRingBinding = "none"
	serveExpertRingBindingDevice serveExpertRingBinding = "device"
	serveExpertRingBindingHost   serveExpertRingBinding = "host"
	serveExpertRingBindingPool   serveExpertRingBinding = "pool"
	serveExpertRingBindingBand   serveExpertRingBinding = "band"
	serveExpertRingBindingFloor  serveExpertRingBinding = "floor"
	serveExpertRingBindingGuard  serveExpertRingBinding = "guard"
)

// serveExpertRingSizing is the size choice and every input it was made from, for the log line.
type serveExpertRingSizing struct {
	Mode serveExpertRingMode
	// Binding is the bound an auto ring took; none for token, explicit and unmeasured auto.
	Binding    serveExpertRingBinding
	RingBytes  int64
	TokenBytes int64
	Floor      int64
	Ceiling    int64
	// DeviceBound / HostBound are the room each bound leaves the ring; Known=false when the
	// bound could not be measured and did not constrain the choice.
	DeviceBound, HostBound           int64
	DeviceBoundKnown, HostBoundKnown bool
	DeviceAvail                      int64
	NonRingDevice, NonRingHost       int64
	HostAvail, HostMargin            int64
	RSSAnonPeak                      int64
	HostGatheredBytes                int64
	HostGatheredReserve              int64
	SharedPool                       bool
	// GuardTrimBytes is how much the load-time guard trimmed an auto ring below the bounds.
	GuardTrimBytes int64
	// PoolCapBytes = PoolFraction x host MemTotal bounds an auto ring; 0 when not applied.
	PoolFraction float64
	PoolCapBytes int64
}

// sizeServeExpertRing picks the ring for a non-token mode from the plan the token ring was
// admitted in (nonRing: every plan row except the ring). Token mode returns tokenRing unchanged.
func sizeServeExpertRing(size serveExpertRingSize, f ggufload.ActivatedExpertFit, tokenRing int64, nonRing compute.MemoryPlan, deviceAvail int64, host serveExpertRingHost) (serveExpertRingSizing, error) {
	s := serveExpertRingSizing{
		Mode: size.Mode, RingBytes: tokenRing, TokenBytes: tokenRing,
		Floor: f.ActivatedLayerBytes, Ceiling: f.RoutedBandBytes,
		DeviceAvail: deviceAvail, NonRingDevice: nonRing.DeviceTotal(), NonRingHost: nonRing.HostTotal(),
		HostAvail: host.Avail, RSSAnonPeak: max(host.RSSAnonPeak, 0),
		HostGatheredBytes: f.HostGatheredBytes, HostGatheredReserve: f.HostGatheredPageReserve(),
		SharedPool: host.SharedPool,
	}
	if s.Mode == "" {
		s.Mode = serveExpertRingToken
	}
	s.Binding = serveExpertRingBindingNone
	if s.Mode == serveExpertRingToken {
		return s, nil
	}
	if deviceAvail > 0 {
		s.DeviceBound, s.DeviceBoundKnown = deviceAvail-s.NonRingDevice, true
	}
	if host.Known && host.Avail > 0 {
		s.HostMargin = serveExpertRingHostMargin(host.Total)
		// The host-gathered page reserve is already a host-scoped row of nonRing.
		room := host.Avail - s.RSSAnonPeak - s.HostMargin - s.NonRingHost
		if host.SharedPool {
			room -= s.NonRingDevice
		}
		s.HostBound, s.HostBoundKnown = room, true
	}
	switch s.Mode {
	case serveExpertRingAuto:
		if !s.DeviceBoundKnown && !s.HostBoundKnown {
			// Nothing measurable to size against: keep the admitted token ring.
			return s, nil
		}
		ring, binding := int64(math.MaxInt64), serveExpertRingBindingNone
		take := func(bound int64, b serveExpertRingBinding) {
			if bound < ring {
				ring, binding = bound, b
			}
		}
		if s.DeviceBoundKnown {
			take(s.DeviceBound, serveExpertRingBindingDevice)
		}
		if s.HostBoundKnown {
			take(s.HostBound, serveExpertRingBindingHost)
		}
		if host.Known && host.Total > 0 {
			s.PoolFraction = size.PoolFraction
			if s.PoolFraction <= 0 {
				s.PoolFraction = serveExpertRingDefaultPoolFraction(host)
			}
			s.PoolCapBytes = int64(s.PoolFraction * float64(host.Total))
			take(s.PoolCapBytes, serveExpertRingBindingPool)
		}
		take(s.Ceiling, serveExpertRingBindingBand)
		s.RingBytes = ring
		if s.RingBytes < s.Floor {
			s.RingBytes, binding = s.Floor, serveExpertRingBindingFloor
		}
		s.Binding = binding
	case serveExpertRingExplicit:
		s.RingBytes = min(size.Bytes, s.Ceiling)
	default:
		return s, fmt.Errorf("%w: mode %q", ErrServeExpertRingSize, size.Mode)
	}
	if s.RingBytes < s.Floor {
		return s, fmt.Errorf("%w: %d bytes is below one layer's activated experts (%d bytes)", ErrServeExpertRingSize, s.RingBytes, s.Floor)
	}
	return s, s.refuseIfOverBound(nonRing)
}

// refuseIfOverBound returns the typed refusal for a ring larger than a measured bound.
func (s serveExpertRingSizing) refuseIfOverBound(nonRing compute.MemoryPlan) error {
	plan := append(append(compute.MemoryPlan(nil), nonRing...), compute.MemoryDemand{Class: compute.MemoryWeights, Bytes: s.RingBytes, Detail: serveActivatedRingDetail, Scope: compute.MemoryScopeDevice})
	if s.DeviceBoundKnown && s.RingBytes > s.DeviceBound {
		return serveExpertRingTooBig(s, &compute.FitError{Verdict: compute.FitTooBig, Want: s.NonRingDevice + s.RingBytes, Avail: s.DeviceAvail, Demands: plan, Scope: compute.MemoryScopeDevice})
	}
	if s.HostBoundKnown && s.RingBytes > s.HostBound {
		want := s.RingBytes + s.NonRingHost + s.RSSAnonPeak
		if s.SharedPool {
			want += s.NonRingDevice
		}
		return serveExpertRingTooBig(s, &compute.FitError{Verdict: compute.FitTooBig, Want: want, Avail: s.HostAvail - s.HostMargin, Demands: plan, Scope: compute.MemoryScopeHost})
	}
	return nil
}

func serveExpertRingTooBig(s serveExpertRingSizing, fe *compute.FitError) error {
	return fmt.Errorf("%w (mode=%s ring=%d bytes): %w", ErrServeExpertRingTooBig, s.Mode, s.RingBytes, fe)
}

// Text renders the inputs of the size choice for the startup log line.
func (s serveExpertRingSizing) Text() string {
	bound := func(v int64, known bool) string {
		if !known {
			return "unmeasured"
		}
		return bytesText(uint64(max(v, 0)))
	}
	mode := s.Mode
	if mode == "" {
		mode = serveExpertRingToken
	}
	binding := s.Binding
	if binding == "" {
		binding = serveExpertRingBindingNone
	}
	return fmt.Sprintf("ring size mode=%s binding=%s chosen=%s token=%s floor=%s band=%s device_bound=%s (device_avail=%s non_ring_device=%s) host_bound=%s (mem_available=%s rss_anon_peak=%s margin=%s non_ring_host=%s shared_pool=%t) pool_cap=%s guard_trim=%s host_gathered=%s page_reserve=%s",
		mode, binding, bytesText(uint64(max(s.RingBytes, 0))), bytesText(uint64(max(s.TokenBytes, 0))),
		bytesText(uint64(max(s.Floor, 0))), bytesText(uint64(max(s.Ceiling, 0))),
		bound(s.DeviceBound, s.DeviceBoundKnown), bytesText(uint64(max(s.DeviceAvail, 0))), bytesText(uint64(max(s.NonRingDevice, 0))),
		bound(s.HostBound, s.HostBoundKnown), bytesText(uint64(max(s.HostAvail, 0))), bytesText(uint64(s.RSSAnonPeak)),
		bytesText(uint64(max(s.HostMargin, 0))), bytesText(uint64(max(s.NonRingHost, 0))), s.SharedPool,
		s.poolCapText(),
		bytesText(uint64(max(s.GuardTrimBytes, 0))),
		bytesText(uint64(max(s.HostGatheredBytes, 0))), bytesText(uint64(max(s.HostGatheredReserve, 0))))
}

func (s serveExpertRingSizing) poolCapText() string {
	if s.PoolCapBytes <= 0 {
		return "none"
	}
	return fmt.Sprintf("%d (fraction=%.2f)", s.PoolCapBytes, s.PoolFraction)
}

// serveExpertRingBytesFlag is the flag name, kept in one place so registration and refusals agree.
const serveExpertRingBytesFlag = "expert-ring-bytes"

// applyServeExpertRingBytes validates the flag before the load and carries it to the load path
// through FakExpertRingBytesEnv, normalized, so the flag wins over a pre-set environment value.
func applyServeExpertRingBytes(v string) error {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	size, err := parseServeExpertRingSize(v)
	if err != nil {
		return fmt.Errorf("fak serve: --%s: %w", serveExpertRingBytesFlag, err)
	}
	return os.Setenv(FakExpertRingBytesEnv, size.String())
}

// enableServeSharedExpertRing gives the loaded model ONE routed-expert ring for the serve's
// backend, so every request attaches to the same residency instead of building and freeing a
// fresh ring: an expert paged in by request N is a hit for request N+1, and concurrent sessions
// share one budget instead of overcommitting it N times.
func enableServeSharedExpertRing(m *fakmodel.Model, be compute.Backend) error {
	if m == nil || be == nil {
		return nil
	}
	if _, err := m.EnableSharedExpertRing(be); err != nil {
		return fmt.Errorf("fak serve: model-scoped activated-expert ring: %w", err)
	}
	return nil
}
