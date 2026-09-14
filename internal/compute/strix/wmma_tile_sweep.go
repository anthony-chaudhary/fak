package strix

import (
	"fmt"
	"time"
)

// Model-only roofline efficiency parameters.
//
// These two constants feed the asserted roofline in wave32_wmma.go's buildTelemetry
// and buildAttentionTelemetry. They are MODEL-ONLY: they were chosen to reproduce an
// expected roofline shape and are NOT measured against actual gfx1151 silicon. They
// stay in the tree until a physical device kernel witnesses the real value, at which
// point a measured sweep result should replace them. Any telemetry derived from them
// must carry EfficiencyLabelSourceModelOnly, never "measured".
const (
	// ModelOnlyWave32ComputeEfficiency is a MODEL-ONLY roofline parameter: the fraction
	// of peak WMMA BF16 throughput assumed to be achievable in practice. NOT measured on
	// gfx1151 hardware.
	ModelOnlyWave32ComputeEfficiency = 0.85

	// ModelOnlySustainedBWEfficiency is a MODEL-ONLY roofline parameter: the fraction of
	// peak unified-bus bandwidth assumed sustainable. NOT measured on gfx1151 hardware.
	ModelOnlySustainedBWEfficiency = 0.80
)

// Efficiency label values used by the WMMA tile sweep to state provenance honestly.
const (
	// EfficiencyLabelMeasured marks a TFLOPs figure computed from wall-clock execution
	// of a real device kernel.
	EfficiencyLabelMeasured = "measured"

	// EfficiencyLabelModelOnly marks a value that is an asserted model parameter, not a
	// measurement of the hardware.
	EfficiencyLabelModelOnly = "model-only"
)

// Execution-class labels distinguishing a real device command buffer dispatch from a
// pure-Go simulation and from an unavailable device path. Kept as plain strings so a
// sweep row stays decoupled from the EvidenceClass type used by RunWMMAMeasured.
const (
	// SweepExecutionClassDeviceSimulated marks the pure-Go simulator path: numerically
	// modeled, never dispatched to a GPU.
	SweepExecutionClassDeviceSimulated = "simulated"

	// SweepExecutionClassDeviceUnavailable marks the case where neither a real device
	// kernel nor a simulator run was possible.
	SweepExecutionClassDeviceUnavailable = "unavailable"

	// SweepExecutionClassDeviceMeasured marks a real, timed device-kernel execution.
	SweepExecutionClassDeviceMeasured = "measured"
)

// EfficiencyLabelSource returns "measured" when a sweep result was produced from a real
// timed device kernel, and "model-only" otherwise. It is the single provenance helper the
// telemetry use sites and tests should call instead of hardcoding a literal.
func EfficiencyLabelSource(measured bool) string {
	if measured {
		return EfficiencyLabelMeasured
	}
	return EfficiencyLabelModelOnly
}

// WMMATileSweepResult is one row of a WMMA tile-geometry sweep. MeasuredTFLOPs is the
// ACTUAL measured throughput and is zero whenever no real device kernel ran, in which
// case EfficiencyLabel is "model-only".
type WMMATileSweepResult struct {
	TileM           int
	TileN           int
	TileK           int
	Primitive       string
	TotalFLOPs      float64
	Elapsed         time.Duration
	MeasuredTFLOPs  float64
	ExecutionClass  string
	EfficiencyLabel string
}

// WMMADeviceKernelFn runs one real device kernel for the given tile geometry and returns
// the wall-clock duration it actually took. Implementations must execute the opcode on a
// device; a nil function means no device kernel is available.
type WMMADeviceKernelFn func(M, N, K, tileM, tileN, tileK int) (time.Duration, error)

// WMMASweepOptions controls a tile sweep.
type WMMASweepOptions struct {
	// DeviceKernel, when non-nil, is the real on-device kernel runner. Its returned
	// duration is the measured wall clock and drives MeasuredTFLOPs.
	DeviceKernel WMMADeviceKernelFn

	// ForceNoDevice, when true, suppresses DeviceKernel even if one is supplied so the
	// caller can witness the honest model-only path.
	ForceNoDevice bool

	// Simulate, when true and no device kernel is used, still runs the pure-Go simulator
	// for numerical correctness. The result is still labeled "model-only".
	Simulate bool
}

// sweepGeometries returns the tile geometries swept by SweepWMMABFTiles: the 16x16x16 and
// 16x16x32 WMMA primitives.
func sweepGeometries() []Wave32WMMAConfig {
	return []Wave32WMMAConfig{
		DefaultWave32WMMAConfig(),
		Wave32WMMAConfig16x16x32(),
	}
}

// SweepWMMABFTiles measures actual BF16 WMMA throughput across the 16x16x16 and 16x16x32
// tile geometries for an M x N x K problem.
//
// When opts supplies a device kernel it is timed and MeasuredTFLOPs is derived from the
// real elapsed duration. When no device kernel is available (or ForceNoDevice is set) the
// result is honestly labeled "model-only" with MeasuredTFLOPs == 0; the simulator may
// still run when opts.Simulate is set but its output is never reported as measured.
func SweepWMMABFTiles(M, N, K int, opts WMMASweepOptions) ([]WMMATileSweepResult, error) {
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, ErrInvalidDimensions
	}

	useDevice := opts.DeviceKernel != nil && !opts.ForceNoDevice

	geoms := sweepGeometries()
	results := make([]WMMATileSweepResult, 0, len(geoms))

	for _, cfg := range geoms {
		if err := cfg.Validate(); err != nil {
			return nil, fmt.Errorf("strix/wmma sweep: invalid tile geometry %dx%dx%d: %w",
				cfg.TileM, cfg.TileN, cfg.TileK, err)
		}

		flops := 2.0 * float64(M) * float64(N) * float64(K)

		res := WMMATileSweepResult{
			TileM:           cfg.TileM,
			TileN:           cfg.TileN,
			TileK:           cfg.TileK,
			Primitive:       fmt.Sprintf("%dx%dx%d", cfg.TileM, cfg.TileN, cfg.TileK),
			TotalFLOPs:      flops,
			MeasuredTFLOPs:  0,
			ExecutionClass:  SweepExecutionClassDeviceUnavailable,
			EfficiencyLabel: EfficiencyLabelModelOnly,
		}

		if useDevice {
			elapsed, err := opts.DeviceKernel(M, N, K, cfg.TileM, cfg.TileN, cfg.TileK)
			if err != nil {
				return nil, fmt.Errorf("strix/wmma sweep: device kernel %s failed: %w", res.Primitive, err)
			}
			res.Elapsed = elapsed
			if elapsed > 0 {
				res.MeasuredTFLOPs = flops / elapsed.Seconds() / 1e12
			}
			res.ExecutionClass = SweepExecutionClassDeviceMeasured
			res.EfficiencyLabel = EfficiencyLabelSource(true)
			results = append(results, res)
			continue
		}

		if opts.Simulate {
			sim := NewWave32RetiledMatMul(cfg)
			a := make([]uint16, M*K)
			b := make([]uint16, K*N)
			if _, _, err := sim.MatMulBF16(M, N, K, a, b); err != nil {
				return nil, fmt.Errorf("strix/wmma sweep: simulator %s failed: %w", res.Primitive, err)
			}
			res.ExecutionClass = SweepExecutionClassDeviceSimulated
		}

		res.EfficiencyLabel = EfficiencyLabelSource(false)
		results = append(results, res)
	}

	return results, nil
}
