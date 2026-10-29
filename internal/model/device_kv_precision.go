package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrDeviceKVPrecisionUnrealized refuses a device KV tier the session's HAL store cannot
// hold for this geometry. Callers fall back to f32 rather than mixing row encodings.
var ErrDeviceKVPrecisionUnrealized = errors.New("model: device kv precision not realized")

// cudaDeviceKVBackend is the HAL backend name whose KVStore stores 16-bit attended K/V
// (compute/cuda_kv.go NewKV). Every other HAL store panics on a non-f32 KVConfig.
const cudaDeviceKVBackend = "cuda"

// DeviceKVPrecision maps a serve KV tier to the compute tier a HAL KVStore holds.
// ok is false for host-cache-only tiers (q8_0, q4_0).
func DeviceKVPrecision(prec KVPrecision) (compute.KVPrecision, bool) {
	switch prec {
	case "", KVPrecisionFP32:
		return compute.KVPrecisionF32, true
	case KVPrecisionFP16:
		return compute.KVPrecisionF16, true
	case KVPrecisionBF16:
		return compute.KVPrecisionBF16, true
	default:
		return compute.KVPrecisionF32, false
	}
}

// SetDeviceKVPrecision re-creates this HAL session's still-empty KV store at prec so every
// append and attention read goes through the typed store. It refuses (leaving the f32 store
// in place) when the backend or geometry cannot realize prec or the store already holds rows.
func (s *Session) SetDeviceKVPrecision(prec KVPrecision) error {
	if s == nil || s.Backend == nil || s.halKV == nil || s.M == nil {
		return fmt.Errorf("%w: session has no HAL KV store", ErrDeviceKVPrecisionUnrealized)
	}
	if reason := s.M.Cfg.DeviceKVRefusal(s.Backend.Name(), prec); reason != "" {
		return fmt.Errorf("%w: %s", ErrDeviceKVPrecisionUnrealized, reason)
	}
	dev, _ := DeviceKVPrecision(prec)
	if dev == s.halKVPrecision {
		return nil
	}
	if s.halKV.Len() != 0 || s.graphCaptured {
		return fmt.Errorf("%w: HAL KV store already holds %d positions", ErrDeviceKVPrecisionUnrealized, s.halKV.Len())
	}
	next := newHALKVStoreAt(s.Backend, s.M.Cfg, dev)
	if next == nil {
		return fmt.Errorf("%w: backend %q returned no KV store", ErrDeviceKVPrecisionUnrealized, s.Backend.Name())
	}
	s.halKV.Free()
	s.halKV, s.halKVPrecision = next, dev
	return nil
}

// HALKVPrecision reports the compute tier this session's HAL KV store was created at.
func (s *Session) HALKVPrecision() compute.KVPrecision {
	if s == nil {
		return compute.KVPrecisionF32
	}
	return s.halKVPrecision
}
