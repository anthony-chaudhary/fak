package agent

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ErrKVPrecisionUnrealized is the closed refusal for a --kv-precision tier the
// planner's forward path cannot store. It is checked at planner construction so a
// serve refuses at startup instead of crash-looping on the first request.
var ErrKVPrecisionUnrealized = errors.New("agent: kv precision not realized by the forward path")

// Forward-path names carried by KVPrecisionUnrealizedError.Path.
const (
	KVPrecisionPathHost      = "host"
	KVPrecisionPathDeviceHAL = "device-hal"
	KVPrecisionPathMetal     = "metal"
)

// KVPrecisionUnrealizedError names the refused tier, the forward path, and why.
type KVPrecisionUnrealizedError struct {
	Precision model.KVPrecision
	Path      string
	Reason    string
}

func (e *KVPrecisionUnrealizedError) Error() string {
	return fmt.Sprintf("--kv-precision=%s is not realized on the %s path (%s); use f32", e.Precision, e.Path, e.Reason)
}

func (e *KVPrecisionUnrealizedError) Unwrap() error { return ErrKVPrecisionUnrealized }

// ValidateKVPrecision reports whether a planner over cfg on (backend, metal) can
// realize prec. f32 (or unset) always passes. fp16/bf16 on the device HAL pass
// (realized by the CUDA store, else f32 via pickDeviceKVTier). q8_0 is realized only
// by the host KVCache on architectures whose forward goes through its precision-aware
// accessors; device HAL and Metal stores append f32 rows, so q8 there would silently
// mix f32 and packed rows. Other host tiers are declared but not realized yet.
func ValidateKVPrecision(cfg model.Config, prec model.KVPrecision, backend compute.Backend, metal bool) error {
	if prec == "" || prec == model.KVPrecisionFP32 {
		return nil
	}
	path := KVPrecisionPathHost
	switch {
	case backend != nil:
		path = KVPrecisionPathDeviceHAL
	case metal:
		path = KVPrecisionPathMetal
	}
	refuse := func(reason string) error {
		return &KVPrecisionUnrealizedError{Precision: prec, Path: path, Reason: reason}
	}
	if path == KVPrecisionPathDeviceHAL && deviceKV16(prec) {
		// The HAL store realizes 16-bit K/V where the backend supports it;
		// pickDeviceKVTier falls back to f32 elsewhere, never mixing rows.
		return nil
	}
	if prec != model.KVPrecisionQ8_0 {
		return refuse("tier declared but not realized by the native KV cache")
	}
	if reason := cfg.QuantizedKVUnsupportedReason(); reason != "" {
		return refuse(reason)
	}
	if path != KVPrecisionPathHost {
		return refuse("device KV store appends f32 rows")
	}
	return nil
}

// KVPrecisionConfigError is the construction-time verdict of ValidateKVPrecision for
// this planner; nil when the configured tier is realized.
func (p *InKernelPlanner) KVPrecisionConfigError() error {
	if p == nil {
		return nil
	}
	return p.kvPrecisionErr
}

func plannerKVPrecisionError(m *model.Model, prec model.KVPrecision, backend compute.Backend, metal bool) error {
	var cfg model.Config
	if m != nil {
		cfg = m.Cfg
	}
	return ValidateKVPrecision(cfg, prec, backend, metal)
}
