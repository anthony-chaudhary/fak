package l3kv

// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0. See
// https://www.apache.org/licenses/LICENSE-2.0.
// Adapted from Mooncake2d41f1a62cf49cf464a8e2acee800708f34ee4af,
// tent/src/transport/fabric/{fabric_context,fabric_transport}.cpp.
// Changes: a pure Go host planning contract with explicit residency and checked
// metadata. This does not register memory, bind CUDA, issue RDMA or own buffers.

import (
	"errors"
	"math"
)

type RegistrationMemoryClass uint8

const (
	RegistrationMemoryUnknown RegistrationMemoryClass = iota
	RegistrationMemoryHost
	RegistrationMemoryCUDA
)

type RegistrationInterface string

const (
	RegistrationInterfaceSystem RegistrationInterface = "system"
	RegistrationInterfaceCUDA   RegistrationInterface = "cuda"
)

var (
	ErrMemoryRegistrationInput    = errors.New("l3kv: invalid memory registration descriptor")
	ErrMemoryTransportUnavailable = errors.New("l3kv: memory registration transport unavailable")
	ErrDeviceMemoryUnsupported    = errors.New("l3kv: transport cannot register CUDA device memory")
)

// RegistrationBuffer is caller-supplied metadata, not proof of allocation or
// pointer residency. Address is opaque to this host component. CUDADevice must
// be -1 for host memory and nonnegative for CUDA memory (including device0).
type RegistrationBuffer struct {
	Class      RegistrationMemoryClass
	Address    uint64
	Length     uint64
	CUDADevice int32
}

// RegistrationContextCaps describes one installed transport context. The
// eventual transport must obtain these facts from its provider, not infer them
// from a device name. No probing occurs in this package.
type RegistrationContextCaps struct {
	Ready      bool
	CUDAMemory bool
}

type MemoryRegistrationCaps struct {
	CUDAEnabled bool // corresponding backend support was built/enabled
	// Contexts must be the complete installed context set, not a filtered
	// capable subset. This host contract cannot discover omitted contexts.
	Contexts []RegistrationContextCaps
}

// MemoryRegistrationPlan owns only scalar metadata. Interface is a semantic
// host-or-CUDA classification, not a libfabric enum value or registration key.
// A successful plan establishes no resource lifetime, permission or transfer.
type MemoryRegistrationPlan struct {
	Buffer       RegistrationBuffer
	Interface    RegistrationInterface
	ContextCount int
}

// PlanMemoryRegistration preserves buffer residency instead of silently staging
// device bytes through host memory. Every context must be ready, and
// CUDA memory requires both built CUDA support and device-memory capability on
// every context, as in Mooncake's aggregate hmem_ and registerMemory gates.
// Unlike its transport-dispatch skip, this closed contract explicitly refuses
// unsupported CUDA memory so a caller cannot treat a skipped registration as
// success. Actual address/ordinal validation and resource ownership remain owed
// by any future real registration implementation. Inputs are never mutated.
func PlanMemoryRegistration(buffer RegistrationBuffer, caps MemoryRegistrationCaps) (MemoryRegistrationPlan, error) {
	if buffer.Address == 0 || buffer.Length == 0 || buffer.Length > math.MaxUint64-buffer.Address {
		return MemoryRegistrationPlan{}, ErrMemoryRegistrationInput
	}
	var iface RegistrationInterface
	switch buffer.Class {
	case RegistrationMemoryHost:
		if buffer.CUDADevice != -1 {
			return MemoryRegistrationPlan{}, ErrMemoryRegistrationInput
		}
		iface = RegistrationInterfaceSystem
	case RegistrationMemoryCUDA:
		if buffer.CUDADevice < 0 {
			return MemoryRegistrationPlan{}, ErrMemoryRegistrationInput
		}
		iface = RegistrationInterfaceCUDA
	default:
		return MemoryRegistrationPlan{}, ErrMemoryRegistrationInput
	}
	if len(caps.Contexts) == 0 {
		return MemoryRegistrationPlan{}, ErrMemoryTransportUnavailable
	}
	for _, context := range caps.Contexts {
		if !context.Ready {
			return MemoryRegistrationPlan{}, ErrMemoryTransportUnavailable
		}
	}
	if buffer.Class == RegistrationMemoryCUDA {
		if !caps.CUDAEnabled {
			return MemoryRegistrationPlan{}, ErrDeviceMemoryUnsupported
		}
		for _, context := range caps.Contexts {
			if !context.CUDAMemory {
				return MemoryRegistrationPlan{}, ErrDeviceMemoryUnsupported
			}
		}
	}
	return MemoryRegistrationPlan{Buffer: buffer, Interface: iface, ContextCount: len(caps.Contexts)}, nil
}
