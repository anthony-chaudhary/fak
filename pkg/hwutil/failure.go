// Closed DCGM health-failure field catalog for GPU receipts.
//
// Direct port of the DCGM_FR_* health-failure field set from
// NVIDIA/dcgm-exporter internal/pkg/collector/gpu_health_collector.go
// (healthCheckErrorToStringMap), pinned at
// 4e663a526c4624ad0e18ad816f09b7be2c5348a7:
// https://github.com/NVIDIA/dcgm-exporter/blob/4e663a526c4624ad0e18ad816f09b7be2c5348a7/internal/pkg/collector/gpu_health_collector.go
// The numeric values are the DCGM HealthCheckErrorCode enum as published by
// NVIDIA/go-dcgm v1.4712.2 pkg/dcgm/const.go at that pin — the same binding
// dcgm-exporter consumes — not the upstream map's source line numbers.
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
package hwutil

import (
	"errors"
	"fmt"
)

// FailureCode is one closed DCGM health-failure field name. Code strings
// cross the wire and are matched by exact value; prose around them is not
// part of the contract.
type FailureCode string

// FailureClass is the closed fault-class vocabulary grouping the catalog
// rows the way production SRE receipts group them.
type FailureClass string

const (
	// ClassXID marks driver-reported XID error events on the GPU.
	ClassXID FailureClass = "xid"
	// ClassSXID marks NVSwitch/fabric SXID error events.
	ClassSXID FailureClass = "sxid"
	// ClassRowRemap marks HBM row-remap failure states.
	ClassRowRemap FailureClass = "row_remap"
	// ClassNVLink marks NVLink CRC/BER threshold violations.
	ClassNVLink FailureClass = "nvlink"
	// ClassNVSwitch marks NVSwitch-down and fabric-manager training failures.
	ClassNVSwitch FailureClass = "nvswitch"
	// ClassBus marks the GPU fallen off the bus condition.
	ClassBus FailureClass = "bus"
	// ClassRecovery marks the GPU-recovery action rows.
	ClassRecovery FailureClass = "recovery"
)

// The closed DCGM_FR_* health-failure field set: XID, SXID, row-remap,
// NVLink CRC/BER thresholds, NVSwitch-down, fallen-off-bus, and the
// GPU-recovery actions. Adding or removing a constant is a contract change.
const (
	FailureNVLinkCRCThreshold    FailureCode = "DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD"
	FailureNVLinkCritical        FailureCode = "DCGM_FR_NVLINK_ERROR_CRITICAL"
	FailureRowRemap              FailureCode = "DCGM_FR_ROW_REMAP_FAILURE"
	FailureRowRemapUncorrectable FailureCode = "DCGM_FR_UNCORRECTABLE_ROW_REMAP"
	FailureRowRemapPending       FailureCode = "DCGM_FR_PENDING_ROW_REMAP"
	FailureNVSwitchDown          FailureCode = "DCGM_FR_NVSWITCH_NVLINK_DOWN"
	FailureXID                   FailureCode = "DCGM_FR_XID_ERROR"
	FailureSXID                  FailureCode = "DCGM_FR_SXID_ERROR"
	FailureFabricTraining        FailureCode = "DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR"
	FailureNVLinkBEREffective    FailureCode = "DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD"
	FailureFallenOffBus          FailureCode = "DCGM_FR_FALLEN_OFF_BUS"
	FailureNVLinkBERSymbol       FailureCode = "DCGM_FR_NVLINK_SYMBOL_BER_THRESHOLD"
	FailureRecoveryReset         FailureCode = "DCGM_FR_GPU_RECOVERY_RESET"
	FailureRecoveryReboot        FailureCode = "DCGM_FR_GPU_RECOVERY_REBOOT"
	FailureRecoveryDrainP2P      FailureCode = "DCGM_FR_GPU_RECOVERY_DRAIN_P2P"
	FailureRecoveryDrainReset    FailureCode = "DCGM_FR_GPU_RECOVERY_DRAIN_RESET"
)

// Failure is one catalog row: the closed code, its DCGM numeric field ID,
// and its closed class.
type Failure struct {
	Code  FailureCode
	ID    int
	Class FailureClass
}

// FailureCatalog is the closed DCGM_FR_* health-failure field set, ordered
// by DCGM field ID. Receipts record only these fields; a receipt naming any
// other DCGM failure reason is malformed against this plane.
var FailureCatalog = []Failure{
	{Code: FailureNVLinkCRCThreshold, ID: 70, Class: ClassNVLink},
	{Code: FailureNVLinkCritical, ID: 71, Class: ClassNVLink},
	{Code: FailureRowRemap, ID: 80, Class: ClassRowRemap},
	{Code: FailureRowRemapUncorrectable, ID: 84, Class: ClassRowRemap},
	{Code: FailureRowRemapPending, ID: 85, Class: ClassRowRemap},
	{Code: FailureNVSwitchDown, ID: 88, Class: ClassNVSwitch},
	{Code: FailureXID, ID: 101, Class: ClassXID},
	{Code: FailureSXID, ID: 109, Class: ClassSXID},
	{Code: FailureFabricTraining, ID: 112, Class: ClassNVSwitch},
	{Code: FailureNVLinkBEREffective, ID: 119, Class: ClassNVLink},
	{Code: FailureFallenOffBus, ID: 120, Class: ClassBus},
	{Code: FailureNVLinkBERSymbol, ID: 121, Class: ClassNVLink},
	{Code: FailureRecoveryReset, ID: 125, Class: ClassRecovery},
	{Code: FailureRecoveryReboot, ID: 126, Class: ClassRecovery},
	{Code: FailureRecoveryDrainP2P, ID: 127, Class: ClassRecovery},
	{Code: FailureRecoveryDrainReset, ID: 128, Class: ClassRecovery},
}

var failuresByID = func() map[int]Failure {
	m := make(map[int]Failure, len(FailureCatalog))
	for _, row := range FailureCatalog {
		m[row.ID] = row
	}
	return m
}()

var failuresByCode = func() map[FailureCode]Failure {
	m := make(map[FailureCode]Failure, len(FailureCatalog))
	for _, row := range FailureCatalog {
		m[row.Code] = row
	}
	return m
}()

// ErrUnknownFailure is the sentinel for a DCGM failure field ID or wire name
// outside the closed catalog.
var ErrUnknownFailure = errors.New("hwutil: unknown DCGM failure code")

// unknownFailureError carries which lookup missed so Error() stays useful
// while errors.Is keeps matching the sentinel.
type unknownFailureError struct {
	id   int // >= 0 when the lookup was by numeric ID
	name string
}

func (e *unknownFailureError) Error() string {
	if e.id >= 0 {
		return fmt.Sprintf("hwutil: unknown DCGM failure code id %d", e.id)
	}
	return fmt.Sprintf("hwutil: unknown DCGM failure code %q", e.name)
}

// Unwrap exposes the wrapped sentinel for errors.Is.
func (e *unknownFailureError) Unwrap() error { return ErrUnknownFailure }

// FailureByID returns the catalog row carrying DCGM numeric field ID id. An
// ID outside the closed set returns an error wrapping ErrUnknownFailure.
func FailureByID(id int) (Failure, error) {
	if row, ok := failuresByID[id]; ok {
		return row, nil
	}
	return Failure{}, &unknownFailureError{id: id}
}

// ParseFailure parses a closed DCGM_FR_* wire name into its catalog row. A
// name outside the closed set returns an error wrapping ErrUnknownFailure.
func ParseFailure(name string) (Failure, error) {
	if row, ok := failuresByCode[FailureCode(name)]; ok {
		return row, nil
	}
	return Failure{}, &unknownFailureError{name: name}
}

// ID returns the DCGM numeric field ID carried by c, or -1 when c is outside
// the closed catalog.
func (c FailureCode) ID() int {
	if row, ok := failuresByCode[c]; ok {
		return row.ID
	}
	return -1
}

// Class returns the closed fault class of c, or "" when c is outside the
// closed catalog.
func (c FailureCode) Class() FailureClass {
	if row, ok := failuresByCode[c]; ok {
		return row.Class
	}
	return ""
}

// Valid reports whether c is inside the closed catalog.
func (c FailureCode) Valid() bool {
	_, ok := failuresByCode[c]
	return ok
}
