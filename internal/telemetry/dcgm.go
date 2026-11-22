// DIRECT-PORT of NVIDIA/go-dcgm pkg/dcgm/const_fields.go at
// bc5d307bece3d373de882978c8503a1c8613e232 (Apache-2.0), https://github.com/NVIDIA/go-dcgm.
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// Ported: the DCGM_FI_DEV_* field IDs and names for the health and thermal
// fields, with each field's current upstream name and its deprecated alias.
// Changes were made.
//
// What fak adds: a fail-closed seam around that vocabulary. A DCGM reading only
// means something when the dcgm host-engine service is present, so every read
// path refuses with ErrDCGMServiceAbsent instead of skipping or zero-filling;
// an absent service must show up in a receipt, never as a healthy device. A
// field ID or name outside the pinned set is refused the same way, so a drifted
// or invented field can never be rendered into a record. The field groups and
// the health bounds are fak-local: upstream field groups are runtime handles
// from FieldGroupCreate, not fixed IDs.

package telemetry

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ErrDCGMServiceAbsent is returned by every read path when the dcgm host-engine
// service is not present. Fail-closed: the refusal is the contract, so callers
// branch on the error rather than on an empty record.
var ErrDCGMServiceAbsent = errors.New("telemetry: dcgm service absent")

// ErrDCGMFieldUnknown is returned by FieldByID and FieldByName for a field
// outside the pinned health/thermal vocabulary.
var ErrDCGMFieldUnknown = errors.New("telemetry: dcgm field unknown")

// FieldGroup is a fak-local classification of the pinned fields. It is not a
// DCGM field-group handle: upstream groups are created at runtime.
type FieldGroup int

const (
	// FieldGroupHealth classifies the error and row-remap fields.
	FieldGroupHealth FieldGroup = iota + 1
	// FieldGroupThermal classifies the temperature and power fields.
	FieldGroupThermal
)

// String renders the group for operator text.
func (g FieldGroup) String() string {
	switch g {
	case FieldGroupHealth:
		return "health"
	case FieldGroupThermal:
		return "thermal"
	default:
		return fmt.Sprintf("group(%d)", int(g))
	}
}

// FieldID is a DCGM field ID (the upstream dcgm.Short values).
type FieldID int

// The pinned field IDs, verbatim from upstream. A drifted ID would silently
// change what a receipt says about a device, so TestDCGMTelemetry asserts them.
const (
	// FieldMemoryTemp is DCGM_FI_DEV_MEMORY_TEMP_CELSIUS.
	FieldMemoryTemp FieldID = 140
	// FieldGpuTemp is DCGM_FI_DEV_GPU_TEMP_CELSIUS.
	FieldGpuTemp FieldID = 150
	// FieldPowerDraw is DCGM_FI_DEV_BOARD_POWER_WATTS.
	FieldPowerDraw FieldID = 155
	// FieldXidErrors is DCGM_FI_DEV_XID_ERROR; the value is the last XID code,
	// not a count.
	FieldXidErrors FieldID = 230
	// FieldUncorrectableRemappedRows is DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL.
	FieldUncorrectableRemappedRows FieldID = 393
	// FieldCorrectableRemappedRows is DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL.
	FieldCorrectableRemappedRows FieldID = 394
	// FieldRowRemapFailed is DCGM_FI_DEV_ROW_REMAP_FAILED.
	FieldRowRemapFailed FieldID = 395
	// FieldNvlinkReplayErrors is DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL.
	FieldNvlinkReplayErrors FieldID = 429
	// FieldNvlinkRecoveryErrors is DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL.
	FieldNvlinkRecoveryErrors FieldID = 439
)

// Field is one row of the pinned DCGM field vocabulary.
type Field struct {
	// ID is the DCGM field ID.
	ID FieldID
	// Group is the fak-local group the field is classified under.
	Group FieldGroup
	// Name is the current upstream constant name.
	Name string
	// LegacyName is the deprecated upstream alias for the same ID, which
	// dcgm-exporter counter files still use.
	LegacyName string
	// Unit is the unit the value is reported in.
	Unit string
}

var healthThermalFields = []Field{
	{ID: FieldMemoryTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS", LegacyName: "DCGM_FI_DEV_MEMORY_TEMP", Unit: "C"},
	{ID: FieldGpuTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_GPU_TEMP_CELSIUS", LegacyName: "DCGM_FI_DEV_GPU_TEMP", Unit: "C"},
	{ID: FieldPowerDraw, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_BOARD_POWER_WATTS", LegacyName: "DCGM_FI_DEV_POWER_USAGE", Unit: "W"},
	{ID: FieldXidErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_XID_ERROR", LegacyName: "DCGM_FI_DEV_XID_ERRORS", Unit: "xid"},
	{ID: FieldUncorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL", LegacyName: "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
	{ID: FieldCorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL", LegacyName: "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
	{ID: FieldRowRemapFailed, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ROW_REMAP_FAILED", LegacyName: "DCGM_FI_DEV_ROW_REMAP_FAILURE", Unit: "bool"},
	{ID: FieldNvlinkReplayErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL", LegacyName: "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL", Unit: "count"},
	{ID: FieldNvlinkRecoveryErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL", LegacyName: "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL", Unit: "count"},
}

// HealthThermalFields returns the pinned health/thermal field vocabulary,
// sorted by field ID. The returned slice is a copy; mutating it does not
// affect the vocabulary.
func HealthThermalFields() []Field {
	out := make([]Field, len(healthThermalFields))
	copy(out, healthThermalFields)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// FieldByID looks up one pinned field by its DCGM field ID, fail-closed: an ID
// outside the pinned vocabulary is refused with ErrDCGMFieldUnknown rather than
// guessed.
func FieldByID(id FieldID) (Field, error) {
	for _, f := range healthThermalFields {
		if f.ID == id {
			return f, nil
		}
	}
	return Field{}, fmt.Errorf("%w: %d", ErrDCGMFieldUnknown, int(id))
}

// FieldByName looks up one pinned field by its current upstream name or its
// deprecated alias, fail-closed like FieldByID.
func FieldByName(name string) (Field, error) {
	for _, f := range healthThermalFields {
		if f.Name == name || f.LegacyName == name {
			return f, nil
		}
	}
	return Field{}, fmt.Errorf("%w: %q", ErrDCGMFieldUnknown, name)
}

// FieldsInGroup returns the pinned fields of one group, sorted by field ID.
// An unknown group comes back empty, not an error: the group is a filter, and
// the closed vocabulary is the authority.
func FieldsInGroup(g FieldGroup) []Field {
	var out []Field
	for _, f := range healthThermalFields {
		if f.Group == g {
			out = append(out, f)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// FieldValue is one sampled DCGM field value for one device.
type FieldValue struct {
	// Field is the pinned field the value belongs to.
	Field Field `json:"field"`
	// Value is the sampled value in the field's unit.
	Value float64 `json:"value"`
}

// DeviceRecord is one device's DCGM health/thermal reading: the device
// identity plus the pinned field values, so a perfledger or hostdiag consumer
// can fold a DCGM reading into its own record without a second device model.
type DeviceRecord struct {
	// Index is the device index the service reported.
	Index int `json:"index"`
	// UUID is the device UUID the service reported.
	UUID string `json:"uuid,omitempty"`
	// Values is one row per pinned field the service reported for the device.
	Values []FieldValue `json:"values"`
}

// Service is the fail-closed seam over the dcgm host-engine service.
type Service struct {
	// Present is whether the dcgm host-engine service is reachable.
	Present bool
	// HostEngine is the host-engine identity the service reported, when present.
	HostEngine string
}

// Probe reports whether the dcgm host-engine service is present. An empty host
// engine is absent.
func Probe(hostEngine string) Service {
	if hostEngine == "" {
		return Service{Present: false}
	}
	return Service{Present: true, HostEngine: hostEngine}
}

// Snapshot returns one DeviceRecord per device the service reported, fail-closed:
// an absent service is refused with ErrDCGMServiceAbsent rather than skipped, so
// a receipt can never read an absent service as a healthy device.
func (s Service) Snapshot(devices []DeviceRecord) ([]DeviceRecord, error) {
	if !s.Present {
		return nil, ErrDCGMServiceAbsent
	}
	out := make([]DeviceRecord, len(devices))
	copy(out, devices)
	return out, nil
}

// HealthVerdict is the folded health state of a set of device records.
type HealthVerdict int

const (
	// VerdictUnknown is the verdict when no records were read (service absent).
	VerdictUnknown HealthVerdict = iota
	// VerdictHealthy is the verdict when every read value is within its pinned
	// healthy bound.
	VerdictHealthy
	// VerdictWarn is the verdict when a read value is at or above its warn bound.
	VerdictWarn
	// VerdictCritical is the verdict when a read value is at or above its
	// critical bound.
	VerdictCritical
)

// String renders the verdict for operator text.
func (v HealthVerdict) String() string {
	switch v {
	case VerdictHealthy:
		return "healthy"
	case VerdictWarn:
		return "warn"
	case VerdictCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// fieldBounds is the fak-local per-field warn/critical bound the verdict folds
// against. A value at or above the critical bound is critical; at or above the
// warn bound is warn; below the warn bound is healthy.
type fieldBounds struct {
	warn     float64
	critical float64
}

var healthBounds = map[FieldID]fieldBounds{
	FieldCorrectableRemappedRows:   {warn: 1, critical: 10},
	FieldUncorrectableRemappedRows: {warn: 1, critical: 1},
	FieldRowRemapFailed:            {warn: 1, critical: 1},
	FieldXidErrors:                 {warn: 1, critical: 1},
	FieldNvlinkReplayErrors:        {warn: 1, critical: 10},
	FieldNvlinkRecoveryErrors:      {warn: 1, critical: 10},
	FieldGpuTemp:                   {warn: 83, critical: 93},
	FieldMemoryTemp:                {warn: 95, critical: 105},
	FieldPowerDraw:                 {warn: 0, critical: 0}, // power has no health bound; never trips
}

// HealthVerdictOf folds a set of device records into one verdict: the worst
// bound any read value tripped. An empty set is VerdictUnknown, so a caller
// that read nothing (service absent) can never report healthy.
func HealthVerdictOf(records []DeviceRecord) HealthVerdict {
	if len(records) == 0 {
		return VerdictUnknown
	}
	verdict := VerdictHealthy
	for _, r := range records {
		for _, v := range r.Values {
			bound, ok := healthBounds[v.Field.ID]
			if !ok || bound.critical <= 0 {
				continue
			}
			switch {
			case v.Value >= bound.critical:
				if verdict < VerdictCritical {
					verdict = VerdictCritical
				}
			case v.Value >= bound.warn:
				if verdict < VerdictWarn {
					verdict = VerdictWarn
				}
			}
		}
	}
	return verdict
}

// Render formats a snapshot as operator-readable text, one line per device,
// worst field first. An empty snapshot is reported explicitly rather than as a
// bare empty list.
func (s Service) Render(records []DeviceRecord) string {
	if len(records) == 0 {
		return "dcgm: no device records (service absent)"
	}
	verdict := HealthVerdictOf(records)
	var b strings.Builder
	fmt.Fprintf(&b, "dcgm (%d devices, %s):", len(records), verdict)
	for _, r := range records {
		fmt.Fprintf(&b, " dev%d", r.Index)
		// Highest value first so a hot device is the first thing read.
		vals := make([]FieldValue, len(r.Values))
		copy(vals, r.Values)
		sort.Slice(vals, func(i, j int) bool { return vals[i].Value > vals[j].Value })
		for _, v := range vals {
			fmt.Fprintf(&b, " %s=%.1f%s", v.Field.Name, v.Value, v.Field.Unit)
		}
	}
	return b.String()
}
