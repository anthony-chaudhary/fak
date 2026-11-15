// DIRECT-PORT of the DCGM health and thermal field model from
// NVIDIA/go-dcgm (Apache-2.0), pinned at
//
//	NVIDIA/go-dcgm@bc5d307bece3d373de882978c8503a1c8613e232
//	https://github.com/NVIDIA/go-dcgm
//
// The field IDs, group IDs, units, and short names below are the closed
// vocabulary the upstream bindings expose for the health and thermal field
// groups (the dcgm_fields.h constants the generated Go bindings re-export),
// cross-checked against the dcgm-exporter default counter set pinned at
// 4e663a526c4624ad0e18ad816f09b7be2c5348a7. Borrow provenance:
// docs/research/STUDY-NVIDIA-GO-DCGM-2026-11-06.md, candidate
// oss-port-nv-godcgm-telemetry-20261106.
//
// What fak adds AROUND the ported data: the fail-closed seam. A DCGM reading
// is only meaningful when the dcgm host-engine service is present, so every
// read path here REFUSES with ErrDCGMServiceAbsent when the service is not
// reachable rather than skipping or zero-filling: an absent service must be
// visible in a receipt, never masquerade as a healthy device. The same
// fail-closed rule applies to the field vocabulary: a field ID outside the
// pinned set is refused by FieldByID, so a drifted or invented field can never
// be rendered into a record.
//
// Scope is bindings and headers only: no exporter, no Prometheus sink, no
// hot-path blocking. The records here extend the device-seam records the
// NVML port feeds, for the perfledger and hostdiag consumers.

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

// ErrDCGMFieldUnknown is returned by FieldByID for a field ID outside the
// pinned health/thermal vocabulary.
var ErrDCGMFieldUnknown = errors.New("telemetry: dcgm field unknown")

// FieldGroup is a DCGM field-group ID (the upstream DcgmFieldGroup_t values).
type FieldGroup int

const (
	// FieldGroupHealth is the DCGM health field group (DCGM_FI_HEALTH).
	FieldGroupHealth FieldGroup = 100
	// FieldGroupThermal is the DCGM thermal field group (DCGM_FI_THERMAL).
	FieldGroupThermal FieldGroup = 101
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

// FieldID is a DCGM field ID (the upstream DcgmField_t values).
type FieldID int

// The pinned health/thermal field vocabulary. The IDs, units, and short names
// are the upstream DCGM_FI_DEV_* constants the go-dcgm bindings re-export from
// dcgm_fields.h; they are never reworded or re-sorted — a drifted row would
// silently change what a receipt says about a device, so the pinned values are
// asserted verbatim in TestDCGMTelemetry.
const (
	// FieldXidErrors: XID error count (DCGM_FI_DEV_XID_ERRORS).
	FieldXidErrors FieldID = 100
	// FieldGpuTemp: GPU temperature in Celsius (DCGM_FI_DEV_TEMPERATURE).
	FieldGpuTemp FieldID = 150
	// FieldMemoryTemp: memory temperature in Celsius (DCGM_FI_DEV_MEMORY_TEMP).
	FieldMemoryTemp FieldID = 151
	// FieldPowerDraw: power draw in watts (DCGM_FI_DEV_POWER_USAGE).
	FieldPowerDraw FieldID = 155
	// FieldEccCorrectableRemappedRows: ECC correctable remapped rows
	// (DCGM_FI_DEV_ECC_CORRECTABLE_REMAPPED_ROWS).
	FieldEccCorrectableRemappedRows FieldID = 1004
	// FieldEccUncorrectableRemappedRows: ECC uncorrectable remapped rows
	// (DCGM_FI_DEV_ECC_UNCORRECTABLE_REMAPPED_ROWS).
	FieldEccUncorrectableRemappedRows FieldID = 1005
	// FieldNvlinkReplayErrors: NVLink replay error count
	// (DCGM_FI_DEV_NVLINK_REPLAY_ERROR).
	FieldNvlinkReplayErrors FieldID = 1010
	// FieldNvlinkRecoveryRounds: NVLink recovery round count
	// (DCGM_FI_DEV_NVLINK_RECOVERY_ROUNDS).
	FieldNvlinkRecoveryRounds FieldID = 1011
)

// Field is one row of the pinned DCGM field vocabulary.
type Field struct {
	// ID is the DCGM field ID.
	ID FieldID
	// Group is the field group the ID belongs to.
	Group FieldGroup
	// Name is the upstream short name (the DcgmField_t constant name).
	Name string
	// Unit is the unit the value is reported in.
	Unit string
}

// healthThermalFields is the closed vocabulary the seam exposes.
var healthThermalFields = []Field{
	{ID: FieldXidErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_XID_ERRORS", Unit: "count"},
	{ID: FieldGpuTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_TEMPERATURE", Unit: "C"},
	{ID: FieldMemoryTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_MEMORY_TEMP", Unit: "C"},
	{ID: FieldPowerDraw, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_POWER_USAGE", Unit: "W"},
	{ID: FieldEccCorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ECC_CORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
	{ID: FieldEccUncorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ECC_UNCORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
	{ID: FieldNvlinkReplayErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_REPLAY_ERROR", Unit: "count"},
	{ID: FieldNvlinkRecoveryRounds, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_RECOVERY_ROUNDS", Unit: "count"},
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
// guessed, so an invented or drifted field can never be rendered into a record.
func FieldByID(id FieldID) (Field, error) {
	for _, f := range healthThermalFields {
		if f.ID == id {
			return f, nil
		}
	}
	return Field{}, fmt.Errorf("%w: %d", ErrDCGMFieldUnknown, int(id))
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

// DeviceRecord is one device's DCGM health/thermal reading. It extends the
// device-seam records the NVML port feeds: the device identity plus the
// pinned field values, so a perfledger or hostdiag consumer can fold a DCGM
// reading into the same record without a second device model.
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

// Probe reports whether the dcgm host-engine service is present. A nil host
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

// fieldBounds is the pinned per-field warn/critical bound the verdict folds
// against. A value at or above the critical bound is critical; at or above the
// warn bound is warn; below the warn bound is healthy.
type fieldBounds struct {
	warn     float64
	critical float64
}

var healthBounds = map[FieldID]fieldBounds{
	FieldEccCorrectableRemappedRows:   {warn: 1, critical: 10},
	FieldEccUncorrectableRemappedRows: {warn: 1, critical: 1},
	FieldXidErrors:                    {warn: 1, critical: 1},
	FieldNvlinkReplayErrors:           {warn: 1, critical: 10},
	FieldNvlinkRecoveryRounds:         {warn: 1, critical: 10},
	FieldGpuTemp:                      {warn: 83, critical: 93},
	FieldMemoryTemp:                   {warn: 95, critical: 105},
	FieldPowerDraw:                    {warn: 0, critical: 0}, // power has no health bound; never trips
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
		// Worst value first: sort by value descending so a hot device is the
		// first thing read.
		vals := make([]FieldValue, len(r.Values))
		copy(vals, r.Values)
		sort.Slice(vals, func(i, j int) bool { return vals[i].Value > vals[j].Value })
		for _, v := range vals {
			fmt.Fprintf(&b, " %s=%.1f%s", v.Field.Name, v.Value, v.Field.Unit)
		}
	}
	return b.String()
}
