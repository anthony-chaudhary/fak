package telemetry

import (
	"errors"
	"strings"
	"testing"
)

// TestDCGMTelemetry exercises the DCGM health/thermal field seam: the pinned
// field vocabulary, the fail-closed absent-service refusal, and the fail-closed
// unknown-field refusal.
func TestDCGMTelemetry(t *testing.T) {
	t.Run("pinned_vocabulary", func(t *testing.T) {
		fields := HealthThermalFields()
		if len(fields) != 8 {
			t.Fatalf("expected 8 pinned fields, got %d", len(fields))
		}
		// The pinned IDs are the authority; assert a few verbatim so a drifted
		// row is caught.
		want := map[FieldID]Field{
			FieldXidErrors:                    {ID: FieldXidErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_XID_ERRORS", Unit: "count"},
			FieldGpuTemp:                      {ID: FieldGpuTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_TEMPERATURE", Unit: "C"},
			FieldMemoryTemp:                   {ID: FieldMemoryTemp, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_MEMORY_TEMP", Unit: "C"},
			FieldPowerDraw:                    {ID: FieldPowerDraw, Group: FieldGroupThermal, Name: "DCGM_FI_DEV_POWER_USAGE", Unit: "W"},
			FieldEccCorrectableRemappedRows:   {ID: FieldEccCorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ECC_CORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
			FieldEccUncorrectableRemappedRows: {ID: FieldEccUncorrectableRemappedRows, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_ECC_UNCORRECTABLE_REMAPPED_ROWS", Unit: "rows"},
			FieldNvlinkReplayErrors:           {ID: FieldNvlinkReplayErrors, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_REPLAY_ERROR", Unit: "count"},
			FieldNvlinkRecoveryRounds:         {ID: FieldNvlinkRecoveryRounds, Group: FieldGroupHealth, Name: "DCGM_FI_DEV_NVLINK_RECOVERY_ROUNDS", Unit: "count"},
		}
		for _, f := range fields {
			w, ok := want[f.ID]
			if !ok {
				t.Fatalf("unexpected field ID %d in pinned vocabulary", int(f.ID))
			}
			if f != w {
				t.Fatalf("field %d = %+v, want %+v", int(f.ID), f, w)
			}
		}
		// Sorted by ID ascending.
		for i := 1; i < len(fields); i++ {
			if fields[i-1].ID >= fields[i].ID {
				t.Fatalf("fields not sorted by ID: %d then %d", int(fields[i-1].ID), int(fields[i].ID))
			}
		}
	})

	t.Run("field_by_id", func(t *testing.T) {
		f, err := FieldByID(FieldGpuTemp)
		if err != nil {
			t.Fatalf("FieldByID(%d) error: %v", int(FieldGpuTemp), err)
		}
		if f.Name != "DCGM_FI_DEV_TEMPERATURE" || f.Unit != "C" {
			t.Fatalf("FieldByID(%d) = %+v, want DCGM_FI_DEV_TEMPERATURE C", int(FieldGpuTemp), f)
		}
	})

	t.Run("fields_in_group", func(t *testing.T) {
		health := FieldsInGroup(FieldGroupHealth)
		if len(health) != 5 {
			t.Fatalf("expected 5 health fields, got %d", len(health))
		}
		thermal := FieldsInGroup(FieldGroupThermal)
		if len(thermal) != 3 {
			t.Fatalf("expected 3 thermal fields, got %d", len(thermal))
		}
		for _, f := range health {
			if f.Group != FieldGroupHealth {
				t.Fatalf("health group returned non-health field %d", int(f.ID))
			}
		}
		for _, f := range thermal {
			if f.Group != FieldGroupThermal {
				t.Fatalf("thermal group returned non-thermal field %d", int(f.ID))
			}
		}
	})

	t.Run("absent_service_refuses", func(t *testing.T) {
		svc := Probe("")
		if svc.Present {
			t.Fatalf("Probe(\"\") should be absent, got present")
		}
		records := []DeviceRecord{
			{Index: 0, UUID: "GPU-0", Values: []FieldValue{{Field: Field{ID: FieldGpuTemp}, Value: 45}}},
		}
		got, err := svc.Snapshot(records)
		if !errors.Is(err, ErrDCGMServiceAbsent) {
			t.Fatalf("absent service Snapshot: err = %v, want ErrDCGMServiceAbsent", err)
		}
		if got != nil {
			t.Fatalf("absent service Snapshot returned %d records, want nil", len(got))
		}
		// An absent service must never render as a healthy device.
		if got := svc.Render(nil); !strings.Contains(got, "service absent") {
			t.Fatalf("absent service Render = %q, want 'service absent'", got)
		}
	})

	t.Run("present_service_populates_seam", func(t *testing.T) {
		svc := Probe("host-engine-1")
		if !svc.Present || svc.HostEngine != "host-engine-1" {
			t.Fatalf("Probe(host-engine-1) = %+v, want present", svc)
		}
		devs := []DeviceRecord{
			{
				Index: 0,
				UUID:  "GPU-0",
				Values: []FieldValue{
					{Field: Field{ID: FieldGpuTemp, Name: "DCGM_FI_DEV_TEMPERATURE", Unit: "C"}, Value: 45},
					{Field: Field{ID: FieldXidErrors, Name: "DCGM_FI_DEV_XID_ERRORS", Unit: "count"}, Value: 0},
				},
			},
			{
				Index: 1,
				UUID:  "GPU-1",
				Values: []FieldValue{
					{Field: Field{ID: FieldGpuTemp, Name: "DCGM_FI_DEV_TEMPERATURE", Unit: "C"}, Value: 95},
				},
			},
		}
		got, err := svc.Snapshot(devs)
		if err != nil {
			t.Fatalf("present service Snapshot error: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("expected 2 device records, got %d", len(got))
		}
		if got[0].Index != 0 || got[0].UUID != "GPU-0" {
			t.Fatalf("device 0 = %+v, want index 0 uuid GPU-0", got[0])
		}
		if got[1].Values[0].Value != 95 {
			t.Fatalf("device 1 temp = %v, want 95", got[1].Values[0].Value)
		}
		// The snapshot must not alias the caller's slice: mutating the input
		// must not change the returned record.
		devs[0].Index = 99
		if got[0].Index != 0 {
			t.Fatalf("Snapshot aliased caller slice: got[0].Index = %d after input mutation", got[0].Index)
		}
	})

	t.Run("unknown_field_refuses", func(t *testing.T) {
		// A field ID outside the pinned vocabulary is refused, not guessed.
		_, err := FieldByID(FieldID(99999))
		if !errors.Is(err, ErrDCGMFieldUnknown) {
			t.Fatalf("FieldByID(99999): err = %v, want ErrDCGMFieldUnknown", err)
		}
		// A negative ID is refused too.
		if _, err := FieldByID(FieldID(-1)); !errors.Is(err, ErrDCGMFieldUnknown) {
			t.Fatalf("FieldByID(-1): err = %v, want ErrDCGMFieldUnknown", err)
		}
	})

	t.Run("health_verdict", func(t *testing.T) {
		// No records -> unknown, never healthy.
		if v := HealthVerdictOf(nil); v != VerdictUnknown {
			t.Fatalf("HealthVerdictOf(nil) = %v, want unknown", v)
		}
		// All within bounds -> healthy.
		healthy := []DeviceRecord{
			{Index: 0, Values: []FieldValue{
				{Field: Field{ID: FieldGpuTemp}, Value: 45},
				{Field: Field{ID: FieldXidErrors}, Value: 0},
			}},
		}
		if v := HealthVerdictOf(healthy); v != VerdictHealthy {
			t.Fatalf("healthy records verdict = %v, want healthy", v)
		}
		// GPU temp at warn bound -> warn.
		warn := []DeviceRecord{
			{Index: 0, Values: []FieldValue{{Field: Field{ID: FieldGpuTemp}, Value: 83}}},
		}
		if v := HealthVerdictOf(warn); v != VerdictWarn {
			t.Fatalf("warn records verdict = %v, want warn", v)
		}
		// GPU temp at critical bound -> critical.
		crit := []DeviceRecord{
			{Index: 0, Values: []FieldValue{{Field: Field{ID: FieldGpuTemp}, Value: 93}}},
		}
		if v := HealthVerdictOf(crit); v != VerdictCritical {
			t.Fatalf("critical records verdict = %v, want critical", v)
		}
		// Worst bound wins: a critical device among healthy ones -> critical.
		mixed := append(append([]DeviceRecord{}, healthy...), crit...)
		if v := HealthVerdictOf(mixed); v != VerdictCritical {
			t.Fatalf("mixed records verdict = %v, want critical", v)
		}
	})

	t.Run("render", func(t *testing.T) {
		svc := Probe("host-engine-1")
		records := []DeviceRecord{
			{
				Index: 0,
				Values: []FieldValue{
					{Field: Field{ID: FieldGpuTemp, Name: "DCGM_FI_DEV_TEMPERATURE", Unit: "C"}, Value: 45},
					{Field: Field{ID: FieldXidErrors, Name: "DCGM_FI_DEV_XID_ERRORS", Unit: "count"}, Value: 0},
				},
			},
		}
		out := svc.Render(records)
		if !strings.Contains(out, "dcgm (1 devices, healthy)") {
			t.Fatalf("Render = %q, want 'dcgm (1 devices, healthy)'", out)
		}
		if !strings.Contains(out, "dev0") {
			t.Fatalf("Render = %q, want 'dev0'", out)
		}
		if !strings.Contains(out, "DCGM_FI_DEV_TEMPERATURE=45.0C") {
			t.Fatalf("Render = %q, want 'DCGM_FI_DEV_TEMPERATURE=45.0C'", out)
		}
	})
}
