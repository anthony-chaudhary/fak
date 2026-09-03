package dataslot

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDataSlotDescriptor_Validate(t *testing.T) {
	valid := DataSlotDescriptor{
		ID:             "sqlite:dev.db",
		Family:         FamilySQLite,
		Status:         StatusReady,
		SourceArtifact: "dev.db",
		LocalPath:      "dev.db",
		ReadOnly:       true,
	}

	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid descriptor to pass validation, got: %v", err)
	}

	// Missing ID
	noID := valid
	noID.ID = "   "
	if err := noID.Validate(); err == nil || !strings.Contains(err.Error(), "descriptor ID is required") {
		t.Fatalf("expected missing ID error, got: %v", err)
	}

	// Invalid Family
	badFamily := valid
	badFamily.Family = "oracle"
	if err := badFamily.Validate(); err == nil || !strings.Contains(err.Error(), "unknown database family") {
		t.Fatalf("expected unknown database family error, got: %v", err)
	}

	// Invalid Status
	badStatus := valid
	badStatus.Status = "running"
	if err := badStatus.Validate(); err == nil || !strings.Contains(err.Error(), "unknown status") {
		t.Fatalf("expected unknown status error, got: %v", err)
	}
}

func TestDataSlotDescriptor_StatusPredicates(t *testing.T) {
	tests := []struct {
		status      SlotStatus
		wantDormant bool
		wantReady   bool
		wantActive  bool
	}{
		{StatusAbsent, false, false, false},
		{StatusUnmaterialized, true, false, false},
		{StatusReady, true, true, false},
		{StatusActive, false, false, true},
	}

	for _, tc := range tests {
		t.Run(string(tc.status), func(t *testing.T) {
			d := DataSlotDescriptor{Status: tc.status}
			if got := d.IsDormant(); got != tc.wantDormant {
				t.Errorf("IsDormant() = %t, want %t", got, tc.wantDormant)
			}
			if got := d.IsReady(); got != tc.wantReady {
				t.Errorf("IsReady() = %t, want %t", got, tc.wantReady)
			}
			if got := d.IsActive(); got != tc.wantActive {
				t.Errorf("IsActive() = %t, want %t", got, tc.wantActive)
			}
		})
	}
}

func TestDataSlotDescriptor_JSONRoundtrip(t *testing.T) {
	orig := DataSlotDescriptor{
		ID:              "prisma:prisma/schema.prisma",
		Family:          FamilyPostgres,
		Status:          StatusUnmaterialized,
		SourceArtifact:  "prisma/schema.prisma",
		MigrationEngine: MigrationPrisma,
		MigrationPath:   "prisma",
		ReadOnly:        true,
		Metadata: map[string]any{
			"env_var": "DATABASE_URL",
		},
	}

	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var decoded DataSlotDescriptor
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if decoded.ID != orig.ID {
		t.Errorf("ID = %q, want %q", decoded.ID, orig.ID)
	}
	if decoded.Family != orig.Family {
		t.Errorf("Family = %q, want %q", decoded.Family, orig.Family)
	}
	if decoded.Status != orig.Status {
		t.Errorf("Status = %q, want %q", decoded.Status, orig.Status)
	}
	if decoded.MigrationEngine != orig.MigrationEngine {
		t.Errorf("MigrationEngine = %q, want %q", decoded.MigrationEngine, orig.MigrationEngine)
	}
	if !decoded.ReadOnly {
		t.Errorf("ReadOnly = false, want true")
	}
}

func TestConstantsEnumeration(t *testing.T) {
	families := []DatabaseFamily{FamilySQLite, FamilyDuckDB, FamilyPostgres, FamilyMySQL, FamilyRedis}
	for _, fam := range families {
		if !ValidFamilies[fam] {
			t.Errorf("family %q missing from ValidFamilies", fam)
		}
	}

	statuses := []SlotStatus{StatusAbsent, StatusUnmaterialized, StatusReady, StatusActive}
	for _, st := range statuses {
		if !ValidStatuses[st] {
			t.Errorf("status %q missing from ValidStatuses", st)
		}
	}
}
