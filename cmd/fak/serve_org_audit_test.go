package main

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestLoadServeOrgAuditConfigAcceptsMissingEnrollment(t *testing.T) {
	t.Setenv(policy.OrgEnrollmentPathEnv, filepath.Join(t.TempDir(), "missing-enrollment.json"))

	got, err := loadServeOrgAuditConfig()
	if err != nil {
		t.Fatalf("unenrolled serve startup: %v", err)
	}
	if !reflect.DeepEqual(got, auditreceipt.Config{}) {
		t.Fatalf("unenrolled audit config = %+v, want zero config", got)
	}
}
