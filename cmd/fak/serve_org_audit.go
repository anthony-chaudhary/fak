package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func loadServeOrgAuditConfig() (auditreceipt.Config, error) {
	enrollment, enrolled, err := policy.LoadOrgEnrollment(policy.OrgEnrollmentPath())
	if err != nil {
		return auditreceipt.Config{}, err
	}
	if !enrolled || strings.TrimSpace(enrollment.AuditURL) == "" {
		return auditreceipt.Config{}, nil
	}
	redact := adjudicator.Default.PolicySnapshot().RedactFields
	return auditreceipt.Config{
		Endpoint:     enrollment.AuditURL,
		DeviceID:     enrollment.DeviceID,
		BufferPath:   policy.OrgAuditBufferPath(),
		RedactFields: redact,
	}, nil
}
