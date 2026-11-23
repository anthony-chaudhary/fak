package modelroute

import "github.com/anthony-chaudhary/fak/pkg/routepolicy"

// AuditRecord and the Decision/Manifest digest methods live in pkg/routepolicy.
type AuditRecord = routepolicy.AuditRecord

func ParseAuditRecord(b []byte) (AuditRecord, error) { return routepolicy.ParseAuditRecord(b) }
