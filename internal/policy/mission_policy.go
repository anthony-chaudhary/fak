package policy

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// Re-export ReasonMissionWriteSetViolation and ReasonMissionWriteSetViolationName from adjudicator (#12276).
const (
	ReasonMissionWriteSetViolation     abi.ReasonCode = adjudicator.ReasonMissionWriteSetViolation
	ReasonMissionWriteSetViolationName string         = adjudicator.ReasonMissionWriteSetViolationName
)

// MissionContract defines dynamic file write-sets and asymmetric directory extraction boundaries (#12276).
type MissionContract = adjudicator.MissionContract

// NewMissionContract builds a MissionContract with designated write-set and extract-into directories (#12276).
func NewMissionContract(writeSet, extractInto []string) MissionContract {
	return MissionContract{
		WriteSet:    append([]string(nil), writeSet...),
		ExtractInto: append([]string(nil), extractInto...),
	}
}
