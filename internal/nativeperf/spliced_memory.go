package nativeperf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// VendorMemoryEvidenceSchema defines the schema for vendor memory evidence documents.
const VendorMemoryEvidenceSchema = "fak-native-vendor-memory-evidence/v1"

// MemoryEvidenceVerdict expresses the coherence verdict for vendor memory profiler evidence.
type MemoryEvidenceVerdict string

const (
	// MemoryVerdictCoherent indicates all sections, stamps, pass identities, and time orders reconcile.
	MemoryVerdictCoherent MemoryEvidenceVerdict = "COHERENT"
	// MemoryVerdictSpliced indicates evidence of tampering, cross-run/cross-pass splicing, or sequence violation.
	MemoryVerdictSpliced MemoryEvidenceVerdict = "SPLICED"
	// MemoryVerdictUnknown indicates the evidence is missing, unparseable, or lacks verifiable stamps.
	MemoryVerdictUnknown MemoryEvidenceVerdict = "UNKNOWN"
)

// MemorySectionStamp is a cryptographic or vendor-signed stamp binding a memory section to a pass.
type MemorySectionStamp struct {
	SectionID string `json:"section_id"`
	PassID    string `json:"pass_id"`
	PassIndex int    `json:"pass_index"`
	RunID     string `json:"run_id"`
	DeviceID  string `json:"device_id,omitempty"`
	Hash      string `json:"hash"`
}

// VendorMemoryRecord represents an individual memory profile section or allocation observation.
type VendorMemoryRecord struct {
	SectionName      string             `json:"section_name"`
	PassID           string             `json:"pass_id"`
	PassIndex        int                `json:"pass_index"`
	RunID            string             `json:"run_id"`
	DeviceID         string             `json:"device_id,omitempty"`
	StartTimestampNS int64              `json:"start_timestamp_ns"`
	EndTimestampNS   int64              `json:"end_timestamp_ns"`
	AllocatedBytes   uint64             `json:"allocated_bytes"`
	FreedBytes       uint64             `json:"freed_bytes"`
	PeakBytes        uint64             `json:"peak_bytes"`
	Stamp            MemorySectionStamp `json:"stamp"`
	Metadata         map[string]string  `json:"metadata,omitempty"`
}

// VendorMemoryEvidence is the top-level container for vendor memory profiler output.
type VendorMemoryEvidence struct {
	Schema      string               `json:"schema"`
	Vendor      string               `json:"vendor"`
	RunID       string               `json:"run_id"`
	DeviceID    string               `json:"device_id,omitempty"`
	TotalPasses int                  `json:"total_passes"`
	Sections    []VendorMemoryRecord `json:"sections"`
	CapturedAt  string               `json:"captured_at,omitempty"`
}

// MemorySplicingCheckResult holds the typed verdict and diagnostic reasons for the check.
type MemorySplicingCheckResult struct {
	Verdict       MemoryEvidenceVerdict `json:"verdict"`
	Reasons       []string              `json:"reasons,omitempty"`
	TotalSections int                   `json:"total_sections"`
	PassCount     int                   `json:"pass_count"`
}

// ComputeSectionStamp calculates the canonical SHA-256 digest of a memory record's invariant fields.
func ComputeSectionStamp(record VendorMemoryRecord) string {
	payload := fmt.Sprintf(
		"name=%s|pass=%s|idx=%d|run=%s|dev=%s|start=%d|end=%d|alloc=%d|freed=%d|peak=%d",
		record.SectionName,
		record.PassID,
		record.PassIndex,
		record.RunID,
		record.DeviceID,
		record.StartTimestampNS,
		record.EndTimestampNS,
		record.AllocatedBytes,
		record.FreedBytes,
		record.PeakBytes,
	)
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}

// BindVendorMemoryRecord constructs a VendorMemoryRecord and computes its canonical SectionStamp.
func BindVendorMemoryRecord(sectionName, passID string, passIndex int, runID, deviceID string, startNS, endNS int64, allocatedBytes, freedBytes, peakBytes uint64) VendorMemoryRecord {
	rec := VendorMemoryRecord{
		SectionName:      sectionName,
		PassID:           passID,
		PassIndex:        passIndex,
		RunID:            runID,
		DeviceID:         deviceID,
		StartTimestampNS: startNS,
		EndTimestampNS:   endNS,
		AllocatedBytes:   allocatedBytes,
		FreedBytes:       freedBytes,
		PeakBytes:        peakBytes,
	}
	rec.Stamp = MemorySectionStamp{
		SectionID: fmt.Sprintf("%s-p%d", sectionName, passIndex),
		PassID:    passID,
		PassIndex: passIndex,
		RunID:     runID,
		DeviceID:  deviceID,
		Hash:      ComputeSectionStamp(rec),
	}
	return rec
}

// VerifySectionStamp verifies that a record's stamp matches its content and metadata.
func VerifySectionStamp(record VendorMemoryRecord) bool {
	if record.Stamp.Hash == "" {
		return false
	}
	if record.Stamp.PassID != record.PassID ||
		record.Stamp.PassIndex != record.PassIndex ||
		(record.Stamp.RunID != "" && record.Stamp.RunID != record.RunID) ||
		(record.Stamp.DeviceID != "" && record.Stamp.DeviceID != record.DeviceID) {
		return false
	}
	expected := ComputeSectionStamp(record)
	return strings.EqualFold(record.Stamp.Hash, expected)
}

// MarshalVendorMemoryEvidence serializes VendorMemoryEvidence to formatted JSON bytes.
func MarshalVendorMemoryEvidence(evidence *VendorMemoryEvidence) ([]byte, error) {
	if evidence == nil {
		return nil, fmt.Errorf("nativeperf: cannot marshal nil vendor memory evidence")
	}
	return json.Marshal(evidence)
}

// ParseVendorMemoryEvidence deserializes raw bytes into a VendorMemoryEvidence structure.
func ParseVendorMemoryEvidence(data []byte) (*VendorMemoryEvidence, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("nativeperf: empty vendor memory evidence data")
	}
	var ev VendorMemoryEvidence
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("nativeperf: unmarshal vendor memory evidence: %w", err)
	}
	return &ev, nil
}

// ParseAndCheckVendorMemoryEvidence parses raw JSON bytes and returns the splicing verdict.
func ParseAndCheckVendorMemoryEvidence(data []byte) MemorySplicingCheckResult {
	ev, err := ParseVendorMemoryEvidence(data)
	if err != nil {
		return MemorySplicingCheckResult{
			Verdict: MemoryVerdictUnknown,
			Reasons: []string{err.Error()},
		}
	}
	return CheckVendorMemoryEvidence(ev)
}

// CheckVendorMemoryEvidence inspects memory evidence records for section stamp fidelity,
// pass identity consistency, monotonic ordering, and cross-pass splicing anomalies.
func CheckVendorMemoryEvidence(evidence *VendorMemoryEvidence) MemorySplicingCheckResult {
	if prelimRes, ok := validateVendorMemoryEvidencePrelims(evidence); !ok {
		return prelimRes
	}

	res := MemorySplicingCheckResult{
		TotalSections: len(evidence.Sections),
	}

	var reasons []string

	// Track whether any stamps exist
	stampedCount := 0
	for _, sec := range evidence.Sections {
		if sec.Stamp.Hash != "" {
			stampedCount++
		}
	}

	// If no sections have stamps at all, we cannot verify coherence -> UNKNOWN
	if stampedCount == 0 {
		res.Verdict = MemoryVerdictUnknown
		res.Reasons = []string{"vendor memory evidence has no section stamps to evaluate"}
		return res
	}

	// Partial stamping: some stamped, some missing -> SPLICED
	if stampedCount > 0 && stampedCount < len(evidence.Sections) {
		reasons = append(reasons, fmt.Sprintf(
			"inconsistent stamping: %d/%d sections have stamps", stampedCount, len(evidence.Sections),
		))
	}

	passIndexToID := make(map[int]string)
	passIDToIndex := make(map[string]int)
	seenStamps := make(map[string]string)      // stampHash -> "pass:secName"
	seenSectionInPass := make(map[string]bool) // "passIdx:secName" -> true

	lastPassIndex := -1
	passEnds := make(map[int]int64)
	passStarts := make(map[int]int64)

	for i, sec := range evidence.Sections {
		secDesc := fmt.Sprintf("section[%d](name=%q, pass_idx=%d, pass_id=%q)", i, sec.SectionName, sec.PassIndex, sec.PassID)

		if strings.TrimSpace(sec.SectionName) == "" {
			reasons = append(reasons, fmt.Sprintf("%s has empty section_name", secDesc))
		}
		if strings.TrimSpace(sec.PassID) == "" {
			reasons = append(reasons, fmt.Sprintf("%s has empty pass_id", secDesc))
		}
		if sec.PassIndex < 0 {
			reasons = append(reasons, fmt.Sprintf("%s has negative pass_index %d", secDesc, sec.PassIndex))
		}
		if sec.StartTimestampNS < 0 || sec.EndTimestampNS < 0 {
			reasons = append(reasons, fmt.Sprintf("%s has negative timestamp: start=%d ns, end=%d ns", secDesc, sec.StartTimestampNS, sec.EndTimestampNS))
		}

		// 1. Check top-level RunID and DeviceID consistency
		if evidence.RunID != "" && sec.RunID != "" && sec.RunID != evidence.RunID {
			reasons = append(reasons, fmt.Sprintf(
				"%s run_id %q does not match top-level run_id %q", secDesc, sec.RunID, evidence.RunID,
			))
		}
		if evidence.DeviceID != "" && sec.DeviceID != "" && sec.DeviceID != evidence.DeviceID {
			reasons = append(reasons, fmt.Sprintf(
				"%s device_id %q does not match top-level device_id %q", secDesc, sec.DeviceID, evidence.DeviceID,
			))
		}

		// 2. Check Stamp integrity
		if stampReasons := checkSectionStampIntegrity(sec, secDesc, seenStamps); len(stampReasons) > 0 {
			reasons = append(reasons, stampReasons...)
		}

		// 3. Duplicate section in same pass
		passSecKey := fmt.Sprintf("%d:%s", sec.PassIndex, sec.SectionName)
		if seenSectionInPass[passSecKey] {
			reasons = append(reasons, fmt.Sprintf(
				"%s duplicate section name within pass %d", secDesc, sec.PassIndex,
			))
		}
		seenSectionInPass[passSecKey] = true

		// 4. Pass Identity Consistency
		if priorID, exists := passIndexToID[sec.PassIndex]; exists {
			if priorID != sec.PassID {
				reasons = append(reasons, fmt.Sprintf(
					"%s pass_index %d carries contradictory pass_id %q (previously %q)",
					secDesc, sec.PassIndex, sec.PassID, priorID,
				))
			}
		} else {
			passIndexToID[sec.PassIndex] = sec.PassID
		}

		if priorIdx, exists := passIDToIndex[sec.PassID]; exists {
			if priorIdx != sec.PassIndex {
				reasons = append(reasons, fmt.Sprintf(
					"%s pass_id %q carries contradictory pass_index %d (previously %d)",
					secDesc, sec.PassID, sec.PassIndex, priorIdx,
				))
			}
		} else {
			passIDToIndex[sec.PassID] = sec.PassIndex
		}

		// 5. Timestamps within section
		if sec.StartTimestampNS > 0 && sec.EndTimestampNS > 0 && sec.EndTimestampNS < sec.StartTimestampNS {
			reasons = append(reasons, fmt.Sprintf(
				"%s inverted timestamps: start=%d ns, end=%d ns", secDesc, sec.StartTimestampNS, sec.EndTimestampNS,
			))
		}

		// 6. Section ordering in stream: pass index should not regress backwards
		if sec.PassIndex < lastPassIndex {
			reasons = append(reasons, fmt.Sprintf(
				"%s pass_index %d regressed from prior pass_index %d", secDesc, sec.PassIndex, lastPassIndex,
			))
		}
		lastPassIndex = sec.PassIndex

		// Track pass start/end bounds
		if sec.StartTimestampNS > 0 {
			if curStart, ok := passStarts[sec.PassIndex]; !ok || sec.StartTimestampNS < curStart {
				passStarts[sec.PassIndex] = sec.StartTimestampNS
			}
		}
		if sec.EndTimestampNS > 0 {
			if curEnd, ok := passEnds[sec.PassIndex]; !ok || sec.EndTimestampNS > curEnd {
				passEnds[sec.PassIndex] = sec.EndTimestampNS
			}
		}

		// 7. Memory byte count sanity
		if sec.FreedBytes > 0 && sec.AllocatedBytes > 0 && sec.FreedBytes > sec.AllocatedBytes {
			reasons = append(reasons, fmt.Sprintf(
				"%s freed bytes (%d) exceed allocated bytes (%d)", secDesc, sec.FreedBytes, sec.AllocatedBytes,
			))
		}
	}

	// 8. Cross-pass temporal ordering: Pass N start must not precede Pass N-1 end
	// Collect sorted unique pass indices
	var passIndices []int
	for idx := range passIndexToID {
		passIndices = append(passIndices, idx)
	}
	res.PassCount = len(passIndices)
	sort.Ints(passIndices)

	for i := 1; i < len(passIndices); i++ {
		prevIdx := passIndices[i-1]
		currIdx := passIndices[i]
		prevEnd := passEnds[prevIdx]
		currStart := passStarts[currIdx]

		if prevEnd > 0 && currStart > 0 && currStart < prevEnd {
			reasons = append(reasons, fmt.Sprintf(
				"temporal overlap or regression: pass %d start (%d ns) precedes pass %d end (%d ns)",
				currIdx, currStart, prevIdx, prevEnd,
			))
		}
	}

	// Check total passes consistency if declared
	if evidence.TotalPasses > 0 && res.PassCount > evidence.TotalPasses {
		reasons = append(reasons, fmt.Sprintf(
			"observed pass count %d exceeds declared total_passes %d", res.PassCount, evidence.TotalPasses,
		))
	}

	if len(reasons) > 0 {
		res.Verdict = MemoryVerdictSpliced
		res.Reasons = reasons
		return res
	}

	res.Verdict = MemoryVerdictCoherent
	return res
}

func validateVendorMemoryEvidencePrelims(evidence *VendorMemoryEvidence) (MemorySplicingCheckResult, bool) {
	if evidence == nil {
		return MemorySplicingCheckResult{
			Verdict: MemoryVerdictUnknown,
			Reasons: []string{"evidence is nil"},
		}, false
	}
	if evidence.Schema != VendorMemoryEvidenceSchema {
		return MemorySplicingCheckResult{
			Verdict: MemoryVerdictUnknown,
			Reasons: []string{fmt.Sprintf("schema must be %q, got %q", VendorMemoryEvidenceSchema, evidence.Schema)},
		}, false
	}
	if len(evidence.Sections) == 0 {
		return MemorySplicingCheckResult{
			Verdict: MemoryVerdictUnknown,
			Reasons: []string{"zero memory sections present in evidence"},
		}, false
	}
	if strings.TrimSpace(evidence.Vendor) == "" {
		return MemorySplicingCheckResult{
			Verdict: MemoryVerdictUnknown,
			Reasons: []string{"vendor not specified in evidence"},
		}, false
	}
	return MemorySplicingCheckResult{}, true
}

func checkSectionStampIntegrity(sec VendorMemoryRecord, secDesc string, seenStamps map[string]string) []string {
	var reasons []string
	if sec.Stamp.Hash == "" {
		return nil
	}
	if sec.Stamp.PassID != sec.PassID {
		reasons = append(reasons, fmt.Sprintf(
			"%s stamp pass_id %q mismatches record pass_id %q", secDesc, sec.Stamp.PassID, sec.PassID,
		))
	}
	if sec.Stamp.PassIndex != sec.PassIndex {
		reasons = append(reasons, fmt.Sprintf(
			"%s stamp pass_index %d mismatches record pass_index %d", secDesc, sec.Stamp.PassIndex, sec.PassIndex,
		))
	}
	if sec.Stamp.RunID != "" && sec.RunID != "" && sec.Stamp.RunID != sec.RunID {
		reasons = append(reasons, fmt.Sprintf(
			"%s stamp run_id %q mismatches record run_id %q", secDesc, sec.Stamp.RunID, sec.RunID,
		))
	}
	if sec.Stamp.DeviceID != "" && sec.DeviceID != "" && sec.Stamp.DeviceID != sec.DeviceID {
		reasons = append(reasons, fmt.Sprintf(
			"%s stamp device_id %q mismatches record device_id %q", secDesc, sec.Stamp.DeviceID, sec.DeviceID,
		))
	}

	// Validate hash
	expectedHash := ComputeSectionStamp(sec)
	if !strings.EqualFold(sec.Stamp.Hash, expectedHash) {
		reasons = append(reasons, fmt.Sprintf(
			"%s stamp hash mismatch: computed %s, found %s", secDesc, expectedHash, sec.Stamp.Hash,
		))
	}

	// Check stamp reuse across different pass/section
	currentTag := fmt.Sprintf("pass%d:%s", sec.PassIndex, sec.SectionName)
	if priorTag, exists := seenStamps[sec.Stamp.Hash]; exists && priorTag != currentTag {
		reasons = append(reasons, fmt.Sprintf(
			"%s reuses stamp hash from %s", secDesc, priorTag,
		))
	}
	seenStamps[sec.Stamp.Hash] = currentTag
	return reasons
}
