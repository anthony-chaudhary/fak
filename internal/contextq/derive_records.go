// Package contextq derives bounded, immutable observations from addressable context.
package contextq

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	// RecordPlanSchema is the only plan schema understood by DeriveRecords.
	RecordPlanSchema = "fak.context-query-plan/1"
	// DerivedRecordViewSchema identifies the immutable derived-view receipt.
	DerivedRecordViewSchema = "fak.context-derived-view/1"
)

// RecordOperation is the closed v1 record-query vocabulary.
type RecordOperation string

const (
	RecordOperationProject    RecordOperation = "project"
	RecordOperationGroupCount RecordOperation = "group_count"
)

// RecordEqualFilter retains records whose named field is exactly Value. V1 deliberately
// accepts strings only: coercion would make replay depend on parser conventions.
type RecordEqualFilter struct {
	Field string `json:"field"`
	Value string `json:"value"`
}

// RecordPlan describes one deterministic operation over an NDJSON object stream.
// Project and GroupBy are mutually exclusive according to Operation. Project
// fields are canonicalized into lexical order before hashing and encoding.
type RecordPlan struct {
	Schema    string             `json:"schema"`
	Operation RecordOperation    `json:"operation"`
	Filter    *RecordEqualFilter `json:"filter,omitempty"`
	Project   []string           `json:"project,omitempty"`
	GroupBy   string             `json:"group_by,omitempty"`
}

// RecordLimits defines hard refusal bounds. Zero is invalid rather than unbounded so a
// caller cannot accidentally create an ungoverned query with a zero value.
type RecordLimits struct {
	MaxSourceBytes int64 `json:"max_source_bytes"`
	MaxOutputBytes int64 `json:"max_output_bytes"`
	MaxRecords     int   `json:"max_records"`
	MaxWorkUnits   int64 `json:"max_work_units"`
}

// RecordAccounting separates source size, visible derived size, record counts, and
// deterministic work. A work unit is one record decode, filter comparison, or
// projected/grouped field access.
type RecordAccounting struct {
	SourceBytes   int64 `json:"source_bytes"`
	OutputBytes   int64 `json:"output_bytes"`
	RecordsRead   int   `json:"records_read"`
	RecordsMatch  int   `json:"records_matched"`
	RecordsOutput int   `json:"records_output"`
	WorkUnits     int64 `json:"work_units"`
}

// DerivedRecordView is the payload-free lineage and accounting receipt for Output.
// DerivationDigest hashes the canonical RecordPlan bytes used to produce Output,
// separating the derivation specification from the source and output byte digests.
// Output is omitted from JSON because an inline abi.Ref can contain the derived
// bytes; callers explicitly resolve it when they intend to admit the view.
type DerivedRecordView struct {
	Schema           string           `json:"schema"`
	SourceDigest     string           `json:"source_digest"`
	DerivationDigest string           `json:"derivation_digest"`
	OutputDigest     string           `json:"output_digest"`
	Taint            abi.TaintLabel   `json:"taint"`
	Scope            abi.ShareScope   `json:"scope"`
	Truncated        bool             `json:"truncated"`
	Accounting       RecordAccounting `json:"accounting"`
	Output           abi.Ref          `json:"-"`
}

// DeriveReason is the closed refusal vocabulary for DeriveRecords.
type DeriveReason string

const (
	DeriveReasonCanceled             DeriveReason = "canceled"
	DeriveReasonResolverMissing      DeriveReason = "resolver_missing"
	DeriveReasonLimitsInvalid        DeriveReason = "limits_invalid"
	DeriveReasonTaintInvalid         DeriveReason = "taint_invalid"
	DeriveReasonScopeInvalid         DeriveReason = "scope_invalid"
	DeriveReasonSourceQuarantined    DeriveReason = "source_quarantined"
	DeriveReasonDigestInvalid        DeriveReason = "digest_invalid"
	DeriveReasonDigestMismatch       DeriveReason = "digest_mismatch"
	DeriveReasonLengthMismatch       DeriveReason = "length_mismatch"
	DeriveReasonSourceLimit          DeriveReason = "source_limit"
	DeriveReasonRecordLimit          DeriveReason = "record_limit"
	DeriveReasonWorkLimit            DeriveReason = "work_limit"
	DeriveReasonOutputLimit          DeriveReason = "output_limit"
	DeriveReasonResolveFailed        DeriveReason = "resolve_failed"
	DeriveReasonMalformedSource      DeriveReason = "malformed_source"
	DeriveReasonRecordInvalid        DeriveReason = "record_invalid"
	DeriveReasonUnsupportedPlan      DeriveReason = "unsupported_plan"
	DeriveReasonPutFailed            DeriveReason = "put_failed"
	DeriveReasonOutputDigestMismatch DeriveReason = "output_digest_mismatch"
)

// DeriveRefusal is a typed, fail-closed query error.
type DeriveRefusal struct {
	Reason DeriveReason
	Detail string
}

func (e *DeriveRefusal) Error() string {
	if e.Detail == "" {
		return "contextq: " + string(e.Reason)
	}
	return "contextq: " + string(e.Reason) + ": " + e.Detail
}

// DeriveRecords resolves one immutable NDJSON source, executes the closed v1
// plan, and stores the canonical derived bytes through resolver. No successful
// result is partial: every exceeded bound returns a DeriveRefusal and Truncated is
// therefore always false on success.
func DeriveRecords(ctx context.Context, resolver abi.Resolver, source abi.Ref, plan RecordPlan, limits RecordLimits) (DerivedRecordView, error) {
	if err := contextRefusal(ctx); err != nil {
		return DerivedRecordView{}, err
	}
	if resolver == nil {
		return DerivedRecordView{}, refuse(DeriveReasonResolverMissing, "an abi.Resolver is required")
	}
	if limits.MaxSourceBytes <= 0 || limits.MaxOutputBytes <= 0 || limits.MaxRecords <= 0 || limits.MaxWorkUnits <= 0 {
		return DerivedRecordView{}, refuse(DeriveReasonLimitsInvalid, "all limits must be positive")
	}
	if !validTaintLabel(source.Taint) {
		return DerivedRecordView{}, refuse(DeriveReasonTaintInvalid, "source taint is outside the closed ABI vocabulary")
	}
	if !validShareScope(source.Scope) {
		return DerivedRecordView{}, refuse(DeriveReasonScopeInvalid, "source scope is outside the closed ABI vocabulary")
	}
	if source.Taint == abi.TaintQuarantined {
		return DerivedRecordView{}, refuse(DeriveReasonSourceQuarantined, "quarantined refs cannot be queried")
	}
	if !validDigest(source.Digest) {
		return DerivedRecordView{}, refuse(DeriveReasonDigestInvalid, "source digest must be 64 lowercase hexadecimal characters")
	}
	if source.Len < 0 {
		return DerivedRecordView{}, refuse(DeriveReasonLengthMismatch, "source length is negative")
	}
	if source.Len > limits.MaxSourceBytes {
		return DerivedRecordView{}, refuse(DeriveReasonSourceLimit, "declared source length exceeds max_source_bytes")
	}

	canonicalPlan, err := normalizePlan(plan)
	if err != nil {
		return DerivedRecordView{}, err
	}
	planBytes, err := json.Marshal(canonicalPlan)
	if err != nil {
		return DerivedRecordView{}, refuse(DeriveReasonUnsupportedPlan, "canonical plan encoding failed")
	}

	body, err := resolver.Resolve(ctx, source)
	if err != nil {
		if cerr := contextRefusal(ctx); cerr != nil {
			return DerivedRecordView{}, cerr
		}
		return DerivedRecordView{}, refuse(DeriveReasonResolveFailed, err.Error())
	}
	if err := contextRefusal(ctx); err != nil {
		return DerivedRecordView{}, err
	}
	if int64(len(body)) > limits.MaxSourceBytes {
		return DerivedRecordView{}, refuse(DeriveReasonSourceLimit, "resolved source exceeds max_source_bytes")
	}
	if source.Len != int64(len(body)) {
		return DerivedRecordView{}, refuse(DeriveReasonLengthMismatch, "declared source length does not match resolved bytes")
	}
	if digest(body) != source.Digest {
		return DerivedRecordView{}, refuse(DeriveReasonDigestMismatch, "source digest does not match resolved bytes")
	}

	result, accounting, err := execute(ctx, body, canonicalPlan, limits)
	if err != nil {
		return DerivedRecordView{}, err
	}
	if int64(len(result)) > limits.MaxOutputBytes {
		return DerivedRecordView{}, refuse(DeriveReasonOutputLimit, "derived bytes exceed max_output_bytes")
	}
	accounting.SourceBytes = int64(len(body))
	accounting.OutputBytes = int64(len(result))

	outDigest := digest(result)
	output, err := resolver.Put(ctx, result)
	if err != nil {
		if cerr := contextRefusal(ctx); cerr != nil {
			return DerivedRecordView{}, cerr
		}
		return DerivedRecordView{}, refuse(DeriveReasonPutFailed, err.Error())
	}
	if err := contextRefusal(ctx); err != nil {
		return DerivedRecordView{}, err
	}
	if !validDigest(output.Digest) || output.Digest != outDigest || output.Len != int64(len(result)) {
		return DerivedRecordView{}, refuse(DeriveReasonOutputDigestMismatch, "resolver returned an inconsistent output ref")
	}
	// Resolver.Put defaults to private/tainted. The derived view inherits the
	// exact source labels so it is never widened or promoted by derivation.
	output.Taint = source.Taint
	output.Scope = source.Scope

	return DerivedRecordView{
		Schema:           DerivedRecordViewSchema,
		SourceDigest:     source.Digest,
		DerivationDigest: digest(planBytes),
		OutputDigest:     outDigest,
		Taint:            source.Taint,
		Scope:            source.Scope,
		Truncated:        false,
		Accounting:       accounting,
		Output:           output,
	}, nil
}

func validTaintLabel(label abi.TaintLabel) bool {
	switch label {
	case abi.TaintTainted, abi.TaintTrusted, abi.TaintQuarantined:
		return true
	default:
		return false
	}
}

func validShareScope(scope abi.ShareScope) bool {
	switch scope {
	case abi.ScopeAgent, abi.ScopeFleet, abi.ScopeTenant:
		return true
	default:
		return false
	}
}

func execute(ctx context.Context, body []byte, plan RecordPlan, limits RecordLimits) ([]byte, RecordAccounting, error) {
	var accounting RecordAccounting
	projected := make([]map[string]json.RawMessage, 0)
	groups := make(map[string]int)

	lines := bytes.Split(body, []byte{'\n'})
	for i, rawLine := range lines {
		if i == len(lines)-1 && len(rawLine) == 0 {
			continue
		}
		if err := contextRefusal(ctx); err != nil {
			return nil, RecordAccounting{}, err
		}
		line := bytes.TrimSpace(rawLine)
		if len(line) == 0 {
			return nil, RecordAccounting{}, refuse(DeriveReasonMalformedSource, fmt.Sprintf("record %d is blank", i+1))
		}
		if accounting.RecordsRead >= limits.MaxRecords {
			return nil, RecordAccounting{}, refuse(DeriveReasonRecordLimit, "source contains more records than max_records")
		}
		if err := charge(&accounting, limits, 1); err != nil {
			return nil, RecordAccounting{}, err
		}
		record, err := decodeRecord(line)
		if err != nil {
			return nil, RecordAccounting{}, refuse(DeriveReasonMalformedSource, fmt.Sprintf("record %d: %v", i+1, err))
		}
		accounting.RecordsRead++

		matched := true
		if plan.Filter != nil {
			if err := charge(&accounting, limits, 1); err != nil {
				return nil, RecordAccounting{}, err
			}
			matched, err = fieldEquals(record, *plan.Filter)
			if err != nil {
				return nil, RecordAccounting{}, refuse(DeriveReasonRecordInvalid, fmt.Sprintf("record %d: %v", i+1, err))
			}
		}
		if !matched {
			continue
		}
		accounting.RecordsMatch++

		switch plan.Operation {
		case RecordOperationProject:
			if err := charge(&accounting, limits, int64(len(plan.Project))); err != nil {
				return nil, RecordAccounting{}, err
			}
			row := make(map[string]json.RawMessage, len(plan.Project))
			for _, field := range plan.Project {
				value, ok := record[field]
				if !ok {
					return nil, RecordAccounting{}, refuse(DeriveReasonRecordInvalid, fmt.Sprintf("record %d has no projected field %q", i+1, field))
				}
				row[field] = append(json.RawMessage(nil), value...)
			}
			projected = append(projected, row)
		case RecordOperationGroupCount:
			if err := charge(&accounting, limits, 1); err != nil {
				return nil, RecordAccounting{}, err
			}
			value, err := stringField(record, plan.GroupBy)
			if err != nil {
				return nil, RecordAccounting{}, refuse(DeriveReasonRecordInvalid, fmt.Sprintf("record %d: %v", i+1, err))
			}
			groups[value]++
		}
	}

	var result []byte
	var err error
	switch plan.Operation {
	case RecordOperationProject:
		accounting.RecordsOutput = len(projected)
		result, err = json.Marshal(projected)
	case RecordOperationGroupCount:
		keys := make([]string, 0, len(groups))
		for key := range groups {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		rows := make([]groupCount, 0, len(keys))
		for _, key := range keys {
			rows = append(rows, groupCount{Group: key, Count: groups[key]})
		}
		accounting.RecordsOutput = len(rows)
		result, err = json.Marshal(rows)
	}
	if err != nil {
		return nil, RecordAccounting{}, refuse(DeriveReasonRecordInvalid, "derived output encoding failed")
	}
	return result, accounting, nil
}

type groupCount struct {
	Group string `json:"group"`
	Count int    `json:"count"`
}

func normalizePlan(plan RecordPlan) (RecordPlan, error) {
	if plan.Schema != RecordPlanSchema {
		return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "unsupported plan schema")
	}
	if plan.Filter != nil {
		field := strings.TrimSpace(plan.Filter.Field)
		if field == "" {
			return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "filter field is required")
		}
		filter := *plan.Filter
		filter.Field = field
		plan.Filter = &filter
	}
	plan.GroupBy = strings.TrimSpace(plan.GroupBy)
	switch plan.Operation {
	case RecordOperationProject:
		if plan.GroupBy != "" || len(plan.Project) == 0 {
			return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "project requires fields and forbids group_by")
		}
		fields := append([]string(nil), plan.Project...)
		seen := make(map[string]struct{}, len(fields))
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
			if fields[i] == "" {
				return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "project fields must be non-empty")
			}
			if _, duplicate := seen[fields[i]]; duplicate {
				return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "project fields must be unique")
			}
			seen[fields[i]] = struct{}{}
		}
		sort.Strings(fields)
		plan.Project = fields
	case RecordOperationGroupCount:
		if plan.GroupBy == "" || len(plan.Project) != 0 {
			return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "group_count requires group_by and forbids project")
		}
	default:
		return RecordPlan{}, refuse(DeriveReasonUnsupportedPlan, "unsupported operation")
	}
	return plan, nil
}

func decodeRecord(line []byte) (map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(line))
	start, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := start.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("record must be a JSON object")
	}
	record := make(map[string]json.RawMessage)
	for dec.More() {
		keyToken, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := keyToken.(string)
		if !ok {
			return nil, fmt.Errorf("object key is not a string")
		}
		if _, duplicate := record[key]; duplicate {
			return nil, fmt.Errorf("duplicate field %q", key)
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		record[key] = append(json.RawMessage(nil), value...)
	}
	end, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := end.(json.Delim); !ok || delim != '}' {
		return nil, fmt.Errorf("record object did not terminate")
	}
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("record has trailing JSON")
		}
		return nil, err
	}
	return record, nil
}

func fieldEquals(record map[string]json.RawMessage, filter RecordEqualFilter) (bool, error) {
	raw, ok := record[filter.Field]
	if !ok {
		return false, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return false, fmt.Errorf("filter field %q must be a string", filter.Field)
	}
	return value == filter.Value, nil
}

func stringField(record map[string]json.RawMessage, field string) (string, error) {
	raw, ok := record[field]
	if !ok {
		return "", fmt.Errorf("group field %q is missing", field)
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", fmt.Errorf("group field %q must be a string", field)
	}
	return value, nil
}

func charge(accounting *RecordAccounting, limits RecordLimits, units int64) error {
	if units < 0 || accounting.WorkUnits > limits.MaxWorkUnits-units {
		return refuse(DeriveReasonWorkLimit, "query work exceeds max_work_units")
	}
	accounting.WorkUnits += units
	return nil
}

func contextRefusal(ctx context.Context) error {
	if ctx == nil {
		return refuse(DeriveReasonCanceled, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return refuse(DeriveReasonCanceled, err.Error())
	}
	return nil
}

func refuse(reason DeriveReason, detail string) error {
	return &DeriveRefusal{Reason: reason, Detail: detail}
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func validDigest(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
