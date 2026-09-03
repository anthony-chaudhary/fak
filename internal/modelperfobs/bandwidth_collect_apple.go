package modelperfobs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"
)

const (
	AppleMemoryImportSchema   = "fak-apple-memory-counter-import/1"
	AppleMemoryArtifactSchema = "fak-apple-unified-memory-counters/1"

	maxAppleMemoryCounterImportBytes   = 16 << 20
	appleMemoryTemporalAttributionOnly = "request phase is temporal attribution only; package/system traffic is not process or model traffic"
)

type AppleMemoryImportFormat string

const (
	AppleMemoryFormatAuto        AppleMemoryImportFormat = "auto"
	AppleMemoryFormatGenericJSON AppleMemoryImportFormat = "generic-json"
)

type AppleMemoryScope struct {
	Kind string `json:"kind"`
	ID   string `json:"id,omitempty"`
}

type AppleMemoryImportOptions struct {
	Format           AppleMemoryImportFormat
	Provider         string
	ProviderVersion  string
	Scope            AppleMemoryScope
	CaptureStartedAt time.Time
	CaptureEndedAt   time.Time
	Interval         time.Duration
	Phase            RequestPhase
	Shape            RequestShape
	TheoreticalGBS   *float64
	MeasuredHostGBS  *float64
}

type AppleMemoryRawValue struct {
	CapturedAt    string `json:"captured_at"`
	Value         uint64 `json:"value"`
	ResetObserved *bool  `json:"reset_observed,omitempty"`
}

type AppleMemoryCounterArtifact struct {
	Direction          string                `json:"direction"`
	Semantic           string                `json:"semantic"`
	RawField           string                `json:"raw_field"`
	Unit               string                `json:"unit"`
	Scope              AppleMemoryScope      `json:"scope"`
	RawValues          []AppleMemoryRawValue `json:"raw_values,omitempty"`
	RateBytesPerSecond *uint64               `json:"rate_bytes_per_second,omitempty"`
	IntervalBytes      *uint64               `json:"interval_bytes,omitempty"`
	ResetObserved      *bool                 `json:"reset_observed,omitempty"`
	ByteProvenance     string                `json:"byte_provenance"`
}

type AppleMemoryArtifact struct {
	Schema              string                       `json:"schema"`
	Provider            string                       `json:"provider"`
	ProviderVersion     string                       `json:"provider_version"`
	ImportFormat        AppleMemoryImportFormat      `json:"import_format"`
	Scope               AppleMemoryScope             `json:"scope"`
	CaptureStartedAt    string                       `json:"capture_started_at"`
	CaptureEndedAt      string                       `json:"capture_ended_at"`
	IntervalNS          int64                        `json:"interval_ns"`
	CaptureStatus       string                       `json:"capture_status"`
	CaptureMetadata     map[string]string            `json:"capture_metadata"`
	RawProviderFields   map[string]json.RawMessage   `json:"raw_provider_fields"`
	Counters            []AppleMemoryCounterArtifact `json:"counters"`
	ReadBytesPerSecond  *uint64                      `json:"read_bytes_per_second,omitempty"`
	WriteBytesPerSecond *uint64                      `json:"write_bytes_per_second,omitempty"`
	TotalBytesPerSecond *uint64                      `json:"total_bytes_per_second,omitempty"`
	ReadBytes           *uint64                      `json:"read_bytes,omitempty"`
	WriteBytes          *uint64                      `json:"write_bytes,omitempty"`
	TotalBytes          *uint64                      `json:"total_bytes,omitempty"`
	PhaseAttribution    string                       `json:"phase_attribution"`
}

type genericAppleMemoryImport struct {
	Schema            string                      `json:"schema"`
	Provider          string                      `json:"provider"`
	ProviderVersion   string                      `json:"provider_version"`
	Scope             AppleMemoryScope            `json:"scope"`
	Capture           genericAppleMemoryCapture   `json:"capture"`
	Counters          []genericAppleMemoryCounter `json:"counters"`
	RawProviderFields map[string]json.RawMessage  `json:"raw_provider_fields"`
}

type genericAppleMemoryCapture struct {
	Status     string            `json:"status"`
	StartedAt  string            `json:"started_at"`
	EndedAt    string            `json:"ended_at"`
	IntervalNS int64             `json:"interval_ns"`
	Metadata   map[string]string `json:"metadata"`
	Error      string            `json:"error,omitempty"`
}

type genericAppleMemoryCounter struct {
	Direction          string                `json:"direction"`
	Semantic           string                `json:"semantic"`
	Field              string                `json:"field"`
	Unit               string                `json:"unit"`
	Scope              AppleMemoryScope      `json:"scope"`
	RateBytesPerSecond *uint64               `json:"rate_bytes_per_second,omitempty"`
	Snapshots          []AppleMemoryRawValue `json:"snapshots,omitempty"`
}

// ImportAppleMemoryCounters imports a stable normalized envelope around Apple
// counter evidence. A provider adapter must establish byte units before
// producing this schema; this importer never derives traffic from power,
// utilization, capacity, process/storage I/O, or a theoretical roofline.
func ImportAppleMemoryCounters(r io.Reader, o AppleMemoryImportOptions) (BandwidthCollection, error) {
	if r == nil {
		return BandwidthCollection{}, errors.New("Apple memory counter input is nil")
	}
	normalizeAppleMemoryOptions(&o)
	if err := validateAppleMemoryOptions(o); err != nil {
		return BandwidthCollection{}, err
	}
	data, err := io.ReadAll(io.LimitReader(r, maxAppleMemoryCounterImportBytes+1))
	if err != nil {
		return BandwidthCollection{}, err
	}
	if len(data) > maxAppleMemoryCounterImportBytes {
		return BandwidthCollection{}, fmt.Errorf("Apple memory counter input exceeds %d bytes", maxAppleMemoryCounterImportBytes)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return BandwidthCollection{}, errors.New("Apple memory counter input is empty")
	}
	format, err := detectAppleMemoryFormat(data, o.Format)
	if err != nil {
		return BandwidthCollection{}, err
	}
	if format != AppleMemoryFormatGenericJSON {
		return BandwidthCollection{}, fmt.Errorf("unsupported Apple memory import format %q", format)
	}
	artifact, err := parseGenericAppleMemoryCounters(data, o)
	if err != nil {
		return BandwidthCollection{}, err
	}
	return appleMemoryCollection(artifact, o)
}

func normalizeAppleMemoryOptions(o *AppleMemoryImportOptions) {
	o.Provider = strings.TrimSpace(o.Provider)
	o.ProviderVersion = strings.TrimSpace(o.ProviderVersion)
	o.Scope.Kind = strings.TrimSpace(o.Scope.Kind)
	o.Scope.ID = strings.TrimSpace(o.Scope.ID)
	if o.Format == "" {
		o.Format = AppleMemoryFormatAuto
	}
}

func validateAppleMemoryOptions(o AppleMemoryImportOptions) error {
	if o.Provider == "" {
		return errors.New("Apple memory counter provider is required")
	}
	if o.ProviderVersion == "" {
		return errors.New("Apple memory counter provider version is required")
	}
	if err := validateAppleMemoryScope(o.Scope); err != nil {
		return err
	}
	switch o.Format {
	case AppleMemoryFormatAuto, AppleMemoryFormatGenericJSON:
	default:
		return fmt.Errorf("unsupported Apple memory import format %q", o.Format)
	}
	if o.CaptureStartedAt.IsZero() != o.CaptureEndedAt.IsZero() {
		return errors.New("Apple capture start and end must be supplied together")
	}
	if !o.CaptureStartedAt.IsZero() && !o.CaptureEndedAt.After(o.CaptureStartedAt) {
		return errors.New("Apple capture end must be after start")
	}
	if o.Interval < 0 {
		return errors.New("Apple capture interval must not be negative")
	}
	if o.Interval > 0 && !o.CaptureStartedAt.IsZero() && o.CaptureEndedAt.Sub(o.CaptureStartedAt) != o.Interval {
		return errors.New("Apple capture interval conflicts with capture bounds")
	}
	for name, value := range map[string]*float64{
		"theoretical host memory roofline": o.TheoreticalGBS,
		"measured host memory roofline":    o.MeasuredHostGBS,
	} {
		if value != nil && (*value <= 0 || math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return fmt.Errorf("%s must be positive and finite", name)
		}
	}
	return validateSample(BandwidthSample{
		Phase: o.Phase, Shape: o.Shape,
		Provenance: BandwidthProvenance{Source: "apple-memory-counters"},
	})
}

func validateAppleMemoryScope(scope AppleMemoryScope) error {
	switch scope.Kind {
	case "system", "package":
		if scope.ID != "" {
			return fmt.Errorf("%s Apple memory scope must not have an id", scope.Kind)
		}
	default:
		return fmt.Errorf("Apple memory scope must be system or package, got %q", scope.Kind)
	}
	return nil
}

func detectAppleMemoryFormat(data []byte, requested AppleMemoryImportFormat) (AppleMemoryImportFormat, error) {
	if requested != "" && requested != AppleMemoryFormatAuto {
		return requested, nil
	}
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return "", errors.New("Apple memory counter input is empty")
	}
	if trimmed[0] == '{' {
		return AppleMemoryFormatGenericJSON, nil
	}
	return "", errors.New("Apple memory counter input is not the supported generic JSON schema")
}

func validateAppleMemoryJSONShape(data []byte) error {
	if err := rejectDuplicateJSONNames(data); err != nil {
		return fmt.Errorf("parse generic Apple memory counters: %w", err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return fmt.Errorf("parse generic Apple memory counters: %w", err)
	}
	if err := requireExactJSONKeys("top level", top,
		"schema", "provider", "provider_version", "scope", "capture", "counters", "raw_provider_fields"); err != nil {
		return err
	}
	if err := validateAppleScopeJSON("scope", top["scope"]); err != nil {
		return err
	}
	var capture map[string]json.RawMessage
	if err := json.Unmarshal(top["capture"], &capture); err != nil {
		return fmt.Errorf("generic Apple capture must be an object: %w", err)
	}
	if err := requireExactJSONKeys("capture", capture,
		"status", "started_at", "ended_at", "interval_ns", "metadata", "error"); err != nil {
		return err
	}
	var counters []json.RawMessage
	if err := json.Unmarshal(top["counters"], &counters); err != nil {
		return fmt.Errorf("generic Apple counters must be an array: %w", err)
	}
	for i, rawCounter := range counters {
		var counter map[string]json.RawMessage
		if err := json.Unmarshal(rawCounter, &counter); err != nil {
			return fmt.Errorf("generic Apple counter %d must be an object: %w", i+1, err)
		}
		path := fmt.Sprintf("counter %d", i+1)
		if err := requireExactJSONKeys(path, counter,
			"direction", "semantic", "field", "unit", "scope", "rate_bytes_per_second", "snapshots"); err != nil {
			return err
		}
		if err := validateAppleScopeJSON(path+" scope", counter["scope"]); err != nil {
			return err
		}
		if rawSnapshots, ok := counter["snapshots"]; ok {
			var snapshots []json.RawMessage
			if err := json.Unmarshal(rawSnapshots, &snapshots); err != nil {
				return fmt.Errorf("generic Apple %s snapshots must be an array: %w", path, err)
			}
			for j, rawSnapshot := range snapshots {
				var snapshot map[string]json.RawMessage
				if err := json.Unmarshal(rawSnapshot, &snapshot); err != nil {
					return fmt.Errorf("generic Apple %s snapshot %d must be an object: %w", path, j+1, err)
				}
				if err := requireExactJSONKeys(fmt.Sprintf("%s snapshot %d", path, j+1), snapshot,
					"captured_at", "value", "reset_observed"); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func validateAppleScopeJSON(path string, raw json.RawMessage) error {
	var scope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &scope); err != nil {
		return fmt.Errorf("generic Apple %s must be an object: %w", path, err)
	}
	return requireExactJSONKeys(path, scope, "kind", "id")
}

func requireExactJSONKeys(path string, object map[string]json.RawMessage, allowed ...string) error {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedSet[key] = struct{}{}
	}
	for key := range object {
		if _, ok := allowedSet[key]; !ok {
			return fmt.Errorf("generic Apple %s has unknown or non-exact field %q", path, key)
		}
	}
	return nil
}

func rejectDuplicateJSONNames(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	var walk func(json.Token) error
	var walkNext func() error
	walkNext = func() error {
		valueToken, err := dec.Token()
		if err != nil {
			return err
		}
		return walk(valueToken)
	}
	walk = func(token json.Token) error {
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]struct{})
			for dec.More() {
				nameToken, err := dec.Token()
				if err != nil {
					return err
				}
				name, ok := nameToken.(string)
				if !ok {
					return errors.New("JSON object name is not a string")
				}
				if _, exists := seen[name]; exists {
					return fmt.Errorf("duplicate JSON object name %q", name)
				}
				seen[name] = struct{}{}
				if err := walkNext(); err != nil {
					return err
				}
			}
			end, err := dec.Token()
			if err != nil {
				return err
			}
			if end != json.Delim('}') {
				return errors.New("JSON object is malformed")
			}
		case '[':
			for dec.More() {
				if err := walkNext(); err != nil {
					return err
				}
			}
			end, err := dec.Token()
			if err != nil {
				return err
			}
			if end != json.Delim(']') {
				return errors.New("JSON array is malformed")
			}
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
		return nil
	}
	first, err := dec.Token()
	if err != nil {
		return err
	}
	if err := walk(first); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func parseGenericAppleMemoryCounters(data []byte, o AppleMemoryImportOptions) (AppleMemoryArtifact, error) {
	if err := validateAppleMemoryJSONShape(data); err != nil {
		return AppleMemoryArtifact{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var input genericAppleMemoryImport
	if err := dec.Decode(&input); err != nil {
		return AppleMemoryArtifact{}, fmt.Errorf("parse generic Apple memory counters: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return AppleMemoryArtifact{}, err
	}
	if input.Schema != AppleMemoryImportSchema {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple memory counter schema must be %q", AppleMemoryImportSchema)
	}
	if input.Provider != o.Provider {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple memory provider %q does not match declared provider %q", input.Provider, o.Provider)
	}
	if input.ProviderVersion != o.ProviderVersion {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple memory provider version %q does not match declared version %q", input.ProviderVersion, o.ProviderVersion)
	}
	if input.Scope != o.Scope {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple memory scope %+v does not match declared scope %+v", input.Scope, o.Scope)
	}
	switch input.Capture.Status {
	case "ok":
		if input.Capture.Error != "" {
			return AppleMemoryArtifact{}, errors.New("successful generic Apple capture must not contain an error")
		}
	case "permission-denied":
		return AppleMemoryArtifact{}, fmt.Errorf("Apple memory provider capture is permission-gated: %s", strings.TrimSpace(input.Capture.Error))
	case "unsupported":
		return AppleMemoryArtifact{}, fmt.Errorf("Apple memory provider capture is unsupported: %s", strings.TrimSpace(input.Capture.Error))
	default:
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple capture status must be ok, permission-denied, or unsupported, got %q", input.Capture.Status)
	}
	started, err := time.Parse(time.RFC3339Nano, input.Capture.StartedAt)
	if err != nil {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple capture start must be RFC3339: %w", err)
	}
	ended, err := time.Parse(time.RFC3339Nano, input.Capture.EndedAt)
	if err != nil {
		return AppleMemoryArtifact{}, fmt.Errorf("generic Apple capture end must be RFC3339: %w", err)
	}
	if !ended.After(started) {
		return AppleMemoryArtifact{}, errors.New("generic Apple capture end must be after start")
	}
	if input.Capture.IntervalNS <= 0 || time.Duration(input.Capture.IntervalNS) != ended.Sub(started) {
		return AppleMemoryArtifact{}, errors.New("generic Apple capture interval must be positive and match capture bounds")
	}
	if !o.CaptureStartedAt.IsZero() && !o.CaptureStartedAt.Equal(started) {
		return AppleMemoryArtifact{}, errors.New("generic Apple capture start conflicts with CLI capture metadata")
	}
	if !o.CaptureEndedAt.IsZero() && !o.CaptureEndedAt.Equal(ended) {
		return AppleMemoryArtifact{}, errors.New("generic Apple capture end conflicts with CLI capture metadata")
	}
	if o.Interval > 0 && o.Interval != time.Duration(input.Capture.IntervalNS) {
		return AppleMemoryArtifact{}, errors.New("generic Apple capture interval conflicts with CLI capture metadata")
	}
	metadata, err := validateAppleCaptureMetadata(input.Capture.Metadata)
	if err != nil {
		return AppleMemoryArtifact{}, err
	}
	rawFields, err := cloneRawProviderFields(input.RawProviderFields)
	if err != nil {
		return AppleMemoryArtifact{}, err
	}
	artifact := newAppleMemoryArtifact(o, started, ended)
	artifact.CaptureMetadata = metadata
	artifact.RawProviderFields = rawFields
	for _, counter := range input.Counters {
		parsed, err := parseGenericAppleMemoryCounter(counter, input.Scope, started, ended)
		if err != nil {
			return AppleMemoryArtifact{}, err
		}
		if _, ok := rawFields[parsed.RawField]; !ok {
			return AppleMemoryArtifact{}, fmt.Errorf("generic Apple raw_provider_fields is missing counter field %q", parsed.RawField)
		}
		if err := validateAppleRawCounterBinding(rawFields[parsed.RawField], parsed); err != nil {
			return AppleMemoryArtifact{}, err
		}
		artifact.Counters = append(artifact.Counters, parsed)
	}
	if err := finishAppleMemoryArtifact(&artifact); err != nil {
		return AppleMemoryArtifact{}, err
	}
	return artifact, nil
}

func validateAppleCaptureMetadata(input map[string]string) (map[string]string, error) {
	required := []string{"os_version", "hardware_model", "capture_method"}
	out := make(map[string]string, len(input))
	for key, value := range input {
		if key == "" || key != strings.TrimSpace(key) {
			return nil, errors.New("generic Apple capture metadata key must be non-empty and whitespace-exact")
		}
		out[key] = value
	}
	for _, key := range required {
		if strings.TrimSpace(out[key]) == "" {
			return nil, fmt.Errorf("generic Apple capture metadata requires non-empty %q", key)
		}
	}
	return out, nil
}

func cloneRawProviderFields(input map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	if len(input) == 0 {
		return nil, errors.New("generic Apple import requires raw_provider_fields")
	}
	out := make(map[string]json.RawMessage, len(input))
	for key, value := range input {
		if key == "" || key != strings.TrimSpace(key) {
			return nil, errors.New("generic Apple raw provider field name must be non-empty and whitespace-exact")
		}
		if len(bytes.TrimSpace(value)) == 0 || !json.Valid(value) {
			return nil, fmt.Errorf("generic Apple raw provider field %q is not valid JSON", key)
		}
		out[key] = append(json.RawMessage(nil), value...)
	}
	return out, nil
}

func validateAppleRawCounterBinding(raw json.RawMessage, counter AppleMemoryCounterArtifact) error {
	if counter.RateBytesPerSecond != nil {
		var value uint64
		if err := json.Unmarshal(raw, &value); err != nil {
			return fmt.Errorf("generic Apple raw provider field %q must be the exact unsigned byte rate: %w", counter.RawField, err)
		}
		if value != *counter.RateBytesPerSecond {
			return fmt.Errorf("generic Apple raw provider field %q value does not match normalized byte rate", counter.RawField)
		}
		return nil
	}
	var rawObject map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rawObject); err != nil {
		return fmt.Errorf("generic Apple raw provider field %q must contain exact start/end byte values: %w", counter.RawField, err)
	}
	if err := requireExactJSONKeys("raw provider field "+counter.RawField, rawObject, "start", "end"); err != nil {
		return err
	}
	var start, end uint64
	if err := json.Unmarshal(rawObject["start"], &start); err != nil {
		return fmt.Errorf("generic Apple raw provider field %q start must be an unsigned byte count: %w", counter.RawField, err)
	}
	if err := json.Unmarshal(rawObject["end"], &end); err != nil {
		return fmt.Errorf("generic Apple raw provider field %q end must be an unsigned byte count: %w", counter.RawField, err)
	}
	if len(counter.RawValues) != 2 || start != counter.RawValues[0].Value || end != counter.RawValues[1].Value {
		return fmt.Errorf("generic Apple raw provider field %q values do not match normalized snapshots", counter.RawField)
	}
	return nil
}

func parseGenericAppleMemoryCounter(counter genericAppleMemoryCounter, scope AppleMemoryScope, started, ended time.Time) (AppleMemoryCounterArtifact, error) {
	if counter.Field == "" || counter.Field != strings.TrimSpace(counter.Field) {
		return AppleMemoryCounterArtifact{}, errors.New("generic Apple counter field must be non-empty and whitespace-exact")
	}
	if err := rejectNonMemoryTrafficField(counter.Field); err != nil {
		return AppleMemoryCounterArtifact{}, err
	}
	if counter.Scope != scope {
		return AppleMemoryCounterArtifact{}, fmt.Errorf("mixed Apple memory counter scopes: %+v and %+v", scope, counter.Scope)
	}
	if counter.Direction != "read" && counter.Direction != "write" {
		return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter direction must be read or write, got %q", counter.Direction)
	}
	if err := validateAppleCounterFieldDirection(counter.Field, counter.Direction); err != nil {
		return AppleMemoryCounterArtifact{}, err
	}
	wantSemantic := "memory-read-bytes"
	if counter.Direction == "write" {
		wantSemantic = "memory-write-bytes"
	}
	if counter.Semantic != wantSemantic {
		return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple %s counter semantic must be %q, got %q", counter.Direction, wantSemantic, counter.Semantic)
	}
	out := AppleMemoryCounterArtifact{
		Direction: counter.Direction,
		Semantic:  counter.Semantic,
		RawField:  counter.Field,
		Unit:      counter.Unit,
		Scope:     counter.Scope,
	}
	if err := validateAppleCounterShape(counter); err != nil {
		return AppleMemoryCounterArtifact{}, err
	}
	switch counter.Unit {
	case "bytes_per_second":
		out.RateBytesPerSecond = counter.RateBytesPerSecond
		out.ByteProvenance = "direct-byte-rate"
	case "bytes":
		for i, snapshot := range counter.Snapshots {
			captured, err := time.Parse(time.RFC3339Nano, snapshot.CapturedAt)
			if err != nil {
				return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter %q snapshot %d time must be RFC3339: %w", counter.Field, i+1, err)
			}
			want := started
			if i == 1 {
				want = ended
			}
			if !captured.Equal(want) {
				return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter %q snapshot %d does not match capture bounds", counter.Field, i+1)
			}
			if snapshot.ResetObserved == nil {
				return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter %q reset state is ambiguous", counter.Field)
			}
			if *snapshot.ResetObserved {
				return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter %q reset during capture", counter.Field)
			}
		}
		startValue := counter.Snapshots[0].Value
		endValue := counter.Snapshots[1].Value
		if endValue < startValue {
			return AppleMemoryCounterArtifact{}, fmt.Errorf("generic Apple counter %q decreased without a supported wrap contract", counter.Field)
		}
		delta := endValue - startValue
		reset := false
		out.RawValues = append([]AppleMemoryRawValue(nil), counter.Snapshots...)
		out.IntervalBytes = &delta
		out.ResetObserved = &reset
		out.ByteProvenance = "monotonic-byte-delta"
	default:
		return AppleMemoryCounterArtifact{}, fmt.Errorf("unsupported Apple memory counter unit %q for %q", counter.Unit, counter.Field)
	}
	return out, nil
}

func appleCounterHasRate(counter genericAppleMemoryCounter) bool {
	return counter.RateBytesPerSecond != nil
}

func validateAppleCounterShape(counter genericAppleMemoryCounter) error {
	switch counter.Unit {
	case "bytes_per_second":
		if !appleCounterHasRate(counter) {
			return fmt.Errorf("generic Apple counter %q is missing rate_bytes_per_second", counter.Field)
		}
		if len(counter.Snapshots) != 0 {
			return fmt.Errorf("generic Apple counter %q mixes a direct rate with snapshots", counter.Field)
		}
	case "bytes":
		if counter.RateBytesPerSecond != nil {
			return fmt.Errorf("generic Apple counter %q mixes byte snapshots with a direct rate", counter.Field)
		}
		if len(counter.Snapshots) != 2 {
			return fmt.Errorf("generic Apple counter %q requires exactly two byte snapshots", counter.Field)
		}
	}
	return nil
}

func rejectNonMemoryTrafficField(field string) error {
	token := normalizedAppleCounterField(field)
	for strings.Contains(token, "__") {
		token = strings.ReplaceAll(token, "__", "_")
	}
	for _, forbidden := range []string{
		"process_io", "disk", "storage", "filesystem", "file_io",
		"power", "watt", "utilization", "capacity", "roofline",
		"resident", "allocated", "allocation", "footprint", "available", "free_bytes", "used_bytes", "total_bytes",
	} {
		if strings.Contains(token, forbidden) {
			return fmt.Errorf("Apple counter field %q names unsupported process/storage I/O, power, utilization, capacity, or roofline evidence", field)
		}
	}
	if token == "read_bytes" || token == "write_bytes" {
		return fmt.Errorf("Apple counter field %q is ambiguous process/storage I/O evidence", field)
	}
	return nil
}

func validateAppleCounterFieldDirection(field, direction string) error {
	token := "_" + normalizedAppleCounterField(field) + "_"
	opposite := "_write_"
	oppositeShort := "_wr_"
	if direction == "write" {
		opposite = "_read_"
		oppositeShort = "_rd_"
	}
	if strings.Contains(token, opposite) || strings.Contains(token, oppositeShort) {
		return fmt.Errorf("Apple counter field %q conflicts with declared %s direction", field, direction)
	}
	return nil
}

func normalizedAppleCounterField(field string) string {
	token := strings.NewReplacer("-", "_", ".", "_", "/", "_", " ", "_").Replace(strings.ToLower(field))
	for strings.Contains(token, "__") {
		token = strings.ReplaceAll(token, "__", "_")
	}
	return strings.Trim(token, "_")
}

func newAppleMemoryArtifact(o AppleMemoryImportOptions, started, ended time.Time) AppleMemoryArtifact {
	return AppleMemoryArtifact{
		Schema:           AppleMemoryArtifactSchema,
		Provider:         o.Provider,
		ProviderVersion:  o.ProviderVersion,
		ImportFormat:     AppleMemoryFormatGenericJSON,
		Scope:            o.Scope,
		CaptureStartedAt: started.UTC().Format(time.RFC3339Nano),
		CaptureEndedAt:   ended.UTC().Format(time.RFC3339Nano),
		IntervalNS:       ended.Sub(started).Nanoseconds(),
		CaptureStatus:    "ok",
		Counters:         make([]AppleMemoryCounterArtifact, 0, 2),
		PhaseAttribution: appleMemoryTemporalAttributionOnly,
	}
}

func finishAppleMemoryArtifact(artifact *AppleMemoryArtifact) error {
	if artifact == nil {
		return errors.New("Apple memory artifact is nil")
	}
	if artifact.IntervalNS <= 0 {
		return errors.New("Apple memory artifact interval must be positive")
	}
	var read, write *AppleMemoryCounterArtifact
	for i := range artifact.Counters {
		counter := &artifact.Counters[i]
		if counter.RawField == "" {
			return errors.New("Apple memory raw counter field is required")
		}
		if counter.Scope != artifact.Scope {
			return errors.New("mixed Apple memory counter scopes")
		}
		switch counter.Direction {
		case "read":
			if read != nil {
				return errors.New("Apple memory capture has duplicate read counters")
			}
			read = counter
		case "write":
			if write != nil {
				return errors.New("Apple memory capture has duplicate write counters")
			}
			write = counter
		default:
			return fmt.Errorf("Apple memory counter direction must be read or write, got %q", counter.Direction)
		}
	}
	if read == nil || write == nil {
		return errors.New("Apple memory capture requires one read/write counter pair")
	}
	if read.ByteProvenance != write.ByteProvenance {
		return errors.New("Apple memory read/write pair has mixed byte provenance")
	}
	artifact.Counters = []AppleMemoryCounterArtifact{*read, *write}
	if read.RateBytesPerSecond != nil {
		if write.RateBytesPerSecond == nil {
			return errors.New("Apple memory direct-rate read/write pair is incomplete")
		}
		total, err := addAppleUint64(*read.RateBytesPerSecond, *write.RateBytesPerSecond)
		if err != nil {
			return errors.New("Apple memory total byte rate overflows uint64")
		}
		readRate, writeRate := *read.RateBytesPerSecond, *write.RateBytesPerSecond
		artifact.ReadBytesPerSecond, artifact.WriteBytesPerSecond, artifact.TotalBytesPerSecond = &readRate, &writeRate, &total
		return nil
	}
	if read.IntervalBytes == nil || write.IntervalBytes == nil {
		return errors.New("Apple memory interval-byte read/write pair is incomplete")
	}
	total, err := addAppleUint64(*read.IntervalBytes, *write.IntervalBytes)
	if err != nil {
		return errors.New("Apple memory total interval bytes overflow uint64")
	}
	readBytes, writeBytes := *read.IntervalBytes, *write.IntervalBytes
	artifact.ReadBytes, artifact.WriteBytes, artifact.TotalBytes = &readBytes, &writeBytes, &total
	return nil
}

func appleMemoryCollection(artifact AppleMemoryArtifact, o AppleMemoryImportOptions) (BandwidthCollection, error) {
	var read, write float64
	var source string
	if artifact.ReadBytesPerSecond != nil {
		read = float64(*artifact.ReadBytesPerSecond) / 1e9
		write = float64(*artifact.WriteBytesPerSecond) / 1e9
		source = "apple-memory-direct-byte-rates"
	} else {
		seconds := float64(artifact.IntervalNS) / float64(time.Second)
		read = float64(*artifact.ReadBytes) / seconds / 1e9
		write = float64(*artifact.WriteBytes) / seconds / 1e9
		source = "apple-memory-monotonic-byte-deltas"
	}
	total := read + write
	scope := BandwidthScope{Kind: artifact.Scope.Kind, ID: artifact.Scope.ID}
	rooflines := Rooflines{
		TheoreticalGBS:         cloneFloat(o.TheoreticalGBS),
		MeasuredSustainableGBS: cloneFloat(o.MeasuredHostGBS),
	}
	sample := BandwidthSample{
		Phase: o.Phase, Shape: o.Shape,
		Provenance: BandwidthProvenance{
			Source: source, Device: "apple-unified-memory", Collector: artifact.Provider, SampledAt: artifact.CaptureEndedAt,
		},
		Rooflines: rooflines,
		Live: LiveBandwidth{
			ReadGBS: &read, WriteGBS: &write, TotalGBS: &total, Scope: &scope,
		},
	}
	capture := BandwidthCapture{
		Schema: BandwidthSchema, Engine: "fak-native",
		Trigger: TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 1e100, ResourceUtilization: 1},
		Samples: []BandwidthSample{sample},
	}
	report, err := AnalyzeBandwidth(capture)
	if err != nil {
		return BandwidthCollection{}, err
	}
	intervalMS := artifact.IntervalNS / int64(time.Millisecond)
	if artifact.IntervalNS%int64(time.Millisecond) != 0 {
		intervalMS++
	}
	return BandwidthCollection{
		Schema: BandwidthCollectionSchema, Engine: "fak-native", Collector: artifact.Provider,
		IntervalMS: intervalMS,
		Availability: CollectorAvailability{
			DRAMCounters: true,
		},
		Capture:             capture,
		Report:              report,
		AppleMemoryArtifact: &artifact,
	}, nil
}

func addAppleUint64(a, b uint64) (uint64, error) {
	if a > math.MaxUint64-b {
		return 0, errors.New("uint64 addition overflow")
	}
	return a + b, nil
}
