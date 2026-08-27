package trajectory

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// AuditSchemaBaselineSchema versions the checked-in provider transcript shape
// baseline independently from the audit artifact that reports drift against it.
const AuditSchemaBaselineSchema = "fak-trajectory-schema-baseline/1"

//go:embed audit_schema_baseline.v1.json
var defaultAuditSchemaBaselineJSON []byte

// AuditBuildIdentity is the explicit provider/harness provenance available in
// a transcript. Empty build fields mean the harness did not record them; they
// are never inferred from a model name or local installation.
type AuditBuildIdentity struct {
	Provider      string `json:"provider,omitempty"`
	ProviderBuild string `json:"provider_build,omitempty"`
	Harness       string `json:"harness"`
	HarnessBuild  string `json:"harness_build,omitempty"`
}

// AuditShapeField is one stable JSON field path and the wire types observed at
// that path. Arbitrary prompt/tool payloads are opaque leaves so user data does
// not masquerade as provider schema.
type AuditShapeField struct {
	Path  string   `json:"path"`
	Types []string `json:"types"`
}

// AuditSchemaEvent is the normalized shape of one transcript event type.
type AuditSchemaEvent struct {
	Source    string               `json:"source"`
	EventType string               `json:"event_type"`
	Builds    []AuditBuildIdentity `json:"builds"`
	Fields    []AuditShapeField    `json:"fields"`
}

// AuditSchemaBaseline is the checked-in parser contract used for drift reads.
type AuditSchemaBaseline struct {
	Schema string             `json:"schema"`
	Events []AuditSchemaEvent `json:"events"`
}

// AuditEventSchemaRow makes current event shapes queryable in audit JSONL.
type AuditEventSchemaRow struct {
	Schema string `json:"schema"`
	Kind   string `json:"kind"`
	AuditSchemaEvent
}

// AuditSchemaDriftRow distinguishes safe field additions from changes that can
// invalidate parser semantics. ParserSurface and FixtureSurface make every row
// directly dispatchable instead of leaving an operator to rediscover ownership.
type AuditSchemaDriftRow struct {
	Schema          string               `json:"schema"`
	Kind            string               `json:"kind"`
	Source          string               `json:"source"`
	EventType       string               `json:"event_type"`
	Change          string               `json:"change"`
	Compatibility   string               `json:"compatibility"`
	BaselineBuilds  []AuditBuildIdentity `json:"baseline_builds,omitempty"`
	CurrentBuilds   []AuditBuildIdentity `json:"current_builds,omitempty"`
	AdditiveFields  []string             `json:"additive_fields,omitempty"`
	BreakingChanges []string             `json:"breaking_changes,omitempty"`
	ParserSurface   string               `json:"parser_surface"`
	FixtureSurface  string               `json:"fixture_surface"`
	ProposedAction  string               `json:"proposed_action"`
}

type auditShapeSet map[string]map[string]struct{}

// DefaultAuditSchemaBaseline loads the embedded, checked-in parser baseline.
func DefaultAuditSchemaBaseline() (AuditSchemaBaseline, error) {
	return DecodeAuditSchemaBaseline(strings.NewReader(string(defaultAuditSchemaBaselineJSON)))
}

// ReadAuditSchemaBaseline loads an operator-supplied versioned baseline.
func ReadAuditSchemaBaseline(r io.Reader) (AuditSchemaBaseline, error) {
	return DecodeAuditSchemaBaseline(r)
}

// DecodeAuditSchemaBaseline validates and normalizes one baseline document.
func DecodeAuditSchemaBaseline(r io.Reader) (AuditSchemaBaseline, error) {
	var baseline AuditSchemaBaseline
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&baseline); err != nil {
		return AuditSchemaBaseline{}, fmt.Errorf("trajectory audit: decode schema baseline: %w", err)
	}
	if baseline.Schema != AuditSchemaBaselineSchema {
		return AuditSchemaBaseline{}, fmt.Errorf("trajectory audit: schema baseline %q has no reader", baseline.Schema)
	}
	for i := range baseline.Events {
		event := &baseline.Events[i]
		if strings.TrimSpace(event.Source) == "" || strings.TrimSpace(event.EventType) == "" {
			return AuditSchemaBaseline{}, fmt.Errorf("trajectory audit: schema baseline event %d needs source and event_type", i)
		}
		normalizeAuditSchemaEvent(event)
	}
	sortAuditSchemaEvents(baseline.Events)
	for i := 1; i < len(baseline.Events); i++ {
		previous, current := baseline.Events[i-1], baseline.Events[i]
		if previous.Source == current.Source && previous.EventType == current.EventType {
			return AuditSchemaBaseline{}, fmt.Errorf("trajectory audit: duplicate schema baseline event %s/%s", current.Source, current.EventType)
		}
	}
	return baseline, nil
}

func observeAuditBuildIdentity(source string, record map[string]any, state *auditParseState) {
	identity := state.buildIdentity
	identity.Harness = source
	switch source {
	case AuditSourceClaude:
		if value := auditFirstString(record, "version", "cli_version"); value != "" {
			identity.HarnessBuild = value
		}
		if value := auditFirstString(record, "provider", "model_provider"); value != "" {
			identity.Provider = value
		}
		if value := auditFirstString(record, "provider_version", "provider_build"); value != "" {
			identity.ProviderBuild = value
		}
		if message, ok := record["message"].(map[string]any); ok {
			if identity.Provider == "" {
				identity.Provider = auditFirstString(message, "provider", "model_provider")
			}
			if identity.ProviderBuild == "" {
				identity.ProviderBuild = auditFirstString(message, "provider_version", "provider_build")
			}
		}
	case AuditSourceCodex:
		recordType, _ := record["type"].(string)
		payload, _ := record["payload"].(map[string]any)
		if recordType == "session_meta" && !state.codexPrimaryMetaSeen {
			identity.HarnessBuild = auditFirstString(payload, "cli_version", "version")
			identity.Provider = auditFirstString(payload, "model_provider", "provider")
			identity.ProviderBuild = auditFirstString(payload, "model_provider_version", "provider_version", "provider_build")
		} else if recordType == "turn_context" {
			if identity.Provider == "" {
				identity.Provider = auditFirstString(payload, "model_provider", "provider")
			}
			if identity.ProviderBuild == "" {
				identity.ProviderBuild = auditFirstString(payload, "model_provider_version", "provider_version", "provider_build")
			}
		}
	}
	state.buildIdentity = identity
	state.buildIdentities[identity] = struct{}{}
}

func auditFirstString(object map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := object[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func observeAuditEventShape(record map[string]any, state *auditParseState) {
	eventType := auditRecordType(record)
	shape := state.schemaShapes[eventType]
	if shape == nil {
		shape = auditShapeSet{}
		state.schemaShapes[eventType] = shape
	}
	auditWalkShape("$", record, shape)
}

func auditWalkShape(path string, value any, shape auditShapeSet) {
	typeName := auditJSONType(value)
	types := shape[path]
	if types == nil {
		types = map[string]struct{}{}
		shape[path] = types
	}
	types[typeName] = struct{}{}
	object, ok := value.(map[string]any)
	if !ok {
		return
	}
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		childPath := path + "." + key
		child := object[key]
		childType := auditJSONType(child)
		childTypes := shape[childPath]
		if childTypes == nil {
			childTypes = map[string]struct{}{}
			shape[childPath] = childTypes
		}
		childTypes[childType] = struct{}{}
		if !auditOpaqueShapeField(key) {
			if _, nested := child.(map[string]any); nested {
				auditWalkShape(childPath, child, shape)
			}
		}
	}
}

func auditOpaqueShapeField(key string) bool {
	switch key {
	case "arguments", "content", "input", "output", "result", "text":
		return true
	default:
		return false
	}
}

func auditJSONType(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case json.Number, float64, float32, int, int64, int32, uint, uint64, uint32:
		return "number"
	default:
		return "unknown"
	}
}

func auditCurrentSchemaEvents(transcripts []AuditTranscriptRow) []AuditSchemaEvent {
	byEvent := map[string]*AuditSchemaEvent{}
	shapeSets := map[string]auditShapeSet{}
	for _, transcript := range transcripts {
		for eventType, shape := range transcript.schemaShapes {
			key := transcript.Source + "\x00" + eventType
			event := byEvent[key]
			if event == nil {
				event = &AuditSchemaEvent{Source: transcript.Source, EventType: eventType}
				byEvent[key] = event
				shapeSets[key] = auditShapeSet{}
			}
			event.Builds = appendAuditBuilds(event.Builds, transcript.BuildIdentities...)
			mergeAuditShapeSet(shapeSets[key], shape)
		}
	}
	events := make([]AuditSchemaEvent, 0, len(byEvent))
	for key, event := range byEvent {
		event.Fields = auditShapeFields(shapeSets[key])
		normalizeAuditSchemaEvent(event)
		events = append(events, *event)
	}
	sortAuditSchemaEvents(events)
	return events
}

func compareAuditSchema(current []AuditSchemaEvent, baseline AuditSchemaBaseline) []AuditSchemaDriftRow {
	currentByKey := make(map[string]AuditSchemaEvent, len(current))
	baselineByKey := make(map[string]AuditSchemaEvent, len(baseline.Events))
	for _, event := range current {
		currentByKey[event.Source+"\x00"+event.EventType] = event
	}
	for _, event := range baseline.Events {
		baselineByKey[event.Source+"\x00"+event.EventType] = event
	}
	keys := make(map[string]struct{}, len(currentByKey)+len(baselineByKey))
	for key := range currentByKey {
		keys[key] = struct{}{}
	}
	for key := range baselineByKey {
		keys[key] = struct{}{}
	}
	ordered := make([]string, 0, len(keys))
	for key := range keys {
		ordered = append(ordered, key)
	}
	sort.Strings(ordered)

	var rows []AuditSchemaDriftRow
	for _, key := range ordered {
		currentEvent, currentOK := currentByKey[key]
		baselineEvent, baselineOK := baselineByKey[key]
		source, eventType := currentEvent.Source, currentEvent.EventType
		if !currentOK {
			source, eventType = baselineEvent.Source, baselineEvent.EventType
		}
		parser, fixture := auditSchemaOwner(source)
		row := AuditSchemaDriftRow{
			Schema: AuditSchema, Kind: "schema_drift", Source: source, EventType: eventType,
			BaselineBuilds: baselineEvent.Builds, CurrentBuilds: currentEvent.Builds,
			ParserSurface: parser, FixtureSurface: fixture,
		}
		switch {
		case !baselineOK:
			row.Change = "new_event"
			row.Compatibility = "breaking"
			row.BreakingChanges = []string{"event type is not mapped by the checked-in baseline"}
		case !currentOK:
			row.Change = "removed_event"
			row.Compatibility = "breaking"
			row.BreakingChanges = []string{"baseline event type was not observed in the selected corpus"}
		default:
			row.AdditiveFields, row.BreakingChanges = auditCompareShapeFields(currentEvent.Fields, baselineEvent.Fields)
			if len(row.AdditiveFields) == 0 && len(row.BreakingChanges) == 0 {
				continue
			}
			row.Change = "shape_changed"
			row.Compatibility = "additive"
			if len(row.BreakingChanges) > 0 {
				row.Compatibility = "breaking"
			}
		}
		row.ProposedAction = fmt.Sprintf("update %s and %s; review semantics, then refresh internal/trajectory/audit_schema_baseline.v1.json (%s)", parser, fixture, AuditSchemaBaselineSchema)
		rows = append(rows, row)
	}
	return rows
}

func auditSchemaBaselineForPresentSources(baseline AuditSchemaBaseline, denominators []AuditDenominatorRow) AuditSchemaBaseline {
	present := make(map[string]bool, len(denominators))
	for _, denominator := range denominators {
		present[denominator.Source] = denominator.RootPresent
	}
	filtered := AuditSchemaBaseline{Schema: baseline.Schema}
	for _, event := range baseline.Events {
		if present[event.Source] {
			filtered.Events = append(filtered.Events, event)
		}
	}
	return filtered
}

func auditCompareShapeFields(current, baseline []AuditShapeField) ([]string, []string) {
	currentByPath := make(map[string][]string, len(current))
	baselineByPath := make(map[string][]string, len(baseline))
	for _, field := range current {
		currentByPath[field.Path] = field.Types
	}
	for _, field := range baseline {
		baselineByPath[field.Path] = field.Types
	}
	var additive, breaking []string
	for path, currentTypes := range currentByPath {
		baselineTypes, exists := baselineByPath[path]
		if !exists {
			additive = append(additive, fmt.Sprintf("%s (%s)", path, strings.Join(currentTypes, "|")))
			continue
		}
		if !auditStringSubset(currentTypes, baselineTypes) {
			breaking = append(breaking, fmt.Sprintf("%s type %s, baseline %s", path, strings.Join(currentTypes, "|"), strings.Join(baselineTypes, "|")))
		}
	}
	for path, baselineTypes := range baselineByPath {
		if _, exists := currentByPath[path]; !exists {
			breaking = append(breaking, fmt.Sprintf("removed %s (%s)", path, strings.Join(baselineTypes, "|")))
		}
	}
	sort.Strings(additive)
	sort.Strings(breaking)
	return additive, breaking
}

func auditSchemaOwner(source string) (string, string) {
	switch source {
	case AuditSourceCodex:
		return "internal/trajectory/audit_parse.go:parseCodexAuditRecord", "internal/trajectory/testdata/audit/codex/sessions"
	default:
		return "internal/trajectory/audit_parse.go:parseClaudeAuditRecord", "internal/trajectory/testdata/audit/claude/projects"
	}
}

func mergeAuditShapeSet(dst, src auditShapeSet) {
	for path, types := range src {
		if dst[path] == nil {
			dst[path] = map[string]struct{}{}
		}
		for typeName := range types {
			dst[path][typeName] = struct{}{}
		}
	}
}

func auditShapeFields(shape auditShapeSet) []AuditShapeField {
	fields := make([]AuditShapeField, 0, len(shape))
	for path, typeSet := range shape {
		types := make([]string, 0, len(typeSet))
		for typeName := range typeSet {
			types = append(types, typeName)
		}
		sort.Strings(types)
		fields = append(fields, AuditShapeField{Path: path, Types: types})
	}
	sort.Slice(fields, func(i, j int) bool { return fields[i].Path < fields[j].Path })
	return fields
}

func normalizeAuditSchemaEvent(event *AuditSchemaEvent) {
	event.Builds = appendAuditBuilds(nil, event.Builds...)
	for i := range event.Fields {
		sort.Strings(event.Fields[i].Types)
	}
	sort.Slice(event.Fields, func(i, j int) bool { return event.Fields[i].Path < event.Fields[j].Path })
}

func appendAuditBuilds(dst []AuditBuildIdentity, values ...AuditBuildIdentity) []AuditBuildIdentity {
	seen := make(map[AuditBuildIdentity]struct{}, len(dst)+len(values))
	for _, value := range dst {
		seen[value] = struct{}{}
	}
	for _, value := range values {
		if value.Harness == "" {
			continue
		}
		seen[value] = struct{}{}
	}
	dst = dst[:0]
	for value := range seen {
		dst = append(dst, value)
	}
	sort.Slice(dst, func(i, j int) bool { return auditBuildKey(dst[i]) < auditBuildKey(dst[j]) })
	return dst
}

func auditBuildKey(identity AuditBuildIdentity) string {
	return identity.Provider + "\x00" + identity.ProviderBuild + "\x00" + identity.Harness + "\x00" + identity.HarnessBuild
}

func auditStringSubset(values, allowed []string) bool {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, value := range allowed {
		allowedSet[value] = struct{}{}
	}
	for _, value := range values {
		if _, ok := allowedSet[value]; !ok {
			return false
		}
	}
	return true
}

func sortAuditSchemaEvents(events []AuditSchemaEvent) {
	sort.Slice(events, func(i, j int) bool {
		if events[i].Source != events[j].Source {
			return events[i].Source < events[j].Source
		}
		return events[i].EventType < events[j].EventType
	})
}
