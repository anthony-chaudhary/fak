package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

const auditConformanceSchema = "fak-trajectory-attribution-conformance/1"

type auditConformanceCorpus struct {
	Schema   string                    `json:"schema"`
	Fixtures []auditConformanceFixture `json:"fixtures"`
}

type auditConformanceFixture struct {
	Name                       string                              `json:"name"`
	Source                     string                              `json:"source"`
	Root                       string                              `json:"root"`
	Path                       string                              `json:"path"`
	Features                   []string                            `json:"features"`
	Coverage                   auditConformanceCoverage            `json:"coverage"`
	ExpectedNormalizedEvents   []auditConformanceEvent             `json:"expected_normalized_events"`
	ExpectedStorageRows        []auditConformanceStorageRow        `json:"expected_storage_rows"`
	ExpectedTotals             auditConformanceTotals              `json:"expected_totals"`
	ExpectedDuplicateDecisions []auditConformanceDuplicateDecision `json:"expected_duplicate_decisions"`
	ExpectedRefusals           []string                            `json:"expected_refusals"`
}

type auditConformanceCoverage struct {
	RowSubtypes          []string `json:"row_subtypes"`
	ContentSubtypes      []string `json:"content_subtypes"`
	MessageBlockSubtypes []string `json:"message_block_subtypes"`
	EventSubtypes        []string `json:"event_subtypes"`
}

type auditConformanceEvent struct {
	Line             int    `json:"line"`
	Ordinal          int    `json:"ordinal"`
	Subtype          string `json:"subtype"`
	Category         string `json:"category"`
	Tool             string `json:"tool,omitempty"`
	CallID           string `json:"call_id,omitempty"`
	ContentBytes     int64  `json:"content_bytes"`
	Visible          bool   `json:"visible"`
	Visibility       string `json:"visibility"`
	VisibilityReason string `json:"visibility_reason"`
	Decision         string `json:"decision"`
}

type auditConformanceStorageRow struct {
	Subtype string `json:"subtype"`
	Bytes   int64  `json:"bytes"`
	Records int    `json:"records"`
}

type auditConformanceTotals struct {
	Records         int              `json:"records"`
	UsageSeen       int              `json:"usage_seen"`
	UsageExact      int              `json:"usage_exact"`
	UsageApplied    int              `json:"usage_applied"`
	DuplicateUsage  int              `json:"duplicate_usage"`
	RefusedRecords  int              `json:"refused_records"`
	ToolCalls       int              `json:"tool_calls"`
	ToolErrors      int              `json:"tool_errors"`
	Tokens          AuditTokens      `json:"tokens"`
	Distribution    map[string]int64 `json:"distribution_bytes"`
	ToolBytes       map[string]int64 `json:"tool_bytes"`
	ToolCallsByName map[string]int   `json:"tool_calls_by_name"`
	StorageBytes    int64            `json:"storage_bytes"`
}

type auditConformanceDuplicateDecision struct {
	Line     int    `json:"line"`
	Kind     string `json:"kind"`
	Identity string `json:"identity"`
	Decision string `json:"decision"`
}

func TestAuditAttributionConformanceCorpus(t *testing.T) {
	corpus := loadAuditConformanceCorpus(t)
	if corpus.Schema != auditConformanceSchema {
		t.Fatalf("corpus schema = %q, want %q", corpus.Schema, auditConformanceSchema)
	}
	if len(corpus.Fixtures) != 2 {
		t.Fatalf("fixtures = %d, want one Claude and one Codex fixture", len(corpus.Fixtures))
	}

	for _, fixture := range corpus.Fixtures {
		fixture := fixture
		t.Run(fixture.Name, func(t *testing.T) {
			validateAuditConformanceDeclarations(t, fixture)
			root := filepath.Join("testdata", "audit", "conformance", filepath.FromSlash(fixture.Root))
			path := filepath.Join(root, filepath.FromSlash(fixture.Path))

			events, decisions := readAuditConformanceEvents(t, fixture.Source, path)
			if !reflect.DeepEqual(events, fixture.ExpectedNormalizedEvents) {
				got, _ := json.MarshalIndent(events, "", "  ")
				want, _ := json.MarshalIndent(fixture.ExpectedNormalizedEvents, "", "  ")
				t.Fatalf("normalized events differ\ngot:\n%s\nwant:\n%s", got, want)
			}
			if !reflect.DeepEqual(decisions, fixture.ExpectedDuplicateDecisions) {
				t.Fatalf("duplicate decisions = %+v, want %+v", decisions, fixture.ExpectedDuplicateDecisions)
			}

			result := runAuditConformanceFixture(t, fixture, root)
			assertAuditConformanceTotals(t, fixture, result)
			assertAuditConformanceDeterministic(t, fixture, root, result)
		})
	}
}

func TestAuditAttributionConformanceCoverageMatrix(t *testing.T) {
	corpus := loadAuditConformanceCorpus(t)
	features := map[string]struct{}{}
	for _, fixture := range corpus.Fixtures {
		for _, feature := range fixture.Features {
			features[feature] = struct{}{}
		}
		actual := deriveAuditConformanceCoverage(t, fixture)
		switch fixture.Source {
		case AuditSourceClaude:
			assertDeclaredConformanceCoverage(t, "Claude row subtypes", fixture.Coverage.RowSubtypes, actual.RowSubtypes)
			assertDeclaredConformanceCoverage(t, "Claude content subtypes", fixture.Coverage.ContentSubtypes, actual.ContentSubtypes)
			assertConformanceSet(t, "Claude exercised row subtypes", actual.RowSubtypes, auditClaudeRowSubtypes)
			assertConformanceSet(t, "Claude exercised content subtypes", actual.ContentSubtypes, auditClaudeBlockSubtypes)
		case AuditSourceCodex:
			assertDeclaredConformanceCoverage(t, "Codex row subtypes", fixture.Coverage.RowSubtypes, actual.RowSubtypes)
			assertDeclaredConformanceCoverage(t, "Codex response-item subtypes", fixture.Coverage.ContentSubtypes, actual.ContentSubtypes)
			assertDeclaredConformanceCoverage(t, "Codex message-block subtypes", fixture.Coverage.MessageBlockSubtypes, actual.MessageBlockSubtypes)
			assertDeclaredConformanceCoverage(t, "Codex event-message subtypes", fixture.Coverage.EventSubtypes, actual.EventSubtypes)
			assertConformanceSet(t, "Codex exercised row subtypes", actual.RowSubtypes, auditCodexRowSubtypes)
			assertConformanceSet(t, "Codex exercised response-item subtypes", actual.ContentSubtypes, auditCodexResponseItemSubtypes)
			assertConformanceSet(t, "Codex exercised message-block subtypes", actual.MessageBlockSubtypes, auditCodexMessageBlockSubtypes)
			assertConformanceSet(t, "Codex exercised event-message subtypes", actual.EventSubtypes, auditCodexEventMessageSubtypes)
		default:
			t.Fatalf("fixture %q has unsupported source %q", fixture.Name, fixture.Source)
		}
	}
	wantFeatures := []string{"empty_content", "future_unknown_types", "malformed_rows", "mirrors", "mixed_messages", "nested_malformed", "out_of_order_tools", "unicode", "unknown_message_blocks"}
	assertConformanceSet(t, "corpus features", setKeys(features), stringSet(wantFeatures))
}

func deriveAuditConformanceCoverage(t *testing.T, fixture auditConformanceFixture) auditConformanceCoverage {
	t.Helper()
	path := filepath.Join("testdata", "audit", "conformance", filepath.FromSlash(fixture.Root), filepath.FromSlash(fixture.Path))
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	rows := map[string]struct{}{}
	content := map[string]struct{}{}
	messageBlocks := map[string]struct{}{}
	events := map[string]struct{}{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var root map[string]json.RawMessage
		if json.Unmarshal(bytes.TrimSpace(scanner.Bytes()), &root) != nil {
			continue
		}
		rowType := rawString(root["type"])
		if fixture.Source == AuditSourceClaude {
			if _, registered := auditClaudeRowSubtypes[rowType]; registered {
				rows[rowType] = struct{}{}
			}
			if rowType != "assistant" && rowType != "user" {
				continue
			}
			var message map[string]json.RawMessage
			if json.Unmarshal(root["message"], &message) != nil {
				continue
			}
			var blocks []map[string]json.RawMessage
			if json.Unmarshal(message["content"], &blocks) != nil {
				continue
			}
			for _, block := range blocks {
				blockType := rawString(block["type"])
				if _, registered := auditClaudeBlockSubtypes[blockType]; registered {
					content[blockType] = struct{}{}
				}
			}
			continue
		}

		if _, registered := auditCodexRowSubtypes[rowType]; registered {
			rows[rowType] = struct{}{}
		}
		var payload map[string]json.RawMessage
		if json.Unmarshal(root["payload"], &payload) != nil {
			continue
		}
		switch rowType {
		case "response_item":
			itemType := rawString(payload["type"])
			if _, registered := auditCodexResponseItemSubtypes[itemType]; registered {
				content[itemType] = struct{}{}
			}
			if itemType != "message" {
				continue
			}
			var blocks []map[string]json.RawMessage
			if json.Unmarshal(payload["content"], &blocks) != nil {
				continue
			}
			for _, block := range blocks {
				blockType := rawString(block["type"])
				if _, registered := auditCodexMessageBlockSubtypes[blockType]; registered {
					messageBlocks[blockType] = struct{}{}
				}
			}
		case "event_msg":
			eventType := rawString(payload["type"])
			if _, registered := auditCodexEventMessageSubtypes[eventType]; registered {
				events[eventType] = struct{}{}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return auditConformanceCoverage{
		RowSubtypes: setKeys(rows), ContentSubtypes: setKeys(content),
		MessageBlockSubtypes: setKeys(messageBlocks), EventSubtypes: setKeys(events),
	}
}

func loadAuditConformanceCorpus(t *testing.T) auditConformanceCorpus {
	t.Helper()
	path := filepath.Join("testdata", "audit", "conformance", "expectations.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var corpus auditConformanceCorpus
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&corpus); err != nil {
		t.Fatalf("decode conformance expectations: %v", err)
	}
	return corpus
}

func validateAuditConformanceDeclarations(t *testing.T, fixture auditConformanceFixture) {
	t.Helper()
	if fixture.Name == "" || fixture.Source == "" || fixture.Root == "" || fixture.Path == "" {
		t.Fatalf("fixture identity is incomplete: %+v", fixture)
	}
	if len(fixture.Features) == 0 || len(fixture.ExpectedNormalizedEvents) == 0 || len(fixture.ExpectedStorageRows) == 0 || len(fixture.ExpectedDuplicateDecisions) == 0 {
		t.Fatalf("fixture %q must declare features, events, storage, and duplicate decisions", fixture.Name)
	}
	for i, event := range fixture.ExpectedNormalizedEvents {
		if event.Line <= 0 || event.Ordinal < 0 || event.Subtype == "" || event.Visibility == "" || event.VisibilityReason == "" || event.Decision == "" {
			t.Fatalf("fixture %q event %d lacks coordinate, subtype, visibility provenance, or decision: %+v", fixture.Name, i, event)
		}
		switch event.Visibility {
		case "inferred_model_visible", "explicit_storage_only", "unresolved":
		default:
			t.Fatalf("fixture %q event %d has unversioned visibility %q", fixture.Name, i, event.Visibility)
		}
		if event.Visible != (event.Visibility == "inferred_model_visible" || event.Category == "visible_unknown") {
			t.Fatalf("fixture %q event %d visibility projection is inconsistent: %+v", fixture.Name, i, event)
		}
	}
	if fixture.ExpectedTotals.Distribution == nil || fixture.ExpectedTotals.ToolBytes == nil || fixture.ExpectedTotals.ToolCallsByName == nil {
		t.Fatalf("fixture %q must declare all aggregate projections", fixture.Name)
	}
}

func readAuditConformanceEvents(t *testing.T, source, path string) ([]auditConformanceEvent, []auditConformanceDuplicateDecision) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	var events []auditConformanceEvent
	var decisions []auditConformanceDuplicateDecision
	seenCalls := map[string]struct{}{}
	pendingResults := map[string]struct{}{}
	seenUsage := map[string]string{}
	scanner := bufio.NewScanner(file)
	line := 0
	for scanner.Scan() {
		line++
		raw := bytes.TrimSpace(scanner.Bytes())
		var root map[string]json.RawMessage
		if len(raw) == 0 || json.Unmarshal(raw, &root) != nil {
			continue
		}
		typ := rawString(root["type"])
		payload := root["payload"]
		if source == AuditSourceClaude {
			payload = root["message"]
		}
		classified := classifyDistributionEvents(source, typ, payload, root)
		for ordinal, event := range classified {
			decision := auditConformanceEventDecision(source, event, seenCalls, pendingResults)
			events = append(events, auditConformanceEvent{
				Line: line, Ordinal: ordinal, Subtype: event.subtype, Category: event.category,
				Tool: event.tool, CallID: event.id, ContentBytes: int64(len(event.content)), Visible: event.visible,
				Visibility: event.visibility, VisibilityReason: event.visibilityReason, Decision: decision,
			})
		}
		decisions = append(decisions, auditConformanceDecisions(source, line, root, seenUsage)...)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return events, decisions
}

func auditConformanceEventDecision(source string, event auditClassifiedEvent, seenCalls, pendingResults map[string]struct{}) string {
	if event.category == "tool_result" && event.id != "" {
		if _, seen := seenCalls[event.id]; !seen {
			pendingResults[event.id] = struct{}{}
			return "retained_pending_call_link"
		}
	}
	if event.category == "tool_call" && event.id != "" {
		if _, duplicate := seenCalls[event.id]; duplicate {
			return "retained_duplicate_payload"
		}
		seenCalls[event.id] = struct{}{}
		if _, pending := pendingResults[event.id]; pending {
			delete(pendingResults, event.id)
			return "retained_resolved_pending_result"
		}
	}
	if event.visibilityReason == "empty_message_content" {
		return "retained_content_empty"
	}
	if event.visibility == "unresolved" {
		return "retained_unknown"
	}
	if event.visibilityReason == "known_event_mirror" || (source == AuditSourceClaude && event.subtype == "attachment/deferred_tools_delta") {
		return "retained_storage_mirror"
	}
	if event.subtype == "event_msg/user_message" || event.subtype == "event_msg/agent_message" {
		return "retained_unlinked_mirror"
	}
	if !event.visible {
		return "retained_storage"
	}
	return "retained"
}

func auditConformanceDecisions(source string, line int, root map[string]json.RawMessage, seenUsage map[string]string) []auditConformanceDuplicateDecision {
	typ := rawString(root["type"])
	if source == AuditSourceClaude {
		if typ == "attachment" {
			var attachment map[string]json.RawMessage
			_ = json.Unmarshal(root["attachment"], &attachment)
			if rawString(attachment["type"]) == "deferred_tools_delta" {
				return []auditConformanceDuplicateDecision{{line, "mirror", "attachment/deferred_tools_delta", "retained_storage_mirror"}}
			}
		}
		if typ != "assistant" {
			return nil
		}
		var message map[string]json.RawMessage
		_ = json.Unmarshal(root["message"], &message)
		id := rawString(message["id"])
		if id == "" || len(message["usage"]) == 0 {
			return nil
		}
		usage := string(message["usage"])
		if previous, seen := seenUsage[id]; seen && previous == usage {
			return []auditConformanceDuplicateDecision{{line, "usage", id, "suppressed_exact_duplicate"}}
		}
		seenUsage[id] = usage
		return nil
	}

	var payload map[string]json.RawMessage
	_ = json.Unmarshal(root["payload"], &payload)
	if typ == "response_item" {
		itemType := rawString(payload["type"])
		if itemType == "function_call" || itemType == "custom_tool_call" {
			id := rawString(payload["call_id"])
			key := "call:" + id
			if _, seen := seenUsage[key]; seen {
				return []auditConformanceDuplicateDecision{{line, "tool_call", id, "suppressed_same_call_id_for_tool_count"}}
			}
			seenUsage[key] = itemType
		}
		return nil
	}
	if typ != "event_msg" {
		return nil
	}
	eventType := rawString(payload["type"])
	identity := "event_msg/" + eventType
	switch eventType {
	case "user_message", "agent_message":
		return []auditConformanceDuplicateDecision{{line, "mirror", identity, "retained_unlinked_mirror"}}
	case "item_started", "item_completed":
		var item map[string]json.RawMessage
		_ = json.Unmarshal(payload["item"], &item)
		identity += "/" + firstNonempty(rawString(item["type"]), "unknown_item")
		return []auditConformanceDuplicateDecision{{line, "mirror", identity, "retained_storage_mirror"}}
	}
	return nil
}

func runAuditConformanceFixture(t *testing.T, fixture auditConformanceFixture, root string) AuditResult {
	t.Helper()
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: fixture.Source, Root: root, RootLabel: fixture.Root}}})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func assertAuditConformanceTotals(t *testing.T, fixture auditConformanceFixture, result AuditResult) {
	t.Helper()
	if len(result.Denominators) != 1 || len(result.Transcripts) != 1 {
		t.Fatalf("fixture %q result shape: denominators=%d transcripts=%d", fixture.Name, len(result.Denominators), len(result.Transcripts))
	}
	denominator := result.Denominators[0]
	row := result.Transcripts[0]
	want := fixture.ExpectedTotals
	if denominator.Records != want.Records || denominator.UsageRecordsSeen != want.UsageSeen || denominator.UsageRecordsExact != want.UsageExact || denominator.UsageRecordsApplied != want.UsageApplied || denominator.DuplicateUsageRecords != want.DuplicateUsage || denominator.RefusedRecords != want.RefusedRecords {
		t.Fatalf("fixture %q denominator = %+v, want records=%d usage=%d/%d/%d duplicates=%d refused=%d", fixture.Name, denominator, want.Records, want.UsageSeen, want.UsageExact, want.UsageApplied, want.DuplicateUsage, want.RefusedRecords)
	}
	if row.Tokens != want.Tokens || row.ToolCalls != want.ToolCalls || row.ToolErrors != want.ToolErrors {
		t.Fatalf("fixture %q transcript totals = tokens:%+v calls:%d errors:%d, want tokens:%+v calls:%d errors:%d", fixture.Name, row.Tokens, row.ToolCalls, row.ToolErrors, want.Tokens, want.ToolCalls, want.ToolErrors)
	}
	if got := distributionByteMap(row.Distribution); !reflect.DeepEqual(got, want.Distribution) {
		t.Fatalf("fixture %q distribution = %+v, want %+v", fixture.Name, got, want.Distribution)
	}
	if got := distributionByteMap(row.ToolDistribution); !reflect.DeepEqual(got, want.ToolBytes) {
		t.Fatalf("fixture %q tool bytes = %+v, want %+v", fixture.Name, got, want.ToolBytes)
	}
	if got := distributionCallMap(row.ToolDistribution); !reflect.DeepEqual(got, want.ToolCallsByName) {
		t.Fatalf("fixture %q tool calls = %+v, want %+v", fixture.Name, got, want.ToolCallsByName)
	}
	gotStorage := make(map[string]auditConformanceStorageRow, len(row.StorageDistribution))
	var storageBytes int64
	for _, storage := range row.StorageDistribution {
		gotStorage[storage.Subtype] = auditConformanceStorageRow{Subtype: storage.Subtype, Bytes: storage.Bytes, Records: storage.Records}
		storageBytes += storage.Bytes
	}
	wantStorage := make(map[string]auditConformanceStorageRow, len(fixture.ExpectedStorageRows))
	for _, storage := range fixture.ExpectedStorageRows {
		wantStorage[storage.Subtype] = storage
	}
	if !reflect.DeepEqual(gotStorage, wantStorage) || storageBytes != want.StorageBytes {
		t.Fatalf("fixture %q storage = %+v total=%d, want %+v total=%d", fixture.Name, gotStorage, storageBytes, wantStorage, want.StorageBytes)
	}
	gotRefusals := make([]string, len(result.Refusals))
	for i, refusal := range result.Refusals {
		gotRefusals[i] = refusal.Code
	}
	if !reflect.DeepEqual(gotRefusals, fixture.ExpectedRefusals) {
		t.Fatalf("fixture %q refusals = %v, want %v", fixture.Name, gotRefusals, fixture.ExpectedRefusals)
	}
	if result.Summary.DistributionProvenance == "" || result.Summary.DistributionUnit != AuditDistributionUnit || result.Summary.StorageUnit != AuditStorageUnit {
		t.Fatalf("fixture %q summary lost units or provenance: %+v", fixture.Name, result.Summary)
	}
}

func assertAuditConformanceDeterministic(t *testing.T, fixture auditConformanceFixture, root string, first AuditResult) {
	t.Helper()
	second := runAuditConformanceFixture(t, fixture, root)
	var firstJSON, secondJSON bytes.Buffer
	if err := WriteAuditJSONL(&firstJSON, first); err != nil {
		t.Fatal(err)
	}
	if err := WriteAuditJSONL(&secondJSON, second); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstJSON.Bytes(), secondJSON.Bytes()) {
		t.Fatalf("fixture %q output is nondeterministic\nfirst:\n%s\nsecond:\n%s", fixture.Name, firstJSON.Bytes(), secondJSON.Bytes())
	}
}

func assertConformanceSet(t *testing.T, name string, got []string, want map[string]struct{}) {
	t.Helper()
	got = append([]string(nil), got...)
	sort.Strings(got)
	wantKeys := setKeys(want)
	if !reflect.DeepEqual(got, wantKeys) {
		t.Fatalf("%s coverage = %v, registered = %v; update the conformance fixture whenever a parser subtype changes", name, got, wantKeys)
	}
}

func assertDeclaredConformanceCoverage(t *testing.T, name string, declared, exercised []string) {
	t.Helper()
	declared = append([]string(nil), declared...)
	exercised = append([]string(nil), exercised...)
	sort.Strings(declared)
	sort.Strings(exercised)
	if !reflect.DeepEqual(declared, exercised) {
		t.Fatalf("%s declaration = %v, actual fixture coverage = %v; subtype names alone do not count as exercised", name, declared, exercised)
	}
}

func setKeys(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for value := range values {
		keys = append(keys, value)
	}
	sort.Strings(keys)
	return keys
}

func stringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func distributionByteMap(rows []AuditDistributionRow) map[string]int64 {
	result := make(map[string]int64, len(rows))
	for _, row := range rows {
		result[row.Name] = row.Bytes
	}
	return result
}

func distributionCallMap(rows []AuditDistributionRow) map[string]int {
	result := make(map[string]int, len(rows))
	for _, row := range rows {
		result[row.Name] = row.Calls
	}
	return result
}

func TestAuditConformanceFixturePathsArePortable(t *testing.T) {
	corpus := loadAuditConformanceCorpus(t)
	for _, fixture := range corpus.Fixtures {
		if strings.Contains(fixture.Root, `\`) || strings.Contains(fixture.Path, `\`) || filepath.IsAbs(fixture.Root) || filepath.IsAbs(fixture.Path) {
			t.Fatalf("fixture %q contains a machine-private or non-portable path", fixture.Name)
		}
	}
}
