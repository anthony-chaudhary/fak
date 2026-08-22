package toolcallcontrol

import (
	"encoding/json"
	"strings"
	"testing"
)

func projectionContract(maximum int64) StructuredProjectionContract {
	return StructuredProjectionContract{
		ItemsPointer: "/data/issues", CompletenessPointer: "/data/complete",
		FieldPointers:      []string{"/id", "/url", "/title"},
		IdentifierPointers: []string{"/id"}, CitationPointers: []string{"/url"},
		MaximumItems: maximum,
	}
}

func TestProjectStructuredResponseReducesAndLabelsLargeResult(t *testing.T) {
	payload := []byte(`{"data":{"complete":true,"issues":[{"id":101,"url":"https://example/101","title":"first","body":"` + strings.Repeat("noise", 40) + `"},{"id":102,"url":"https://example/102","title":"second","body":"` + strings.Repeat("noise", 40) + `"},{"id":103,"url":"https://example/103","title":"third","body":"` + strings.Repeat("noise", 40) + `"}]},"debug":"` + strings.Repeat("trace", 40) + `"}`)
	got := ProjectStructuredResponse(payload, projectionContract(2))
	if got.Receipt.Disposition != ResponseProject || got.Receipt.Completeness != ProjectionPartial {
		t.Fatalf("receipt = %#v", got.Receipt)
	}
	if got.Receipt.OriginalItems != 3 || got.Receipt.ProjectedItems != 2 || got.Receipt.ProjectedBytes >= got.Receipt.OriginalBytes {
		t.Fatalf("receipt = %#v", got.Receipt)
	}
	var envelope struct {
		Schema             string                 `json:"schema"`
		Completeness       ProjectionCompleteness `json:"completeness"`
		SourceCompleteness bool                   `json:"source_completeness"`
		OriginalItems      int64                  `json:"original_items"`
		ReturnedItems      int64                  `json:"returned_items"`
		Items              []map[string]any       `json:"items"`
	}
	if err := json.Unmarshal(got.Payload, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Schema != StructuredProjectionSchema || !envelope.SourceCompleteness || envelope.OriginalItems != 3 || envelope.ReturnedItems != 2 {
		t.Fatalf("envelope = %#v", envelope)
	}
	if envelope.Items[0]["id"] != float64(101) || envelope.Items[0]["url"] != "https://example/101" || envelope.Items[0]["title"] != "first" {
		t.Fatalf("first item = %#v", envelope.Items[0])
	}
	if _, leaked := envelope.Items[0]["body"]; leaked {
		t.Fatalf("unselected body leaked: %#v", envelope.Items[0])
	}
}

func TestProjectStructuredResponsePreservesCompleteLabelWithoutItemCut(t *testing.T) {
	payload := []byte(`{"data":{"complete":"authoritative","issues":[{"id":"a","url":"cite:a","title":"A","body":"` + strings.Repeat("x", 300) + `"}]}}`)
	got := ProjectStructuredResponse(payload, projectionContract(10))
	if got.Receipt.Disposition != ResponseProject || got.Receipt.Completeness != ProjectionComplete {
		t.Fatalf("receipt = %#v", got.Receipt)
	}
	if !strings.Contains(string(got.Payload), `"source_completeness":"authoritative"`) {
		t.Fatalf("source completeness lost: %s", got.Payload)
	}
}

func TestProjectStructuredResponseFallsBackByteForByte(t *testing.T) {
	payload := []byte(`{"data":{"complete":true,"issues":[{"id":1,"title":"no citation"}]}}`)
	tests := []struct {
		name     string
		payload  []byte
		contract StructuredProjectionContract
		reason   string
	}{
		{name: "missing required citation", payload: payload, contract: projectionContract(1), reason: "required_field_missing"},
		{name: "missing completeness", payload: []byte(`{"data":{"issues":[]}}`), contract: projectionContract(1), reason: "completeness_missing"},
		{name: "invalid json", payload: []byte(`{"data":`), contract: projectionContract(1), reason: "invalid_json"},
		{name: "uncontracted citation", payload: payload, contract: StructuredProjectionContract{ItemsPointer: "/data/issues", CompletenessPointer: "/data/complete", FieldPointers: []string{"/id"}, IdentifierPointers: []string{"/id"}, CitationPointers: []string{"/url"}, MaximumItems: 1}, reason: "invalid_contract"},
		{name: "not smaller", payload: []byte(`{"data":{"complete":true,"issues":[]}}`), contract: projectionContract(1), reason: "projection_not_smaller"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ProjectStructuredResponse(test.payload, test.contract)
			if got.Receipt.Disposition != ResponsePass || got.Receipt.Reason != test.reason {
				t.Fatalf("receipt = %#v", got.Receipt)
			}
			if string(got.Payload) != string(test.payload) {
				t.Fatalf("fallback changed payload: got %q want %q", got.Payload, test.payload)
			}
		})
	}
}

func TestProjectStructuredResponseSupportsEscapedPointers(t *testing.T) {
	payload := []byte(`{"root/items":{"done~state":true,"rows":[{"item/id":"a","cite~url":"u","extra":"` + strings.Repeat("x", 300) + `"}]}}`)
	contract := StructuredProjectionContract{
		ItemsPointer: "/root~1items/rows", CompletenessPointer: "/root~1items/done~0state",
		FieldPointers: []string{"/item~1id", "/cite~0url"}, IdentifierPointers: []string{"/item~1id"},
		CitationPointers: []string{"/cite~0url"}, MaximumItems: 1,
	}
	got := ProjectStructuredResponse(payload, contract)
	if got.Receipt.Disposition != ResponseProject || !strings.Contains(string(got.Payload), `"item/id":"a"`) || !strings.Contains(string(got.Payload), `"cite~url":"u"`) {
		t.Fatalf("result = %#v payload=%s", got.Receipt, got.Payload)
	}
}
