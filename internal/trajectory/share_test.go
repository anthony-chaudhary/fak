package trajectory

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestExportViewProducesStableRedactedBundle(t *testing.T) {
	at := time.Date(2026, 8, 17, 17, 0, 0, 0, time.UTC)
	events := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"done","api_key":"secret-value"}`)}
	canonical, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	projected, receipt, err := CompileView(events, EndUserView())
	if err != nil {
		t.Fatal(err)
	}
	first, err := ExportView(projected, receipt)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ExportView(projected, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("share bundle is not deterministic")
	}
	if err := first.Verify(); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(first.JSONL, []byte("secret-value")) {
		t.Fatalf("unredacted secret exported: %s", first.JSONL)
	}
	if !bytes.Contains(first.JSONL, []byte("[REDACTED]")) {
		t.Fatalf("redaction marker missing: %s", first.JSONL)
	}
	if first.Manifest.TrajectoryDigest != receipt.TrajectoryDigest || first.Manifest.ProjectionDigest != receipt.ProjectionDigest || first.Manifest.ViewSpecDigest != receipt.ViewSpecDigest {
		t.Fatalf("manifest lost transform provenance: %#v", first.Manifest)
	}
	after, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(canonical, after) {
		t.Fatal("export mutated canonical evidence")
	}
}

func TestExportViewRejectsProjectionReceiptMismatch(t *testing.T) {
	at := time.Date(2026, 8, 17, 17, 0, 0, 0, time.UTC)
	events := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"done"}`)}
	projected, receipt, err := CompileView(events, EndUserView())
	if err != nil {
		t.Fatal(err)
	}
	projected[0].Event.Payload = json.RawMessage(`{"role":"assistant","text":"tampered"}`)
	if _, err := ExportView(projected, receipt); err == nil {
		t.Fatal("mismatched projection exported")
	}
}

func TestShareBundleDetectsByteAndManifestTampering(t *testing.T) {
	at := time.Date(2026, 8, 17, 17, 0, 0, 0, time.UTC)
	events := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"done"}`)}
	projected, receipt, err := CompileView(events, EndUserView())
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := ExportView(projected, receipt)
	if err != nil {
		t.Fatal(err)
	}
	bytesTampered := bundle
	bytesTampered.JSONL = append([]byte(nil), bundle.JSONL...)
	bytesTampered.JSONL[0] ^= 1
	if err := bytesTampered.Verify(); err == nil {
		t.Fatal("tampered bytes verified")
	}
	manifestTampered := bundle
	manifestTampered.Manifest.Audience = "other"
	if err := manifestTampered.Verify(); err == nil {
		t.Fatal("tampered manifest verified")
	}
}
