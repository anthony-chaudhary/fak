package promptmmu

import "testing"

func TestSerializeAuditStable(t *testing.T) {
	audit := AuditSerialization([]byte("stable prompt bytes"), []byte("stable prompt bytes"))
	if audit.Status != SerializationStable {
		t.Fatalf("status = %q, want %q", audit.Status, SerializationStable)
	}
	if !audit.Range.IsZero() {
		t.Fatalf("stable audit carried a changed range: %+v", audit.Range)
	}
}

func TestSerializeAuditReportsChangedByteRange(t *testing.T) {
	audit := AuditSerialization([]byte("prefix OLD suffix"), []byte("prefix NEW suffix"))
	if audit.Status != SerializationChanged {
		t.Fatalf("status = %q, want %q", audit.Status, SerializationChanged)
	}
	if audit.Range.Start != len("prefix ") || audit.Range.OriginalEnd != len("prefix OLD") || audit.Range.CandidateEnd != len("prefix NEW") {
		t.Fatalf("changed range = %+v, want the OLD/NEW byte interval", audit.Range)
	}
}

func TestSerializeAuditJSONRemarshalReportsUnstableBytes(t *testing.T) {
	raw := []byte(`{"z":2,"a":1}`)
	audit := AuditJSONRemarshal(raw)
	if audit.Status != SerializationChanged {
		t.Fatalf("status = %q, want %q for reordered JSON", audit.Status, SerializationChanged)
	}
	if audit.Range.IsZero() || audit.Range.OriginalEnd <= audit.Range.Start || audit.Range.CandidateEnd <= audit.Range.Start {
		t.Fatalf("changed range must be non-empty, got %+v", audit.Range)
	}
}

func TestSerializeAuditJSONRemarshalInvalidJSON(t *testing.T) {
	audit := AuditJSONRemarshal([]byte(`{"not":`))
	if audit.Status != SerializationInvalidJSON {
		t.Fatalf("status = %q, want %q", audit.Status, SerializationInvalidJSON)
	}
}
