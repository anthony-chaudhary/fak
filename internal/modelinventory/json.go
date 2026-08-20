package modelinventory

import (
	"bytes"
	"encoding/json"
	"io"
	"time"
)

// CanonicalJSON validates and renders a byte-stable, newline-terminated artifact.
func (in Inventory) CanonicalJSON() ([]byte, Diagnostics) {
	canonical := canonicalizeInventory(in)
	asOf, err := time.Parse(time.RFC3339, canonical.AsOf)
	if err != nil {
		return nil, Diagnostics{newDiagnostic(
			CodeEvidenceMalformed, "", "as_of", "", "inventory as_of is not RFC3339",
			"write one explicit RFC3339 UTC timestamp",
		)}
	}
	if diagnostics := canonical.ValidateAt(asOf); len(diagnostics) != 0 {
		return nil, diagnostics
	}
	type wireInventory Inventory
	b, err := json.MarshalIndent(wireInventory(canonical), "", "  ")
	if err != nil {
		return nil, Diagnostics{newDiagnostic(CodeInvalidJSON, "", "inventory", "", "inventory cannot be encoded", "repair the typed inventory value")}
	}
	return append(b, '\n'), nil
}

// MarshalJSON enforces the same validation floor as CanonicalJSON, preventing a
// caller from serializing mutated or credential-bearing inventory state.
func (in Inventory) MarshalJSON() ([]byte, error) {
	canonical := canonicalizeInventory(in)
	asOf, err := time.Parse(time.RFC3339, canonical.AsOf)
	if err != nil {
		return nil, Diagnostics{newDiagnostic(CodeEvidenceMalformed, "", "as_of", "", "inventory as_of is not RFC3339", "write one explicit RFC3339 UTC timestamp")}
	}
	if diagnostics := canonical.ValidateAt(asOf); len(diagnostics) != 0 {
		return nil, diagnostics
	}
	type wireInventory Inventory
	return json.Marshal(wireInventory(canonical))
}

// ParseJSON performs strict independent read-back and rechecks evidence at asOf.
func ParseJSON(data []byte, asOf time.Time) (Inventory, Diagnostics) {
	type wireInventory Inventory
	var wire wireInventory
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return Inventory{}, Diagnostics{newDiagnostic(CodeInvalidJSON, "", "inventory", "", "inventory is not one strict JSON document", "regenerate it from typed observations")}
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return Inventory{}, Diagnostics{newDiagnostic(CodeInvalidJSON, "", "inventory", "", "inventory contains trailing JSON data", "keep exactly one JSON document")}
	}
	canonical := canonicalizeInventory(Inventory(wire))
	if diagnostics := canonical.ValidateAt(asOf); len(diagnostics) != 0 {
		return Inventory{}, diagnostics
	}
	return canonical, nil
}

// CanonicalJSON renders diagnostics in their stable rejection order.
func (ds Diagnostics) CanonicalJSON() []byte {
	b, err := json.MarshalIndent(ds.sorted(), "", "  ")
	if err != nil {
		return []byte("[]\n")
	}
	return append(b, '\n')
}
