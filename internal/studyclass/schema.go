package studyclass

import (
	_ "embed"
	"encoding/json"
)

//go:embed classification.schema.json
var schemaDocument []byte

// SchemaDocument returns an independent copy of the deterministic full-output
// Draft 2020-12 JSON Schema document.
func SchemaDocument() []byte { return append([]byte(nil), schemaDocument...) }

// SchemaObject returns the same schema as a generic machine-readable object.
func SchemaObject() map[string]any {
	var out map[string]any
	if err := json.Unmarshal(schemaDocument, &out); err != nil {
		panic("studyclass: embedded schema is invalid: " + err.Error())
	}
	return out
}
