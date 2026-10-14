package traceabi_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/traceabi"
)

// fak-test:runtime fast est=10ms
func TestExternalConsumer(t *testing.T) {
	t.Parallel()
	e := traceabi.CausalEnvelope{
		Schema: traceabi.Schema, TraceID: "native/trace", TraceFormat: traceabi.TraceOpaque,
		ParentStatus: traceabi.ParentUnknown, Sequence: 1, LogicalID: "native/event",
		Operation: "consumer", Phase: "observed", SourceSchema: "external/1", SourceVersion: "1",
		MappingStatus: traceabi.MappingUnknown,
		Correlations:  []traceabi.Correlation{{Kind: traceabi.CorrelationProducerNative, Value: " bytes/<unchanged> "}},
	}
	wire, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var decoded traceabi.CausalEnvelope
	if err := json.Unmarshal(wire, &decoded); err != nil || !reflect.DeepEqual(e, decoded) {
		t.Fatalf("external consumer lost opaque values: %+v, %v", decoded, err)
	}
	descriptor := traceabi.CausalEnvelopeSchemaDescriptor()
	if descriptor.EnvelopeSchema != e.Schema || len(descriptor.CorrelationKinds) != 7 || len(descriptor.Required)+len(descriptor.Optional) != reflect.TypeOf(e).NumField() {
		t.Fatalf("incomplete public schema descriptor: %+v", descriptor)
	}
	// Each descriptor is caller-owned; changing one must not rewrite the ABI.
	descriptor.Required[0] = "changed"
	if traceabi.CausalEnvelopeSchemaDescriptor().Required[0] != "schema" {
		t.Fatal("schema descriptor shares mutable global storage")
	}
}
