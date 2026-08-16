package quantobs

import (
	"encoding/json"
	"fmt"
)

type wireEvent struct {
	SchemaVersion string `json:"schema_version"`
	Kind          string `json:"kind"`
	Outcome       string `json:"outcome"`
	Value         string `json:"value"`
	Recipe        string `json:"recipe"`
	Evidence      string `json:"evidence"`
	Envelope      string `json:"hardware_envelope"`
	Reason        string `json:"reason"`
}

type wireResult struct {
	SchemaVersion string      `json:"schema_version"`
	Outcome       string      `json:"outcome"`
	Reason        string      `json:"reason"`
	Events        []wireEvent `json:"events"`
}

// MarshalJSON emits the stable public schema and rejects impossible output
// values rather than serializing integer fallbacks.
func (r Result) MarshalJSON() ([]byte, error) {
	outcome, ok := outcomeName(r.Outcome)
	if !ok {
		return nil, fmt.Errorf("quantobs: invalid result outcome %d", r.Outcome)
	}
	reason, ok := reasonName(r.Reason)
	if !ok {
		return nil, fmt.Errorf("quantobs: invalid result reason %d", r.Reason)
	}
	wire := wireResult{SchemaVersion: SchemaVersion, Outcome: outcome, Reason: reason, Events: make([]wireEvent, len(r.Events))}
	for i, event := range r.Events {
		converted, err := event.wire()
		if err != nil {
			return nil, fmt.Errorf("quantobs: event %d: %w", i, err)
		}
		wire.Events[i] = converted
	}
	return json.Marshal(wire)
}

func (e Event) wire() (wireEvent, error) {
	kind, ok := kindName(e.Kind)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid kind %d", e.Kind)
	}
	outcome, ok := outcomeName(e.Outcome)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid outcome %d", e.Outcome)
	}
	value, ok := codeName(e.Value)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid value %d", e.Value)
	}
	recipe, ok := recipeName(e.Recipe)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid recipe %d", e.Recipe)
	}
	evidence, ok := evidenceName(e.Evidence)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid evidence %d", e.Evidence)
	}
	envelope, ok := envelopeName(e.Envelope)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid envelope %d", e.Envelope)
	}
	reason, ok := reasonName(e.Reason)
	if !ok {
		return wireEvent{}, fmt.Errorf("invalid reason %d", e.Reason)
	}
	return wireEvent{SchemaVersion, kind, outcome, value, recipe, evidence, envelope, reason}, nil
}

func kindName(v EventKind) (string, bool) {
	return lookup(uint8(v), []string{"", "artifact_format", "effective_precision", "runtime_delegation", "conversion", "memory_residency", "refusal_reason"})
}

func outcomeName(v Outcome) (string, bool) {
	return lookup(uint8(v), []string{"", "observed", "abstained", "refused"})
}

func evidenceName(v Evidence) (string, bool) {
	return lookup(uint8(v), []string{"", "artifact_metadata", "recipe_declaration", "runtime_report", "conversion_record", "measured_hardware", "adjudication"})
}

func envelopeName(v Envelope) (string, bool) {
	return lookup(uint8(v), []string{"", "not_applicable", "unmeasured", "measured"})
}

func codeName(v Code) (string, bool) {
	names := map[Code]string{
		CodeUnknown: "unknown", CodeNotApplicable: "not_applicable",
		CodeGGUF: "gguf", CodeSafeTensors: "safetensors", CodeONNX: "onnx", CodeTorchScript: "torchscript",
		CodeFP32: "fp32", CodeFP16: "fp16", CodeBF16: "bf16", CodeINT8: "int8", CodeINT4: "int4", CodeFP8: "fp8", CodeMixed: "mixed",
		CodeRecipeNone: "none", CodeWeightOnly: "weight_only", CodeWeightActivation: "weight_activation", CodeKVCache: "kv_cache", CodeHybrid: "hybrid",
		CodeRuntimeLocal: "local", CodeRuntimeDelegated: "delegated",
		CodeConversionNone: "none", CodeConversionLossless: "lossless", CodeConversionRequantized: "requantized", CodeConversionDequantized: "dequantized",
		CodeResidencyHost: "host", CodeResidencyAccelerator: "accelerator", CodeResidencySplit: "split", CodeResidencyStorage: "storage",
		CodeNoRefusal: "none", CodeUnknownSchema: "unknown_schema", CodeUnknownInput: "unknown_input", CodeUnsupportedCombination: "unsupported_combination",
	}
	name, ok := names[v]
	return name, ok
}

func recipeName(v Code) (string, bool) {
	if v == CodeNotApplicable {
		return "not_applicable", true
	}
	if oneOf(v, CodeRecipeNone, CodeWeightOnly, CodeWeightActivation, CodeKVCache, CodeHybrid) {
		return codeName(v)
	}
	return "", false
}

func reasonName(v Code) (string, bool) {
	if oneOf(v, CodeNoRefusal, CodeUnknownSchema, CodeUnknownInput, CodeUnsupportedCombination) {
		return codeName(v)
	}
	return "", false
}

func lookup(index uint8, values []string) (string, bool) {
	if int(index) >= len(values) || values[index] == "" {
		return "", false
	}
	return values[index], true
}
