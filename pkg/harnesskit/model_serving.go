package harnesskit

// LocalModelDeclaration is the immutable, caller-authored model-serving input.
// It identifies an existing GGUF artifact; acquisition and runtime launch are
// separate lifecycle steps.
type LocalModelDeclaration struct {
	Schema          string   `json:"schema"`
	ModelID         string   `json:"model_id"`
	GGUFPath        string   `json:"gguf_path"`
	GGUFSHA256      string   `json:"gguf_sha256"`
	Runtime         string   `json:"runtime"`
	ContextTokens   int      `json:"context_tokens"`
	RequiredDevices []string `json:"required_devices,omitempty"`
}
