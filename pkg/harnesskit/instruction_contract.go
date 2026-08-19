package harnesskit

// InstructionCompositionContract describes the public dynamic-instruction boundary.
type InstructionCompositionContract struct {
	SchemaVersion string `json:"schema_version"`
	Resolution    string `json:"resolution"`
	Composition   string `json:"composition"`
	Ownership     string `json:"ownership"`
	Security      string `json:"security"`
	Determinism   string `json:"determinism"`
	Cancellation  string `json:"cancellation"`
	Compatibility string `json:"compatibility"`
	Errors        []Code `json:"errors"`
}

// PublicInstructionContract returns the normative dynamic-instruction contract.
func PublicInstructionContract() InstructionCompositionContract {
	return InstructionCompositionContract{
		SchemaVersion: InstructionContractVersion,
		Resolution:    "the host invokes InstructionProvider.Resolve at declared run, thread, or turn boundaries",
		Composition:   "providers return typed fragments; the host validates, orders, fingerprints, and realizes them without opaque prompt mutation",
		Ownership:     "providers own application fragments; the host owns policy, stable-prefix admission, final serialization, and invocation authority",
		Security:      "only host-trusted fragments may enter the stable prefix; untrusted fragments cannot claim positive precedence",
		Determinism:   "identical typed inputs and provider output produce byte-stable ordering and SHA-256 snapshot fingerprints",
		Cancellation:  "resolution accepts context.Context and returns CodeCanceled with the cancellation cause",
		Compatibility: "same schema_version is additive; changed ordering, authority, or fingerprint semantics require a new schema_version",
		Errors:        []Code{CodeInvalid, CodeUnsupported, CodeConflict, CodeDenied, CodeCanceled, CodeInternal},
	}
}
