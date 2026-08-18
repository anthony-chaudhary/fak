package stackresolve

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const ComponentContractSchema = "fak-component-contract/1"

// ComponentContract is the independently publishable unit of stack compatibility.
// Requires and conflicts are hard constraints; recommends and optional relations are hints.
type ComponentContract struct {
	Schema    string    `json:"schema"`
	Component Component `json:"component"`
}

// ParseComponentContract decodes one standalone component declaration and applies
// the same validation used by full stack manifests.
func ParseComponentContract(raw []byte) (ComponentContract, error) {
	var contract ComponentContract
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&contract); err != nil {
		return ComponentContract{}, fmt.Errorf("component contract: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return ComponentContract{}, fmt.Errorf("component contract: %w", err)
	}
	if contract.Schema != ComponentContractSchema {
		return ComponentContract{}, fmt.Errorf("component contract: schema %q, want %q", contract.Schema, ComponentContractSchema)
	}
	if err := validateComponent(contract.Component, "component contract"); err != nil {
		return ComponentContract{}, fmt.Errorf("component contract: %w", err)
	}
	return contract, nil
}

// ComposeContracts turns independently authored contracts into the existing
// resolver manifest without weakening duplicate-ID or relation validation.
func ComposeContracts(workload string, roots []string, contracts []ComponentContract) (Manifest, error) {
	if strings.TrimSpace(workload) == "" {
		return Manifest{}, fmt.Errorf("compose component contracts: workload is required")
	}
	if len(roots) == 0 {
		return Manifest{}, fmt.Errorf("compose component contracts: at least one root is required")
	}
	manifest := Manifest{Schema: ManifestSchema, Workload: workload, Roots: append([]string(nil), roots...)}
	seen := make(map[string]bool, len(contracts))
	for _, contract := range contracts {
		if contract.Schema != ComponentContractSchema {
			return Manifest{}, fmt.Errorf("component contract: schema %q, want %q", contract.Schema, ComponentContractSchema)
		}
		if err := validateComponent(contract.Component, "component contract"); err != nil {
			return Manifest{}, fmt.Errorf("component contract: %w", err)
		}
		if seen[contract.Component.ID] {
			return Manifest{}, fmt.Errorf("compose component contracts: duplicate component id %q", contract.Component.ID)
		}
		seen[contract.Component.ID] = true
		manifest.Components = append(manifest.Components, contract.Component)
	}
	return manifest, nil
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return err
	}
	return fmt.Errorf("multiple JSON values")
}
