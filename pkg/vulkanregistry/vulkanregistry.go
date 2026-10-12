// Package vulkanregistry exports the identity of the Vulkan SPIR-V shader
// registry the fak builder compiles and stamps today, so consumers that admit
// fak Vulkan build receipts track the builder instead of copying its constants.
package vulkanregistry

import "github.com/anthony-chaudhary/fak/internal/computebuild"

// Identity names one trusted shader registry and the receipt schema that
// stamps it. SHA256 hashes the sorted module stems as a JSON string array.
type Identity struct {
	ReceiptSchema string `json:"receipt_schema"`
	ID            string `json:"id"`
	SHA256        string `json:"sha256"`
	ModuleCount   int    `json:"module_count"`
}

// Current returns the registry identity the builder stamps today.
func Current() (Identity, error) {
	schema, registry, err := computebuild.CurrentVulkanBuildIdentity()
	if err != nil {
		return Identity{}, err
	}
	return Identity{ReceiptSchema: schema, ID: registry.ID, SHA256: registry.SHA256, ModuleCount: registry.ModuleCount}, nil
}
