//go:build !darwin

package modelperfobs

import (
	"context"
	"fmt"
)

const AppleMemoryCurrentProviderSchema = "fak-apple-memory-current-provider/1"

const (
	AppleMemoryUnavailableUnsupported      = "unsupported"
	AppleMemoryUnavailablePermissionDenied = "permission-denied"
	AppleMemoryUnavailableMalformed        = "malformed"
	AppleMemoryUnavailableAmbiguous        = "ambiguous"
)

type AppleMemoryProviderEvidence struct {
	Schema             string           `json:"schema"`
	Provider           string           `json:"provider"`
	ProviderVersion    string           `json:"provider_version"`
	Status             string           `json:"status"`
	UnavailableReason  string           `json:"unavailable_reason,omitempty"`
	Detail             string           `json:"detail,omitempty"`
	Scope              AppleMemoryScope `json:"scope"`
	IntervalNS         int64            `json:"interval_ns"`
	RateOrDelta        string           `json:"rate_or_delta"`
	ResetSemantics     string           `json:"reset_semantics"`
	AdvertisedSamplers []string         `json:"advertised_samplers,omitempty"`
	RawFields          []string         `json:"raw_fields"`
	RawUnits           []string         `json:"raw_units"`
	ObservedAt         string           `json:"observed_at,omitempty"`
}

type AppleMemoryProviderUnavailableError struct {
	Evidence AppleMemoryProviderEvidence
	Cause    error
}

func (e *AppleMemoryProviderUnavailableError) Error() string {
	return fmt.Sprintf("Apple unified-memory bandwidth unavailable (%s): %s", e.Evidence.UnavailableReason, e.Evidence.Detail)
}
func (e *AppleMemoryProviderUnavailableError) Unwrap() error { return e.Cause }

func CollectCurrentAppleMemoryBandwidth(context.Context, AppleMemoryImportOptions) (BandwidthCollection, error) {
	return BandwidthCollection{}, &AppleMemoryProviderUnavailableError{Evidence: AppleMemoryProviderEvidence{
		Schema: AppleMemoryCurrentProviderSchema, Provider: "apple-powermetrics", ProviderVersion: "unavailable",
		Status: "unavailable", UnavailableReason: AppleMemoryUnavailableUnsupported,
		Detail: "Apple powermetrics unified-memory bandwidth collection is available only on Darwin",
		Scope:  AppleMemoryScope{Kind: "system"}, RateOrDelta: "unavailable",
		ResetSemantics: "unavailable; no byte counter was accepted", RawFields: []string{}, RawUnits: []string{},
	}}
}
