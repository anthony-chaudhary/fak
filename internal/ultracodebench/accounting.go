package ultracodebench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
)

const AccountingSchema = "fak.ultracode.accounting.v1"

type AccountingAvailability string

const (
	AccountingUnavailable AccountingAvailability = "unavailable"
	AccountingPartial     AccountingAvailability = "partial"
	AccountingAvailable   AccountingAvailability = "available"
)

type AccountingAuthority string

const (
	AuthorityUnreported      AccountingAuthority = "unreported"
	AuthorityProviderUsage   AccountingAuthority = "provider_usage"
	AuthorityProviderBilling AccountingAuthority = "provider_billing"
)

type TokenAccounting struct {
	Availability   AccountingAvailability `json:"availability"`
	Authority      AccountingAuthority    `json:"authority"`
	ArtifactDigest string                 `json:"artifact_digest,omitempty"`
	Coverage       float64                `json:"coverage"`
	Value          *int64                 `json:"value,omitempty"`
	Reason         string                 `json:"reason,omitempty"`
}

type SpendAccounting struct {
	Availability   AccountingAvailability `json:"availability"`
	Authority      AccountingAuthority    `json:"authority"`
	ArtifactDigest string                 `json:"artifact_digest,omitempty"`
	Coverage       float64                `json:"coverage"`
	ValueUSD       *float64               `json:"value_usd,omitempty"`
	Reason         string                 `json:"reason,omitempty"`
}

// AccountingReceipt is the redacted provider join. Every axis keeps its own
// provenance and coverage; no total-token field is promoted into billed tokens.
type AccountingReceipt struct {
	Schema           string          `json:"schema"`
	InputTokens      TokenAccounting `json:"input_tokens"`
	OutputTokens     TokenAccounting `json:"output_tokens"`
	CacheReadTokens  TokenAccounting `json:"cache_read_tokens"`
	CacheWriteTokens TokenAccounting `json:"cache_write_tokens"`
	BilledTokens     TokenAccounting `json:"billed_tokens"`
	SpendUSD         SpendAccounting `json:"spend_usd"`
}

func DecodeAccounting(data []byte) (AccountingReceipt, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	var receipt AccountingReceipt
	if err := dec.Decode(&receipt); err != nil {
		return AccountingReceipt{}, fmt.Errorf("decode accounting receipt: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return AccountingReceipt{}, fmt.Errorf("decode accounting receipt: trailing JSON value")
		}
		return AccountingReceipt{}, fmt.Errorf("decode accounting receipt: %w", err)
	}
	return receipt, receipt.Validate()
}

func (r *AccountingReceipt) UnmarshalJSON(data []byte) error {
	type wire AccountingReceipt
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var decoded wire
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing JSON value")
		}
		return err
	}
	receipt := AccountingReceipt(decoded)
	if err := receipt.Validate(); err != nil {
		return err
	}
	*r = receipt
	return nil
}

func (r AccountingReceipt) Validate() error {
	if r.Schema != AccountingSchema {
		return fmt.Errorf("accounting schema must be %q", AccountingSchema)
	}
	for name, axis := range map[string]TokenAccounting{
		"input_tokens": r.InputTokens, "output_tokens": r.OutputTokens,
		"cache_read_tokens": r.CacheReadTokens, "cache_write_tokens": r.CacheWriteTokens,
		"billed_tokens": r.BilledTokens,
	} {
		if err := axis.validate(name); err != nil {
			return err
		}
	}
	return r.SpendUSD.validate("spend_usd")
}

func (a TokenAccounting) validate(name string) error {
	if err := validateAccountingMetadata(name, a.Availability, a.Authority, a.ArtifactDigest, a.Coverage, a.Reason, a.Value != nil); err != nil {
		return err
	}
	if a.Value != nil && *a.Value < 0 {
		return fmt.Errorf("accounting %s value cannot be negative", name)
	}
	return nil
}

func (a SpendAccounting) validate(name string) error {
	if err := validateAccountingMetadata(name, a.Availability, a.Authority, a.ArtifactDigest, a.Coverage, a.Reason, a.ValueUSD != nil); err != nil {
		return err
	}
	if a.ValueUSD != nil && (*a.ValueUSD < 0 || math.IsNaN(*a.ValueUSD) || math.IsInf(*a.ValueUSD, 0)) {
		return fmt.Errorf("accounting %s value_usd must be finite and non-negative", name)
	}
	return nil
}

func validateAccountingMetadata(name string, availability AccountingAvailability, authority AccountingAuthority, digest string, coverage float64, reason string, hasValue bool) error {
	if authority != AuthorityUnreported && authority != AuthorityProviderUsage && authority != AuthorityProviderBilling {
		return fmt.Errorf("accounting %s authority %q is invalid", name, authority)
	}
	if math.IsNaN(coverage) || math.IsInf(coverage, 0) || coverage < 0 || coverage > 1 {
		return fmt.Errorf("accounting %s coverage must be in [0,1]", name)
	}
	if authority != AuthorityUnreported && !validTrajectoryDigest(digest) {
		return fmt.Errorf("accounting %s artifact_digest must be a sha256 digest", name)
	}
	switch availability {
	case AccountingAvailable:
		if authority == AuthorityUnreported || coverage != 1 || !hasValue {
			return fmt.Errorf("accounting %s available value requires reported authority, full coverage, and a value", name)
		}
	case AccountingPartial:
		if authority == AuthorityUnreported || coverage <= 0 || coverage >= 1 || !hasValue || strings.TrimSpace(reason) == "" {
			return fmt.Errorf("accounting %s partial value requires reported authority, partial coverage, a value, and a reason", name)
		}
	case AccountingUnavailable:
		if coverage != 0 || hasValue || strings.TrimSpace(reason) == "" {
			return fmt.Errorf("accounting %s unavailable value requires zero coverage, no value, and a reason", name)
		}
	default:
		return fmt.Errorf("accounting %s availability %q is invalid", name, availability)
	}
	return nil
}

func missingAccountingReceipt() AccountingReceipt {
	token := TokenAccounting{Availability: AccountingUnavailable, Authority: AuthorityUnreported, Reason: "accounting receipt missing"}
	return AccountingReceipt{
		Schema:      AccountingSchema,
		InputTokens: token, OutputTokens: token, CacheReadTokens: token, CacheWriteTokens: token, BilledTokens: token,
		SpendUSD: SpendAccounting{Availability: AccountingUnavailable, Authority: AuthorityUnreported, Reason: "accounting receipt missing"},
	}
}

func normalizeAccounting(receipt AccountingReceipt) (AccountingReceipt, error) {
	if receipt.Schema == "" {
		return missingAccountingReceipt(), nil
	}
	return receipt, receipt.Validate()
}

func tokenAccounting(value int64, authority AccountingAuthority, digest string) TokenAccounting {
	return TokenAccounting{Availability: AccountingAvailable, Authority: authority, ArtifactDigest: digest, Coverage: 1, Value: &value}
}

func spendAccounting(value float64, authority AccountingAuthority, digest string) SpendAccounting {
	return SpendAccounting{Availability: AccountingAvailable, Authority: authority, ArtifactDigest: digest, Coverage: 1, ValueUSD: &value}
}

func knownAccounting(input, output, cacheRead, cacheWrite, billed int64, spend float64, digest string) AccountingReceipt {
	return AccountingReceipt{
		Schema:           AccountingSchema,
		InputTokens:      tokenAccounting(input, AuthorityProviderUsage, digest),
		OutputTokens:     tokenAccounting(output, AuthorityProviderUsage, digest),
		CacheReadTokens:  tokenAccounting(cacheRead, AuthorityProviderUsage, digest),
		CacheWriteTokens: tokenAccounting(cacheWrite, AuthorityProviderUsage, digest),
		BilledTokens:     tokenAccounting(billed, AuthorityProviderBilling, digest),
		SpendUSD:         spendAccounting(spend, AuthorityProviderBilling, digest),
	}
}
