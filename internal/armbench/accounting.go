package armbench

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
)

const AccountingReceiptSchema = "fak.armbench.accounting-receipt/1"

// AccountingMetric names one independently authoritative usage field. Keeping
// the fields separate prevents an available token count from laundering a
// missing cache or cost count into a complete-looking receipt.
type AccountingMetric string

const (
	MetricInputTokens      AccountingMetric = "input_tokens"
	MetricOutputTokens     AccountingMetric = "output_tokens"
	MetricCacheReadTokens  AccountingMetric = "cache_read_tokens"
	MetricCacheWriteTokens AccountingMetric = "cache_write_tokens"
	MetricCacheHits        AccountingMetric = "cache_hits"
	MetricCacheMisses      AccountingMetric = "cache_misses"
	MetricCostUSD          AccountingMetric = "cost_usd"
)

var accountingMetrics = []AccountingMetric{
	MetricInputTokens,
	MetricOutputTokens,
	MetricCacheReadTokens,
	MetricCacheWriteTokens,
	MetricCacheHits,
	MetricCacheMisses,
	MetricCostUSD,
}

type AccountingAvailability string

const (
	AvailabilityMissing   AccountingAvailability = "missing"
	AvailabilityAvailable AccountingAvailability = "available"
	AvailabilityDegraded  AccountingAvailability = "degraded"
	AvailabilityConflict  AccountingAvailability = "conflict"
)

type AccountingAuthority string

const (
	AuthorityUnknown           AccountingAuthority = "unknown"
	AuthorityStepSum           AccountingAuthority = "step_sum"
	AuthorityHarnessAggregate  AccountingAuthority = "harness_aggregate"
	AuthorityProviderAggregate AccountingAuthority = "provider_aggregate"
)

// AccountingArtifact binds a number to the raw provider or harness artifact
// that supplied it. Ref locates the artifact; SHA256 makes that reference
// stable even when its storage path is mutable.
type AccountingArtifact struct {
	Ref    string `json:"ref"`
	SHA256 string `json:"sha256"`
}

// ArtifactFor returns a content-bound artifact reference.
func ArtifactFor(ref string, content []byte) AccountingArtifact {
	digest := sha256.Sum256(content)
	return AccountingArtifact{Ref: ref, SHA256: "sha256:" + hex.EncodeToString(digest[:])}
}

// AccountingCoverage states which population the source covers. A complete
// aggregate is selected ahead of an incomplete sum even when the sum comes from
// a nominally higher rung.
type AccountingCoverage struct {
	Scope    string `json:"scope"`
	Observed int    `json:"observed"`
	Expected int    `json:"expected"`
}

func (c AccountingCoverage) Complete() bool {
	return c.Expected > 0 && c.Observed == c.Expected
}

// AccountingValues uses pointers so an observed zero and an absent field have
// different wire representations.
type AccountingValues struct {
	InputTokens      *float64 `json:"input_tokens"`
	OutputTokens     *float64 `json:"output_tokens"`
	CacheReadTokens  *float64 `json:"cache_read_tokens"`
	CacheWriteTokens *float64 `json:"cache_write_tokens"`
	CacheHits        *float64 `json:"cache_hits"`
	CacheMisses      *float64 `json:"cache_misses"`
	CostUSD          *float64 `json:"cost_usd"`
}

// AccountingSource is one independently captured view of usage. Reconciliation
// retains every source instead of replacing the fallback in place.
type AccountingSource struct {
	Authority AccountingAuthority `json:"authority"`
	Artifact  AccountingArtifact  `json:"artifact"`
	Coverage  AccountingCoverage  `json:"coverage"`
	Values    AccountingValues    `json:"values"`
}

type AccountingObservation struct {
	Value     float64             `json:"value"`
	Authority AccountingAuthority `json:"authority"`
	Artifact  AccountingArtifact  `json:"artifact"`
	Coverage  AccountingCoverage  `json:"coverage"`
}

type AccountingDiscrepancy struct {
	Selected AccountingObservation `json:"selected"`
	Other    AccountingObservation `json:"other"`
	Delta    float64               `json:"delta"`
	Material bool                  `json:"material"`
	Detail   string                `json:"detail"`
}

// AccountingField is the publishable decision for one field. Value is nil for
// missing accounting and non-nil for an observed zero.
type AccountingField struct {
	Availability  AccountingAvailability  `json:"availability"`
	Value         *float64                `json:"value"`
	Unit          string                  `json:"unit"`
	Authority     AccountingAuthority     `json:"authority"`
	Artifact      AccountingArtifact      `json:"artifact"`
	Coverage      AccountingCoverage      `json:"coverage"`
	Sources       []AccountingObservation `json:"sources"`
	Discrepancies []AccountingDiscrepancy `json:"discrepancies"`
	RefusalReason string                  `json:"refusal_reason,omitempty"`
}

// AccountingReceipt carries independently reconciled token, cache, and cost
// fields. Callers compare fields, never the receipt as one all-or-nothing blob.
type AccountingReceipt struct {
	Schema           string          `json:"schema"`
	InputTokens      AccountingField `json:"input_tokens"`
	OutputTokens     AccountingField `json:"output_tokens"`
	CacheReadTokens  AccountingField `json:"cache_read_tokens"`
	CacheWriteTokens AccountingField `json:"cache_write_tokens"`
	CacheHits        AccountingField `json:"cache_hits"`
	CacheMisses      AccountingField `json:"cache_misses"`
	CostUSD          AccountingField `json:"cost_usd"`
}

// ReconcileAccounting selects the most complete, authoritative source for
// each field and keeps every competing observation in the receipt.
func ReconcileAccounting(sources []AccountingSource) (AccountingReceipt, error) {
	for i, source := range sources {
		if err := validateAccountingSource(source); err != nil {
			return AccountingReceipt{}, fmt.Errorf("accounting source %d: %w", i, err)
		}
	}
	receipt := missingAccountingReceipt()
	for _, metric := range accountingMetrics {
		observations := make([]AccountingObservation, 0, len(sources))
		for _, source := range sources {
			if value := source.Values.field(metric); value != nil {
				observations = append(observations, AccountingObservation{
					Value: *value, Authority: source.Authority,
					Artifact: source.Artifact, Coverage: source.Coverage,
				})
			}
		}
		receipt.setField(metric, reconcileAccountingField(metric, observations))
	}
	return receipt, nil
}

// AggregateAccounting sums trial receipts without promoting incomplete,
// mixed-authority, or conflicting fields to publishable totals.
func AggregateAccounting(receipts []AccountingReceipt) AccountingReceipt {
	aggregate := missingAccountingReceipt()
	for _, metric := range accountingMetrics {
		fields := make([]AccountingField, 0, len(receipts))
		for _, receipt := range receipts {
			field, _ := normalizedAccountingReceipt(receipt).Field(metric)
			fields = append(fields, field)
		}
		aggregate.setField(metric, aggregateAccountingField(metric, fields))
	}
	return aggregate
}

// Field returns the named field without reflection so adapters share the exact
// metric vocabulary used by reconciliation.
func (r AccountingReceipt) Field(metric AccountingMetric) (AccountingField, bool) {
	switch metric {
	case MetricInputTokens:
		return r.InputTokens, true
	case MetricOutputTokens:
		return r.OutputTokens, true
	case MetricCacheReadTokens:
		return r.CacheReadTokens, true
	case MetricCacheWriteTokens:
		return r.CacheWriteTokens, true
	case MetricCacheHits:
		return r.CacheHits, true
	case MetricCacheMisses:
		return r.CacheMisses, true
	case MetricCostUSD:
		return r.CostUSD, true
	default:
		return AccountingField{}, false
	}
}

func (r *AccountingReceipt) setField(metric AccountingMetric, field AccountingField) {
	switch metric {
	case MetricInputTokens:
		r.InputTokens = field
	case MetricOutputTokens:
		r.OutputTokens = field
	case MetricCacheReadTokens:
		r.CacheReadTokens = field
	case MetricCacheWriteTokens:
		r.CacheWriteTokens = field
	case MetricCacheHits:
		r.CacheHits = field
	case MetricCacheMisses:
		r.CacheMisses = field
	case MetricCostUSD:
		r.CostUSD = field
	}
}

// Validate replays the receipt decision from its retained observations. It
// catches hand-edited values and authority labels before a run is published.
func (r AccountingReceipt) Validate() error {
	if r.Schema != AccountingReceiptSchema {
		return fmt.Errorf("accounting schema %q, want %q", r.Schema, AccountingReceiptSchema)
	}
	for _, metric := range accountingMetrics {
		field, _ := r.Field(metric)
		for i, observation := range field.Sources {
			source := AccountingSource{
				Authority: observation.Authority,
				Artifact:  observation.Artifact,
				Coverage:  observation.Coverage,
			}
			source.Values.setField(metric, observation.Value)
			if err := validateAccountingSource(source); err != nil {
				return fmt.Errorf("%s source %d: %w", metric, i, err)
			}
		}
		want := reconcileAccountingField(metric, append([]AccountingObservation(nil), field.Sources...))
		if !sameAccountingDecision(field, want) {
			return fmt.Errorf("%s decision does not reconcile from retained sources", metric)
		}
	}
	return nil
}

// AccountingFieldComparison is the common claim gate used by ArmBench and
// AgenticBench. Delta is right minus left and exists only for comparable data.
type AccountingFieldComparison struct {
	Comparable bool     `json:"comparable"`
	Delta      *float64 `json:"delta"`
	Detail     string   `json:"detail"`
}

func CompareAccountingFields(metric AccountingMetric, left, right AccountingField) AccountingFieldComparison {
	if _, ok := metricUnit(metric); !ok {
		return AccountingFieldComparison{Detail: fmt.Sprintf("unknown accounting metric %q", metric)}
	}
	if left.Availability != AvailabilityAvailable || right.Availability != AvailabilityAvailable || left.Value == nil || right.Value == nil {
		return AccountingFieldComparison{Detail: fmt.Sprintf("availability is not comparable: left=%s right=%s", left.Availability, right.Availability)}
	}
	if left.Authority != right.Authority || left.Authority == AuthorityUnknown {
		return AccountingFieldComparison{Detail: fmt.Sprintf("authority is incomparable: left=%s right=%s", left.Authority, right.Authority)}
	}
	if !left.Coverage.Complete() || !right.Coverage.Complete() || left.Coverage.Scope != right.Coverage.Scope || left.Coverage.Expected != right.Coverage.Expected {
		return AccountingFieldComparison{Detail: fmt.Sprintf("coverage is incomparable: left=%s %d/%d right=%s %d/%d", left.Coverage.Scope, left.Coverage.Observed, left.Coverage.Expected, right.Coverage.Scope, right.Coverage.Observed, right.Coverage.Expected)}
	}
	delta := *right.Value - *left.Value
	return AccountingFieldComparison{Comparable: true, Delta: &delta, Detail: "authority and complete coverage reconcile"}
}

func reconcileAccountingField(metric AccountingMetric, observations []AccountingObservation) AccountingField {
	unit, _ := metricUnit(metric)
	field := AccountingField{Availability: AvailabilityMissing, Unit: unit, Authority: AuthorityUnknown, Sources: observations}
	if len(observations) == 0 {
		field.RefusalReason = "no accounting source reported this field"
		return field
	}
	sort.SliceStable(observations, func(i, j int) bool {
		a, b := observations[i], observations[j]
		if a.Coverage.Complete() != b.Coverage.Complete() {
			return a.Coverage.Complete()
		}
		if authorityRank(a.Authority) != authorityRank(b.Authority) {
			return authorityRank(a.Authority) > authorityRank(b.Authority)
		}
		if a.Artifact.Ref != b.Artifact.Ref {
			return a.Artifact.Ref < b.Artifact.Ref
		}
		return a.Value < b.Value
	})
	field.Sources = observations
	selected := observations[0]
	value := selected.Value
	field.Value = &value
	field.Authority = selected.Authority
	field.Artifact = selected.Artifact
	field.Coverage = selected.Coverage
	field.Availability = AvailabilityAvailable
	if !selected.Coverage.Complete() {
		field.Availability = AvailabilityDegraded
		field.RefusalReason = fmt.Sprintf("selected fallback covers %d/%d %s", selected.Coverage.Observed, selected.Coverage.Expected, selected.Coverage.Scope)
	}
	for _, other := range observations[1:] {
		if !materiallyDifferent(metric, selected.Value, other.Value) {
			continue
		}
		material := selected.Coverage.Complete() && other.Coverage.Complete()
		detail := "complete source supersedes incomplete fallback"
		if material {
			detail = "complete sources disagree"
			field.Availability = AvailabilityConflict
			field.RefusalReason = detail
		}
		field.Discrepancies = append(field.Discrepancies, AccountingDiscrepancy{
			Selected: selected, Other: other, Delta: other.Value - selected.Value,
			Material: material, Detail: detail,
		})
	}
	return field
}

func aggregateAccountingField(metric AccountingMetric, fields []AccountingField) AccountingField {
	unit, _ := metricUnit(metric)
	result := AccountingField{
		Availability:  AvailabilityMissing,
		Unit:          unit,
		Authority:     AuthorityUnknown,
		Coverage:      AccountingCoverage{Scope: "armbench_trials", Expected: len(fields)},
		RefusalReason: "no trial reported this field",
	}
	if len(fields) == 0 {
		return result
	}
	var sum float64
	values := 0
	complete := 0
	conflict := false
	degraded := false
	authority := AuthorityUnknown
	artifactParts := make([]string, 0, len(fields))
	for _, field := range fields {
		result.Sources = append(result.Sources, field.Sources...)
		result.Discrepancies = append(result.Discrepancies, field.Discrepancies...)
		if field.Value == nil {
			degraded = true
			continue
		}
		values++
		sum += *field.Value
		artifactParts = append(artifactParts, field.Artifact.Ref+"@"+field.Artifact.SHA256)
		if field.Availability == AvailabilityAvailable {
			complete++
		} else {
			degraded = true
		}
		if field.Availability == AvailabilityConflict {
			conflict = true
		}
		if authority == AuthorityUnknown {
			authority = field.Authority
		} else if authority != field.Authority {
			authority = AuthorityUnknown
			degraded = true
		}
	}
	result.Coverage.Observed = complete
	if values == 0 {
		return result
	}
	result.Value = &sum
	result.Authority = authority
	sort.Strings(artifactParts)
	result.Artifact = ArtifactFor("armbench://trial-sum/"+string(metric), []byte(strings.Join(artifactParts, "\n")))
	result.Availability = AvailabilityAvailable
	result.RefusalReason = ""
	if degraded || complete != len(fields) || authority == AuthorityUnknown {
		result.Availability = AvailabilityDegraded
		result.RefusalReason = fmt.Sprintf("only %d/%d trial fields have complete, same-authority accounting", complete, len(fields))
	}
	if conflict {
		result.Availability = AvailabilityConflict
		result.RefusalReason = "at least one trial has materially conflicting accounting sources"
	}
	return result
}

func validateAccountingSource(source AccountingSource) error {
	if authorityRank(source.Authority) == 0 {
		return fmt.Errorf("unknown authority %q", source.Authority)
	}
	if strings.TrimSpace(source.Artifact.Ref) == "" {
		return fmt.Errorf("artifact ref is empty")
	}
	if !validAccountingSHA256(source.Artifact.SHA256) {
		return fmt.Errorf("artifact SHA-256 %q is not sha256:<64 lowercase hex>", source.Artifact.SHA256)
	}
	if strings.TrimSpace(source.Coverage.Scope) == "" || source.Coverage.Expected <= 0 || source.Coverage.Observed < 0 || source.Coverage.Observed > source.Coverage.Expected {
		return fmt.Errorf("invalid coverage %+v", source.Coverage)
	}
	seen := false
	for _, metric := range accountingMetrics {
		value := source.Values.field(metric)
		if value == nil {
			continue
		}
		seen = true
		if math.IsNaN(*value) || math.IsInf(*value, 0) || *value < 0 {
			return fmt.Errorf("%s value %v must be finite and non-negative", metric, *value)
		}
		if metric != MetricCostUSD && math.Trunc(*value) != *value {
			return fmt.Errorf("%s value %v must be an integer count", metric, *value)
		}
	}
	if !seen {
		return fmt.Errorf("source reports no accounting fields")
	}
	return nil
}

func (v AccountingValues) field(metric AccountingMetric) *float64 {
	switch metric {
	case MetricInputTokens:
		return v.InputTokens
	case MetricOutputTokens:
		return v.OutputTokens
	case MetricCacheReadTokens:
		return v.CacheReadTokens
	case MetricCacheWriteTokens:
		return v.CacheWriteTokens
	case MetricCacheHits:
		return v.CacheHits
	case MetricCacheMisses:
		return v.CacheMisses
	case MetricCostUSD:
		return v.CostUSD
	default:
		return nil
	}
}

func (v *AccountingValues) setField(metric AccountingMetric, value float64) {
	copy := value
	switch metric {
	case MetricInputTokens:
		v.InputTokens = &copy
	case MetricOutputTokens:
		v.OutputTokens = &copy
	case MetricCacheReadTokens:
		v.CacheReadTokens = &copy
	case MetricCacheWriteTokens:
		v.CacheWriteTokens = &copy
	case MetricCacheHits:
		v.CacheHits = &copy
	case MetricCacheMisses:
		v.CacheMisses = &copy
	case MetricCostUSD:
		v.CostUSD = &copy
	}
}

func missingAccountingReceipt() AccountingReceipt {
	r := AccountingReceipt{Schema: AccountingReceiptSchema}
	for _, metric := range accountingMetrics {
		unit, _ := metricUnit(metric)
		r.setField(metric, AccountingField{
			Availability:  AvailabilityMissing,
			Unit:          unit,
			Authority:     AuthorityUnknown,
			RefusalReason: "no accounting source reported this field",
		})
	}
	return r
}

func normalizedAccountingReceipt(receipt AccountingReceipt) AccountingReceipt {
	if receipt.Schema == "" {
		return missingAccountingReceipt()
	}
	return receipt
}

func metricUnit(metric AccountingMetric) (string, bool) {
	switch metric {
	case MetricInputTokens, MetricOutputTokens, MetricCacheReadTokens, MetricCacheWriteTokens:
		return "tokens", true
	case MetricCacheHits, MetricCacheMisses:
		return "events", true
	case MetricCostUSD:
		return "usd", true
	default:
		return "", false
	}
}

func authorityRank(authority AccountingAuthority) int {
	switch authority {
	case AuthorityStepSum:
		return 1
	case AuthorityHarnessAggregate:
		return 2
	case AuthorityProviderAggregate:
		return 3
	default:
		return 0
	}
}

func materiallyDifferent(metric AccountingMetric, a, b float64) bool {
	delta := math.Abs(a - b)
	if metric != MetricCostUSD {
		return delta >= 1
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return delta > 1e-9 && (scale == 0 || delta/scale > 1e-6)
}

func validAccountingSHA256(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil && value == strings.ToLower(value)
}

func sameAccountingDecision(got, want AccountingField) bool {
	if got.Availability != want.Availability || got.Authority != want.Authority || got.Unit != want.Unit || got.Artifact != want.Artifact || got.Coverage != want.Coverage || got.RefusalReason != want.RefusalReason {
		return false
	}
	if (got.Value == nil) != (want.Value == nil) || (got.Value != nil && *got.Value != *want.Value) {
		return false
	}
	if len(got.Discrepancies) != len(want.Discrepancies) {
		return false
	}
	for i := range got.Discrepancies {
		if got.Discrepancies[i] != want.Discrepancies[i] {
			return false
		}
	}
	return true
}
