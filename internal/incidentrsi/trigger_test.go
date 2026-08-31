package incidentrsi

import (
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"
)

func validTrigger() Trigger {
	first := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	observed := first.Add(10 * time.Second)
	cooldownUntil := observed.Add(5 * time.Minute)
	return Trigger{
		Schema: Schema,
		Source: Source{
			Component:       "guard-hook",
			Operation:       "post-tool",
			IncidentClass:   IncidentHookFailure,
			ProducerSchema:  "fak-hook-fault-1",
			ProducerVersion: "r42-gabcdef0",
		},
		Fingerprint: strings.Repeat("a", 64),
		Privacy: Privacy{
			Classification: PrivacyContentFree,
			Redaction:      RedactionSHA256,
		},
		Observation: Observation{
			ID:              "obs:01J6NQ",
			ObservedAt:      observed,
			OccurrenceCount: 3,
			BurstCount:      3,
			FirstSeen:       first,
			LastSeen:        observed,
		},
		DebounceInputs: DebounceInputs{
			Threshold:               3,
			CollectionWindowSeconds: 30,
			MaxWaitSeconds:          120,
			CooldownSeconds:         300,
		},
		DebounceResult: DebounceResult{
			State:          StateThresholdReady,
			WindowStarted:  first,
			WindowEnds:     first.Add(30 * time.Second),
			MaxWaitEnds:    first.Add(120 * time.Second),
			CooldownUntil:  &cooldownUntil,
			NextEligibleAt: cooldownUntil,
		},
		Decision: DecisionAdmit,
		Reason:   ReasonThreshold,
		Refs: DurableRefs{
			LedgerEvent:   "ledger:event-17",
			IssueAction:   IssueCreate,
			Issue:         "github:10495",
			Route:         "route:incident-rsi",
			LaunchReceipt: "launch:01J6NR",
		},
		OriginalProductFailurePreserved: true,
	}
}

func TestDeterministicJSONRoundTrip(t *testing.T) {
	trigger := validTrigger()
	first, err := trigger.Encode()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeStrict(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := decoded.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("round trip changed bytes:\n%s\n%s", first, second)
	}
}

func TestRequiredFieldsAndSchemaMajor(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Trigger)
	}{
		{"empty schema", func(v *Trigger) { v.Schema = "" }},
		{"v0", func(v *Trigger) { v.Schema = "fak-incident-rsi-trigger/0" }},
		{"v2", func(v *Trigger) { v.Schema = "fak-incident-rsi-trigger/2" }},
		{"unknown major", func(v *Trigger) { v.Schema = "other/1" }},
		{"component", func(v *Trigger) { v.Source.Component = "" }},
		{"observation id", func(v *Trigger) { v.Observation.ID = "" }},
		{"preservation", func(v *Trigger) { v.OriginalProductFailurePreserved = false }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := validTrigger()
			tc.edit(&v)
			if err := v.Validate(); err == nil {
				t.Fatal("Validate succeeded")
			}
		})
	}
}

func TestClosedEnumsRejectUnknownValues(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Trigger)
	}{
		{"incident class", func(v *Trigger) { v.Source.IncidentClass = "OTHER" }},
		{"privacy", func(v *Trigger) { v.Privacy.Classification = "REDACTED" }},
		{"redaction", func(v *Trigger) { v.Privacy.Redaction = "HASH" }},
		{"state", func(v *Trigger) { v.DebounceResult.State = "READY" }},
		{"decision", func(v *Trigger) { v.Decision = "RUN" }},
		{"reason", func(v *Trigger) { v.Reason = "READY" }},
		{"issue action", func(v *Trigger) { v.Refs.IssueAction = "UPSERT" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := validTrigger()
			tc.edit(&v)
			if err := v.Validate(); err == nil {
				t.Fatal("Validate succeeded")
			}
		})
	}
}

func TestFingerprintAndPrivacyBounds(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Trigger)
	}{
		{"short fingerprint", func(v *Trigger) { v.Fingerprint = "abc" }},
		{"uppercase fingerprint", func(v *Trigger) { v.Fingerprint = strings.Repeat("A", 64) }},
		{"raw operation", func(v *Trigger) { v.Source.Operation = `open C:\\Users\\alice\\secret.txt` }},
		{"raw producer", func(v *Trigger) { v.Source.ProducerVersion = "token=secret" }},
		{"oversize component", func(v *Trigger) { v.Source.Component = strings.Repeat("a", 65) }},
		{"raw observation", func(v *Trigger) { v.Observation.ID = "error: payment card declined" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := validTrigger()
			tc.edit(&v)
			if err := v.Validate(); err == nil {
				t.Fatal("Validate succeeded")
			}
		})
	}
}

func TestTemporalAndCountInvariants(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Trigger)
	}{
		{"first after last", func(v *Trigger) { v.Observation.FirstSeen = v.Observation.LastSeen.Add(time.Second) }},
		{"last differs from observed", func(v *Trigger) { v.Observation.LastSeen = v.Observation.ObservedAt.Add(-time.Second) }},
		{"zero occurrence", func(v *Trigger) { v.Observation.OccurrenceCount = 0 }},
		{"burst exceeds occurrence", func(v *Trigger) { v.Observation.BurstCount = v.Observation.OccurrenceCount + 1 }},
		{"window mismatch", func(v *Trigger) { v.DebounceResult.WindowEnds = v.DebounceResult.WindowEnds.Add(time.Second) }},
		{"max wait shorter than window", func(v *Trigger) { v.DebounceInputs.MaxWaitSeconds = 10 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := validTrigger()
			tc.edit(&v)
			if err := v.Validate(); err == nil {
				t.Fatal("Validate succeeded")
			}
		})
	}
}

func TestDebounceThresholdMaxWaitAndCooldownRemainDistinct(t *testing.T) {
	t.Run("collect", func(t *testing.T) {
		v := collectingTrigger()
		if err := v.Validate(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("max wait admits below threshold", func(t *testing.T) {
		v := validTrigger()
		v.DebounceInputs.Threshold = 10
		v.Observation.ObservedAt = v.DebounceResult.MaxWaitEnds
		v.Observation.LastSeen = v.Observation.ObservedAt
		until := v.Observation.ObservedAt.Add(time.Duration(v.DebounceInputs.CooldownSeconds) * time.Second)
		v.DebounceResult.State = StateMaxWaitReady
		v.DebounceResult.CooldownUntil = &until
		v.DebounceResult.NextEligibleAt = until
		v.Reason = ReasonMaxWait
		if err := v.Validate(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("cooldown wins over threshold", func(t *testing.T) {
		v := validTrigger()
		until := v.Observation.ObservedAt.Add(time.Minute)
		v.DebounceResult.State = StateCooldownSuppressed
		v.DebounceResult.CooldownUntil = &until
		v.DebounceResult.NextEligibleAt = until
		v.Decision = DecisionSuppress
		v.Reason = ReasonCooldown
		v.Refs = DurableRefs{IssueAction: IssueNone}
		if err := v.Validate(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("cooldown cannot masquerade as collection window", func(t *testing.T) {
		v := collectingTrigger()
		until := v.Observation.ObservedAt.Add(time.Minute)
		v.DebounceResult.CooldownUntil = &until
		if err := v.Validate(); err == nil {
			t.Fatal("Validate succeeded")
		}
	})
}

func collectingTrigger() Trigger {
	v := validTrigger()
	v.Observation.OccurrenceCount = 1
	v.Observation.BurstCount = 1
	v.Observation.ObservedAt = v.Observation.FirstSeen.Add(5 * time.Second)
	v.Observation.LastSeen = v.Observation.ObservedAt
	v.DebounceResult.State = StateCollecting
	v.DebounceResult.CooldownUntil = nil
	v.DebounceResult.NextEligibleAt = v.DebounceResult.WindowEnds
	v.Decision = DecisionCollect
	v.Reason = ReasonBelowThreshold
	v.Refs = DurableRefs{IssueAction: IssueNone}
	return v
}

func TestConditionalDurableReferences(t *testing.T) {
	t.Run("admit requires every reference", func(t *testing.T) {
		fields := []func(*Trigger){
			func(v *Trigger) { v.Refs.LedgerEvent = "" },
			func(v *Trigger) { v.Refs.IssueAction = IssueNone },
			func(v *Trigger) { v.Refs.Issue = "" },
			func(v *Trigger) { v.Refs.Route = "" },
			func(v *Trigger) { v.Refs.LaunchReceipt = "" },
		}
		for i, edit := range fields {
			v := validTrigger()
			edit(&v)
			if err := v.Validate(); err == nil {
				t.Fatalf("case %d succeeded", i)
			}
		}
	})

	t.Run("non-admit cannot claim references", func(t *testing.T) {
		v := collectingTrigger()
		v.Refs.LedgerEvent = "ledger:event-17"
		if err := v.Validate(); err == nil {
			t.Fatal("Validate succeeded")
		}
	})
}

func TestBoundedStructuredSideEffectFailures(t *testing.T) {
	v := validTrigger()
	v.SideEffectFailures = []SideEffectFailure{{Stage: StageLaunch, Code: "TIMEOUT", Retryable: true}}
	if err := v.Validate(); err != nil {
		t.Fatal(err)
	}

	v = validTrigger()
	v.SideEffectFailures = make([]SideEffectFailure, maxSideEffectFailures+1)
	for i := range v.SideEffectFailures {
		v.SideEffectFailures[i] = SideEffectFailure{Stage: StageIssue, Code: "FAILED"}
	}
	if err := v.Validate(); err == nil {
		t.Fatal("oversize failure list accepted")
	}

	v = validTrigger()
	v.SideEffectFailures = []SideEffectFailure{{Stage: StageLaunch, Code: "raw failure: secret"}}
	if err := v.Validate(); err == nil {
		t.Fatal("raw failure content accepted")
	}
}

func TestDecodeStrictRejectsUnknownFields(t *testing.T) {
	encoded, err := validTrigger().Encode()
	if err != nil {
		t.Fatal(err)
	}
	withUnknown := bytes.Replace(encoded, []byte(`"schema":`), []byte(`"future_field":true,"schema":`), 1)
	if _, err := DecodeStrict(withUnknown); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestIdentityIsDeterministicAndContentSensitive(t *testing.T) {
	v := validTrigger()
	first, err := v.Identity()
	if err != nil {
		t.Fatal(err)
	}
	second, err := v.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if first != second || len(first) != 64 {
		t.Fatalf("identity not deterministic: %q %q", first, second)
	}
	v.Observation.ID = "obs:01J6NS"
	changed, err := v.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if changed == first {
		t.Fatal("identity did not change with contract content")
	}
}

func TestConcurrentReadOnlyValidation(t *testing.T) {
	v := validTrigger()
	const workers = 64
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := v.Validate(); err != nil {
				errs <- err
				return
			}
			_, err := v.Identity()
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}
