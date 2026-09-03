package orchestration

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestValidateEffectCohortVerifiedTwoChildren(t *testing.T) {
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	admitted := testAdmittedEffects()
	receipts := []EffectReceipt{
		testEffectReceipt(admitted[1], now.Add(-time.Minute)),
		testEffectReceipt(admitted[0], now.Add(-2*time.Minute)),
	}

	got := ValidateEffectCohort("run-8841", admitted, receipts, now, 5*time.Minute)
	if got.State != EffectVerified {
		t.Fatalf("state = %q, want %q: %+v", got.State, EffectVerified, got)
	}
	if len(got.Children) != 2 || got.Children[0].ChildID != "child-a" || got.Children[1].ChildID != "child-b" {
		t.Fatalf("children not deterministically sorted: %+v", got.Children)
	}
	for _, child := range got.Children {
		if child.State != EffectVerified {
			t.Errorf("child %q state = %q, want VERIFIED", child.ChildID, child.State)
		}
	}
}

func TestValidateEffectCohortAdversarial(t *testing.T) {
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	baseAdmitted := testAdmittedEffects()
	baseReceipts := []EffectReceipt{
		testEffectReceipt(baseAdmitted[0], now.Add(-time.Minute)),
		testEffectReceipt(baseAdmitted[1], now.Add(-time.Minute)),
	}

	tests := []struct {
		name      string
		mutate    func(*[]AdmittedEffect, *[]EffectReceipt)
		want      EffectVerificationState
		wantChild EffectVerificationState
	}{
		{name: "self authored witness", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].Witness.AuthorChildID = "child-a" }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "authority is child", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].Witness.AuthorityID = "child-a" }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "missing child", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { *r = (*r)[:1] }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "duplicate receipt", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { *r = append(*r, (*r)[0]) }, want: EffectFailed, wantChild: EffectFailed},
		{name: "unlaunched child", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].ChildID = "child-x" }, want: EffectFailed, wantChild: EffectFailed},
		{name: "not admitted", mutate: func(a *[]AdmittedEffect, _ *[]EffectReceipt) { (*a)[0].Admitted = false }, want: EffectFailed, wantChild: EffectFailed},
		{name: "run mismatch", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].RunID = "other-run" }, want: EffectFailed, wantChild: EffectFailed},
		{name: "successor mismatch", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].SuccessorID = "other-successor" }, want: EffectFailed, wantChild: EffectFailed},
		{name: "effect mismatch", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].EffectClass = "other-effect" }, want: EffectFailed, wantChild: EffectFailed},
		{name: "expected digest mismatch", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].ExpectedScrubbedDigest = digestB }, want: EffectFailed, wantChild: EffectFailed},
		{name: "observed digest mismatch", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].ObservedScrubbedDigest = digestB }, want: EffectFailed, wantChild: EffectFailed},
		{name: "stale", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].ObservedAt = now.Add(-6 * time.Minute) }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "future", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].ObservedAt = now.Add(time.Nanosecond) }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "unsupported schema", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].Schema = "v2" }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "unsupported state", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].State = "SUCCESS" }, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "unknown reconciliation", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) {
			(*r)[0].Reconciliation = EffectUnobserved
			(*r)[0].State = EffectUnknown
		}, want: EffectUnknown, wantChild: EffectUnknown},
		{name: "explicit failure", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) {
			(*r)[0].Reconciliation = EffectDiverged
			(*r)[0].State = EffectFailed
		}, want: EffectFailed, wantChild: EffectFailed},
		{name: "raw private field", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) {
			(*r)[0].UnsupportedFields = []string{"raw_log"}
		}, want: EffectFailed, wantChild: EffectFailed},
		{name: "activation is not readback", mutate: func(_ *[]AdmittedEffect, r *[]EffectReceipt) { (*r)[0].Witness = EffectWitnessAuthority{} }, want: EffectUnknown, wantChild: EffectUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admitted := append([]AdmittedEffect(nil), baseAdmitted...)
			receipts := append([]EffectReceipt(nil), baseReceipts...)
			tt.mutate(&admitted, &receipts)
			got := ValidateEffectCohort("run-8841", admitted, receipts, now, 5*time.Minute)
			if got.State != tt.want {
				t.Fatalf("state = %q, want %q: %+v", got.State, tt.want, got)
			}
			found := false
			for _, child := range got.Children {
				if child.State == tt.wantChild {
					found = true
				}
			}
			if !found {
				t.Fatalf("no child state %q in %+v", tt.wantChild, got.Children)
			}
		})
	}
}

func TestValidateEffectCohortDeterministicAndPure(t *testing.T) {
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	admitted := testAdmittedEffects()
	receipts := []EffectReceipt{testEffectReceipt(admitted[1], now), testEffectReceipt(admitted[0], now)}
	admittedBefore := append([]AdmittedEffect(nil), admitted...)
	receiptsBefore := append([]EffectReceipt(nil), receipts...)

	first := ValidateEffectCohort("run-8841", admitted, receipts, now, time.Minute)
	second := ValidateEffectCohort("run-8841", admitted, receipts, now, time.Minute)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same values produced different results:\n%+v\n%+v", first, second)
	}
	if !reflect.DeepEqual(admitted, admittedBefore) || !reflect.DeepEqual(receipts, receiptsBefore) {
		t.Fatal("validator mutated its inputs")
	}
}

func TestDecodeEffectReceiptJSONRejectsRawPrivateAndTrailingData(t *testing.T) {
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	receipt := testEffectReceipt(testAdmittedEffects()[0], now)
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEffectReceiptJSON(data)
	if err != nil || !reflect.DeepEqual(decoded, receipt) {
		t.Fatalf("decode valid receipt = (%+v, %v)", decoded, err)
	}

	for _, data := range [][]byte{
		append(data[:len(data)-1], []byte(`,"raw_log":"private token"}`)...),
		append(append([]byte(nil), data...), []byte(` {}`)...),
	} {
		if _, err := DecodeEffectReceiptJSON(data); err == nil || err.Error() != "invalid effect receipt" {
			t.Fatalf("unsafe input error = %v, want privacy-safe generic rejection", err)
		}
	}
}

func TestEffectReceiptPrivacySurface(t *testing.T) {
	typ := reflect.TypeOf(EffectReceipt{})
	for i := 0; i < typ.NumField(); i++ {
		name := typ.Field(i).Name
		for _, forbidden := range []string{"Path", "Log", "Payload", "Command", "Callback", "Content", "Private"} {
			if containsFold(name, forbidden) {
				t.Fatalf("receipt exposes raw/private surface %q", name)
			}
		}
	}
	if canonicalSHA256("ABCDEF0000000000000000000000000000000000000000000000000000000000") || canonicalSHA256("not-a-digest") {
		t.Fatal("scrubbed digest validation accepted a non-canonical digest")
	}
}

func containsFold(value, part string) bool {
	for i := 0; i+len(part) <= len(value); i++ {
		if equalFoldASCII(value[i:i+len(part)], part) {
			return true
		}
	}
	return false
}

func equalFoldASCII(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

const (
	digestA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	digestB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

func testAdmittedEffects() []AdmittedEffect {
	return []AdmittedEffect{
		{RunID: "run-8841", ChildID: "child-a", SuccessorID: "successor-a", EffectClass: "landed-commit", ExpectedScrubbedDigest: digestA, Admitted: true, Activated: true, ChildReportedSuccess: true},
		{RunID: "run-8841", ChildID: "child-b", SuccessorID: "successor-b", EffectClass: "captured-behavior", ExpectedScrubbedDigest: digestB, Admitted: true, Activated: true, ChildReportedSuccess: true},
	}
}

func testEffectReceipt(admitted AdmittedEffect, observedAt time.Time) EffectReceipt {
	return EffectReceipt{
		Schema: EffectReceiptSchema, RunID: admitted.RunID, ChildID: admitted.ChildID,
		SuccessorID: admitted.SuccessorID, EffectClass: admitted.EffectClass,
		ExpectedScrubbedDigest: admitted.ExpectedScrubbedDigest, ObservedScrubbedDigest: admitted.ExpectedScrubbedDigest,
		ReadbackMethod: "dos-verify", Witness: EffectWitnessAuthority{AuthorityID: "independent-judge"},
		Reconciliation: EffectReconciled, State: EffectVerified, ObservedAt: observedAt,
	}
}
