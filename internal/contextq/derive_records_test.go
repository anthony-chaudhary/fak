package contextq

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestDerivedRecordViewSpine(t *testing.T) {
	t.Run("group count lineage and accounting", testDerivedRecordViewGroupCount)
	t.Run("projection canonicalization", testDeriveRecordsProjectionCanonicalizesFields)
	t.Run("fail closed matrix", testDeriveRecordsRefusesInvalidInputs)
}

func testDerivedRecordViewGroupCount(t *testing.T) {
	body := []byte(strings.Join([]string{
		`{"id":"a-1","status":"failed","owner":"alice","detail":"distant evidence one"}`,
		`{"id":"b-1","status":"ok","owner":"bob","detail":"not selected"}`,
		`{"id":"b-2","status":"failed","owner":"bob","detail":"distant evidence two"}`,
		`{"id":"c-1","status":"ok","owner":"carol","detail":"not selected"}`,
		`{"id":"a-2","status":"failed","owner":"alice","detail":"distant evidence three"}`,
		`{"id":"d-1","status":"ok","owner":"dora","detail":"not selected"}`,
	}, "\n") + "\n")
	resolver, source := testSource(body)
	source.Taint = abi.TaintTrusted
	source.Scope = abi.ScopeTenant
	plan := RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationGroupCount, Filter: &RecordEqualFilter{Field: "status", Value: "failed"}, GroupBy: "owner"}

	first, err := DeriveRecords(context.Background(), resolver, source, plan, generousLimits())
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveRecords(context.Background(), resolver, source, plan, generousLimits())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("replay changed view:\nfirst  %#v\nsecond %#v", first, second)
	}
	got, err := resolver.Resolve(context.Background(), first.Output)
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"group":"alice","count":2},{"group":"bob","count":1}]`
	if string(got) != want {
		t.Fatalf("derived bytes = %s, want %s", got, want)
	}
	if first.Schema != DerivedRecordViewSchema || first.SourceDigest != source.Digest || first.DerivationDigest == "" || first.OutputDigest != digest(got) {
		t.Fatalf("lineage incomplete: %#v", first)
	}
	if first.Taint != source.Taint || first.Scope != source.Scope || first.Output.Taint != source.Taint || first.Output.Scope != source.Scope {
		t.Fatalf("labels not inherited: view=%#v output=%#v", first, first.Output)
	}
	if first.Truncated {
		t.Fatal("successful exact result must not be marked truncated")
	}
	wantAccounting := RecordAccounting{
		SourceBytes: int64(len(body)), OutputBytes: int64(len(got)),
		RecordsRead: 6, RecordsMatch: 3, RecordsOutput: 2, WorkUnits: 15,
	}
	if first.Accounting != wantAccounting {
		t.Fatalf("accounting = %#v, want %#v", first.Accounting, wantAccounting)
	}
	if first.Accounting.OutputBytes >= first.Accounting.SourceBytes {
		t.Fatalf("derived view did not reduce visible bytes: %#v", first.Accounting)
	}
}

func testDeriveRecordsProjectionCanonicalizesFields(t *testing.T) {
	body := []byte("{\"id\":\"a\",\"status\":\"failed\",\"owner\":\"alice\"}\n{\"id\":\"b\",\"status\":\"ok\",\"owner\":\"bob\"}\n")
	resolver, source := testSource(body)
	filter := &RecordEqualFilter{Field: "status", Value: "failed"}
	a, err := DeriveRecords(context.Background(), resolver, source, RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationProject, Filter: filter, Project: []string{"owner", "id"}}, generousLimits())
	if err != nil {
		t.Fatal(err)
	}
	b, err := DeriveRecords(context.Background(), resolver, source, RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationProject, Filter: filter, Project: []string{"id", "owner"}}, generousLimits())
	if err != nil {
		t.Fatal(err)
	}
	if a.DerivationDigest != b.DerivationDigest || a.OutputDigest != b.OutputDigest {
		t.Fatalf("equivalent projections diverged: a=%#v b=%#v", a, b)
	}
	got, _ := resolver.Resolve(context.Background(), a.Output)
	if string(got) != `[{"id":"a","owner":"alice"}]` {
		t.Fatalf("projection = %s", got)
	}
}

func testDeriveRecordsRefusesInvalidInputs(t *testing.T) {
	validBody := []byte("{\"status\":\"failed\",\"owner\":\"alice\",\"detail\":\"long enough output input\"}\n")
	validPlan := RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationGroupCount, GroupBy: "owner"}

	tests := []struct {
		name   string
		reason DeriveReason
		run    func() error
	}{
		{"nil context", DeriveReasonCanceled, func() error {
			r, s := testSource(validBody)
			_, err := DeriveRecords(nil, r, s, validPlan, generousLimits())
			return err
		}},
		{"canceled", DeriveReasonCanceled, func() error {
			r, s := testSource(validBody)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err := DeriveRecords(ctx, r, s, validPlan, generousLimits())
			return err
		}},
		{"resolver missing", DeriveReasonResolverMissing, func() error {
			_, s := testSource(validBody)
			_, err := DeriveRecords(context.Background(), nil, s, validPlan, generousLimits())
			return err
		}},
		{"limits invalid", DeriveReasonLimitsInvalid, func() error {
			r, s := testSource(validBody)
			_, err := DeriveRecords(context.Background(), r, s, validPlan, RecordLimits{})
			return err
		}},
		{"taint invalid", DeriveReasonTaintInvalid, func() error {
			r, s := testSource(validBody)
			s.Taint = abi.TaintLabel(255)
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			if r.resolveCalls != 0 {
				t.Fatalf("source with invalid taint was resolved")
			}
			return err
		}},
		{"scope invalid", DeriveReasonScopeInvalid, func() error {
			r, s := testSource(validBody)
			s.Scope = abi.ShareScope(255)
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			if r.resolveCalls != 0 {
				t.Fatalf("source with invalid scope was resolved")
			}
			return err
		}},
		{"quarantine", DeriveReasonSourceQuarantined, func() error {
			r, s := testSource(validBody)
			s.Taint = abi.TaintQuarantined
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			if r.resolveCalls != 0 {
				t.Fatalf("quarantined source was resolved")
			}
			return err
		}},
		{"digest missing", DeriveReasonDigestInvalid, func() error {
			r, s := testSource(validBody)
			s.Digest = ""
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"digest prefixed", DeriveReasonDigestInvalid, func() error {
			r, s := testSource(validBody)
			s.Digest = "sha256:" + s.Digest
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"digest uppercase", DeriveReasonDigestInvalid, func() error {
			r, s := testSource(validBody)
			s.Digest = strings.ToUpper(s.Digest)
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"digest mismatch", DeriveReasonDigestMismatch, func() error {
			r, s := testSource(validBody)
			s.Digest = strings.Repeat("0", 64)
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"length negative", DeriveReasonLengthMismatch, func() error {
			r, s := testSource(validBody)
			s.Len = -1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"length mismatch", DeriveReasonLengthMismatch, func() error {
			r, s := testSource(validBody)
			s.Len--
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"declared source limit", DeriveReasonSourceLimit, func() error {
			r, s := testSource(validBody)
			limits := generousLimits()
			limits.MaxSourceBytes = s.Len - 1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, limits)
			return err
		}},
		{"resolved source limit", DeriveReasonSourceLimit, func() error {
			r, s := testSource(validBody)
			s.Len = 1
			limits := generousLimits()
			limits.MaxSourceBytes = 1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, limits)
			return err
		}},
		{"record limit", DeriveReasonRecordLimit, func() error {
			body := []byte("{\"owner\":\"a\"}\n{\"owner\":\"b\"}\n")
			r, s := testSource(body)
			limits := generousLimits()
			limits.MaxRecords = 1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, limits)
			return err
		}},
		{"work limit", DeriveReasonWorkLimit, func() error {
			r, s := testSource(validBody)
			limits := generousLimits()
			limits.MaxWorkUnits = 1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, limits)
			return err
		}},
		{"output limit", DeriveReasonOutputLimit, func() error {
			r, s := testSource(validBody)
			limits := generousLimits()
			limits.MaxOutputBytes = 1
			_, err := DeriveRecords(context.Background(), r, s, validPlan, limits)
			return err
		}},
		{"malformed json", DeriveReasonMalformedSource, func() error {
			r, s := testSource([]byte("{bad}\n"))
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"blank interior record", DeriveReasonMalformedSource, func() error {
			r, s := testSource([]byte("{\"owner\":\"a\"}\n\n{\"owner\":\"b\"}\n"))
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"non object", DeriveReasonMalformedSource, func() error {
			r, s := testSource([]byte("[1,2]\n"))
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"duplicate field", DeriveReasonMalformedSource, func() error {
			r, s := testSource([]byte("{\"owner\":\"a\",\"owner\":\"b\"}\n"))
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"unsupported schema", DeriveReasonUnsupportedPlan, func() error {
			r, s := testSource(validBody)
			plan := validPlan
			plan.Schema = "v2"
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"unsupported operation", DeriveReasonUnsupportedPlan, func() error {
			r, s := testSource(validBody)
			plan := validPlan
			plan.Operation = "eval"
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"project without fields", DeriveReasonUnsupportedPlan, func() error {
			r, s := testSource(validBody)
			plan := RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationProject}
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"duplicate projection", DeriveReasonUnsupportedPlan, func() error {
			r, s := testSource(validBody)
			plan := RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationProject, Project: []string{"owner", "owner"}}
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"missing projected field", DeriveReasonRecordInvalid, func() error {
			r, s := testSource(validBody)
			plan := RecordPlan{Schema: RecordPlanSchema, Operation: RecordOperationProject, Project: []string{"missing"}}
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"missing group field", DeriveReasonRecordInvalid, func() error {
			r, s := testSource(validBody)
			plan := validPlan
			plan.GroupBy = "missing"
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"non-string group", DeriveReasonRecordInvalid, func() error {
			r, s := testSource([]byte("{\"owner\":1}\n"))
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"non-string filter", DeriveReasonRecordInvalid, func() error {
			r, s := testSource([]byte("{\"owner\":\"a\",\"status\":1}\n"))
			plan := validPlan
			plan.Filter = &RecordEqualFilter{Field: "status", Value: "failed"}
			_, err := DeriveRecords(context.Background(), r, s, plan, generousLimits())
			return err
		}},
		{"resolve failure", DeriveReasonResolveFailed, func() error {
			r, s := testSource(validBody)
			r.resolveErr = errors.New("offline")
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"put failure", DeriveReasonPutFailed, func() error {
			r, s := testSource(validBody)
			r.putErr = errors.New("full")
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
		{"bad output ref", DeriveReasonOutputDigestMismatch, func() error {
			r, s := testSource(validBody)
			bad := abi.Ref{Digest: strings.Repeat("0", 64), Len: 1}
			r.putRef = &bad
			_, err := DeriveRecords(context.Background(), r, s, validPlan, generousLimits())
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.run()
			var refusal *DeriveRefusal
			if !errors.As(err, &refusal) {
				t.Fatalf("error = %v, want *DeriveRefusal", err)
			}
			if refusal.Reason != tt.reason {
				t.Fatalf("reason = %q, want %q (error %v)", refusal.Reason, tt.reason, err)
			}
		})
	}
}

func generousLimits() RecordLimits {
	return RecordLimits{MaxSourceBytes: 1 << 20, MaxOutputBytes: 1 << 20, MaxRecords: 1000, MaxWorkUnits: 10000}
}

func testSource(body []byte) (*testResolver, abi.Ref) {
	r := &testResolver{body: append([]byte(nil), body...)}
	return r, abi.Ref{Kind: abi.RefInline, Digest: digest(body), Inline: append([]byte(nil), body...), Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
}

type testResolver struct {
	body         []byte
	resolveErr   error
	putErr       error
	putRef       *abi.Ref
	resolveCalls int
}

func (r *testResolver) Resolve(ctx context.Context, ref abi.Ref) ([]byte, error) {
	r.resolveCalls++
	if r.resolveErr != nil {
		return nil, r.resolveErr
	}
	if ref.Digest == digest(r.body) || ref.Kind == abi.RefInline {
		if len(ref.Inline) > 0 {
			return append([]byte(nil), ref.Inline...), nil
		}
		return append([]byte(nil), r.body...), nil
	}
	return nil, errors.New("unknown ref")
}

func (r *testResolver) Put(_ context.Context, body []byte) (abi.Ref, error) {
	if r.putErr != nil {
		return abi.Ref{}, r.putErr
	}
	if r.putRef != nil {
		return *r.putRef, nil
	}
	return abi.Ref{Kind: abi.RefInline, Digest: digest(body), Inline: bytes.Clone(body), Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}, nil
}
