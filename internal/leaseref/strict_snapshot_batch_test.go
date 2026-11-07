package leaseref

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime integration est=1s lane=default
func TestStrictSnapshotBatchRealGit(t *testing.T) {
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	payload := `{"holder":"fixture","acquired_unix":1700000000,"ttl_seconds":600,"generation":7}`
	oid := reservedNamespaceGit(t, dir, payload, "hash-object", "-w", "--stdin")
	var updates strings.Builder
	const count = 128
	for i := range count {
		fmt.Fprintf(&updates, "create %slease-%03d %s\n", refPrefix, i, oid)
	}
	reservedNamespaceGit(t, dir, updates.String(), "update-ref", "--stdin")
	before := reservedNamespaceRefs(t, dir)
	var scans, blobs, batches int
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if args[0] == "for-each-ref" {
			scans++
		} else if args[0] == "cat-file" {
			blobs++
		}
		return gitRunner(ctx, dir, args...)
	}
	stdin := func(ctx context.Context, dir, input string, args ...string) (string, int, error) {
		batches++
		if !reflect.DeepEqual(args, []string{"cat-file", "--batch"}) || input != strings.Repeat(oid+"\n", count) {
			t.Fatal("strict batch did not request the enumerated immutable objects")
		}
		return gitStdinRunner(ctx, dir, input, args...)
	}
	start := time.Now()
	want, err := NewWithRunner(run, dir).StrictSnapshot(ctx)
	if err != nil || len(want) != count {
		t.Fatalf("per-ref oracle: records=%d err=%v", len(want), err)
	}
	legacyTime, legacyCalls := time.Since(start), scans+blobs
	scans, blobs = 0, 0
	start = time.Now()
	got, err := NewWithStdinRunner(run, stdin, dir).StrictSnapshot(ctx)
	elapsed := time.Since(start)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("strict batch differs from per-ref oracle: records=%d err=%v", len(got), err)
	}
	if after := reservedNamespaceRefs(t, dir); !reflect.DeepEqual(after, before) {
		t.Fatal("strict snapshot mutated ref identities")
	}
	t.Logf("records=%d legacy_calls=%d legacy_elapsed=%s batch_calls=%d batch_elapsed=%s", count, legacyCalls, legacyTime, scans+blobs+batches, elapsed)
	if scans != 1 || batches != 1 || blobs != 0 {
		t.Fatalf("process counts: scans=%d batches=%d per_ref=%d; want 1, 1, 0", scans, batches, blobs)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestStrictSnapshotBatchRefusesIncompleteEvidence(t *testing.T) {
	oid := strings.Repeat("a", 40)
	ref := refPrefix + "lease-a"
	payload := "{\n\"holder\":\"fixture\",\"generation\":7\n}"
	frame := func(oid, kind, body string) string {
		return fmt.Sprintf("%s %s %d\n%s\n", oid, kind, len(body), body)
	}
	good := frame(oid, "blob", payload)
	for _, tc := range []struct {
		name, enumeration, stream string
		scanCode, batchCode       int
		scanErr, batchErr         error
		wantRef                   string
		wantErr, empty            bool
	}{
		{name: "valid-multiline-id-fill", stream: good},
		{name: "explicit-id", stream: frame(oid, "blob", `{"id":"lease-a"}`)},
		{name: "sha256-object", enumeration: ref + " " + strings.Repeat("b", 64) + "\n", stream: frame(strings.Repeat("b", 64), "blob", payload)},
		{name: "empty-namespace", empty: true},
		{name: "reserved-namespace", enumeration: refPrefix + "epoch/v1/opaque " + oid + "\n", empty: true},
		{name: "missing", stream: oid + " missing\n", wantErr: true, wantRef: ref},
		{name: "wrong-oid", stream: frame(strings.Repeat("b", 40), "blob", payload), wantErr: true, wantRef: ref},
		{name: "wrong-type", stream: frame(oid, "tree", payload), wantErr: true, wantRef: ref},
		{name: "truncated-header", stream: oid, wantErr: true, wantRef: ref},
		{name: "header-fields", stream: oid + " blob\n", wantErr: true, wantRef: ref},
		{name: "negative-length", stream: oid + " blob -1\n", wantErr: true, wantRef: ref},
		{name: "overflow-length", stream: oid + " blob 999999999999999999999\n", wantErr: true, wantRef: ref},
		{name: "truncated-payload", stream: good[:len(good)-2], wantErr: true, wantRef: ref},
		{name: "missing-delimiter", stream: good[:len(good)-1] + "x", wantErr: true, wantRef: ref},
		{name: "trailing-data", stream: good + good, wantErr: true},
		{name: "invalid-json", stream: frame(oid, "blob", "{"), wantErr: true, wantRef: ref},
		{name: "invalid-id", stream: frame(oid, "blob", `{"id":"invalid/id"}`), wantErr: true, wantRef: ref},
		{name: "mismatched-id", stream: frame(oid, "blob", `{"id":"other"}`), wantErr: true, wantRef: ref},
		{name: "no-partial-result", enumeration: ref + " " + oid + "\n" + refPrefix + "lease-b " + oid + "\n", stream: good + oid + " missing\n", wantErr: true, wantRef: refPrefix + "lease-b"},
		{name: "bad-enumeration", enumeration: ref + "\n", wantErr: true, wantRef: ref},
		{name: "invalid-object-id", enumeration: ref + " invalid\n", wantErr: true, wantRef: ref},
		{name: "duplicate-ref", enumeration: strings.Repeat(ref+" "+oid+"\n", 2), wantErr: true, wantRef: ref},
		{name: "foreign-ref", enumeration: "refs/heads/main " + oid + "\n", wantErr: true, wantRef: "refs/heads/main"},
		{name: "enumeration-refused", scanCode: 1, wantErr: true},
		{name: "enumeration-cancelled", scanErr: context.Canceled, wantErr: true},
		{name: "batch-refused", batchCode: 1, stream: good, wantErr: true},
		{name: "batch-cancelled", batchErr: context.Canceled, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scans, batches := 0, 0
			run := func(_ context.Context, _ string, args ...string) (string, int, error) {
				scans++
				if !reflect.DeepEqual(args, []string{"for-each-ref", "--format=%(refname) %(objectname)", refPrefix}) {
					t.Fatalf("unexpected reader or fallback: %v", args)
				}
				enumeration := tc.enumeration
				if enumeration == "" && !tc.empty {
					enumeration = ref + " " + oid + "\n"
				}
				return enumeration, tc.scanCode, tc.scanErr
			}
			stdin := func(_ context.Context, _ string, input string, args ...string) (string, int, error) {
				batches++
				if tc.empty || strings.Contains(input, refPrefix) || !reflect.DeepEqual(args, []string{"cat-file", "--batch"}) {
					t.Fatalf("unexpected batch request: %q %v", input, args)
				}
				return tc.stream, tc.batchCode, tc.batchErr
			}
			got, err := NewWithStdinRunner(run, stdin, "").StrictSnapshot(context.Background())
			if tc.wantErr {
				if err == nil || got != nil {
					t.Fatalf("failed observation returned records=%d err=%v", len(got), err)
				}
				if tc.wantRef != "" {
					var corrupt *CorruptRefError
					if !errors.As(err, &corrupt) || corrupt.Ref != tc.wantRef {
						t.Fatalf("error lost offending ref %q: %v", tc.wantRef, err)
					}
				}
				if tc.scanErr != nil && !errors.Is(err, tc.scanErr) || tc.batchErr != nil && !errors.Is(err, tc.batchErr) {
					t.Fatalf("runner error identity lost: %v", err)
				}
			} else if err != nil || got == nil || tc.empty && len(got) != 0 || !tc.empty && (len(got) != 1 || got[0].ID != "lease-a") {
				t.Fatalf("valid observation: records=%v err=%v", got, err)
			}
			if scans != 1 || batches > 1 {
				t.Fatalf("unexpected retry/fallback: scans=%d batches=%d", scans, batches)
			}
		})
	}
}
