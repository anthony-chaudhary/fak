package relay

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// D3 (issue #1879) done condition: a test asserts the verified-progress read carries no
// claimed field and matches the ledger. These are that witness (run: `go test
// ./internal/relay -run VerifiedProgress`).

// ledgerBackedReader is a hermetic LedgerReader: it projects a fixed JSONL run/intent ledger
// through the SAME ParseLedgerProgress the production reader uses, so the test drives the real
// read path rather than a hand-built step slice. An optional err models an unreachable store.
type ledgerBackedReader struct {
	byRef map[string]string // ledger_ref -> JSONL ledger content
	err   error
}

func (r ledgerBackedReader) ReadProgress(ledgerRef string) ([]ProgressStep, error) {
	if r.err != nil {
		return nil, r.err
	}
	content, ok := r.byRef[ledgerRef]
	if !ok {
		return nil, errors.New("no ledger at " + ledgerRef)
	}
	return ParseLedgerProgress(content), nil
}

// TestVerifiedProgressMatchesLedger pins the "matches the ledger" half of the done condition:
// reading through a cursor's ledger_ref returns EXACTLY the ledger's rows, in ledger order,
// verbatim. It reads through ParseLedgerProgress so the assertion is against the real
// projection of the JSONL, and confirms the steps come only from the ledger content — a
// self-report field has no path in.
func TestVerifiedProgressMatchesLedger(t *testing.T) {
	const ref = ".dos/runs/relay-demo.jsonl"
	// A run/intent ledger with two real progress rows plus noise the reader must drop: a blank
	// line, a malformed line, and a row with no ref (nothing re-readable).
	ledger := strings.Join([]string{
		`{"ref":"0123456789abcdef0123456789abcdef01234567","note":"committed schema"}`,
		``,
		`{"ref":"#1877","note":"D1 reload verifier landed"}`,
		`{malformed`,
		`{"note":"no ref — dropped"}`,
	}, "\n")

	reader := ledgerBackedReader{byRef: map[string]string{ref: ledger}}
	got := ReadVerifiedProgress(ProgressCursor{StartSHA: "HEAD", LedgerRef: ref}, reader)

	if got.Verdict != ProgressVerified {
		t.Fatalf("verdict = %q, want verified (reason=%s)", got.Verdict, got.Reason)
	}
	if got.LedgerRef != ref {
		t.Errorf("ledger_ref = %q, want %q", got.LedgerRef, ref)
	}
	want := []ProgressStep{
		{Ref: "0123456789abcdef0123456789abcdef01234567", Note: "committed schema"},
		{Ref: "#1877", Note: "D1 reload verifier landed"},
	}
	if !reflect.DeepEqual(got.Steps, want) {
		t.Errorf("steps did not match the ledger:\n got  = %+v\n want = %+v", got.Steps, want)
	}
	// The read is a projection of the ledger, so the same ledger parsed directly must equal
	// what the cursor read — no step is added or lost between the ledger and the read.
	if direct := ParseLedgerProgress(ledger); !reflect.DeepEqual(got.Steps, direct) {
		t.Errorf("cursor read diverged from a direct ledger parse:\n cursor=%+v\n direct=%+v", got.Steps, direct)
	}
}

// TestVerifiedProgressFailClosed pins the fail-closed edges that mirror dos_status: a cursor
// with no ledger_ref, and a ledger the reader cannot reach, each return unknown with NO steps
// — never a trusted "zero progress". A false "verified, 0 steps" would let a successor believe
// an unreadable ledger proved no work was done.
func TestVerifiedProgressFailClosed(t *testing.T) {
	reader := ledgerBackedReader{byRef: map[string]string{}}

	if got := ReadVerifiedProgress(ProgressCursor{StartSHA: "HEAD"}, reader); got.Verdict != ProgressUnknown || len(got.Steps) != 0 {
		t.Errorf("empty ledger_ref: verdict=%q steps=%v, want unknown + no steps (reason=%s)", got.Verdict, got.Steps, got.Reason)
	}
	unreachable := ledgerBackedReader{err: errors.New("ledger store unreachable")}
	if got := ReadVerifiedProgress(ProgressCursor{LedgerRef: ".dos/runs/x.jsonl"}, unreachable); got.Verdict != ProgressUnknown || len(got.Steps) != 0 {
		t.Errorf("unreachable ledger must fail closed: verdict=%q steps=%v (reason=%s)", got.Verdict, got.Steps, got.Reason)
	}
}

// TestVerifiedProgressHasNoClaimedField is the structural invariant that gives this rung its
// name: the verified-progress read carries NO `claimed` field. Walking VerifiedProgress
// reflectively (the same walk baton.go's TestBatonHasNoClaimedField uses) means any future
// edit that adds a claimed field — a Go name or a json tag, at any depth — fails here rather
// than silently reopening the self-report door the ledger read exists to keep shut.
func TestVerifiedProgressHasNoClaimedField(t *testing.T) {
	seen := map[reflect.Type]bool{}
	var walk func(rt reflect.Type, path string)
	walk = func(rt reflect.Type, path string) {
		for rt.Kind() == reflect.Ptr || rt.Kind() == reflect.Slice || rt.Kind() == reflect.Array {
			rt = rt.Elem()
		}
		if rt.Kind() != reflect.Struct || seen[rt] {
			return
		}
		seen[rt] = true
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			jsonTag := strings.Split(f.Tag.Get("json"), ",")[0]
			if strings.EqualFold(f.Name, "claimed") || strings.EqualFold(jsonTag, "claimed") {
				t.Errorf("forbidden `claimed` field at %s.%s (json:%q) — progress must be read from the ledger, never self-reported", path, f.Name, jsonTag)
			}
			walk(f.Type, path+"."+f.Name)
		}
	}
	walk(reflect.TypeOf(VerifiedProgress{}), "VerifiedProgress")
}
