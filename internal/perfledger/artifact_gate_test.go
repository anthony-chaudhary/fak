// fak-test:runtime fast est=50ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const gateReceipt = `{"schema":"bench/1","decode_tps":41.5,"e2e_ms":1200,"model":"m-a","note":"","ttft_ms":null,"huge":1e999,"quoted_tps":"41.5"}`

func writeGateArtifact(t *testing.T, body string) (path string, size *int64, sum string) {
	t.Helper()
	path = filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	n := int64(len(body))
	digest := sha256.Sum256([]byte(body))
	return path, &n, hex.EncodeToString(digest[:])
}

func gateCheck(t *testing.T, v ArtifactVerdict, name string) ArtifactCheck {
	t.Helper()
	for _, c := range v.Checks {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("verdict has no %q check: %+v", name, v.Checks)
	return ArtifactCheck{}
}

func TestArtifactGateAcceptsFullyDeclaredArtifact(t *testing.T) {
	path, size, sum := writeGateArtifact(t, gateReceipt)
	v := GateArtifact(ArtifactContract{
		Path:           path,
		SizeBytes:      size,
		SHA256:         strings.ToUpper(sum),
		Magic:          []byte(`{"schema"`),
		ContentType:    ArtifactContentJSON,
		RequiredFields: []string{"decode_tps", "model"},
	})
	if !v.Accepted() || v.Status != ArtifactAccepted {
		t.Fatalf("fully declared, matching artifact was %q: %+v", v.Status, v)
	}
	if v.SHA256 != sum || v.ByteSize != *size {
		t.Fatalf("verdict records sha256=%s size=%d, want %s/%d", v.SHA256, v.ByteSize, sum, *size)
	}
	for _, name := range []string{"artifact.size_bytes", "artifact.sha256", "artifact.magic_bytes", "artifact.content_type", "field.decode_tps", "field.model"} {
		if c := gateCheck(t, v, name); c.Status != ArtifactCheckPassed {
			t.Fatalf("check %s = %q, want passed", name, c.Status)
		}
	}
	// Even a clean pass does not vouch for the numbers themselves.
	if want := []string{UnverifiedMeasurement}; !reflect.DeepEqual(v.Unverified, want) {
		t.Fatalf("Unverified = %v, want %v", v.Unverified, want)
	}
	if got, ok := v.Metric("decode_tps"); !ok || got != 41.5 {
		t.Fatalf("Metric(decode_tps) = %v, %v; want 41.5", got, ok)
	}
}

func TestArtifactGateRefusesContractFailures(t *testing.T) {
	path, size, sum := writeGateArtifact(t, gateReceipt)
	wrongSize := *size + 1
	full := func() ArtifactContract {
		return ArtifactContract{Path: path, SizeBytes: size, SHA256: sum, Magic: []byte("{"), ContentType: ArtifactContentJSON,
			RequiredFields: []string{"decode_tps"}}
	}
	cases := []struct {
		name      string
		mutate    func(*ArtifactContract)
		failed    string
		wantFound string
	}{
		{"size_mismatch", func(c *ArtifactContract) { c.SizeBytes = &wrongSize }, "artifact.size_bytes", ""},
		{"digest_mismatch", func(c *ArtifactContract) { c.SHA256 = strings.Repeat("0", 64) }, "artifact.sha256", ""},
		{"magic_mismatch", func(c *ArtifactContract) { c.Magic = []byte("PK\x03\x04") }, "artifact.magic_bytes", ""},
		{"magic_longer_than_file", func(c *ArtifactContract) { c.Magic = []byte(gateReceipt + "x") }, "artifact.magic_bytes", ""},
		{"field_absent", func(c *ArtifactContract) { c.RequiredFields = append(c.RequiredFields, "prefill_tps") }, "field.prefill_tps", "absent"},
		{"field_null", func(c *ArtifactContract) { c.RequiredFields = append(c.RequiredFields, "ttft_ms") }, "field.ttft_ms", "null"},
		{"field_empty_string", func(c *ArtifactContract) { c.RequiredFields = append(c.RequiredFields, "note") }, "field.note", "empty string"},
		{"field_number_out_of_range", func(c *ArtifactContract) { c.RequiredFields = append(c.RequiredFields, "huge") }, "field.huge", "number out of range"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			contract := full()
			tc.mutate(&contract)
			v := GateArtifact(contract)
			if v.Status != ArtifactRefused || v.Accepted() {
				t.Fatalf("status = %q, want refused", v.Status)
			}
			c := gateCheck(t, v, tc.failed)
			if c.Status != ArtifactCheckFailed {
				t.Fatalf("check %s = %q, want failed", tc.failed, c.Status)
			}
			if tc.wantFound != "" && c.Observed != tc.wantFound {
				t.Fatalf("check %s observed %q, want %q", tc.failed, c.Observed, tc.wantFound)
			}
			// decode_tps itself read back fine in every case; it still must
			// not be readable from a refused artifact.
			if c := gateCheck(t, v, "field.decode_tps"); c.Status != ArtifactCheckPassed {
				t.Fatalf("field.decode_tps = %q, want passed alongside the failing check", c.Status)
			}
			if _, ok := v.Metric("decode_tps"); ok {
				t.Fatal("a refused artifact still yielded a metric")
			}
		})
	}
}

func TestArtifactGateRefusesUnparseableDeclaredJSON(t *testing.T) {
	for name, body := range map[string]string{
		"truncated":     `{"decode_tps": 41.5`,
		"not_an_object": `[1, 2, 3]`,
		"json_null":     `null`,
		"trailing_junk": `{"decode_tps": 41.5} extra`,
	} {
		path, size, _ := writeGateArtifact(t, body)
		v := GateArtifact(ArtifactContract{Path: path, SizeBytes: size, ContentType: ArtifactContentJSON, RequiredFields: []string{"decode_tps"}})
		if v.Status != ArtifactRefused {
			t.Fatalf("%s: status = %q, want refused", name, v.Status)
		}
		if c := gateCheck(t, v, "artifact.content_type"); c.Status != ArtifactCheckFailed {
			t.Fatalf("%s: content_type check = %q, want failed", name, c.Status)
		}
		if c := gateCheck(t, v, "field.decode_tps"); c.Status != ArtifactCheckFailed {
			t.Fatalf("%s: field check = %q, want failed when the artifact does not parse", name, c.Status)
		}
	}
}

func TestArtifactGateAntiTamper(t *testing.T) {
	cases := map[string]func(t *testing.T, path string){
		// The mtime is put back, so only the size comparison can catch this.
		"size_changed": func(t *testing.T, path string) {
			before, err := os.Stat(path)
			if err != nil {
				t.Fatalf("stat: %v", err)
			}
			f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			if _, err := f.WriteString(" "); err != nil {
				t.Fatalf("append: %v", err)
			}
			f.Close()
			if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
				t.Fatalf("chtimes: %v", err)
			}
		},
		// Same size, same bytes: only the modification time moves.
		"mtime_changed": func(t *testing.T, path string) {
			later := time.Now().Add(2 * time.Hour)
			if err := os.Chtimes(path, later, later); err != nil {
				t.Fatalf("chtimes: %v", err)
			}
		},
		"removed": func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatalf("remove: %v", err)
			}
		},
	}
	for name, tamper := range cases {
		t.Run(name, func(t *testing.T) {
			path, size, sum := writeGateArtifact(t, gateReceipt)
			contract := ArtifactContract{Path: path, SizeBytes: size, SHA256: sum, ContentType: ArtifactContentJSON, RequiredFields: []string{"decode_tps"}}
			v := gateArtifact(contract, func() { tamper(t, path) })
			if v.Status != ArtifactRefused {
				t.Fatalf("status = %q after the file changed mid-analysis, want refused", v.Status)
			}
			if len(v.Errors) != 1 || !(strings.Contains(v.Errors[0], "changed while it was being analyzed") || strings.Contains(v.Errors[0], "vanished")) {
				t.Fatalf("errors = %q, want the change named", v.Errors)
			}
			if c := gateCheck(t, v, "artifact.preflight"); c.Status != ArtifactCheckFailed {
				t.Fatalf("preflight check = %q, want failed", c.Status)
			}
			// The bytes that were read matched the contract; they still do
			// not count, and no digest is published for them.
			if v.SHA256 != "" || len(v.Checks) != 1 {
				t.Fatalf("a tampered read published sha256=%q and %d checks", v.SHA256, len(v.Checks))
			}
			if _, ok := v.Metric("decode_tps"); ok {
				t.Fatal("a tampered artifact still yielded a metric")
			}
		})
	}

	// Control: the same contract with no interference is accepted, so the
	// refusals above come from the change and nothing else.
	path, size, sum := writeGateArtifact(t, gateReceipt)
	contract := ArtifactContract{Path: path, SizeBytes: size, SHA256: sum, ContentType: ArtifactContentJSON, RequiredFields: []string{"decode_tps"}}
	if v := gateArtifact(contract, func() {}); !v.Accepted() {
		t.Fatalf("untouched artifact was %q: %+v", v.Status, v)
	}
}

func TestArtifactGateUnjudgedWhenTooLittleIsDeclared(t *testing.T) {
	path, size, sum := writeGateArtifact(t, gateReceipt)
	cases := []struct {
		name           string
		contract       ArtifactContract
		wantUnverified []string
	}{
		{
			name:     "nothing_declared",
			contract: ArtifactContract{Path: path},
			wantUnverified: []string{UnverifiedContentType, UnverifiedDeclaredFields, UnverifiedDeclaredSHA256,
				UnverifiedDeclaredSize, UnverifiedMagicBytes, UnverifiedMeasurement},
		},
		{
			name:     "size_only",
			contract: ArtifactContract{Path: path, SizeBytes: size},
			wantUnverified: []string{UnverifiedContentType, UnverifiedDeclaredFields, UnverifiedDeclaredSHA256,
				UnverifiedMagicBytes, UnverifiedMeasurement},
		},
		{
			name:           "content_but_no_size",
			contract:       ArtifactContract{Path: path, SHA256: sum, ContentType: ArtifactContentJSON, RequiredFields: []string{"model", "decode_tps"}},
			wantUnverified: []string{UnverifiedDeclaredSize, UnverifiedMagicBytes, UnverifiedMeasurement},
		},
		{
			// A type the gate cannot read back establishes nothing, and its
			// fields cannot be checked either.
			name:     "unreadable_content_type",
			contract: ArtifactContract{Path: path, SizeBytes: size, ContentType: "application/x-parquet", RequiredFields: []string{"model"}},
			wantUnverified: []string{UnverifiedContentType, UnverifiedDeclaredFields, UnverifiedDeclaredSHA256,
				UnverifiedMagicBytes, UnverifiedMeasurement},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := GateArtifact(tc.contract)
			if v.Status != ArtifactUnjudged || v.Accepted() {
				t.Fatalf("status = %q, want unjudged and not accepted", v.Status)
			}
			if !reflect.DeepEqual(v.Unverified, tc.wantUnverified) {
				t.Fatalf("Unverified = %v, want %v", v.Unverified, tc.wantUnverified)
			}
			if len(v.Errors) != 0 {
				t.Fatalf("an unjudged artifact reported errors: %q", v.Errors)
			}
			if _, ok := v.Metric("decode_tps"); ok {
				t.Fatal("an unjudged artifact yielded a metric")
			}
		})
	}
	unreadable := GateArtifact(cases[3].contract)
	if c := gateCheck(t, unreadable, "field.model"); c.Status != ArtifactCheckNotApplicable {
		t.Fatalf("field check under an unreadable content type = %q, want not_applicable", c.Status)
	}

	// The smallest contracts that can be accepted: the size plus any one
	// content check.
	for name, contract := range map[string]ArtifactContract{
		"magic":        {Path: path, SizeBytes: size, Magic: []byte("{")},
		"sha256":       {Path: path, SizeBytes: size, SHA256: sum},
		"content_type": {Path: path, SizeBytes: size, ContentType: ArtifactContentJSON},
	} {
		if v := GateArtifact(contract); !v.Accepted() {
			t.Fatalf("size plus %s was %q, want accepted", name, v.Status)
		}
	}
}

func TestArtifactGateNoImputation(t *testing.T) {
	path, size, sum := writeGateArtifact(t, gateReceipt)
	v := GateArtifact(ArtifactContract{Path: path, SizeBytes: size, SHA256: sum, ContentType: ArtifactContentJSON, RequiredFields: []string{"decode_tps", "model", "quoted_tps"}})
	if !v.Accepted() {
		t.Fatalf("status = %q, want accepted", v.Status)
	}
	if got, ok := v.Metric("decode_tps"); !ok || got != 41.5 {
		t.Fatalf("Metric(decode_tps) = %v, %v; want 41.5", got, ok)
	}
	for field, why := range map[string]string{
		"quoted_tps":  "declared but a quoted string, not a number",
		"e2e_ms":      "present in the file but not declared",
		"prefill_tps": "absent from the file",
		"model":       "declared but not a number",
	} {
		if got, ok := v.Metric(field); ok || got != 0 {
			t.Fatalf("Metric(%s) = %v, %v; want no value for a field %s", field, got, ok, why)
		}
	}
}

func TestArtifactGatePreflightRefusals(t *testing.T) {
	dir := t.TempDir()
	for name, path := range map[string]string{
		"missing":   filepath.Join(dir, "nope.json"),
		"directory": dir,
	} {
		v := GateArtifact(ArtifactContract{Path: path})
		if v.Status != ArtifactRefused || len(v.Errors) != 1 {
			t.Fatalf("%s: status=%q errors=%q, want one refusal", name, v.Status, v.Errors)
		}
		if c := gateCheck(t, v, "artifact.preflight"); c.Status != ArtifactCheckFailed {
			t.Fatalf("%s: preflight check = %q, want failed", name, c.Status)
		}
		wantUnverified := []string{UnverifiedContentType, UnverifiedDeclaredFields, UnverifiedDeclaredSHA256, UnverifiedDeclaredSize, UnverifiedMagicBytes, UnverifiedMeasurement}
		if !reflect.DeepEqual(v.Unverified, wantUnverified) {
			t.Fatalf("%s: Unverified = %v, want every item listed for an unread artifact: %v", name, v.Unverified, wantUnverified)
		}
	}
}
