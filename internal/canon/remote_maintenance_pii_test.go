package canon

import "testing"

// fak-test:runtime fast est=0.01s
func TestCanonicalPIIClassesPreserveTargetOverlap(t *testing.T) {
	emailOnly := map[string]bool{PIIClassEmail: true}
	for _, tc := range []struct {
		name       string
		body       string
		wantPII    bool
		wantExcept bool
	}{
		{"clean", "ordinary maintenance receipt", false, false},
		{"ssh-email", "svc@node.local", true, false},
		{"raw-national-id", "svc@123-45-6789.example", true, true},
		{"reversed-national-id", "svc@9876-54-321.example", true, true},
		{"base64-national-id", "svc@MTIzLTQ1LTY3ODkg.example", true, true},
		{"phone", "svc@555-123-4567.example", true, true},
		{"payment-card", "svc@4111-1111-1111-1111.example", true, true},
		{"iban", "svc@GB82WEST12345698765432.example", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			if got := Scan(body).PII; got != tc.wantPII {
				t.Fatalf("Scan().PII = %v, want %v", got, tc.wantPII)
			}
			if got := HasCanonicalPIIExcept(body, nil); got != tc.wantPII {
				t.Fatalf("canonical class presence = %v, want Scan PII %v", got, tc.wantPII)
			}
			if got := HasCanonicalPIIExcept(body, emailOnly); got != tc.wantExcept {
				t.Fatalf("canonical non-email PII = %v, want %v", got, tc.wantExcept)
			}
			if string(body) != tc.body {
				t.Fatal("canonical class detection mutated body")
			}
		})
	}
	t.Run("other-class-exemptions", func(t *testing.T) {
		body := []byte("9876-54-321")
		if !Scan(body).PII || !HasCanonicalPIIExcept(body, emailOnly) {
			t.Fatal("reversed national ID missing from source canonical views")
		}
		if HasCanonicalPIIExcept(body, map[string]bool{PIIClassNationalID: true}) {
			t.Fatal("national-ID class exemption was not applied to canonical views")
		}
		if !HasCanonicalPIIExcept([]byte("svc@node.local"), map[string]bool{PIIClassNationalID: true}) {
			t.Fatal("exempting another class concealed an email")
		}
	})
}
