package modelengine

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestKVLoadCertificateReconstructsOrRefuses(t *testing.T) {
	ids := []int{4, 1, 7, 2}
	cert := NewKVLoadCertificate(ids, "paged-16")
	if cert.MaxBlockID != 7 || cert.Blocks != 4 || cert.CostBytes != 8*4+len("paged-16")+16 {
		t.Fatalf("unexpected certificate: %+v", cert)
	}
	if err := cert.Verify(ids, "paged-16"); err != nil {
		t.Fatalf("exact reconstruction must verify: %v", err)
	}
	for name, bad := range map[string]struct {
		ids    []int
		layout string
	}{
		"reordered": {[]int{1, 4, 7, 2}, "paged-16"},
		"layout":    {ids, "paged-32"},
		"short":     {ids[:3], "paged-16"},
		"oob":       {[]int{4, 1, 9, 2}, "paged-16"},
	} {
		if err := cert.Verify(bad.ids, bad.layout); !errors.Is(err, ErrKVLoadCertificateInvalid) {
			t.Fatalf("%s: expected ErrKVLoadCertificateInvalid, got %v", name, err)
		}
	}
}
