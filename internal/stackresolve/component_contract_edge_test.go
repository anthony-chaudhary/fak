package stackresolve

import (
	"context"
	"strings"
	"testing"
)

func TestPublishedComponentContractsEdgeAndAdversarialInputs(t *testing.T) {
	valid := `{"schema":"fak-component-contract/1","component":{"id":"runtime","kind":"runtime","version":"1","provides":["runtime"],"evidence":{"authority":"fixture","source":"runtime-contract"}}}`
	parseCases := []struct {
		name string
		raw  string
		want string
	}{
		{name: "empty", raw: "", want: "EOF"},
		{name: "malformed", raw: `{"schema":`, want: "unexpected EOF"},
		{name: "hostile unknown", raw: strings.TrimSuffix(valid, "}") + `,"__proto__":{"polluted":true}}`, want: "unknown field"},
		{name: "multiple documents", raw: valid + valid, want: "multiple JSON values"},
		{name: "oversized identifier", raw: `{"schema":"fak-component-contract/1","component":{"id":"` + strings.Repeat("x", 1<<20) + `","kind":"runtime","version":"1"}}`, want: "id"},
	}
	for _, tc := range parseCases {
		t.Run("parse/"+tc.name, func(t *testing.T) {
			_, err := ParseComponentContract([]byte(tc.raw))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseComponentContract error = %v, want text %q", err, tc.want)
			}
		})
	}

	contract, err := ParseComponentContract([]byte(valid))
	if err != nil {
		t.Fatal(err)
	}
	composeCases := []struct {
		name      string
		workload  string
		roots     []string
		contracts []ComponentContract
		want      string
	}{
		{name: "empty workload", roots: []string{"runtime"}, contracts: []ComponentContract{contract}, want: "workload is required"},
		{name: "empty roots", workload: "serve", contracts: []ComponentContract{contract}, want: "root is required"},
		{name: "duplicate contracts", workload: "serve", roots: []string{"runtime"}, contracts: []ComponentContract{contract, contract}, want: "duplicate component id"},
		{name: "hostile root", workload: "serve", roots: []string{"../../escape"}, contracts: []ComponentContract{contract}, want: "root"},
	}
	for _, tc := range composeCases {
		t.Run("compose/"+tc.name, func(t *testing.T) {
			var receipt Receipt
			manifest, err := ComposeContracts(tc.workload, tc.roots, tc.contracts)
			if err == nil {
				receipt, err = Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
			}
			if err != nil && strings.Contains(err.Error(), tc.want) {
				return
			}
			if err != nil || receipt.Status != "refuse" || receipt.Conflict == nil || !strings.Contains(strings.ToLower(receipt.Conflict.Code), tc.want) {
				t.Fatalf("contract outcome error=%v receipt=%+v, want text %q", err, receipt, tc.want)
			}
		})
	}
}
