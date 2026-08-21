package stackresolve

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type refusingContractProvider struct{ err error }

func (refusingContractProvider) Name() string                                      { return "refusing-catalog" }
func (p refusingContractProvider) Components(context.Context) ([]Component, error) { return nil, p.err }

func TestPublishedComponentContractsRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name string
		run  func() error
		want []string
	}{
		{name: "missing workload and roots", run: func() error { _, err := Resolve(context.Background(), "", nil); return err }, want: []string{"workload", "root", "required"}},
		{name: "provider failure", run: func() error {
			_, err := Resolve(context.Background(), "serve", []string{"runtime"}, refusingContractProvider{err: errors.New("catalog unavailable")})
			return err
		}, want: []string{"refusing-catalog", "catalog unavailable"}},
		{name: "missing component fields", run: func() error {
			_, err := ComposeContracts("serve", []string{"runtime"}, []ComponentContract{{Schema: ComponentContractSchema}})
			return err
		}, want: []string{"component", "id", "kind", "version", "required"}},
		{name: "unsupported schema", run: func() error {
			_, err := ParseComponentContract([]byte(`{"schema":"future","component":{}}`))
			return err
		}, want: []string{"schema", "future", ComponentContractSchema}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.run()
			if err == nil {
				t.Fatal("operation accepted refusal input")
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q omits recovery text %q", err, want)
				}
			}
		})
	}
}
