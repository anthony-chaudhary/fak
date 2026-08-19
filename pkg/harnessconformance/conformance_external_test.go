package harnessconformance_test

import (
	"encoding/json"
	hc "github.com/anthony-chaudhary/fak/pkg/harnessconformance"
	"strings"
	"testing"
)

type adapter struct {
	caps   []hc.Capability
	broken hc.Capability
	skip   hc.Capability
}

func (a adapter) Capabilities() []hc.Capability { return a.caps }
func (a adapter) Check(c hc.Capability) (hc.Outcome, string) {
	if c == a.broken {
		return hc.Fail, "seeded failure"
	}
	if c == a.skip {
		return hc.Skip, "not applicable"
	}
	return hc.Pass, ""
}
func all() []hc.Capability { return append([]hc.Capability(nil), hc.Required...) }
func TestStockAndCustomAdaptersCertify(t *testing.T) {
	for _, name := range []string{"stock", "custom"} {
		t.Run(name, func(t *testing.T) {
			cert := hc.Run(adapter{caps: all()})
			if !cert.Full {
				t.Fatalf("certificate=%+v", cert)
			}
			if err := cert.Validate(); err != nil {
				t.Fatal(err)
			}
			data, err := cert.JSON()
			if err != nil {
				t.Fatal(err)
			}
			var decoded hc.Certificate
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded.Contract != hc.ContractVersion || decoded.FixtureDigest == "" {
				t.Fatalf("decoded=%+v", decoded)
			}
		})
	}
}
func TestSeededBadAdapterRejectedPerContractClass(t *testing.T) {
	for _, class := range hc.Required {
		t.Run(string(class), func(t *testing.T) {
			cert := hc.Run(adapter{caps: all(), broken: class})
			err := cert.Validate()
			if err == nil || !strings.Contains(err.Error(), string(class)) {
				t.Fatalf("error=%v checks=%+v", err, cert.Checks)
			}
		})
	}
}
func TestTypedSkipCannotReceiveFullCertificate(t *testing.T) {
	cert := hc.Run(adapter{caps: []hc.Capability{hc.Lifecycle}})
	if cert.Full {
		t.Fatal("partial adapter received full certificate")
	}
	found := false
	for _, check := range cert.Checks {
		if check.Outcome == hc.Skip {
			found = true
		}
	}
	if !found {
		t.Fatal("missing typed skip")
	}
	if err := cert.Validate(); err == nil {
		t.Fatal("partial certificate validated")
	}
}
