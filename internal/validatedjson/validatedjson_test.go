package validatedjson

import (
	"errors"
	"testing"
)

var errInvalid = errors.New("invalid value")

type payload struct {
	Value string `json:"value"`
}

func (p payload) Validate() error {
	if p.Value == "" {
		return errInvalid
	}
	return nil
}

type unmarshalable struct {
	Ch chan int `json:"ch"`
}

func (unmarshalable) Validate() error { return nil }

func TestMarshalEncodesOnlyValidValues(t *testing.T) {
	raw, err := Marshal(payload{Value: "ok"})
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"value":"ok"}` {
		t.Fatalf("compact encoding changed: %s", raw)
	}
	// The error from Validate is returned unwrapped and identity-comparable.
	_, err = Marshal(payload{})
	if !errors.Is(err, errInvalid) {
		t.Fatalf("want the validator's own error, got %v", err)
	}
}

func TestMarshalErrorPassesThroughForValidState(t *testing.T) {
	// A state that passes Validate but cannot be encoded surfaces the
	// json.Marshal error unwrapped.
	if _, err := Marshal(unmarshalable{Ch: make(chan int)}); err == nil {
		t.Fatal("json.Marshal error not propagated")
	}
}
