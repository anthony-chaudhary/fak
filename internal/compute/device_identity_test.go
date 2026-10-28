package compute

import (
	"errors"
	"testing"
)

type identityProvider struct {
	id  DeviceIdentity
	err error
}

func (p *identityProvider) DeviceIdentity() (DeviceIdentity, error) { return p.id, p.err }

// fak-test:runtime fast est=1s
func TestRequireDeviceIdentity(t *testing.T) {
	var nilProvider *identityProvider
	failure := errors.New("provider failure")
	for _, input := range []any{nil, struct{}{}, nilProvider, &identityProvider{}, &identityProvider{err: failure}} {
		id, err := RequireDeviceIdentity(input)
		if !errors.Is(err, ErrDeviceIdentityUnavailable) || id != (DeviceIdentity{}) {
			t.Fatalf("refusal id=%v err=%v", id, err)
		}
	}
	id := DeviceIdentity{UUID: [16]byte{1}}
	got, err := RequireDeviceIdentity(&identityProvider{id: id})
	if err != nil || got != id {
		t.Fatalf("id=%v err=%v", got, err)
	}
	_, err = RequireDeviceIdentity(&identityProvider{err: failure})
	if !errors.Is(err, failure) {
		t.Fatal("provider error lost")
	}
}
