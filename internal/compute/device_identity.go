package compute

import (
	"errors"
	"reflect"
)

// ErrDeviceIdentityUnavailable means no usable execution-device identity was supplied.
var ErrDeviceIdentityUnavailable = errors.New("compute: device identity unavailable")

// DeviceIdentity contains raw execution-device UUID bytes. It does not identify
// the device as a physical GPU or a MIG instance.
type DeviceIdentity struct{ UUID [16]byte }

// DeviceIdentityBackend optionally identifies the backend's execution device.
type DeviceIdentityBackend interface {
	DeviceIdentity() (DeviceIdentity, error)
}

// RequireDeviceIdentity refuses absent, nil, failed, and zero identity providers.
// Device class and binding to a physical pool require separate evidence.
func RequireDeviceIdentity(backend any) (DeviceIdentity, error) {
	p, ok := backend.(DeviceIdentityBackend)
	if !ok || p == nil {
		return DeviceIdentity{}, ErrDeviceIdentityUnavailable
	}
	v := reflect.ValueOf(p)
	if (v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface || v.Kind() == reflect.Map || v.Kind() == reflect.Slice || v.Kind() == reflect.Func || v.Kind() == reflect.Chan) && v.IsNil() {
		return DeviceIdentity{}, ErrDeviceIdentityUnavailable
	}
	id, err := p.DeviceIdentity()
	if err != nil {
		return DeviceIdentity{}, errors.Join(ErrDeviceIdentityUnavailable, err)
	}
	if id.UUID == ([16]byte{}) {
		return DeviceIdentity{}, ErrDeviceIdentityUnavailable
	}
	return id, nil
}
