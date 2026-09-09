//go:build !darwin

package mtpbench

import "errors"

func hardwareIdentity() (HardwareIdentity, error) {
	return HardwareIdentity{}, errors.New("mtpbench: Metal hardware identity unavailable")
}
