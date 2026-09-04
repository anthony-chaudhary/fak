//go:build !darwin && !windows && !linux

package power

func platformAcquire(reason string, flags WakeFlags) (platformLock, error) {
	return &noOpLock{}, nil
}
