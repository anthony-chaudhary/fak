//go:build darwin && !cgo

package power

import "errors"

func acquireDarwinIOKit(reason string, flags WakeFlags) (platformLock, error) {
	return nil, errors.New("cgo is disabled: IOKit power assertion unavailable")
}
