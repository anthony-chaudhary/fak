package rsiloop

import (
	"errors"
)

// IsTransient reports whether err represents a transient error condition (such as
// a transient measurement error, git lock contention, or temporary infrastructure failure).
// It unwraps err and recognizes:
//   - any error for which IsTransientMeasureError(err) is true
//   - any error for which IsGitLockError(err) is true
//   - any error implementing IsTransient() bool that returns true
//   - any error implementing Transient() bool that returns true
//   - any error implementing TransientMeasure() bool that returns true
func IsTransient(err error) bool {
	if err == nil {
		return false
	}
	if IsTransientMeasureError(err) {
		return true
	}
	if IsGitLockError(err) {
		return true
	}
	type isTransient interface {
		IsTransient() bool
	}
	var it isTransient
	if errors.As(err, &it) {
		return it.IsTransient()
	}
	type transient interface {
		Transient() bool
	}
	var tr transient
	if errors.As(err, &tr) {
		return tr.Transient()
	}
	return false
}
