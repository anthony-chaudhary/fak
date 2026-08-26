package compute

import "errors"

var (
	errMetalOwnerTerminal = errors.New("compute: Metal command owner is terminal")
	errMetalOwnerEmpty    = errors.New("compute: Metal command owner has no encoders")
)

type metalOwnerState uint8

const (
	metalOwnerOpen metalOwnerState = iota
	metalOwnerSubmitted
	metalOwnerAborted
)

type metalOwnerLifecycle struct {
	state    metalOwnerState
	encoders int
}

func (s *metalOwnerLifecycle) encode() error {
	if s.state != metalOwnerOpen {
		return errMetalOwnerTerminal
	}
	s.encoders++
	return nil
}

func (s *metalOwnerLifecycle) finish() error {
	if s.state != metalOwnerOpen {
		return errMetalOwnerTerminal
	}
	if s.encoders == 0 {
		return errMetalOwnerEmpty
	}
	s.state = metalOwnerSubmitted
	return nil
}

func (s *metalOwnerLifecycle) abort() error {
	if s.state != metalOwnerOpen {
		return errMetalOwnerTerminal
	}
	s.state = metalOwnerAborted
	return nil
}
