//go:build !darwin || !arm64 || !cgo

package inklingmetal

func (*Model) NewSessionFor(int, string, []int32) (*Session, error) { return nil, ErrUnavailable }
func (*Session) Reused() int                                        { return 0 }
