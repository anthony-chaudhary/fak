package engine

import "github.com/anthony-chaudhary/fak/internal/abi"

type requestFinish struct {
	res *abi.Result
	err error
}

func (f *requestFinish) complete(tokens chan abi.EngineToken, done chan struct{}, res *abi.Result, err error) {
	f.res, f.err = res, err
	close(tokens)
	close(done)
}
