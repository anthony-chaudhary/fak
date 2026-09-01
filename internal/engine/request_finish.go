package engine

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type requestFinish struct {
	once sync.Once
	res  *abi.Result
	err  error
}

func (f *requestFinish) complete(tokens chan abi.EngineToken, done chan struct{}, res *abi.Result, err error) {
	// The first terminal result wins. Closing done publishes the selected fields
	// to receivers; tokens is closed first so done means the whole request ended.
	f.once.Do(func() {
		f.res, f.err = res, err
		close(tokens)
		close(done)
	})
}
