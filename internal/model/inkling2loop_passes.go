package model

import "fmt"

// Inkling2LoopPassesError refuses a per-request loop pass count the loaded
// checkpoint cannot run.
type Inkling2LoopPassesError struct {
	Requested, Max int
}

func (e *Inkling2LoopPassesError) Error() string {
	return fmt.Sprintf("inkling2_loop: %d loop passes requested, checkpoint supports 1..%d", e.Requested, e.Max)
}

// WithPasses returns hp running requested core passes for one request; 0 keeps
// the checkpoint's count. The bound is the loaded loop.count: the controllers
// are shared across passes, so more passes would run, but past the trained
// count the output is unvalidated. KV and short-conv state are per request and
// sized from the returned Passes, so N passes cost Layers+(N-1)*core cache
// slots and as many layer visits per token.
func (hp Inkling2LoopHParams) WithPasses(requested int) (Inkling2LoopHParams, error) {
	if requested == 0 {
		return hp, nil
	}
	if requested < 1 || requested > hp.Passes {
		return hp, &Inkling2LoopPassesError{Requested: requested, Max: hp.Passes}
	}
	hp.Passes = requested
	return hp, nil
}
