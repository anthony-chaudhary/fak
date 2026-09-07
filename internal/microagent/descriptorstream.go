package microagent

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// RunDescriptorStream owns a Host for a finite descriptor stream. next returns
// io.EOF at the end; emit receives every accepted descriptor's terminal result
// until emit itself fails, which stops delivery and returns that failure.
// Both callbacks run synchronously, providing backpressure without a goroutine
// per descriptor. IDs must be unique within the stream, and bases and descriptor
// slice contents must remain immutable while their window is executing.
//
// At most Config.Queue descriptors are materialized at once. Each window finishes
// and its results are consumed before the next is read, bounding jobs, results,
// and descriptor storage independently of stream length. Config.Workers bounds
// execution; Config.Warm enables verified disk restoration. The caller owns the
// band. A matching BaseID selects shared immutable messages, not an engine cache
// guarantee. Gateway implementations remain responsible for actual KV reuse.
func RunDescriptorStream(ctx context.Context, gw Gateway, cfg Config, bases map[string][]agent.Message, next func(context.Context) (Descriptor, error), emit func(Result) error) error {
	if next == nil || emit == nil {
		return errors.New("microagent: descriptor stream requires source and result sink")
	}
	h, err := NewHost(gw, cfg)
	if err != nil {
		return err
	}
	defer h.Close()
	for {
		var sourceErr error
		for i := 0; i < cap(h.queue); i++ {
			if sourceErr = ctx.Err(); sourceErr != nil {
				break
			}
			var d Descriptor
			d, sourceErr = next(ctx)
			if sourceErr != nil {
				break
			}
			if sourceErr = d.Validate(); sourceErr != nil {
				break
			}
			base, ok := bases[d.BaseID]
			if !ok {
				sourceErr = fmt.Errorf("microagent: unknown descriptor base %q", d.BaseID)
				break
			}
			a := &DescriptorAgent{Descriptor: d, Base: base}
			if sourceErr = h.spawnWithParent(ctx, d.ID, a, h.rootCaps); sourceErr != nil {
				break
			}
		}
		// No concurrent Add occurs during Wait: this function owns the Host and
		// completes the enrollment window before consuming its terminal results.
		h.pending.Wait()
		var resultErr error
		for _, result := range h.Reap() {
			if err := emit(result); err != nil {
				return errors.Join(sourceErr, err)
			}
			if result.Err != nil && resultErr == nil {
				resultErr = result.Err
			}
		}
		if errors.Is(sourceErr, io.EOF) {
			return resultErr
		}
		if sourceErr != nil || resultErr != nil {
			return errors.Join(sourceErr, resultErr)
		}
	}
}
