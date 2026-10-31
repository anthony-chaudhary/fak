package gateway

import (
	"context"
	"sync"
)

// runNativeStreamDelivery is the native owned-loop transport lifecycle. The
// actual handler supplies its generator and wire renderers; this helper neither
// changes model/tool semantics nor creates another HTTP writer. A qualified
// async consumer owns both renderers until the terminal join returns. Without
// interruptible writer deadlines the same generator/renderers run synchronously.
//
// Generation and delivery errors remain separate so the handler preserves its
// existing failure classification. A generation panic propagates after cancel
// and join; a delivery panic follows CoalescingSender's existing failure path.
func runNativeStreamDelivery(
	ctx context.Context,
	deadlineQualified bool,
	cancel context.CancelFunc,
	generate func(text, progress func(string) error) error,
	deliverText, deliverProgress func(string) error,
) (generationErr, deliveryErr error) {
	if !deadlineQualified {
		return generate(deliverText, deliverProgress), nil
	}
	stop := sync.OnceFunc(cancel)
	sender := newCoalescingSender(ctx, false, func(chunk DeltaChunk) error {
		if chunk.barrier {
			return deliverProgress(chunk.Text)
		}
		return deliverText(chunk.Text)
	}, stop)
	joined := false
	defer func() {
		if !joined {
			stop()
			_ = sender.Close()
		}
	}()
	generationErr = generate(func(text string) error { return sender.Send(text, 0) }, sender.barrier)
	deliveryErr = sender.Close()
	joined = true
	return generationErr, deliveryErr
}
