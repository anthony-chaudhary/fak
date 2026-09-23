package servewarmup

import (
	"context"
	"io"
	"time"
)

// Run invokes the backend startup warmup once.
func Run(ctx context.Context, run func(context.Context) (time.Duration, error), _ io.Writer) {
	_, _ = run(ctx)
}
