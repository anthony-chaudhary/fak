package agent

import (
	"context"
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
)

type resolvedConsumerContextKey struct{}

type resolvedConsumerOutcome struct {
	contextValue string
	err          error
	panicValue   any
	returned     bool
}

func resolvedConsumerWitness(mode string, extracted, publish, publishContext bool, primary error) (out resolvedConsumerOutcome) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		var err error
		ctx := context.Background()
		defer func() { out.panicValue = recover() }()
		// Model the existing outer deadline observer reading the caller's named err.
		defer func() { out.err = err; out.contextValue, _ = ctx.Value(resolvedConsumerContextKey{}).(string) }()
		work := func(frameCtx *context.Context, slot *error) {
			*frameCtx = context.WithValue(*frameCtx, resolvedConsumerContextKey{}, "derived")
			*slot = primary
			switch mode {
			case "panic":
				panic(primary)
			case "goexit":
				runtime.Goexit()
			}
		}
		if extracted {
			consumer := func(callerCtx *context.Context, callerErr *error) (err error) {
				ctx := *callerCtx
				if publish {
					defer func() {
						*callerErr = err
						if publishContext {
							*callerCtx = ctx
						}
					}()
				}
				work(&ctx, &err)
				return
			}
			err = consumer(&ctx, &err)
		} else {
			work(&ctx, &err)
		}
		out.returned = true
	}()
	<-done
	return
}

// fak-test:runtime fast est=100ms lane=default
func TestResolvedConsumerPreservesDeadlineErrorOnUnwind(t *testing.T) {
	failure := errors.New("deadline normalization input")
	for _, mode := range []string{"return", "panic", "goexit"} {
		old := resolvedConsumerWitness(mode, false, false, false, failure)
		got := resolvedConsumerWitness(mode, true, true, true, failure)
		if got != old || got.err != failure || got.contextValue != "derived" {
			t.Fatalf("%s got=%+v old=%+v", mode, got, old)
		}
	}
	// The first extraction published err but not ctx: even on normal return,
	// the outer deadline callback lost values installed by the consumer.
	for _, mode := range []string{"return", "panic", "goexit"} {
		errorOnly := resolvedConsumerWitness(mode, true, true, false, failure)
		if errorOnly.err != failure || errorOnly.contextValue != "" {
			t.Fatalf("error-only negative control no longer loses final context: %+v", errorOnly)
		}
	}
	broken := resolvedConsumerWitness("panic", true, false, false, failure)
	if broken.err != nil || broken.panicValue != failure || broken.returned || broken.contextValue != "" {
		t.Fatalf("negative control no longer distinguishes lost caller error: %+v", broken)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestResolvedConsumerCallerErrorSourceGuard(t *testing.T) {
	data, err := os.ReadFile("inkernel_planner.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	a := strings.Index(src, "func (p *InKernelPlanner) Complete(ctx")
	b := strings.Index(src, "func (p *InKernelPlanner) completeResolvedNative(")
	if a < 0 || b < a {
		t.Fatal("wrapper/consumer missing")
	}
	wrapper := src[a:b]
	auth := strings.Index(wrapper, "offered := offeredToolsFor(ctx, tools)")
	deadline := strings.Index(wrapper, "defer func() { err = p.executionDeadlineError(ctx, err) }()")
	fold := strings.Index(wrapper, "sp := applySampleOpts(opts...)")
	call := strings.Index(wrapper, "return p.completeResolvedNative(ctx, messages, tools, sp, offered, &ctx, &err, opts...)")
	if auth < 0 || deadline < auth || fold < deadline || call < fold {
		t.Fatal("authorization/deadline/option/error-slot order changed")
	}
	end := strings.Index(src[b:], "func (p *InKernelPlanner) requiresDeviceSerialization()")
	if end < 0 {
		t.Fatal("consumer end missing")
	}
	consumer := src[b : b+end]
	if !strings.Contains(consumer, "(comp *Completion, err error) {\n\tdefer func() { *callerCtx, *callerErr = ctx, err }()") {
		t.Fatal("consumer must publish named error on every unwind")
	}
	if !strings.Contains(consumer, "return p.completeNativeBeamSearch(ctx, messages, tools, sp, opts...)") {
		t.Fatal("beam routing/options changed")
	}
}
