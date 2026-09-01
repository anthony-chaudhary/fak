package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type event struct {
	Type   string
	Detail string
}

type app struct {
	product harnesskit.Product
}

func newApp() (app, error) {
	product, err := harnesskit.New("example/custom-harness-tui", harnesskit.ContractVersion).
		WithProfile(harnesskit.Profile{ID: "offline-learning"}).
		WithTransport(harnesskit.Transport{ID: "line-terminal", Provenance: harnesskit.Provenance{Source: "Go standard input/output", Version: "stdlib"}}).
		Build()
	if err != nil {
		return app{}, err
	}
	return app{product: product}, nil
}

func (a app) run(ctx context.Context, in io.Reader, out io.Writer) error {
	fmt.Fprintln(out, "fak custom harness — terminal example")
	fmt.Fprintln(out, "Type a prompt, or /quit to exit.")
	scanner := bufio.NewScanner(in)
	for {
		fmt.Fprint(out, "> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		prompt := strings.TrimSpace(scanner.Text())
		if prompt == "/quit" {
			fmt.Fprintln(out, "bye")
			return nil
		}
		if prompt == "" {
			fmt.Fprintln(out, "  write a prompt first")
			continue
		}
		events, err := a.runOfflineTurn(ctx, prompt)
		if err != nil {
			return err
		}
		for _, event := range events {
			fmt.Fprintf(out, "  %-15s %s\n", event.Type, event.Detail)
		}
	}
}

func (a app) runOfflineTurn(ctx context.Context, prompt string) ([]event, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	return []event{
		{Type: "turn.started", Detail: prompt},
		{Type: "model.response", Detail: fmt.Sprintf("%s received %q", a.product.Spec().ID, prompt)},
		{Type: "tool.requested", Detail: "record_learning_example"},
		{Type: "tool.completed", Detail: "record_learning_example:ok"},
		{Type: "turn.completed", Detail: "ok"},
	}, nil
}

func main() {
	a, err := newApp()
	if err != nil {
		panic(err)
	}
	if err := a.run(context.Background(), os.Stdin, os.Stdout); err != nil {
		panic(err)
	}
}
