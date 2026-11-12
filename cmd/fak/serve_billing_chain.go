// serve_billing_chain.go mounts the multi-tenant billing gateway chain
// (PLAN-C Phase 3 item 12, ticket bm-p3-12-servechain): it builds the chain
// from the billing runtime item 10 loaded, binds its own listener, and serves
// it for the lifetime of `fak serve`. The chain is a SEPARATE billing plane —
// it proxies to the manifest's declared upstreams and never wraps the
// kernel's own serving listener — so a deployment without --billing is
// byte-for-byte today's behavior.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	billingservechain "github.com/anthony-chaudhary/fak/pkg/billing/servechain"
)

// startServeBillingChain builds the chain from the loaded billing runtime,
// binds addr, and serves it. The returned stop shuts the listener down under
// a bounded wait, closes the chain (draining its settle queue so every
// captured usage lands in the ledger), and reaps the serve goroutine's
// error. It is called exactly once, via defer, when cmdServe returns.
func startServeBillingChain(rt *billingRuntime, addr string, stderr io.Writer) (func(), error) {
	chain, err := billingservechain.New(billingservechain.Config{
		Manifest: rt.Manifest,
		Prices:   rt.PriceBook,
		Ledger:   rt.Ledger,
		Keys:     rt.Keys,
	})
	if err != nil {
		return nil, fmt.Errorf("billing chain: %w", err)
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		chain.Close()
		return nil, fmt.Errorf("billing chain listen %q: %w", addr, err)
	}
	srv := &http.Server{Handler: chain, ReadHeaderTimeout: 30 * time.Second}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	models := make([]string, len(rt.Manifest.Models))
	for i, m := range rt.Manifest.Models {
		models[i] = m.Name
	}
	fmt.Fprintf(stderr, "fak serve: billing chain listening on http://%s (%d models: %s)\n",
		ln.Addr(), len(models), strings.Join(models, ", "))

	return func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		chain.Close()
		if serr := <-serveErr; serr != nil && !errors.Is(serr, http.ErrServerClosed) {
			fmt.Fprintf(stderr, "fak serve: billing chain serve error: %v\n", serr)
		}
	}, nil
}
