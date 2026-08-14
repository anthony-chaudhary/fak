package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runSessionDiscoveryServe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session discovery-serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	listen := fs.String("listen", "0.0.0.0:8443", "TLS listen address")
	cert := fs.String("cert", "", "TLS certificate path")
	key := fs.String("key", "", "TLS private key path")
	sessionID := fs.String("session-id", "cross-device-selfcheck", "logical session ID")
	relayURL := fs.String("relay-url", "", "public https relay URL")
	localToken := fs.String("local-token", "", "local publication/revocation token")
	ttl := fs.Duration("ttl", 10*time.Minute, "discovery capability lifetime")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 || *cert == "" || *key == "" || *relayURL == "" || *localToken == "" {
		return 2
	}
	state := gateway.SessionState{TraceID: *sessionID, Run: "RUNNING", Rev: 1}
	var server *gateway.Server
	server, err := gateway.New(gateway.Config{EngineID: "mock", Model: "selfcheck", ObserveSession: func(context.Context, string) gateway.SessionState { return state }, SteerSession: func(_ context.Context, _, _, text string) error {
		server.RecordSessionTerminalOutput(*sessionID, []byte("remote-input:"+text+"\r\n"))
		state.Rev++
		return nil
	}})
	if err != nil {
		fmt.Fprintf(stderr, "session discovery-serve: %v\n", err)
		return 1
	}
	server.ConfigureSessionClientAuth(*localToken)
	server.RestoreSessionClientState(*sessionID, []byte("portable-session-ready\r\n"), nil)
	source := gateway.SessionPlacement{Provider: "selfcheck-a", AccountRef: "account-a", Model: "model-a", Compute: "source-container", Capabilities: []string{"observe", "replay", "text_input", "detach"}, ContextLimit: 1024, BudgetAvailable: 1, ComputeAvailable: true}
	if err := server.ConfigureSessionMove(*sessionID, source, gateway.SessionMoveHooks{RequestSafePoint: func(context.Context, string) error { return nil }, AdmitDestination: func(context.Context, string, gateway.SessionMoveCheckpoint, gateway.SessionMoveRequest) error {
		return nil
	}, RestoreDestination: func(context.Context, gateway.SessionMoveCheckpoint) error { return nil }}); err != nil {
		fmt.Fprintf(stderr, "session discovery-serve: %v\n", err)
		return 1
	}
	publication, err := server.PublishSessionDiscovery(*sessionID, *relayURL, *ttl)
	if err != nil {
		fmt.Fprintf(stderr, "session discovery-serve: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(publication); err != nil {
		return 1
	}
	if f, ok := stdout.(interface{ Sync() error }); ok {
		_ = f.Sync()
	}
	httpServer := &http.Server{Addr: *listen, Handler: server.Handler(), ReadHeaderTimeout: 5 * time.Second}
	if err := httpServer.ListenAndServeTLS(*cert, *key); err != nil && err != http.ErrServerClosed {
		fmt.Fprintf(stderr, "session discovery-serve: %v\n", err)
		return 1
	}
	return 0
}
