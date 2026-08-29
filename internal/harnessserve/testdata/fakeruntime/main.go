package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"time"
)

func main() {
	address := flag.String("address", "", "loopback listen address")
	mode := flag.String("mode", "normal", "normal|never-ready|bad-probe|ignore-shutdown|sentinel")
	readyDelay := flag.Duration("ready-delay", 40*time.Millisecond, "loading duration")
	probeDelay := flag.Duration("probe-delay", 0, "completion response delay")
	marker := flag.String("marker", "", "lifecycle marker path")
	flag.Parse()

	if *mode == "sentinel" {
		writeMarker(*marker, "sentinel-ready")
		wait := make(chan os.Signal, 1)
		signal.Notify(wait, os.Interrupt)
		<-wait
		return
	}
	host, _, err := net.SplitHostPort(*address)
	if err != nil || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		fmt.Fprintln(os.Stderr, "non-loopback address refused")
		os.Exit(2)
	}
	listener, err := net.Listen("tcp", *address)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	started := time.Now()
	var completions atomic.Int64
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		status := "loading"
		code := http.StatusServiceUnavailable
		if *mode != "never-ready" && time.Since(started) >= *readyDelay {
			status, code = "ready", http.StatusOK
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": status})
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model     string `json:"model"`
			Prompt    string `json:"prompt"`
			MaxTokens int    `json:"max_tokens"`
		}
		if json.NewDecoder(r.Body).Decode(&request) != nil || request.Model == "" || request.Prompt == "" || request.MaxTokens != 1 {
			http.Error(w, "bounded probe required", http.StatusBadRequest)
			return
		}
		if *probeDelay > 0 {
			time.Sleep(*probeDelay)
		}
		completionTokens := 1
		if *mode == "bad-probe" {
			completionTokens = 2
		}
		completions.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]string{{"text": "x"}},
			"usage":   map[string]int{"completion_tokens": completionTokens},
		})
	})

	server := &http.Server{Handler: mux}
	mux.HandleFunc("/shutdown", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Fak-Ownership-Token") == "" || r.Header.Get("X-Fak-Ownership-Token") != os.Getenv("FAK_HARNESS_SERVE_OWNERSHIP_TOKEN") {
			http.Error(w, "ownership token required", http.StatusForbidden)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		if *mode == "ignore-shutdown" {
			return
		}
		writeMarker(*marker, "graceful")
		go func() {
			time.Sleep(5 * time.Millisecond)
			_ = server.Shutdown(context.Background())
		}()
	})
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)
	go func() {
		<-interrupt
		if *mode != "ignore-shutdown" {
			writeMarker(*marker, "graceful")
			_ = server.Shutdown(context.Background())
		}
	}()
	writeMarker(*marker, "listening")
	err = server.Serve(listener)
	if err != nil && err != http.ErrServerClosed && !strings.Contains(err.Error(), "closed") {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_ = completions.Load()
}

func writeMarker(path, value string) {
	if path != "" {
		_ = os.WriteFile(path, []byte(value+"\n"), 0o600)
	}
}
