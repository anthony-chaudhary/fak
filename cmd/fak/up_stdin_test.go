package main

import (
	"context"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=500ms lane=default
func TestTurnkeyUpStdinMode(t *testing.T) {
	const childEnv = "FAK_TEST_UP_STDIN_MODE"
	if mode := os.Getenv(childEnv); mode != "" {
		args := []string{"--mock", "--addr", "127.0.0.1:0", "--memory-gib", "16", "--context", "128", "--gpu-idle-exit", "0", "--max-rss", "0"}
		if mode != "default" {
			args = append(args, "--headless="+mode)
		}
		runTurnkeyUp(os.Stdin, os.Stdout, os.Stderr, args)
		return
	}

	for _, mode := range []string{"default", "true", "false"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			output, err := os.Create(filepath.Join(t.TempDir(), "up.log"))
			if err != nil {
				t.Fatal(err)
			}
			defer output.Close()
			readOutput := func() string {
				t.Helper()
				b, err := os.ReadFile(output.Name())
				if err != nil {
					t.Fatal(err)
				}
				return string(b)
			}
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestTurnkeyUpStdinMode$")
			cmd.Dir = t.TempDir()
			cmd.Env = append(os.Environ(), childEnv+"="+mode, "FAK_UP_MAX_RSS=0", "FAK_UP_CODE_WORKSPACE="+cmd.Dir)
			cmd.Stdout, cmd.Stderr = output, output
			// Default nil Stdin gives the child a real OS file at EOF. An explicit
			// interactive request must still consume a piped /quit; headless must ignore it.
			if mode != "default" {
				cmd.Stdin = strings.NewReader("/quit\n")
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			var waitErr error
			go func() {
				waitErr = cmd.Wait()
				close(done)
			}()
			defer func() {
				cancel()
				<-done
			}()

			if mode == "false" {
				<-done
				text := readOutput()
				if ctx.Err() != nil || waitErr != nil || !strings.Contains(text, "you> ") || !strings.Contains(text, "[READY]") {
					t.Fatalf("explicit interactive mode did not accept /quit: %v, %v\n%s", ctx.Err(), waitErr, text)
				}
				return
			}

			ready := regexp.MustCompile(`\[READY\].* running on (http://127\.0\.0\.1:\d+)`)
			client := &http.Client{Timeout: time.Second}
			for {
				text := readOutput()
				if strings.Contains(text, "Interactive chat") {
					t.Fatalf("non-interactive startup entered the EOF-sensitive REPL:\n%s", text)
				}
				if match := ready.FindStringSubmatch(text); match != nil {
					resp, err := client.Get(match[1] + "/readyz")
					if err == nil {
						_, _ = io.Copy(io.Discard, resp.Body)
						_ = resp.Body.Close()
						if resp.StatusCode == http.StatusOK {
							break
						}
					}
				}
				select {
				case <-done:
					t.Fatalf("ready server exited with non-interactive stdin: %v\n%s", waitErr, readOutput())
				case <-ctx.Done():
					t.Fatalf("ready server unavailable: %v\n%s", ctx.Err(), readOutput())
				case <-time.After(10 * time.Millisecond):
				}
			}
			// READY must outlive the input source, rather than race the REPL's
			// EOF-triggered shutdown for a single successful HTTP request.
			select {
			case <-done:
				t.Fatalf("ready server exited after startup: %v\n%s", waitErr, readOutput())
			case <-time.After(100 * time.Millisecond):
			}
			if runtime.GOOS == "windows" {
				_ = cmd.Process.Kill()
			} else if err := cmd.Process.Signal(os.Interrupt); err != nil {
				t.Fatal(err)
			}
			<-done
			if ctx.Err() != nil || (runtime.GOOS != "windows" && waitErr != nil) {
				t.Fatalf("server did not shut down cleanly: %v, %v\n%s", ctx.Err(), waitErr, readOutput())
			}
		})
	}
}
