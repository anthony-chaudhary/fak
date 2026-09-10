package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

const upBackendAutoChildEnv = "FAK_TEST_UP_BACKEND_AUTO_CHILD"

func TestUpBackendAutoVulkanResidencyLifetime(t *testing.T) {
	tests := []struct {
		name       string
		action     string
		envBackend string
	}{
		{name: "environment Vulkan admits before model load", action: "busy", envBackend: "vulkan"},
		{name: "startup failure releases residency", action: "startup-error", envBackend: "vulkan"},
		{name: "successful shutdown releases residency", action: "shutdown", envBackend: "vulkan"},
		{name: "failed shutdown retains residency", action: "failed-shutdown", envBackend: "vulkan"},
		{name: "idle close releases residency", action: "close", envBackend: "vulkan"},
		{name: "active close retains until handler exits", action: "active-close", envBackend: "vulkan"},
	}
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		tests = append(tests,
			struct {
				name       string
				action     string
				envBackend string
			}{name: "automatic Vulkan admits before model load", action: "busy", envBackend: "auto"},
		)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestUpBackendAutoChild$")
			cmd.Env = upBackendAutoChildEnvironment(map[string]string{
				upBackendAutoChildEnv:             "1",
				"FAK_TEST_UP_BACKEND_AUTO_ACTION": tt.action,
				"FAK_GPU_LEASE":                   filepath.Join(t.TempDir(), "gpu.lease"),
				"FAK_BACKEND":                     tt.envBackend,
				"FAK_VULKAN_SPIRV":                "",
			})
			if out, err := cmd.CombinedOutput(); err != nil {
				if ctx.Err() != nil {
					t.Fatalf("child turnkey residency witness timed out: %v", ctx.Err())
				}
				t.Fatalf("child turnkey residency witness failed: %v\n%s", err, out)
			}
		})
	}
}

func TestUpBackendAutoChild(t *testing.T) {
	if os.Getenv(upBackendAutoChildEnv) != "1" {
		return
	}

	// Registration is confined to this short-lived child. The embedded CPU
	// backend is a resolver sentinel and does not represent a physical device.
	compute.Register(&serveBackendAutoSentinel{Backend: compute.Default(), name: "vulkan"})
	action := os.Getenv("FAK_TEST_UP_BACKEND_AUTO_ACTION")
	modelPath := createUpBackendAutoGGUF(t, action != "startup-error")
	plan := macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: modelPath}, ContextBudgetTokens: 128}
	leasePath := os.Getenv("FAK_GPU_LEASE")

	start := func() *turnkeyServer {
		t.Helper()
		server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false)
		if err != nil {
			t.Fatalf("start turnkey server: %v", err)
		}
		return server
	}
	assertLeaseBusy := func(reason string) {
		t.Helper()
		competing, err := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
		if !errors.Is(err, gpulease.ErrBusy) {
			if err == nil {
				competing.Release()
			}
			t.Fatalf("lease after %s = %v, want busy", reason, err)
		}
	}
	assertLeaseAvailable := func(reason string) {
		t.Helper()
		lease, err := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
		if err != nil {
			t.Fatalf("acquire lease after %s: %v", reason, err)
		}
		lease.Release()
	}

	switch action {
	case "busy":
		held, err := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
		if err != nil {
			t.Fatalf("hold lease before turnkey start: %v", err)
		}
		defer held.Release()
		server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false)
		if server != nil {
			_ = server.Close()
			t.Fatal("turnkey server started while Vulkan residency lease was held")
		}
		if err == nil || !strings.Contains(err.Error(), "residency admission refused before model load") {
			t.Fatalf("start error = %v, want admission refusal before model load", err)
		}
	case "startup-error":
		server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false)
		if server != nil {
			_ = server.Close()
			t.Fatal("turnkey server unexpectedly accepted model without tokenizer")
		}
		if err == nil || !strings.Contains(err.Error(), "has no usable tokenizer") {
			t.Fatalf("start error = %v, want tokenizer startup failure", err)
		}
		assertLeaseAvailable("turnkey startup error")
	case "shutdown":
		server := start()
		assertLeaseBusy("turnkey startup")
		if err := server.Shutdown(context.Background()); err != nil {
			t.Fatalf("shutdown turnkey server: %v", err)
		}
		assertLeaseAvailable("successful graceful shutdown")
	case "failed-shutdown":
		server := start()
		conn, err := net.Dial("tcp", server.Addr())
		if err != nil {
			t.Fatalf("open active connection: %v", err)
		}
		defer conn.Close()
		if _, err := conn.Write([]byte("POST /v1/chat/completions HTTP/1.1\r\nHost: fak\r\nContent-Length: 100\r\nExpect: 100-continue\r\n\r\n")); err != nil {
			t.Fatalf("write request headers: %v", err)
		}
		if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
			t.Fatalf("set read deadline: %v", err)
		}
		reader := bufio.NewReader(conn)
		if line, err := reader.ReadString('\n'); err != nil || !strings.Contains(line, "100 Continue") {
			t.Fatalf("interim response = %q, %v; want 100 Continue", line, err)
		}
		if line, err := reader.ReadString('\n'); err != nil || line != "\r\n" {
			t.Fatalf("interim response terminator = %q, %v; want blank line", line, err)
		}
		if err := conn.SetReadDeadline(time.Time{}); err != nil {
			t.Fatalf("clear read deadline: %v", err)
		}
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("shutdown error = %v, want deadline exceeded", err)
		}
		assertLeaseBusy("failed graceful shutdown")
		if err := conn.Close(); err != nil {
			t.Fatalf("close active connection: %v", err)
		}
		finishCtx, finishCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer finishCancel()
		if err := server.Shutdown(finishCtx); err != nil {
			t.Fatalf("finish graceful shutdown: %v", err)
		}
		assertLeaseAvailable("successful retry of graceful shutdown")
	case "close":
		server := start()
		assertLeaseBusy("turnkey startup")
		if err := server.Close(); err != nil {
			t.Fatalf("close turnkey server: %v", err)
		}
		assertLeaseAvailable("idle server close")
	case "active-close":
		server := start()
		body := newUpBackendAutoBlockingBody()
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		requestDone := make(chan struct{})
		go func() {
			defer close(requestDone)
			server.handleChatCompletions(httptest.NewRecorder(), request)
		}()
		select {
		case <-body.entered:
		case <-time.After(5 * time.Second):
			t.Fatal("chat handler did not begin reading request")
		}
		if err := server.Close(); err != nil {
			t.Fatalf("close turnkey server with active handler: %v", err)
		}
		assertLeaseBusy("forced close with active handler")
		if err := server.Shutdown(context.Background()); err != nil {
			t.Fatalf("shutdown after forced close: %v", err)
		}
		assertLeaseBusy("shutdown while direct handler remains active")
		body.unblock()
		select {
		case <-requestDone:
		case <-time.After(5 * time.Second):
			t.Fatal("chat handler did not exit after body unblocked")
		}
		assertLeaseAvailable("active handler exit")
	default:
		t.Fatalf("unknown child action %q", action)
	}
}

type upBackendAutoBlockingBody struct {
	entered   chan struct{}
	release   chan struct{}
	enterOnce sync.Once
	closeOnce sync.Once
}

func newUpBackendAutoBlockingBody() *upBackendAutoBlockingBody {
	return &upBackendAutoBlockingBody{entered: make(chan struct{}), release: make(chan struct{})}
}

func (b *upBackendAutoBlockingBody) Read([]byte) (int, error) {
	b.enterOnce.Do(func() { close(b.entered) })
	<-b.release
	return 0, io.EOF
}

func (b *upBackendAutoBlockingBody) Close() error {
	b.unblock()
	return nil
}

func (b *upBackendAutoBlockingBody) unblock() {
	b.closeOnce.Do(func() { close(b.release) })
}

func createUpBackendAutoGGUF(t *testing.T, withTokenizer bool) string {
	t.Helper()
	const dim = 256
	const blockBytes = 69632
	kvs := uint64(9)
	if withTokenizer {
		kvs += 3
	}
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 2, kvs)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", dim)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", dim)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", dim)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	if withTokenizer {
		writeKVStringForTest(&b, "tokenizer.ggml.model", "gpt2")
		writeUpBackendAutoStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b", "ab"})
		writeUpBackendAutoStringArray(&b, "tokenizer.ggml.merges", []string{"a b"})
	}
	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ8_0), 0)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ8_0), blockBytes)
	padToAlignmentForTest(&b, 32)
	b.Write(bytes.Repeat([]byte{0}, 2*blockBytes))
	path := filepath.Join(t.TempDir(), "turnkey-q8.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatalf("write GGUF fixture: %v", err)
	}
	return path
}

func writeUpBackendAutoStringArray(b *bytes.Buffer, key string, values []string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeString))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, value := range values {
		writeStringForTest(b, value)
	}
}

func upBackendAutoChildEnvironment(overrides map[string]string) []string {
	env := make([]string, 0, len(os.Environ())+len(overrides))
	for _, entry := range os.Environ() {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		replaced := strings.HasPrefix(strings.ToUpper(key), "FAK_TEST_UP_BACKEND_AUTO_")
		for override := range overrides {
			if strings.EqualFold(key, override) {
				replaced = true
				break
			}
		}
		if replaced {
			continue
		}
		env = append(env, entry)
	}
	for key, value := range overrides {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}
	return env
}
