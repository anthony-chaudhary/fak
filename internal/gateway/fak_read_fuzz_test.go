package gateway

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func FuzzFakReadPayloadRoundTrip(f *testing.F) {
	f.Add("plain.txt", []byte("hello"))
	f.Add("space name.txt", []byte{})
	f.Add("binary.bin", []byte{0, 0xff, 1})
	f.Fuzz(func(t *testing.T, name string, data []byte) {
		if name == "" || strings.ContainsAny(name, `/\\\x00`) {
			t.Skip()
		}
		abi.ResetForTest()
		abi.RegisterRegionBackend(inlineBackend{})
		abi.RegisterAdjudicator(0, readAdj{})
		root := t.TempDir()
		agent.RegisterReadEngine(root)
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Skip()
		}
		srv, err := New(Config{EngineID: "fakread", Model: "m"})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(srv.Close)
		_, env, err := srv.fakRead(context.Background(), name, "fuzz-read", "")
		if err != nil {
			t.Fatal(err)
		}
		body := decodeFakReadPayload(t, env)
		got := []byte(body["content"].(string))
		if encoded, ok := body["content_base64"].(string); ok {
			got, err = base64.StdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatal(err)
			}
		}
		if string(got) != string(data) {
			t.Fatalf("returned bytes differ: got=%d want=%d", len(got), len(data))
		}
		receipt := receiptOf(t, body)
		if receipt["outcome"] != "executed_cold_read" || receipt["bytes"] != float64(len(data)) || receipt["freshness_verified"] != true {
			t.Fatalf("receipt=%v", receipt)
		}
	})
}
