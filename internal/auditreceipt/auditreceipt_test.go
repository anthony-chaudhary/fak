package auditreceipt

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestExporterBuffersRedactsAndReplaysAfterRecovery(t *testing.T) {
	var down atomic.Bool
	down.Store(true)
	var mu sync.Mutex
	var got []Receipt
	var wire []byte
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(b)
		if down.Load() {
			http.Error(w, "down", 503)
			return
		}
		var rec Receipt
		if err := json.Unmarshal(b, &rec); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		mu.Lock()
		wire = append(wire, b...)
		got = append(got, rec)
		mu.Unlock()
		w.WriteHeader(204)
	}))
	defer sink.Close()
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	e, err := New(Config{Endpoint: sink.URL, DeviceID: "dev-secret", BufferPath: path, Capacity: 64, Timeout: 20 * time.Millisecond, RedactFields: []string{"device_id"}})
	if err != nil {
		t.Fatal(err)
	}
	secret := "raw-super-secret-argument"
	start := time.Now()
	if !e.Emit("search_kb", "ALLOW", "POLICY_ALLOW", map[string]int64{}) || !e.Emit("refund_payment", "DENY", "POLICY_BLOCK", map[string]int64{}) {
		t.Fatal("admission refused")
	}
	if time.Since(start) > 10*time.Millisecond {
		t.Fatal("sink latency reached adjudication admission")
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if e.Stats().Buffered >= 2 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	disk, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if contains(disk, []byte(secret)) {
		t.Fatal("raw secret reached buffer")
	}
	if contains(disk, []byte("dev-secret")) {
		t.Fatal("redacted identity reached buffer")
	}
	down.Store(false)
	deadline = time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(got)
		mu.Unlock()
		if n == 2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err = e.Close(ctx); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("got %d receipts, want 2", len(got))
	}
	if contains(wire, []byte(secret)) || contains(wire, []byte("dev-secret")) {
		t.Fatal("private value reached HTTP sink")
	}
	if got[0].Schema != Schema || got[0].DeviceID != "[REDACTED]" {
		t.Fatalf("receipt=%+v", got[0])
	}
}
func contains(hay, needle []byte) bool {
	for i := 0; i+len(needle) <= len(hay); i++ {
		ok := true
		for j := range needle {
			if hay[i+j] != needle[j] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}
func TestReceiptHasNoPayloadOrArgumentsField(t *testing.T) {
	typ := reflect.TypeOf(Receipt{})
	for _, bad := range []string{"Args", "Arguments", "Payload", "Secret"} {
		if _, ok := typ.FieldByName(bad); ok {
			t.Fatalf("receipt exposes %s", bad)
		}
	}
}
