package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func mmapPtr[T any](v T) *T { return &v }

// sessRecs is one (session id, turns) pair in dataset order.
type sessRecs struct {
	id   string
	recs []TurnRecord
}

// buildStore writes a finalized payload_bytes store under dir with the given
// sessions in argument order and returns the opened client plus its store.
func buildStore(t *testing.T, dir string, sessions ...sessRecs) (*DatasetStore, *DatasetClient) {
	t.Helper()
	store, err := NewDatasetStore(dir, "run1", FormatPayloadBytes)
	if err != nil {
		t.Fatalf("NewDatasetStore: %v", err)
	}
	for _, s := range sessions {
		if err := store.AddSession(s.id, s.recs); err != nil {
			t.Fatalf("AddSession(%q): %v", s.id, err)
		}
	}
	if err := store.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	client, err := OpenDatasetClient(store.DataPath(), store.IndexPath())
	if err != nil {
		t.Fatalf("OpenDatasetClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return store, client
}

func TestMmapIndexOffsets(t *testing.T) {
	t.Run("AppendOnlyCumulativeOffsets", func(t *testing.T) {
		p1 := []byte(`{"model":"m","messages":[{"role":"user","content":"one"}],"max_tokens":11}`)
		p2 := []byte(`{"model":"m","messages":[{"role":"user","content":"two"}]}`)
		p3 := []byte(`{"model":"m","input":"three","max_output_tokens":33}`)
		_, client := buildStore(t, t.TempDir(),
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: p1}, {Payload: p2}}},
			sessRecs{id: "s2", recs: []TurnRecord{{Payload: p3}}},
		)
		idx := client.Index()
		if idx.Format != FormatPayloadBytes {
			t.Fatalf("format=%q want %q", idx.Format, FormatPayloadBytes)
		}
		if len(idx.ConversationIDs) != 2 || idx.ConversationIDs[0] != "s1" || idx.ConversationIDs[1] != "s2" {
			t.Fatalf("conversation_ids=%v want [s1 s2] (insertion order)", idx.ConversationIDs)
		}
		wantTotal := len(p1) + len(p2) + len(p3)
		if idx.TotalSize != wantTotal {
			t.Fatalf("total_size=%d want %d", idx.TotalSize, wantTotal)
		}
		po := idx.PayloadOffsets["s1"]
		if len(po) != 2 {
			t.Fatalf("s1 has %d payload offsets, want 2", len(po))
		}
		if po[0].Offset != 0 || po[0].Size != len(p1) {
			t.Fatalf("s1[0]={off:%d size:%d} want {off:0 size:%d}", po[0].Offset, po[0].Size, len(p1))
		}
		if po[1].Offset != len(p1) || po[1].Size != len(p2) {
			t.Fatalf("s1[1]={off:%d size:%d} want {off:%d size:%d}", po[1].Offset, po[1].Size, len(p1), len(p2))
		}
		if got := idx.PayloadOffsets["s2"][0]; got.Offset != len(p1)+len(p2) || got.Size != len(p3) {
			t.Fatalf("s2[0]={off:%d size:%d} want {off:%d size:%d}", got.Offset, got.Size, len(p1)+len(p2), len(p3))
		}
		// Schema parity with the upstream index: the conversation-format
		// offsets map stays present but empty on a payload_bytes dataset.
		if len(idx.Offsets) != 0 {
			t.Fatalf("offsets=%v want empty map", idx.Offsets)
		}
	})

	t.Run("VerbatimReplay", func(t *testing.T) {
		p1 := []byte(`{"model":"m","messages":[{"role":"user","content":"one"}],"max_tokens":11}`)
		p2 := []byte(`{"model":"m","messages":[{"role":"user","content":"two"}]}`)
		p3 := []byte(`{"model":"m","input":"three","max_output_tokens":33}`)
		_, client := buildStore(t, t.TempDir(),
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: p1}, {Payload: p2}}},
			sessRecs{id: "s2", recs: []TurnRecord{{Payload: p3}}},
		)
		cases := []struct {
			session string
			turn    int
			want    []byte
		}{
			{"s1", 0, p1}, {"s1", 1, p2}, {"s2", 0, p3},
		}
		for _, c := range cases {
			got, err := client.PayloadBytes(c.session, c.turn)
			if err != nil {
				t.Fatalf("PayloadBytes(%s,%d): %v", c.session, c.turn, err)
			}
			if !bytes.Equal(got, c.want) {
				t.Fatalf("PayloadBytes(%s,%d)=%q want %q", c.session, c.turn, got, c.want)
			}
		}
		// The mapping must serve the file's own bytes: slice dataset.dat
		// directly at the recorded offsets and compare.
		raw, err := os.ReadFile(client.DataPath())
		if err != nil {
			t.Fatalf("read data file: %v", err)
		}
		idx := client.Index()
		for _, c := range cases {
			pos := idx.PayloadOffsets[c.session][c.turn]
			if !bytes.Equal(raw[pos.Offset:pos.Offset+pos.Size], c.want) {
				t.Fatalf("file slice at %d:%d != payload bytes", pos.Offset, pos.Offset+pos.Size)
			}
		}
	})

	t.Run("SidecarScalarsRoundTrip", func(t *testing.T) {
		recs := []TurnRecord{
			// No explicit scalar: max_tokens must fall back to the wire key.
			{Payload: []byte(`{"max_completion_tokens":7}`)},
			// Full sidecar.
			{Payload: []byte(`{"max_tokens":5}`), Meta: PayloadMeta{
				MaxTokens:            mmapPtr(42),
				TimestampMS:          mmapPtr(1234.5),
				SourceKind:           mmapPtr("weka"),
				AudioDurationSeconds: mmapPtr(2.5),
				NumChoices:           3,
			}},
			// No cap anywhere: nil, with the default num_choices of 1.
			{Payload: []byte(`{}`)},
		}
		_, client := buildStore(t, t.TempDir(), sessRecs{id: "s", recs: recs})

		md0, err := client.PayloadTurn("s", 0)
		if err != nil {
			t.Fatalf("PayloadTurn(s,0): %v", err)
		}
		if md0.MaxTokens == nil || *md0.MaxTokens != 7 {
			t.Fatalf("wire fallback max_tokens=%v want 7", md0.MaxTokens)
		}
		if md0.NumChoices != 1 {
			t.Fatalf("default num_choices=%d want 1", md0.NumChoices)
		}

		md1, err := client.PayloadTurn("s", 1)
		if err != nil {
			t.Fatalf("PayloadTurn(s,1): %v", err)
		}
		if md1.MaxTokens == nil || *md1.MaxTokens != 42 {
			t.Fatalf("max_tokens=%v want 42", md1.MaxTokens)
		}
		if md1.TimestampMS == nil || *md1.TimestampMS != 1234.5 {
			t.Fatalf("timestamp=%v want 1234.5", md1.TimestampMS)
		}
		if md1.SourceKind == nil || *md1.SourceKind != "weka" {
			t.Fatalf("source_kind=%v want weka", md1.SourceKind)
		}
		if md1.AudioDurationSeconds == nil || *md1.AudioDurationSeconds != 2.5 {
			t.Fatalf("audio_duration=%v want 2.5", md1.AudioDurationSeconds)
		}
		if md1.NumChoices != 3 {
			t.Fatalf("num_choices=%d want 3", md1.NumChoices)
		}

		md2, err := client.PayloadTurn("s", 2)
		if err != nil {
			t.Fatalf("PayloadTurn(s,2): %v", err)
		}
		if md2.MaxTokens != nil {
			t.Fatalf("max_tokens=%v want nil", *md2.MaxTokens)
		}
		if md2.NumChoices != 1 {
			t.Fatalf("num_choices=%d want 1", md2.NumChoices)
		}
	})

	t.Run("Sentinels", func(t *testing.T) {
		store, client := buildStore(t, t.TempDir(),
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: []byte(`{"a":1}`)}}},
		)
		// ErrNotFound: unknown session, out-of-range turn, negative turn.
		if _, err := client.PayloadBytes("nope", 0); !errors.Is(err, ErrNotFound) {
			t.Fatalf("unknown session err=%v want ErrNotFound", err)
		}
		if _, err := client.PayloadBytes("s1", 5); !errors.Is(err, ErrNotFound) {
			t.Fatalf("out-of-range turn err=%v want ErrNotFound", err)
		}
		if _, err := client.PayloadTurn("s1", -1); !errors.Is(err, ErrNotFound) {
			t.Fatalf("negative turn err=%v want ErrNotFound", err)
		}
		// ErrAlreadyFinalized: add after finalize, double finalize.
		if err := store.AddSession("late", []TurnRecord{{Payload: []byte(`{}`)}}); !errors.Is(err, ErrAlreadyFinalized) {
			t.Fatalf("add-after-finalize err=%v want ErrAlreadyFinalized", err)
		}
		if err := store.Finalize(); !errors.Is(err, ErrAlreadyFinalized) {
			t.Fatalf("double finalize err=%v want ErrAlreadyFinalized", err)
		}
		// ErrDuplicateID: second add of the same session id.
		dupDir := t.TempDir()
		stDup, err := NewDatasetStore(dupDir, "dup", FormatPayloadBytes)
		if err != nil {
			t.Fatalf("NewDatasetStore: %v", err)
		}
		t.Cleanup(func() { _ = stDup.Close() })
		if err := stDup.AddSession("x", []TurnRecord{{Payload: []byte(`{}`)}}); err != nil {
			t.Fatalf("first AddSession(x): %v", err)
		}
		if err := stDup.AddSession("x", nil); !errors.Is(err, ErrDuplicateID) {
			t.Fatalf("duplicate AddSession(x) err=%v want ErrDuplicateID", err)
		}
		// ErrIndexMismatch: index total_size off by one from the data file.
		misDir := t.TempDir()
		_, misClient := buildStore(t, misDir,
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: []byte(`{"a":1}`)}}},
		)
		tamperIndex(t, misClient.IndexPath(), func(m map[string]any) {
			m["total_size"] = m["total_size"].(float64) + 1
		})
		if _, err := OpenDatasetClient(misClient.DataPath(), misClient.IndexPath()); !errors.Is(err, ErrIndexMismatch) {
			t.Fatalf("tampered total_size err=%v want ErrIndexMismatch", err)
		}
		// ErrCorruptIndex: unparseable index.
		corDir := t.TempDir()
		_, corClient := buildStore(t, corDir,
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: []byte(`{"a":1}`)}}},
		)
		if err := os.WriteFile(corClient.IndexPath(), []byte("{not json"), 0o644); err != nil {
			t.Fatalf("write garbage index: %v", err)
		}
		if _, err := OpenDatasetClient(corClient.DataPath(), corClient.IndexPath()); !errors.Is(err, ErrCorruptIndex) {
			t.Fatalf("garbage index err=%v want ErrCorruptIndex", err)
		}
		// ErrCorruptIndex: num_choices below the schema minimum.
		badDir := t.TempDir()
		_, badClient := buildStore(t, badDir,
			sessRecs{id: "s1", recs: []TurnRecord{{Payload: []byte(`{"a":1}`)}}},
		)
		tamperIndex(t, badClient.IndexPath(), func(m map[string]any) {
			turns := m["payload_offsets"].(map[string]any)["s1"].([]any)
			turns[0].(map[string]any)["num_choices"] = 0
		})
		if _, err := OpenDatasetClient(badClient.DataPath(), badClient.IndexPath()); !errors.Is(err, ErrCorruptIndex) {
			t.Fatalf("num_choices=0 err=%v want ErrCorruptIndex", err)
		}
		// Missing files report os.ErrNotExist.
		if _, err := OpenDatasetClient(filepath.Join(misDir, "missing.dat"), misClient.IndexPath()); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("missing data file err=%v want os.ErrNotExist", err)
		}
	})

	t.Run("EmptyStoreOpensWithoutMmap", func(t *testing.T) {
		dir := t.TempDir()
		store, err := NewDatasetStore(dir, "empty", FormatPayloadBytes)
		if err != nil {
			t.Fatalf("NewDatasetStore: %v", err)
		}
		if err := store.Finalize(); err != nil {
			t.Fatalf("Finalize: %v", err)
		}
		client, err := OpenDatasetClient(store.DataPath(), store.IndexPath())
		if err != nil {
			t.Fatalf("OpenDatasetClient(empty): %v", err)
		}
		idx := client.Index()
		if idx.TotalSize != 0 || len(idx.ConversationIDs) != 0 || len(idx.PayloadOffsets) != 0 {
			t.Fatalf("empty index={ids:%v total:%d payloads:%d} want all empty", idx.ConversationIDs, idx.TotalSize, len(idx.PayloadOffsets))
		}
		if _, err := client.PayloadBytes("s", 0); !errors.Is(err, ErrNotFound) {
			t.Fatalf("lookup on empty store err=%v want ErrNotFound", err)
		}
		if err := client.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if err := client.Close(); err != nil {
			t.Fatalf("second Close must be a no-op, got %v", err)
		}
	})

	t.Run("WireKeyClosedContracts", func(t *testing.T) {
		extract := []struct {
			name string
			body string
			want int
			ok   bool
		}{
			{"max_tokens", `{"max_tokens":5}`, 5, true},
			{"max_completion_tokens", `{"max_completion_tokens":9}`, 9, true},
			{"max_output_tokens", `{"max_output_tokens":3}`, 3, true},
			{"first key wins", `{"max_tokens":5,"max_completion_tokens":9,"max_output_tokens":3}`, 5, true},
			{"bool skipped", `{"max_tokens":true}`, 0, false},
			{"zero rejected", `{"max_tokens":0}`, 0, false},
			{"negative rejected", `{"max_tokens":-4}`, 0, false},
			{"non-integer rejected", `{"max_tokens":1.5}`, 0, false},
			{"string rejected", `{"max_tokens":"8"}`, 0, false},
			{"null skipped", `{"max_tokens":null}`, 0, false},
			{"absent", `{}`, 0, false},
			{"non-object", `[1,2]`, 0, false},
			{"garbage", `nope`, 0, false},
		}
		for _, c := range extract {
			got, ok := MaxTokensFromWirePayload([]byte(c.body))
			if got != c.want || ok != c.ok {
				t.Errorf("%s: MaxTokensFromWirePayload=%v/%v want %v/%v", c.name, got, ok, c.want, c.ok)
			}
		}

		apply := func(body string, value int) map[string]any {
			t.Helper()
			out, err := ApplyMaxTokensToWirePayload([]byte(body), value)
			if err != nil {
				t.Fatalf("ApplyMaxTokensToWirePayload(%s): %v", body, err)
			}
			var m map[string]any
			if err := json.Unmarshal(out, &m); err != nil {
				t.Fatalf("re-parse applied body: %v", err)
			}
			return m
		}
		// Every present max-token key is overwritten; other keys survive.
		m := apply(`{"messages":[],"max_tokens":5,"max_completion_tokens":9}`, 42)
		if m["max_tokens"] != float64(42) || m["max_completion_tokens"] != float64(42) {
			t.Fatalf("applied caps=%v want both 42", m)
		}
		if _, ok := m["messages"]; !ok {
			t.Fatalf("unrelated key dropped: %v", m)
		}
		// No cap present: responses dialect (keyed by "input") gets max_output_tokens.
		m = apply(`{"input":"x"}`, 7)
		if m["max_output_tokens"] != float64(7) {
			t.Fatalf("responses dialect=%v want max_output_tokens=7", m)
		}
		if _, present := m["max_tokens"]; present {
			t.Fatalf("responses dialect must not gain max_tokens: %v", m)
		}
		// No cap present: chat/completions dialect gets max_tokens.
		m = apply(`{"messages":[]}`, 7)
		if m["max_tokens"] != float64(7) {
			t.Fatalf("chat dialect=%v want max_tokens=7", m)
		}
		if _, err := ApplyMaxTokensToWirePayload([]byte(`[1]`), 1); err == nil {
			t.Fatal("non-object body must error")
		}
	})
}

// tamperIndex rewrites the index file at path with fn applied to its decoded
// object.
func tamperIndex(t *testing.T, path string, fn func(map[string]any)) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read index: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("decode index: %v", err)
	}
	fn(m)
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("encode tampered index: %v", err)
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write tampered index: %v", err)
	}
}
