package harnessartifact

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessserve"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type fakeModelMutations struct {
	fail        ModelUpgradePhase
	calls       []string
	stopCount   int
	unloadCount int
}

func (f *fakeModelMutations) step(phase ModelUpgradePhase) error {
	f.calls = append(f.calls, string(phase))
	if f.fail == phase {
		return fmt.Errorf("forced %s failure", phase)
	}
	return nil
}

func (f *fakeModelMutations) Verify(_ context.Context, _ ModelUpgradePlan) error {
	return f.step(UpgradeVerify)
}
func (f *fakeModelMutations) Acquire(_ context.Context, plan ModelUpgradePlan) (string, error) {
	if err := f.step(UpgradeAcquire); err != nil {
		return "", err
	}
	return plan.Candidate.GGUFPath, nil
}
func (f *fakeModelMutations) Admit(_ context.Context, _ ModelUpgradePlan, _ string) (ModelAdmission, error) {
	if err := f.step(UpgradeAdmit); err != nil {
		return ModelAdmission{}, err
	}
	return ModelAdmission{ID: "admission-v2"}, nil
}
func (f *fakeModelMutations) Start(_ context.Context, _ ModelUpgradePlan, _ ModelAdmission) (ModelRuntime, error) {
	if err := f.step(UpgradeStart); err != nil {
		return ModelRuntime{}, err
	}
	return ModelRuntime{ID: "runtime-v2", Endpoint: "http://127.0.0.1:2"}, nil
}
func (f *fakeModelMutations) Probe(_ context.Context, _ ModelRuntime, maxTokens int) (harnessserve.ProbeReceipt, error) {
	if maxTokens != 1 {
		return harnessserve.ProbeReceipt{}, fmt.Errorf("max_tokens=%d", maxTokens)
	}
	if err := f.step(UpgradeProbe); err != nil {
		return harnessserve.ProbeReceipt{}, err
	}
	return harnessserve.ProbeReceipt{HTTPStatus: http.StatusOK, CompletionTokens: 1}, nil
}
func (f *fakeModelMutations) Stop(_ context.Context, _ ModelRuntime) error {
	f.stopCount++
	f.calls = append(f.calls, "stop")
	return nil
}
func (f *fakeModelMutations) Unload(_ context.Context, _ ModelAdmission) error {
	f.unloadCount++
	f.calls = append(f.calls, "unload")
	return nil
}

func TestModelUpgradeRuntimeHelper(t *testing.T) {
	if os.Getenv("FAK_MODEL_UPGRADE_HELPER") != "1" {
		return
	}
	address := os.Args[len(os.Args)-1]
	listener, err := net.Listen("tcp4", address)
	if err != nil {
		os.Exit(21)
	}
	mux := http.NewServeMux()
	server := &http.Server{Handler: mux}
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			MaxTokens int `json:"max_tokens"`
		}
		if json.NewDecoder(r.Body).Decode(&request) != nil || request.MaxTokens != 1 {
			http.Error(w, "one token required", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "v", "tokens": 1}}, "usage": map[string]int{"completion_tokens": 1}})
	})
	mux.HandleFunc("/shutdown", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		go server.Shutdown(context.Background())
	})
	if err := server.Serve(listener); !errors.Is(err, http.ErrServerClosed) {
		os.Exit(22)
	}
	os.Exit(0)
}

func TestModelUpgradeFailuresKeepProvenV1RunningAndPinned(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	supervisor := &harnessserve.Supervisor{}
	v1Serve := harnessserve.Plan{
		Executable: executable,
		Args:       []string{"-test.run=^TestModelUpgradeRuntimeHelper$", "--", "{address}"},
		Env:        append(os.Environ(), "FAK_MODEL_UPGRADE_HELPER=1"), Model: "v1",
		HealthPath: "/health", CompletionPath: "/v1/completions", GracefulShutdownPath: "/shutdown",
		StartupTimeout: 3 * time.Second, ProbeTimeout: time.Second, PollInterval: 10 * time.Millisecond,
		GracefulStopTimeout: time.Second, KillTimeout: time.Second,
	}
	v1Runtime, err := supervisor.Launch(context.Background(), v1Serve)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = supervisor.Stop(context.Background(), v1Runtime.Ownership) })

	dir := t.TempDir()
	v1Path := writeBlob(t, dir, "v1.gguf", []byte("fixed-v1-model"))
	v2Path := writeBlob(t, dir, "v2.gguf", []byte("candidate-v2-model"))
	v1Hash, _, _ := hashFile(v1Path)
	v2Hash, _, _ := hashFile(v2Path)
	v1Declaration := declaration(v1Path, v1Hash, "v1")
	statePath := filepath.Join(dir, "state.json")
	v1State := ModelUpgradeState{Schema: ModelUpgradeStateSchema, Active: ModelRevision{
		Declaration: v1Declaration,
		Receipt: ModelRuntimeReceipt{ID: "receipt-v1", DeclarationSHA256: declarationDigest(t, v1Declaration), BlobSHA256: v1Hash,
			RuntimeID: fmt.Sprintf("pid-%d", v1Runtime.Ownership.PID), AdmissionID: "admission-v1", Endpoint: v1Runtime.Endpoint,
			Ownership: v1Runtime.Ownership, Probe: v1Runtime.Probe, ReadyAt: v1Runtime.ReadyAt},
	}}
	if err := writeModelUpgradeStateAtomic(statePath, v1State); err != nil {
		t.Fatal(err)
	}
	plan, err := PrepareModelUpgrade(ModelUpgradeRequest{StatePath: statePath, Candidate: declaration(v2Path, v2Hash, "v2"), PinnedBlobBytes: int64(len("candidate-v2-model")), Serve: v1Serve})
	if err != nil {
		t.Fatal(err)
	}
	originalState, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}

	phases := []ModelUpgradePhase{UpgradeVerify, UpgradeAcquire, UpgradeAdmit, UpgradeStart, UpgradeProbe, UpgradePromote}
	for _, phase := range phases {
		t.Run(string(phase), func(t *testing.T) {
			mutations := &fakeModelMutations{fail: phase}
			opts := ModelUpgradeApplyOptions{Mutations: mutations, Now: func() time.Time { return time.Unix(42, 0) }}
			if phase == UpgradePromote {
				opts.Promote = func(string, ModelUpgradeState) error { return errors.New("forced atomic rename failure") }
			}
			_, err := ApplyModelUpgrade(context.Background(), plan, opts)
			var refusal *ModelUpgradeRefusal
			if !errors.As(err, &refusal) || refusal.Phase != phase {
				t.Fatalf("error=%v, want typed %s refusal", err, phase)
			}
			currentState, readErr := os.ReadFile(statePath)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if !bytes.Equal(currentState, originalState) {
				t.Fatalf("%s failure changed active state", phase)
			}
			if got, _, hashErr := hashFile(v1Path); hashErr != nil || got != v1Hash {
				t.Fatalf("%s failure changed v1 blob: hash=%s err=%v", phase, got, hashErr)
			}
			probeV1(t, v1Runtime.Endpoint)
			if (phase == UpgradeProbe || phase == UpgradePromote) && (mutations.stopCount != 1 || mutations.unloadCount != 1) {
				t.Fatalf("%s cleanup stop=%d unload=%d", phase, mutations.stopCount, mutations.unloadCount)
			}
		})
	}

	mutations := &fakeModelMutations{}
	upgraded, err := ApplyModelUpgrade(context.Background(), plan, ModelUpgradeApplyOptions{Mutations: mutations, Now: func() time.Time { return time.Unix(42, 0) }})
	if err != nil {
		t.Fatal(err)
	}
	if upgraded.Active.Declaration.ModelID != "v2" || upgraded.Rollback == nil || upgraded.Rollback.Receipt.ID != "receipt-v1" {
		t.Fatalf("upgrade did not atomically retain rollback: %+v", upgraded)
	}
	if upgraded.Active.Receipt.Probe.CompletionTokens != 1 || upgraded.Active.Receipt.BlobSHA256 != v2Hash {
		t.Fatalf("candidate receipt is not pinned/proven: %+v", upgraded.Active.Receipt)
	}
}

func TestModelCleanupPreviewExactlyMatchesFilesystemDeletion(t *testing.T) {
	cache := t.TempDir()
	v1 := writeBlob(t, cache, "v1.gguf", []byte("v1"))
	v2 := writeBlob(t, cache, "v2.gguf", []byte("v2"))
	orphan := writeBlob(t, cache, "orphan.gguf", []byte("orphan"))
	_ = os.WriteFile(filepath.Join(cache, "notes.txt"), []byte("keep"), 0o600)
	preview, err := PreviewModelCleanup(ModelCleanupRequest{Operation: CleanupPurge, CacheDir: cache, Referenced: []string{v1, v2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(preview.Delete) != 1 || preview.Delete[0].Path != orphan {
		t.Fatalf("preview=%+v", preview)
	}
	late := writeBlob(t, cache, "late.gguf", []byte("created-after-preview"))
	before := snapshotFiles(t, cache)
	receipt, err := ApplyModelCleanup(preview)
	if err != nil {
		t.Fatal(err)
	}
	after := snapshotFiles(t, cache)
	deleted := setDifference(before, after)
	want := []string{orphan}
	if !reflect.DeepEqual(deleted, want) || !reflect.DeepEqual(receipt.Deleted, want) {
		t.Fatalf("filesystem deletion=%v receipt=%v preview=%v", deleted, receipt.Deleted, preview.Delete)
	}
	for _, kept := range []string{v1, v2, late, filepath.Join(cache, "notes.txt")} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("kept path %s: %v", kept, err)
		}
	}
	if _, err := PreviewModelCleanup(ModelCleanupRequest{Operation: CleanupEvict, CacheDir: cache, Targets: []string{v1}, Referenced: []string{v1}}); err == nil {
		t.Fatal("evict accepted a referenced blob")
	}
}

func TestStopUnloadEvictAndPurgeRemainDistinct(t *testing.T) {
	mutations := &fakeModelMutations{}
	if err := StopModel(context.Background(), mutations, ModelRuntime{ID: "r"}); err != nil {
		t.Fatal(err)
	}
	if mutations.stopCount != 1 || mutations.unloadCount != 0 {
		t.Fatalf("stop counts: %+v", mutations)
	}
	if err := UnloadModel(context.Background(), mutations, ModelAdmission{ID: "a"}); err != nil {
		t.Fatal(err)
	}
	if mutations.stopCount != 1 || mutations.unloadCount != 1 {
		t.Fatalf("unload counts: %+v", mutations)
	}
	cache := t.TempDir()
	a := writeBlob(t, cache, "a.gguf", []byte("a"))
	b := writeBlob(t, cache, "b.gguf", []byte("b"))
	evict, err := PreviewModelCleanup(ModelCleanupRequest{Operation: CleanupEvict, CacheDir: cache, Targets: []string{a}})
	if err != nil || len(evict.Delete) != 1 || evict.Delete[0].Path != a {
		t.Fatalf("evict=%+v err=%v", evict, err)
	}
	purge, err := PreviewModelCleanup(ModelCleanupRequest{Operation: CleanupPurge, CacheDir: cache})
	if err != nil || len(purge.Delete) != 2 || purge.Delete[1].Path != b {
		t.Fatalf("purge=%+v err=%v", purge, err)
	}
}

func declaration(path, digest, id string) harnesskit.LocalModelDeclaration {
	return harnesskit.LocalModelDeclaration{Schema: LocalModelDeclarationSchema, ModelID: id, GGUFPath: path, GGUFSHA256: digest, Runtime: "fake-runtime", ContextTokens: 128}
}

func declarationDigest(t *testing.T, value harnesskit.LocalModelDeclaration) string {
	t.Helper()
	raw, err := CanonicalLocalModelDeclaration(value)
	if err != nil {
		t.Fatal(err)
	}
	return LocalModelDeclarationDigest(raw)
}

func writeBlob(t *testing.T, dir, name string, contents []byte) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func probeV1(t *testing.T, endpoint string) {
	t.Helper()
	body := strings.NewReader(`{"model":"v1","prompt":"p","max_tokens":1}`)
	resp, err := http.Post(endpoint+"/v1/completions", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &result); err != nil || result.Usage.CompletionTokens != 1 {
		t.Fatalf("v1 one-token probe status=%d body=%s err=%v", resp.StatusCode, raw, err)
	}
}

func snapshotFiles(t *testing.T, root string) []string {
	t.Helper()
	var paths []string
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(paths)
	return paths
}

func setDifference(before, after []string) []string {
	set := map[string]bool{}
	for _, path := range after {
		set[path] = true
	}
	var out []string
	for _, path := range before {
		if !set[path] {
			out = append(out, path)
		}
	}
	return out
}

func TestModelUpgradeHashFixtureIsIndependent(t *testing.T) {
	data := []byte("fixture")
	sum := sha256.Sum256(data)
	encoded := hex.EncodeToString(sum[:])
	decoded, err := hex.DecodeString(encoded)
	if err != nil || !bytes.Equal(decoded, sum[:]) {
		t.Fatalf("digest round trip: encoded=%q decoded=%x err=%v", encoded, decoded, err)
	}
}
