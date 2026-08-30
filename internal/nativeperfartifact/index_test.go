package nativeperfartifact

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	keyA = "npc1_0123456789abcdef0123456789abcdef"
	keyB = "npc1_fedcba9876543210fedcba9876543210"
)

func ready(kind Kind, locator string, expires time.Time) Artifact {
	return Artifact{Kind: kind, Locator: locator, SHA256: strings.Repeat("a", 64), ExpiresAt: expires, State: StateReady}
}

func TestResolveExactCorrelationAndKinds(t *testing.T) {
	now := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	index, err := NewIndex(2)
	if err != nil {
		t.Fatal(err)
	}
	artifacts := []Artifact{
		ready(KindReceipt, "https://artifacts.example.test/native/receipt.json", now.Add(time.Hour)),
		ready(KindMetalProfile, "https://artifacts.example.test/native/metal.tgz", now.Add(time.Hour)),
		ready(KindCUDAProfile, "https://artifacts.example.test/native/cuda.tgz", now.Add(time.Hour)),
		ready(KindKernelTrace, "https://artifacts.example.test/native/kernel-trace.json", now.Add(time.Hour)),
		ready(KindComparisonReport, "https://artifacts.example.test/native/comparison.html", now.Add(time.Hour)),
	}
	if err := index.Add(Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: artifacts}); err != nil {
		t.Fatal(err)
	}
	got, err := index.Resolve(keyA, KindKernelTrace, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Locator != artifacts[3].Locator {
		t.Fatalf("locator = %q, want %q", got.Locator, artifacts[3].Locator)
	}
	if _, err := index.Resolve(keyB, KindKernelTrace, now); !errors.Is(err, ErrNotFound) {
		t.Fatalf("wrong key error = %v", err)
	}
	if _, err := index.Resolve(keyA, Kind("unknown"), now); !errors.Is(err, ErrNotFound) {
		t.Fatalf("wrong kind error = %v", err)
	}
}

func TestResolveFailureStates(t *testing.T) {
	now := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name     string
		artifact Artifact
		want     error
	}{
		{"broken", Artifact{Kind: KindReceipt, State: StateBroken}, ErrBroken},
		{"redacted", Artifact{Kind: KindReceipt, State: StateRedacted}, ErrRedacted},
		{"expired", ready(KindReceipt, "https://artifacts.example.test/native/receipt.json", now), ErrExpired},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			index, _ := NewIndex(1)
			if err := index.Add(Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{test.artifact}}); err != nil {
				t.Fatal(err)
			}
			if _, err := index.Resolve(keyA, KindReceipt, now); !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestAddRejectsUnsafeLocatorsAndEngineFallback(t *testing.T) {
	tests := []struct {
		name   string
		record Record
		want   error
	}{
		{"private credentials", Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "https://user:secret@artifacts.example.test/r.json", time.Time{})}}, ErrPrivate},
		{"private path", Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "https://artifacts.example.test/private/run/r.json", time.Time{})}}, ErrPrivate},
		{"private address", Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "https://10.0.0.8/r.json", time.Time{})}}, ErrPrivate},
		{"untrusted scheme", Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "file:///tmp/r.json", time.Time{})}}, ErrUntrustedScheme},
		{"llama fallback", Record{CorrelationKey: keyA, Engine: "llama.cpp", Artifacts: []Artifact{ready(KindReceipt, "https://artifacts.example.test/r.json", time.Time{})}}, errors.New("engine")},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			index, _ := NewIndex(1)
			err := index.Add(test.record)
			if test.name == "llama fallback" {
				if err == nil || !strings.Contains(err.Error(), "fak-native") {
					t.Fatalf("error = %v", err)
				}
				return
			}
			if !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestIndexIsBoundedAndSnapshotIsDetached(t *testing.T) {
	index, _ := NewIndex(1)
	for _, key := range []string{keyA, keyB} {
		if err := index.Add(Record{CorrelationKey: key, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "https://artifacts.example.test/r.json", time.Time{})}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := index.Resolve(keyA, KindReceipt, time.Time{}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("evicted key error = %v", err)
	}
	snapshot := index.Snapshot()
	if len(snapshot) != 1 || snapshot[0].CorrelationKey != keyB {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	snapshot[0].Artifacts[0].Locator = "mutated"
	got, err := index.Resolve(keyB, KindReceipt, time.Time{})
	if err != nil || got.Locator == "mutated" {
		t.Fatalf("stored artifact mutated: %#v, %v", got, err)
	}
}

func TestIndexRejectsCorrelationCollision(t *testing.T) {
	index, _ := NewIndex(2)
	first := Record{CorrelationKey: keyA, Engine: "fak-native", Artifacts: []Artifact{ready(KindReceipt, "https://artifacts.example.test/first.json", time.Time{})}}
	if err := index.Add(first); err != nil {
		t.Fatal(err)
	}
	if err := index.Add(first); err != nil {
		t.Fatalf("identical add must be idempotent: %v", err)
	}
	second := first
	second.Artifacts = []Artifact{ready(KindReceipt, "https://artifacts.example.test/second.json", time.Time{})}
	if err := index.Add(second); err == nil || !strings.Contains(err.Error(), "collision") {
		t.Fatalf("collision error = %v", err)
	}
}

func TestLiveArtifactSeriesProducesBoundedQwen38MetalMetrics(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	digest := strings.Repeat("a", 64)
	observation := LiveObservation{
		CompletedAt: now.Add(-30 * time.Second),
		ModuleRev:   "internal/nativeperfartifact@r3+g3e0ab906a",
		ModelFamily: "Qwen3.8",
		Backend:     "metal",
		Record: Record{
			CorrelationKey: keyA,
			Engine:         "fak-native",
			Artifacts: []Artifact{
				ready(KindMetalProfile, "https://artifacts.example.test/metal.json", now.Add(time.Hour)),
				ready(KindReceipt, "https://artifacts.example.test/receipt.json", now.Add(time.Hour)),
			},
		},
	}
	for n := range observation.Record.Artifacts {
		observation.Record.Artifacts[n].SHA256 = digest
	}

	got := ProduceLiveSeries(now, 5*time.Minute, observation)
	if got.Status != SeriesProduced || got.UnavailableReason != ReasonNone {
		t.Fatalf("series status = %q/%q, want %q/%q", got.Status, got.UnavailableReason, SeriesProduced, ReasonNone)
	}
	want := "# HELP fak_native_artifact_info Bounded public-safe native artifact evidence state.\n" +
		"# TYPE fak_native_artifact_info gauge\n" +
		`fak_native_artifact_info{engine="fak-native",correlation_key="` + keyA + `",kind="benchmark_receipt",state="ready"} 1` + "\n" +
		`fak_native_artifact_info{engine="fak-native",correlation_key="` + keyA + `",kind="metal_profile_bundle",state="ready"} 1` + "\n"
	if got.Prometheus != want {
		t.Fatalf("Prometheus output mismatch:\n--- got ---\n%s--- want ---\n%s", got.Prometheus, want)
	}
	if again := ProduceLiveSeries(now, 5*time.Minute, observation); again != got {
		t.Fatalf("producer is nondeterministic:\nfirst=%+v\nagain=%+v", got, again)
	}
	for _, forbidden := range []string{digest, "artifacts.example.test", observation.ModuleRev, observation.ModelFamily} {
		if strings.Contains(got.Prometheus, forbidden) {
			t.Fatalf("series leaked %q: %s", forbidden, got.Prometheus)
		}
	}
	series := 0
	for _, line := range strings.Split(got.Prometheus, "\n") {
		if !strings.HasPrefix(line, "fak_native_artifact_info{") {
			continue
		}
		series++
		if labels := strings.Count(line, "="); labels != 4 {
			t.Fatalf("series has %d labels, want 4: %s", labels, line)
		}
	}
	if series != len(observation.Record.Artifacts) || series > MaxArtifacts {
		t.Fatalf("series count = %d, artifacts = %d, maximum = %d", series, len(observation.Record.Artifacts), MaxArtifacts)
	}
}

func TestLiveArtifactSeriesFailsClosedWithTypedUnavailable(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	valid := LiveObservation{
		CompletedAt: now.Add(-30 * time.Second),
		ModuleRev:   "internal/nativeperfartifact@r3+g3e0ab906a",
		ModelFamily: "Qwen3.8",
		Backend:     "metal",
		Record: Record{
			CorrelationKey: keyA,
			Engine:         "fak-native",
			Artifacts: []Artifact{
				ready(KindReceipt, "https://artifacts.example.test/receipt.json", now.Add(time.Hour)),
			},
		},
	}

	tests := []struct {
		name   string
		mutate func(*LiveObservation)
		want   UnavailableReason
	}{
		{
			name: "incomplete identity",
			mutate: func(observation *LiveObservation) {
				observation.ModuleRev = ""
			},
			want: ReasonIncomplete,
		},
		{
			name: "stale observation",
			mutate: func(observation *LiveObservation) {
				observation.CompletedAt = now.Add(-6 * time.Minute)
			},
			want: ReasonStale,
		},
		{
			name: "expired artifact",
			mutate: func(observation *LiveObservation) {
				observation.Record.Artifacts[0].ExpiresAt = now
			},
			want: ReasonStale,
		},
		{
			name: "private locator",
			mutate: func(observation *LiveObservation) {
				observation.Record.Artifacts[0].Locator = "https://artifacts.example.test/private/operator/receipt.json"
			},
			want: ReasonPrivate,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			observation := valid
			observation.Record.Artifacts = append([]Artifact(nil), valid.Record.Artifacts...)
			test.mutate(&observation)
			got := ProduceLiveSeries(now, 5*time.Minute, observation)
			if got.Status != SeriesUnavailable || got.UnavailableReason != test.want {
				t.Fatalf("series status = %q/%q, want %q/%q", got.Status, got.UnavailableReason, SeriesUnavailable, test.want)
			}
			if got.Prometheus != "" {
				t.Fatalf("unavailable evidence emitted metrics: %s", got.Prometheus)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			for _, forbidden := range []string{observation.Record.Artifacts[0].Locator, observation.Record.Artifacts[0].SHA256, observation.ModuleRev} {
				if forbidden != "" && strings.Contains(string(encoded), forbidden) {
					t.Fatalf("unavailable result leaked %q: %s", forbidden, encoded)
				}
			}
		})
	}
}

func TestGrafanaDashboardContractAndFixture(t *testing.T) {
	root := filepath.Join("..", "..")
	contractPath := filepath.Join(root, "tools", "grafana", "provisioning", "contracts", "fak-native-artifacts.json")
	dashboardPath := filepath.Join(root, "tools", "grafana", "dashboards", "fak-native-artifacts.json")
	fixturePath := filepath.Join(root, "tools", "grafana", "provisioning", "fixtures", "fak-native-artifacts.prom")

	var contract struct {
		UID              string   `json:"uid"`
		Engine           string   `json:"engine"`
		RequiredQueries  []string `json:"required_queries"`
		RequiredDataLink string   `json:"required_data_link"`
		ArtifactKinds    []string `json:"artifact_kinds"`
		States           []string `json:"states"`
	}
	readJSON(t, contractPath, &contract)
	if contract.UID != "fak-native-artifacts" || contract.Engine != "fak-native" {
		t.Fatalf("contract identity = %q/%q", contract.UID, contract.Engine)
	}

	var dashboard map[string]any
	readJSON(t, dashboardPath, &dashboard)
	values := collectStrings(dashboard)
	for _, query := range contract.RequiredQueries {
		if containsString(values, query) {
			t.Errorf("live dashboard still queries fixture-only artifact family %q", query)
		}
	}
	if count := countString(values, contract.RequiredDataLink); count != 0 {
		t.Errorf("live dashboard exposes %d artifact data link(s) without a Prometheus artifact exporter", count)
	}
	for _, want := range []string{
		`sum by (backend, forward_path) (fak_native_receipt_requests_total{engine="inkernel",backend=~"$backend",forward_path=~"$forward_path"})`,
		`fak_native_receipt_latest_stale`,
		`Prometheus intentionally does not expose per-artifact correlation keys or locators`,
	} {
		if !anyStringContains(values, want) {
			t.Errorf("live dashboard missing honest receipt evidence %q", want)
		}
	}
	for _, forbidden := range []string{
		"fak_native_artifact_info",
		"https://artifacts.example.invalid/native/",
		"{{correlation_key}}",
		"{{kind}}",
	} {
		if anyStringContains(values, forbidden) {
			t.Errorf("live dashboard exposes unavailable artifact surface %q", forbidden)
		}
	}
	for _, forbidden := range []string{"llama.cpp", "file://", "localhost", "/Users/", "/home/", "token="} {
		if anyStringContains(values, forbidden) {
			t.Errorf("dashboard exposes forbidden material %q", forbidden)
		}
	}

	fixture, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	fixtureText := string(fixture)
	for _, kind := range contract.ArtifactKinds {
		if !strings.Contains(fixtureText, `kind="`+kind+`"`) {
			t.Errorf("fixture missing kind %q", kind)
		}
	}
	for _, state := range contract.States {
		if !strings.Contains(fixtureText, `state="`+state+`"`) {
			t.Errorf("fixture missing state %q", state)
		}
	}
	for _, line := range strings.Split(fixtureText, "\n") {
		if strings.HasPrefix(line, "fak_native_artifact_info{") && !strings.Contains(line, `engine="fak-native"`) {
			t.Errorf("fixture loses fak-native identity: %s", line)
		}
	}
}

func readJSON(t *testing.T, path string, target any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func collectStrings(value any) []string {
	var out []string
	var walk func(any)
	walk = func(item any) {
		switch typed := item.(type) {
		case string:
			out = append(out, typed)
		case []any:
			for _, child := range typed {
				walk(child)
			}
		case map[string]any:
			for _, child := range typed {
				walk(child)
			}
		}
	}
	walk(value)
	return out
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func countString(values []string, want string) int {
	count := 0
	for _, value := range values {
		if value == want {
			count++
		}
	}
	return count
}

func anyStringContains(values []string, want string) bool {
	for _, value := range values {
		if strings.Contains(value, want) {
			return true
		}
	}
	return false
}
