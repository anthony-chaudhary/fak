package vcachesnapshot

import "testing"

func TestCompareLocalKeepsSnapshotAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native bounded fsynced JSONL snapshot", "native"}, {"unbounded append-only JSONL", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"SQLite WAL", "external"}, {"Prometheus TSDB", "external"}, {"ClickHouse", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.InputRows != 5 || a.InputTokens != 1500 || a.StorageBytes <= 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.WriteLatency != 0 || a.ReadLatency != 0 || a.InputRows != 0 || a.RetainedRows != 0 || a.DroppedRows != 0 || a.InputTokens != 0 || a.StorageBytes != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].RetainedRows != 3 || got.Arms[0].DroppedRows != 2 || got.Arms[1].Correct || got.Arms[1].RetainedRows != 5 {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkWriteReadBoundedSnapshot(b *testing.B) {
	turns := boundWindow(snapshotFixture(), 3)
	path := b.TempDir() + "/snapshot.jsonl"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Write(path, turns); err != nil {
			b.Fatal(err)
		}
		if _, ok, err := Read(path); err != nil || !ok {
			b.Fatalf("read ok=%v err=%v", ok, err)
		}
	}
}
