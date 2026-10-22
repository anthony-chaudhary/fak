package compute

import (
	"errors"
	"reflect"
	"testing"
)

type counterReaderTestBackend struct {
	observationTestBackend
	counters BackendCounterSnapshot
	err      error
	reads    int
	live     bool
}

func (b *counterReaderTestBackend) BackendCounters() (BackendCounterSnapshot, error) {
	b.reads++
	return b.counters, b.err
}

type gatedCounterReaderTestBackend struct{ *counterReaderTestBackend }

type submitCounterReaderTestBackend struct {
	*counterReaderTestBackend
	submits uint64
}

func (b *submitCounterReaderTestBackend) DeviceSubmits() uint64 { return b.submits }

func (b gatedCounterReaderTestBackend) DispatchCountsLive() bool { return b.live }

type scalarWorkCounterReaderTestBackend struct {
	*counterReaderTestBackend
	work  DeviceWork
	reads int
}

func (b *scalarWorkCounterReaderTestBackend) DeviceWorkCounters() DeviceWork {
	b.reads++
	return b.work
}

// fak-test:runtime fast est=10ms lane=default
func TestReadBackendCountersSeam(t *testing.T) {
	if got, ok, err := ReadBackendCounters(nil); ok || err != nil || !reflect.DeepEqual(got, BackendCounterRead{}) {
		t.Fatalf("nil backend: %+v %t %v", got, ok, err)
	}
	if got, ok, err := ReadBackendCounters(unsupportedObservationBackend{name: "plain"}); ok || err != nil || !reflect.DeepEqual(got, BackendCounterRead{}) {
		t.Fatalf("unsupported backend: %+v %t %v", got, ok, err)
	}

	want := BackendCounterSnapshot{ComputeDispatches: 5, DispatchSubmits: 2, D2HCount: 3, D2HBytes: 12}
	full := &observationTestBackend{name: "counter-snap", snapshots: []BackendExecutionSnapshot{{
		Identity: BackendRuntimeIdentity{Backend: "counter-snap", Device: "d", Driver: "v", Runtime: "r"},
		Counters: want,
	}}}
	if got, ok, err := ReadBackendCounters(full); !ok || err != nil || !reflect.DeepEqual(got, BackendCounterRead{Counters: want, DispatchCountsLive: true}) || full.calls != 1 {
		t.Fatalf("snapshot fallback: %+v %t %v calls=%d", got, ok, err, full.calls)
	}

	cheap := &counterReaderTestBackend{observationTestBackend: observationTestBackend{name: "counter-cheap"}, counters: want}
	scalars := DeviceWork{Dispatches: 7, ReadbackBytes: 12, SubmitWaitNS: 900}
	work := scalars
	work.DeviceIdleNS, work.SyncObserved, work.PipelinedSubmits = 40, true, 2
	work.SyncSites = []DeviceSyncSite{{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 1, WaitNS: 30, D2HBytes: 12}}
	t.Cleanup(RegisterDeviceWorkReader("counter-cheap", func() DeviceWork { return work }))
	wantRead := BackendCounterRead{Counters: want, DispatchCountsLive: true, Work: scalars, WorkObserved: true}
	wantRead.Work.Backend = "counter-cheap"
	if got, ok, err := ReadBackendCounters(cheap); !ok || err != nil || !reflect.DeepEqual(got, wantRead) || cheap.reads != 1 || cheap.calls != 0 {
		t.Fatalf("cheap reader: %+v %t %v reads=%d snapshots=%d", got, ok, err, cheap.reads, cheap.calls)
	}
	gated := gatedCounterReaderTestBackend{cheap}
	for _, live := range []bool{false, true} {
		cheap.live = live
		if got, ok, err := ReadBackendCounters(gated); !ok || err != nil || got.DispatchCountsLive != live {
			t.Fatalf("dispatch gate live=%t: %+v %t %v", live, got, ok, err)
		}
	}
	submitting := &submitCounterReaderTestBackend{counterReaderTestBackend: cheap, submits: 11}
	wantSubmits := wantRead
	wantSubmits.Submits, wantSubmits.SubmitsObserved = 11, true
	if got, ok, err := ReadBackendCounters(submitting); !ok || err != nil || !reflect.DeepEqual(got, wantSubmits) {
		t.Fatalf("always-on submit count: %+v %t %v", got, ok, err)
	}
	if allocs := testing.AllocsPerRun(100, func() { _, _, _ = ReadBackendCounters(submitting) }); allocs != 0 {
		t.Fatalf("submit count read allocates %.1f per read", allocs)
	}
	scalarSeam := &scalarWorkCounterReaderTestBackend{counterReaderTestBackend: cheap, work: DeviceWork{Dispatches: 9, SubmitWaitNS: 1000, SyncSites: work.SyncSites}}
	wantScalar := wantRead
	wantScalar.Work = DeviceWork{Backend: "counter-cheap", Dispatches: 9, SubmitWaitNS: 1000}
	if got, ok, err := ReadBackendCounters(scalarSeam); !ok || err != nil || !reflect.DeepEqual(got, wantScalar) || scalarSeam.reads != 1 {
		t.Fatalf("scalar device-work seam: %+v %t %v reads=%d", got, ok, err, scalarSeam.reads)
	}
	unregistered := &scalarWorkCounterReaderTestBackend{counterReaderTestBackend: &counterReaderTestBackend{observationTestBackend: observationTestBackend{name: "counter-scalar-unregistered"}}}
	if got, ok, err := ReadBackendCounters(unregistered); !ok || err != nil || got.WorkObserved || !reflect.DeepEqual(got.Work, DeviceWork{}) || unregistered.reads != 0 {
		t.Fatalf("unregistered scalar seam: %+v %t %v reads=%d", got, ok, err, unregistered.reads)
	}
	if allocs := testing.AllocsPerRun(100, func() { _, _, _ = ReadBackendCounters(scalarSeam) }); allocs != 0 {
		t.Fatalf("scalar device-work read allocates %.1f per read", allocs)
	}
	cheap.err = ErrBackendCounterWrap
	if got, ok, err := ReadBackendCounters(cheap); ok || !errors.Is(err, ErrBackendCounterWrap) || !reflect.DeepEqual(got, BackendCounterRead{}) {
		t.Fatalf("cheap reader failure: %+v %t %v", got, ok, err)
	}
	if got, ok := ReadDeviceWork("counter-unregistered"); ok || !reflect.DeepEqual(got, DeviceWork{}) {
		t.Fatalf("unregistered device work: %+v %t", got, ok)
	}
	cheap.err = nil
	if allocs := testing.AllocsPerRun(100, func() { _, _, _ = ReadBackendCounters(cheap) }); allocs != 0 {
		t.Fatalf("cheap counter read allocates %.1f per read", allocs)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBackendCounterArithmeticCoversEveryField(t *testing.T) {
	var a, b, sum BackendCounterSnapshot
	av, bv, sv := reflect.ValueOf(&a).Elem(), reflect.ValueOf(&b).Elem(), reflect.ValueOf(&sum).Elem()
	for i := 0; i < av.NumField(); i++ {
		if av.Field(i).Kind() != reflect.Uint64 {
			t.Fatalf("field %s is not a cumulative uint64 counter", av.Type().Field(i).Name)
		}
		av.Field(i).SetUint(uint64(i + 1))
		bv.Field(i).SetUint(uint64(100 * (i + 1)))
		sv.Field(i).SetUint(uint64(101 * (i + 1)))
	}
	if got := AddBackendCounters(a, b); got != sum {
		t.Fatalf("add omitted a field:\n got=%+v\nwant=%+v", got, sum)
	}
	if got, err := SubBackendCounters(sum, a); err != nil || got != b {
		t.Fatalf("sub: %+v %v", got, err)
	}
	if got, err := SubBackendCounters(a, sum); !errors.Is(err, ErrBackendCounterWrap) || got != (BackendCounterSnapshot{}) {
		t.Fatalf("wrap: %+v %v", got, err)
	}
	if allocs := testing.AllocsPerRun(100, func() {
		d, _ := SubBackendCounters(sum, a)
		sum = AddBackendCounters(sum, d)
		sum, _ = SubBackendCounters(sum, d)
	}); allocs != 0 {
		t.Fatalf("counter arithmetic allocates %.1f per run", allocs)
	}
}
