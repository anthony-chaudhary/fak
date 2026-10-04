package model

import (
	"reflect"
	"testing"
)

// Tests for the owner->host synchronization API added with the restored-prefix
// seeding (#12742); the shared doubles live in qwen35_gdn_restored_seed_test.go.

// Finalize writes the validated owner state into the host cache before
// promotion, so a promotion (seed) failure leaves a correct host cache, and a
// snapshot failure records sequenceFailure so later reads fail closed.
func TestQwen35GDNFinalizeNeverLeavesStaleHostState(t *testing.T) {
	t.Run("promotion seed failure", func(t *testing.T) {
		b := newStatefulGDNBackend()
		installStatefulGDNFactory(t, b)
		m, s := restoredHybridSession(t, 5)
		defer s.Close()
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		want := advanceOwners(s, b, 7)
		b.failSeed = true
		if _, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err == nil {
			t.Fatal("finalize with failing promotion seed reported success")
		}
		if _, selected := s.Qwen35GDNDecodePath(); selected || s.qwen35HAL.sequenceBackend != nil {
			t.Fatal("promotion failure retained owners")
		}
		assertHostLinearEquals(t, "host after promotion failure", m.Cfg, s.Cache, want)
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("host-authoritative snapshot after promotion failure: %v", err)
		}
		assertHostLinearEquals(t, "snapshot after promotion failure", m.Cfg, snap.Cache, want)
		snap.Close()
	})
	t.Run("owner read failure", func(t *testing.T) {
		b := newStatefulGDNBackend()
		installStatefulGDNFactory(t, b)
		_, s := restoredHybridSession(t, 5)
		defer s.Close()
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		for _, owner := range s.qwen35HAL.sequenceLayers {
			if owner.valid() {
				delete(b.state, owner)
				break
			}
		}
		if _, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err == nil {
			t.Fatal("finalize of an unreadable owner reported success")
		}
		if s.qwen35HAL.sequenceFailure == nil {
			t.Fatal("finalize read failure left no failure record")
		}
		if snap, err := s.PrefixSnapshot(); err == nil {
			snap.Close()
			t.Fatal("snapshot after finalize failure did not fail closed")
		}
		if err := s.SyncQwen35ResidentGDNStateToHost(); err == nil {
			t.Fatal("host sync after finalize failure did not fail closed")
		}
	})
}

// The P4 MTP round checkpoints GDN through its own state, so its target
// snapshot must not read resident owners; the generic snapshot still does.
func TestQwen35GDNHostOnlySnapshotSkipsOwnerReadback(t *testing.T) {
	b := newStatefulGDNBackend()
	installStatefulGDNFactory(t, b)
	m, s := restoredHybridSession(t, 5)
	defer s.Close()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	stale := s.Cache.Clone()
	want := advanceOwners(s, b, 9)

	reads := b.snapshots
	host, err := s.PrefixSnapshotHostOnly()
	if err != nil {
		t.Fatal(err)
	}
	if b.snapshots != reads {
		t.Fatalf("host-only snapshot read %d resident owners", b.snapshots-reads)
	}
	assertHostCacheUnchanged(t, "host-only snapshot", m.Cfg, stale, host.Cache)
	host.Close()

	tx, err := NewMTPTransactionWithTarget(s, MTPTransactionConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tx.BeginRound(); err != nil {
		t.Fatalf("MTP BeginRound: %v", err)
	}
	if b.snapshots != reads {
		t.Fatalf("MTP round checkpoint read %d resident owners", b.snapshots-reads)
	}
	tx.Close()

	full, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if b.snapshots == reads {
		t.Fatal("generic PrefixSnapshot skipped the live owner readback")
	}
	assertHostLinearEquals(t, "generic snapshot", m.Cfg, full.Cache, want)
	full.Close()
}

func TestQwen35GDNPrefixSnapshotCarriesLiveOwnerState(t *testing.T) {
	b := newStatefulGDNBackend()
	installStatefulGDNFactory(t, b)
	m, s := restoredHybridSession(t, 5)
	defer s.Close()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	owners := append([]Qwen35GDNAuxState(nil), s.qwen35HAL.sequenceLayers...)
	stale := s.Cache.Clone()

	// Sequence phase: owners advanced, host copy untouched.
	want := advanceOwners(s, b, 1)
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	assertHostLinearEquals(t, "sequence-phase snapshot", m.Cfg, snap.Cache, want)
	assertHostCacheUnchanged(t, "sequence-phase live host cache", m.Cfg, stale, s.Cache)
	if !reflect.DeepEqual(s.qwen35HAL.sequenceLayers, owners) || !s.qwen35HAL.sequenceAccepted {
		t.Fatal("snapshot changed live owner authority")
	}
	plain := m.NewSession()
	if err := snap.Restore(plain); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	assertHostLinearEquals(t, "restored plain session", m.Cfg, plain.Cache, want)
	plain.Close()

	// Finalize promotes AND writes the finalized state into the host cache, so a
	// bare host KV clone at the admission boundary carries it.
	want = advanceOwners(s, b, 2)
	if executed, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	if _, selected := s.Qwen35GDNDecodePath(); !selected {
		t.Fatal("finalize did not promote resident decode")
	}
	assertHostLinearEquals(t, "finalized host cache", m.Cfg, s.Cache, want)

	// Decode phase: owners advance again; snapshot and explicit sync both see it.
	want = advanceOwners(s, b, 3)
	snap, err = s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	assertHostLinearEquals(t, "decode-phase snapshot", m.Cfg, snap.Cache, want)
	snap.Close()
	if err := s.SyncQwen35ResidentGDNStateToHost(); err != nil {
		t.Fatal(err)
	}
	assertHostLinearEquals(t, "explicit host sync", m.Cfg, s.Cache, want)
	if _, selected := s.Qwen35GDNDecodePath(); !selected {
		t.Fatal("host sync released resident decode")
	}

	// Demotion before a batched host prefill hands state back and releases owners.
	want = advanceOwners(s, b, 4)
	if err := s.demoteQwen35ResidentGDNDecodeToHost(); err != nil {
		t.Fatal(err)
	}
	assertHostLinearEquals(t, "demoted host cache", m.Cfg, s.Cache, want)
	if _, selected := s.Qwen35GDNDecodePath(); selected {
		t.Fatal("demotion retained resident decode path")
	}
	for _, owner := range owners {
		if owner.valid() && b.freed[owner] != 1 {
			t.Fatalf("owner %#v free count=%d, want 1", owner, b.freed[owner])
		}
	}
}

func TestQwen35GDNPrefixSnapshotFailsClosedOnUnreadableOwner(t *testing.T) {
	b := newStatefulGDNBackend()
	installStatefulGDNFactory(t, b)
	_, s := restoredHybridSession(t, 5)
	defer s.Close()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	for _, owner := range s.qwen35HAL.sequenceLayers {
		if owner.valid() {
			delete(b.state, owner)
			break
		}
	}
	if snap, err := s.PrefixSnapshot(); err == nil {
		snap.Close()
		t.Fatal("snapshot of an unreadable resident owner published a stale prefix")
	}
	if err := s.SyncQwen35ResidentGDNStateToHost(); err == nil {
		t.Fatal("host sync of an unreadable resident owner reported success")
	}
}
