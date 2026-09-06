package l3kv

import (
	"fmt"
	"sync"
	"testing"
)

func TestAdmissionFilterThrottlesFirstTouch(t *testing.T) {
	filter := NewTinyLFUFilter()

	digest := "span-ephemeral-1001"
	byteSize := 32 * 1024 // 32 KB

	// Pre-touch: 0 touches -> Admit must return false
	if filter.Admit(digest, byteSize) {
		t.Fatalf("expected untouched span to return Admit=false, got true")
	}

	// 1st touch: record touch
	filter.RecordTouch(digest)

	// 1st-touch spans must return Admit=false (throttled from SSD offload)
	if filter.Admit(digest, byteSize) {
		t.Fatalf("expected 1st-touch span to return Admit=false, got true")
	}

	// 2nd touch: record touch
	filter.RecordTouch(digest)

	// 2nd-touch spans must return Admit=true (promoted to SSD offload)
	if !filter.Admit(digest, byteSize) {
		t.Fatalf("expected 2nd-touch span to return Admit=true, got false")
	}

	// Subsequent touches must continue admitting
	filter.RecordTouch(digest)
	if !filter.Admit(digest, byteSize) {
		t.Fatalf("expected 3rd-touch span to return Admit=true, got false")
	}
}

func TestAdmissionFilterAgingAndDecay(t *testing.T) {
	// Small window capacity to trigger aging deterministically
	cfg := TinyLFUConfig{
		Width:     64,
		WindowCap: 20,
		Threshold: 2,
	}
	filter := NewTinyLFUFilter(cfg)

	hotKey := "hot-key-recurrent"
	coldKey := "cold-key-single"

	// Touch hotKey 4 times
	for i := 0; i < 4; i++ {
		filter.RecordTouch(hotKey)
	}
	// Touch coldKey once
	filter.RecordTouch(coldKey)

	// Hot key has 4 touches >= 2 -> Admit=true
	if !filter.Admit(hotKey, 1024) {
		t.Fatalf("expected hotKey to be admitted before decay")
	}
	// Cold key has 1 touch < 2 -> Admit=false
	if filter.Admit(coldKey, 1024) {
		t.Fatalf("expected coldKey to be throttled before decay")
	}

	estBefore := filter.Estimate(hotKey)
	if estBefore < 4 {
		t.Fatalf("expected hotKey estimate >= 4, got %d", estBefore)
	}

	// Add 15 dummy touches to reach 20 samples (triggering aging/decay)
	for i := 0; i < 15; i++ {
		filter.RecordTouch(fmt.Sprintf("churn-%d", i))
	}

	// Decay should have fired: counters halved
	// hotKey was >= 4, now halved to >= 2
	// coldKey was 1, 1 >> 1 = 0 (aged out completely)
	estAfter := filter.Estimate(hotKey)
	if estAfter != estBefore/2 {
		t.Fatalf("expected hotKey estimate to halve from %d to %d, got %d", estBefore, estBefore/2, estAfter)
	}

	coldEst := filter.Estimate(coldKey)
	if coldEst != 0 {
		t.Fatalf("expected coldKey to age out to 0, got %d", coldEst)
	}

	// Trigger second decay: 20 more touches
	for i := 0; i < 20; i++ {
		filter.RecordTouch(fmt.Sprintf("churn2-%d", i))
	}

	// hotKey counter halved again (2 >> 1 = 1) -> now < threshold 2 -> Admit=false
	if filter.Admit(hotKey, 1024) {
		t.Fatalf("expected stale hotKey to be throttled after second decay cycle")
	}
}

func TestAdmissionFilterFrequencyEstimationUnderChurn(t *testing.T) {
	// Standard configuration with 4096 width (16 KB state)
	filter := NewTinyLFUFilter()

	const ephemeralCount = 1000
	admittedEphemeral := 0

	// Ingest ephemeral spans (touched exactly once)
	for i := 0; i < ephemeralCount; i++ {
		d := fmt.Sprintf("ephemeral-span-%d", i)
		filter.RecordTouch(d)
		if filter.Admit(d, 4096) {
			admittedEphemeral++
		}
	}

	// 1st-touch ephemeral spans should achieve >= 10x write reduction (< 100 admitted out of 1000).
	// With 4096 width, collisions across all 4 independent hash seeds are near zero.
	if admittedEphemeral > 5 {
		t.Fatalf("too many ephemeral spans admitted: %d (expected <= 5 for 4096-width sketch, >= 200x write reduction)", admittedEphemeral)
	}

	// Recurrent spans touched multiple times must be admitted
	recurrentKey := "session-system-prompt-prefix"
	for i := 0; i < 5; i++ {
		filter.RecordTouch(recurrentKey)
	}
	if !filter.Admit(recurrentKey, 4096) {
		t.Fatalf("expected recurrent span to be admitted")
	}

	est := filter.Estimate(recurrentKey)
	if est < 5 {
		t.Fatalf("expected recurrent span estimate >= 5, got %d", est)
	}
}

func TestAdmissionFilterEdgeCasesAndConcurrency(t *testing.T) {
	filter := NewTinyLFUFilter()

	// Nil / empty checks
	if filter.Admit("", 1024) {
		t.Fatalf("expected empty digest Admit=false")
	}
	if filter.Admit("valid-digest", -1) {
		t.Fatalf("expected negative byteSize Admit=false")
	}
	filter.RecordTouch("") // should not panic

	var nilFilter *TinyLFUFilter
	if nilFilter.Admit("any", 1024) {
		t.Fatalf("expected nil filter Admit=false")
	}
	nilFilter.RecordTouch("any") // should not panic
	if nilFilter.Estimate("any") != 0 {
		t.Fatalf("expected nil filter Estimate=0")
	}

	// Test AlwaysAdmitFilter
	always := AlwaysAdmitFilter{}
	always.RecordTouch("anything")
	if !always.Admit("anything", 1024) {
		t.Fatalf("expected AlwaysAdmitFilter to admit")
	}

	// Reset test
	filter.RecordTouch("k1")
	filter.RecordTouch("k1")
	if !filter.Admit("k1", 1024) {
		t.Fatalf("expected k1 admitted")
	}
	filter.Reset()
	if filter.Admit("k1", 1024) {
		t.Fatalf("expected k1 throttled after Reset")
	}

	// Concurrent touches and admits
	const goroutines = 16
	const opsPerGoroutine = 200
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				key := fmt.Sprintf("concurrent-key-%d", (id+i)%20)
				filter.RecordTouch(key)
				_ = filter.Admit(key, 1024)
				_ = filter.Estimate(key)
			}
		}(g)
	}
	wg.Wait()
}
