package main

import (
	"net"
	"strings"
	"testing"
	"time"
)

func TestEPJoinTimeoutRejectsBadEnv(t *testing.T) {
	t.Setenv("FAK_EP_JOIN_TIMEOUT_S", "bad")
	if _, err := epJoinTimeout(); err == nil || !strings.Contains(err.Error(), "FAK_EP_JOIN_TIMEOUT_S") {
		t.Fatalf("epJoinTimeout bad env error = %v, want FAK_EP_JOIN_TIMEOUT_S refusal", err)
	}
}

func TestEPRequireDevicePGEnv(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"false", false},
		{"1", true},
		{"true", true},
		{"YES", true},
		{" on ", true},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			t.Setenv("FAK_EP_REQUIRE_DEVICE_PG", tc.raw)
			if got := epRequireDevicePG(); got != tc.want {
				t.Fatalf("epRequireDevicePG(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}

func TestEPJoinTimeoutZeroMakesWorkerDialFailFast(t *testing.T) {
	t.Setenv("FAK_EP_JOIN_TIMEOUT_S", "0")
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen temp addr: %v", err)
	}
	addr := ln.Addr().String()
	ln.Close()
	start := time.Now()
	cfg := epRankConfig{ranks: 2, rank: 1, coordAddr: addr, sharded: true}
	_, err = dialEPGroup(cfg)
	if err == nil {
		t.Fatal("dialEPGroup to absent coordinator succeeded; want timeout/refusal")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatalf("dialEPGroup with zero timeout took %s; want fail-fast", time.Since(start))
	}
	if !strings.Contains(err.Error(), "timed out after 0s") {
		t.Fatalf("dialEPGroup error = %v, want timed out after 0s", err)
	}
}
