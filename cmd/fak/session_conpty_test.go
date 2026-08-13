//go:build windows

package main

import (
	"bytes"
	"testing"
	"time"
)

func TestSessionConPTYCapturesRealTerminalBytes(t *testing.T) {
	got, err := runSessionConPTY("echo fak-conpty-witness", 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(got, []byte("fak-conpty-witness")) {
		t.Fatalf("ConPTY transcript=%q", got)
	}
}
