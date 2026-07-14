//go:build windows

package main

import "testing"

func TestControlTypeNameBoundedMarkerSet(t *testing.T) {
	cases := map[uint32]string{2: "CTRL_CLOSE_EVENT", 5: "CTRL_LOGOFF_EVENT", 6: "CTRL_SHUTDOWN_EVENT", 0: "", 1: ""}
	for in, want := range cases {
		if got := controlTypeName(in); got != want {
			t.Fatalf("controlTypeName(%d)=%q want %q", in, got, want)
		}
	}
}
