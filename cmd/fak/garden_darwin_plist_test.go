package main

import (
	"encoding/xml"
	"reflect"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms
func TestGardenRenderDarwinPlistRoundTrip(t *testing.T) {
	t.Parallel()
	const fakBin = `/opt/fak & co/<"binary">'/fak`
	const root = `/work/<repo> & "quoted"' directory`
	for _, tc := range []struct {
		name string
		live bool
	}{
		{name: "dry-run"},
		{name: "live", live: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plist, err := gardenRenderDarwinPlist(fakBin, root, 2*time.Hour+1500*time.Millisecond, tc.live)
			if err != nil {
				t.Fatal(err)
			}
			var got struct {
				XMLName   xml.Name  `xml:"plist"`
				Keys      []string  `xml:"dict>key"`
				Label     string    `xml:"dict>string"`
				Args      []string  `xml:"dict>array>string"`
				Interval  int64     `xml:"dict>integer"`
				RunAtLoad *struct{} `xml:"dict>true"`
			}
			if err := xml.Unmarshal([]byte(plist), &got); err != nil {
				t.Fatalf("invalid garden plist XML: %v\n%s", err, plist)
			}
			wantArgs := []string{fakBin, "garden", "watchdog", "--repo", root}
			if tc.live {
				wantArgs = append(wantArgs, "--live")
			}
			if !reflect.DeepEqual(got.Args, wantArgs) {
				t.Errorf("ProgramArguments = %q, want %q", got.Args, wantArgs)
			}
			if want := []string{"Label", "ProgramArguments", "StartInterval", "RunAtLoad"}; !reflect.DeepEqual(got.Keys, want) {
				t.Errorf("plist keys = %q, want %q", got.Keys, want)
			}
			if got.Label != "com.fleet.stale-work-garden" {
				t.Errorf("Label = %q, want com.fleet.stale-work-garden", got.Label)
			}
			if got.Interval != 7201 || got.RunAtLoad == nil {
				t.Errorf("schedule = interval %d, RunAtLoad %v; want 7201 seconds and true", got.Interval, got.RunAtLoad != nil)
			}
		})
	}
}
