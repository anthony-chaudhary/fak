package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// runDispatchAt drives the dispatch core and returns stdout, stderr, and the exit code.
func runDispatchAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runDispatch(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// writeCandidates writes a candidates JSON file and returns its path.
func writeCandidates(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "candidates.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// twentyFiveForX builds the headline scenario as a candidates file: 25 units sharing key "X",
// each updated a bit later than the last, so unit-24 is the freshest.
func twentyFiveForX(t *testing.T, now int64) string {
	var cs []dispatchorder.Candidate
	for i := 0; i < 25; i++ {
		cs = append(cs, dispatchorder.Candidate{
			ID:          fmt.Sprintf("%d", 100+i),
			Key:         "X",
			UpdatedUnix: now - 10000 + int64(i)*100,
		})
	}
	b, _ := json.Marshal(cs)
	return writeCandidates(t, string(b))
}

// TestDispatchOrderJSONSupersede is the headline scenario through the CLI: 25 tasks for the same
// target collapse to the freshest; --json reports 1 keep, 24 superseded, and that pick.
func TestDispatchOrderJSONSupersede(t *testing.T) {
	const now = 2_000_000
	path := twentyFiveForX(t, now)
	out, errb, code := runDispatchAt("order", "--in", path, "--now", "2000000", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var res struct {
		dispatchorder.Result
		Pick string `json:"pick"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if res.KeepCount != 1 || res.SupersededCount != 24 {
		t.Errorf("keep %d superseded %d, want 1/24", res.KeepCount, res.SupersededCount)
	}
	if res.Pick != "124" {
		t.Errorf("pick = %q, want 124 (the freshest)", res.Pick)
	}
}

// TestDispatchOrderTable: the human table names the pick and shows a superseded line.
func TestDispatchOrderTable(t *testing.T) {
	const now = 2_000_000
	path := twentyFiveForX(t, now)
	out, _, code := runDispatchAt("order", "--in", path, "--now", "2000000")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out, "pick: 124") {
		t.Errorf("table missing the freshest pick:\n%s", out)
	}
	if !strings.Contains(out, "superseded") || !strings.Contains(out, "24 superseded") {
		t.Errorf("table missing supersede accounting:\n%s", out)
	}
}

// TestDispatchOrderCooldownHolds: with a cooldown window, a freshest unit attempted within it is
// held and nothing is picked (no fallback to an older duplicate).
func TestDispatchOrderCooldownHolds(t *testing.T) {
	const now = 2_000_000
	path := writeCandidates(t, `[
	  {"id":"fresh","key":"X","updated_unix":1999900,"last_attempt_unix":1999940},
	  {"id":"stale","key":"X","updated_unix":1999000}
	]`)
	out, _, code := runDispatchAt("order", "--in", path, "--now", "2000000", "--cooldown-min", "10", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	var res struct {
		Pick         string `json:"pick"`
		CoolingCount int    `json:"cooling_count"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if res.Pick != "" || res.CoolingCount != 1 {
		t.Errorf("pick=%q cooling=%d, want empty/1 (freshest held, no fallback)", res.Pick, res.CoolingCount)
	}
}

func TestDispatchOrderJSONPricesFanoutBeforeLaunch(t *testing.T) {
	const now = 2_000_000
	path := writeCandidates(t, `[
	  {"id":"gateway-old","key":"A","lane":"gateway","tree":["internal/gateway/**"],"updated_unix":1999700},
	  {"id":"gateway-fresh","key":"B","lane":"gateway","tree":["internal/gateway/http.go"],"updated_unix":1999900},
	  {"id":"docs","key":"C","lane":"docs","tree":["docs/**"],"updated_unix":1999800}
	]`)
	out, errb, code := runDispatchAt("order", "--in", path, "--now", fmt.Sprint(now), "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var res struct {
		dispatchorder.Result
		Pick string `json:"pick"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if res.CollisionCount != 1 || res.CollisionsAvoided != 1 || res.LanesUtilized != 2 || res.SerializationWasted != 1 {
		t.Fatalf("fanout counts collision=%d avoided=%d lanes=%d wasted=%d, want 1/1/2/1",
			res.CollisionCount, res.CollisionsAvoided, res.LanesUtilized, res.SerializationWasted)
	}
	if res.Pick != "gateway-fresh" {
		t.Fatalf("pick = %q, want gateway-fresh", res.Pick)
	}
	var old dispatchorder.Ranked
	for _, row := range res.Order {
		if row.ID == "gateway-old" {
			old = row
		}
	}
	if old.Disposition != dispatchorder.DispCollisionRisk || old.Reason != dispatchorder.ReasonCollisionRisk {
		t.Fatalf("gateway-old ranked = %+v, want collision-risk %s", old, dispatchorder.ReasonCollisionRisk)
	}
}

func TestDispatchOrderTableReportsCollisionPricing(t *testing.T) {
	path := writeCandidates(t, `[
	  {"id":"a","key":"A","lane":"gateway","tree":["internal/gateway/**"],"updated_unix":99},
	  {"id":"b","key":"B","lane":"gateway","tree":["internal/gateway/http.go"],"updated_unix":100}
	]`)
	out, _, code := runDispatchAt("order", "--in", path, "--now", "200")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	for _, want := range []string{"collision-risk", "COLLISION_RISK", "collisions_avoided=1", "serialization_wasted=1"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q:\n%s", want, out)
		}
	}
}

// TestDispatchOrderBareArrayAndObject: both the bare array and the {"candidates":[...]} object
// forms parse.
func TestDispatchOrderBareArrayAndObject(t *testing.T) {
	arr := writeCandidates(t, `[{"id":"a","key":"K","updated_unix":10}]`)
	obj := writeCandidates(t, `{"candidates":[{"id":"a","key":"K","updated_unix":10}]}`)
	for _, p := range []string{arr, obj} {
		out, errb, code := runDispatchAt("order", "--in", p, "--now", "100", "--json")
		if code != 0 {
			t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
		}
		if !strings.Contains(out, `"pick": "a"`) {
			t.Errorf("input %s did not pick a:\n%s", p, out)
		}
	}
}

// TestDispatchOrderPreferOldest: --prefer-oldest flips the dispatch order so the oldest-created
// ticket is picked, even when a newer ticket has the freshest UPDATE (which the default
// freshest-first pick selects). Two distinct keys so neither supersedes the other.
func TestDispatchOrderPreferOldest(t *testing.T) {
	path := writeCandidates(t, `[
	  {"id":"old","key":"A","created_unix":100,"updated_unix":200},
	  {"id":"new","key":"B","created_unix":800,"updated_unix":990}
	]`)
	// default: freshest update ("new", updated 990) leads.
	def, _, code := runDispatchAt("order", "--in", path, "--now", "2000", "--json")
	if code != 0 {
		t.Fatalf("default exit = %d, want 0", code)
	}
	if !strings.Contains(def, `"pick": "new"`) {
		t.Errorf("default should pick the freshest-updated ticket:\n%s", def)
	}
	// --prefer-oldest: oldest-created ("old", created 100) leads.
	out, _, code := runDispatchAt("order", "--in", path, "--now", "2000", "--prefer-oldest", "--json")
	if code != 0 {
		t.Fatalf("--prefer-oldest exit = %d, want 0", code)
	}
	if !strings.Contains(out, `"pick": "old"`) {
		t.Errorf("--prefer-oldest should pick the oldest-created ticket:\n%s", out)
	}
}

// TestDispatchUsageErrors covers the exit-2 / exit-1 paths.
func TestDispatchUsageErrors(t *testing.T) {
	if _, _, code := runDispatchAt(); code != 2 {
		t.Errorf("no subcommand: exit = %d, want 2", code)
	}
	if _, _, code := runDispatchAt("frobnicate"); code != 2 {
		t.Errorf("unknown subcommand: exit = %d, want 2", code)
	}
	if _, _, code := runDispatchAt("order", "--in", filepath.Join(t.TempDir(), "nope.json")); code != 1 {
		t.Errorf("missing input file: exit = %d, want 1", code)
	}
	bad := writeCandidates(t, `{not json`)
	if _, _, code := runDispatchAt("order", "--in", bad); code != 1 {
		t.Errorf("malformed json: exit = %d, want 1", code)
	}
}

// TestDispatchHelp: the help subcommand exits 0 and prints the usage.
func TestDispatchHelp(t *testing.T) {
	out, _, code := runDispatchAt("help")
	if code != 0 || !strings.Contains(out, "fak dispatch order") {
		t.Errorf("help exit=%d, missing usage:\n%s", code, out)
	}
}
