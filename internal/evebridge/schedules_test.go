package evebridge

import (
	"bytes"
	"testing"
	"testing/fstest"
)

func TestScheduleInventoryStableProjection(t *testing.T) {
	root := fstest.MapFS{
		"agent/agent.ts":           {},
		"agent/schedules/daily.md": {Data: []byte("---\ncron:  0   9 * * 1-5\n---\nPost the summary.")},
		"agent/schedules/sync.ts":  {Data: []byte(`export default defineSchedule({ cron: "*/5 * * * *", async run({ receive }) { return receive(slack) } })`)},
	}
	first, err := InspectSchedules(root, "eve-dev")
	if err != nil {
		t.Fatal(err)
	}
	second, err := InspectSchedules(root, "eve-dev")
	if err != nil {
		t.Fatal(err)
	}
	if !first.OK || !bytes.Equal(first.JSON(), second.JSON()) {
		t.Fatalf("unstable inventory: %+v", first)
	}
	if len(first.Schedules) != 2 {
		t.Fatalf("schedules = %+v", first.Schedules)
	}
	prompt := scheduleByID(t, first, "daily")
	if prompt.Form != "prompt" || prompt.CronUTC != "0 9 * * 1-5" || prompt.SideEffecting || prompt.Principal.Kind != "app" {
		t.Fatalf("prompt projection = %+v", prompt)
	}
	handler := scheduleByID(t, first, "sync")
	if handler.Form != "handler" || handler.ChannelTarget != "slack" || !handler.SideEffecting || handler.HostProjection != "eve-dev-dispatch" {
		t.Fatalf("handler projection = %+v", handler)
	}
	if handler.Unit.SourcePath != "agent/schedules/sync.ts" || handler.Unit.OverlapPolicy == "" || handler.Unit.MissedRunPolicy == "" {
		t.Fatalf("cron unit = %+v", handler.Unit)
	}
}

func TestScheduleInventoryDiagnostics(t *testing.T) {
	root := fstest.MapFS{
		"agent/agent.ts":                     {},
		"agent/schedules/foo-bar.ts":         {Data: []byte(`defineSchedule({cron:"0 0 * * *", markdown:"a"})`)},
		"agent/schedules/foo_bar.ts":         {Data: []byte(`defineSchedule({cron:"0 1 * * *", markdown:"b"})`)},
		"agent/subagents/a/schedules/bad.ts": {Data: []byte(`defineSchedule({cron:"0 2 * * *", markdown:"c"})`)},
	}
	got, err := InspectSchedules(root, "my-host")
	if err != nil {
		t.Fatal(err)
	}
	if got.OK {
		t.Fatal("colliding/misplaced inventory unexpectedly passed")
	}
	for _, code := range []string{CodeScheduleDuplicate, CodeScheduleRootOnly, CodeScheduleCustomHost} {
		assertScheduleDiagnostic(t, got, code)
	}
}

func TestCompiledSchedulesDisabledAndUTCValidation(t *testing.T) {
	root := fstest.MapFS{".eve/compile/compiled-agent-manifest.json": {Data: []byte(`{"schedules":[{"name":"active","logicalPath":"schedules/active.ts","cron":"0 7 * * *","hasRun":true},{"name":"off","logicalPath":"schedules/off.ts","cron":"0 8 * * *","disabled":true}]}`)}}
	got, err := InspectSchedules(root, "vercel")
	if err != nil {
		t.Fatal(err)
	}
	if !got.OK || len(got.Schedules) != 1 || got.Schedules[0].HostProjection != "vercel-cron" {
		t.Fatalf("compiled projection = %+v", got)
	}
	bad := fstest.MapFS{"agent/agent.ts": {}, "agent/schedules/bad.ts": {Data: []byte(`defineSchedule({cron:"0 9 * *", markdown:"bad"})`)}}
	got, err = InspectSchedules(bad, "eve-start")
	if err != nil {
		t.Fatal(err)
	}
	if got.OK {
		t.Fatal("four-field cron accepted")
	}
	assertScheduleDiagnostic(t, got, CodeScheduleMalformed)
}

func TestRecordDevDispatchLinksSessionToLedger(t *testing.T) {
	inventory, err := InspectSchedules(fstest.MapFS{"agent/agent.ts": {}, "agent/schedules/daily.md": {Data: []byte("cron: 0 9 * * *")}}, "eve-dev")
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := RecordDevDispatch(inventory, "daily", "sess_123")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.SessionID != "sess_123" || receipt.LedgerUnitID != "eve-daily" {
		t.Fatalf("receipt = %+v", receipt)
	}
	if _, err := RecordDevDispatch(inventory, "daily", ""); err == nil {
		t.Fatal("empty session id accepted")
	}
}

func scheduleByID(t *testing.T, inventory ScheduleInventory, id string) ScheduleProjection {
	t.Helper()
	for _, schedule := range inventory.Schedules {
		if schedule.ID == id {
			return schedule
		}
	}
	t.Fatalf("schedule %q absent", id)
	return ScheduleProjection{}
}
func assertScheduleDiagnostic(t *testing.T, inventory ScheduleInventory, code string) {
	t.Helper()
	for _, diagnostic := range inventory.Diagnostics {
		if diagnostic.Code == code {
			return
		}
	}
	t.Fatalf("diagnostic %s absent: %+v", code, inventory.Diagnostics)
}
