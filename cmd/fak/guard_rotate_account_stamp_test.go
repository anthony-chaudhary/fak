package main

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/guardrotate"
)

// #5870: the account-scoped goal park is only correct if $DISPATCH_ACCOUNT names the seat
// that ACTUALLY served. The dispatcher stamps it at spawn time; guard may then rotate off a
// cooling seat before it builds the park template, and on the live fleet it usually does (36
// of 59 resolve units in one day carried a rotation line). Without this re-stamp the park
// records the pre-rotation seat: it walls a healthy account and leaves the walled one free.
func TestGuardRotationRestampsTheServingAccount(t *testing.T) {
	t.Setenv("DISPATCH_ACCOUNT", "aug5-netra") // what the dispatcher stamped at spawn

	if !stampGuardRotationAccount(guardrotate.Note{From: "aug5-netra", To: "july16-netra"}) {
		t.Fatal("a resolved rotation did not re-stamp the serving account")
	}
	if got := os.Getenv("DISPATCH_ACCOUNT"); got != "july16-netra" {
		t.Fatalf("DISPATCH_ACCOUNT=%q want %q — the park would name the seat we rotated OFF", got, "july16-netra")
	}

	// The whole point: a park written from this env now walls the seat that hit the wall,
	// and NOT the one the dispatcher originally picked (which is free to be re-dispatched).
	now := time.Unix(1_800_000_000, 0)
	rec := goalpark.Record{
		Schema: goalpark.Schema, Goal: "gateway",
		Account: os.Getenv("DISPATCH_ACCOUNT"), ParkedUntil: now.Unix() + 3600,
	}
	if !rec.Blocks("july16-netra", now) {
		t.Error("the rotated-onto seat — the one that actually served and hit the wall — is not walled")
	}
	if rec.Blocks("aug5-netra", now) {
		t.Error("the seat we rotated OFF is walled by a wall it never hit")
	}
}

// A degenerate note must leave the dispatcher's stamp alone: clearing it would trade a
// correct identity for an unattributed park, and an unattributed park blocks NOBODY.
func TestGuardRotationWithoutATargetKeepsTheDispatcherStamp(t *testing.T) {
	t.Setenv("DISPATCH_ACCOUNT", "aug5-netra")
	for _, note := range []guardrotate.Note{{}, {From: "aug5-netra"}, {To: "   "}} {
		if stampGuardRotationAccount(note) {
			t.Errorf("note %+v reported a re-stamp it cannot make", note)
		}
		if got := os.Getenv("DISPATCH_ACCOUNT"); got != "aug5-netra" {
			t.Fatalf("note %+v clobbered DISPATCH_ACCOUNT to %q", note, got)
		}
	}
}
