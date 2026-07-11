package accounts

import "testing"

// activeSeat builds an active, serveable-looking Home bound to accountUUID. Email is left empty so
// NameLie() is never triggered by the synthetic seat names used here.
func activeSeat(name, accountUUID string) Home {
	return Home{
		Name:     name,
		Dir:      "/fake/" + name,
		Identity: Identity{AccountUUID: accountUUID, Exists: true, HasCreds: true},
	}
}

func TestDetectEnrollCollision_DuplicateRefuses(t *testing.T) {
	// account u1 is already seat "alpha"; enrolling it again as "beta" is a duplicate → refuse.
	reg := Registry{Homes: []Home{activeSeat("alpha", "u1")}}
	got := DetectEnrollCollision(reg, "beta", ProbedIdentity{AccountUUID: "u1"})
	if got.Kind != EnrollDuplicate {
		t.Fatalf("kind = %q, want %q (detail: %s)", got.Kind, EnrollDuplicate, got.Detail)
	}
	if got.ConflictSeat != "alpha" {
		t.Errorf("conflict seat = %q, want alpha", got.ConflictSeat)
	}
	if !got.Refuse() {
		t.Error("a duplicate must Refuse()")
	}
}

func TestDetectEnrollCollision_FreshAccountOK(t *testing.T) {
	reg := Registry{Homes: []Home{activeSeat("alpha", "u1")}}
	got := DetectEnrollCollision(reg, "beta", ProbedIdentity{AccountUUID: "u2"})
	if got.Kind != EnrollOK {
		t.Fatalf("kind = %q, want ok (detail: %s)", got.Kind, got.Detail)
	}
	if got.Refuse() {
		t.Error("a fresh, unheld account must not Refuse()")
	}
}

func TestDetectEnrollCollision_ReconcileRebindSurfacesNotRefuses(t *testing.T) {
	// seat "alpha" is recorded as u1; reconciling it onto u2 (its live credential moved) is
	// enroll-current correcting the binding — a rebind, surfaced but never refused.
	reg := Registry{Homes: []Home{activeSeat("alpha", "u1")}}
	got := DetectEnrollCollision(reg, "alpha", ProbedIdentity{AccountUUID: "u2"})
	if got.Kind != EnrollRebind {
		t.Fatalf("kind = %q, want rebind (detail: %s)", got.Kind, got.Detail)
	}
	if got.PriorAccount != (Identity{AccountUUID: "u1"}).AccountKey() {
		t.Errorf("prior account = %q, want %q", got.PriorAccount, (Identity{AccountUUID: "u1"}).AccountKey())
	}
	if got.Refuse() {
		t.Error("a rebind (enroll-current's own job) must not Refuse()")
	}
}

func TestDetectEnrollCollision_SameAccountReconcileIsOK(t *testing.T) {
	// Reconciling seat "alpha" onto the SAME account it already holds is a clean no-collision.
	reg := Registry{Homes: []Home{activeSeat("alpha", "u1")}}
	got := DetectEnrollCollision(reg, "alpha", ProbedIdentity{AccountUUID: "u1"})
	if got.Kind != EnrollOK {
		t.Fatalf("kind = %q, want ok (detail: %s)", got.Kind, got.Detail)
	}
}

func TestDetectEnrollCollision_EmptyProbeIsUnknown(t *testing.T) {
	reg := Registry{Homes: []Home{activeSeat("alpha", "u1")}}
	got := DetectEnrollCollision(reg, "beta", ProbedIdentity{})
	if got.Kind != EnrollUnknown {
		t.Fatalf("kind = %q, want unknown (detail: %s)", got.Kind, got.Detail)
	}
	if got.Refuse() {
		t.Error("no probed identity must warn, not refuse")
	}
}

func TestDetectEnrollCollision_InactiveSeatDoesNotCollide(t *testing.T) {
	// A tombstoned seat that once held u1 must not block re-enrolling u1 under a new name.
	dead := activeSeat("alpha", "u1")
	dead.Status = StatusTombstoned
	reg := Registry{Homes: []Home{dead}}
	got := DetectEnrollCollision(reg, "beta", ProbedIdentity{AccountUUID: "u1"})
	if got.Kind != EnrollOK {
		t.Fatalf("kind = %q, want ok — a tombstoned holder is not a live collision (detail: %s)", got.Kind, got.Detail)
	}
}

func TestVerifySeatServable_HappyPath(t *testing.T) {
	reg := Registry{Homes: []Home{activeSeat("seatx", "u1")}}
	rep := VerifySeatServable(reg, "seatx", "accounts:\n  - name: seatx\n", "workers:\n  - seatx\n")
	if !rep.Servable {
		t.Fatalf("want servable; got %+v", rep)
	}
	if !rep.DosServable || !rep.InDosView || !rep.InJobView {
		t.Errorf("all three witnesses should hold: %+v", rep)
	}
}

func TestVerifySeatServable_MissingFromJobView(t *testing.T) {
	reg := Registry{Homes: []Home{activeSeat("seatx", "u1")}}
	rep := VerifySeatServable(reg, "seatx", "accounts:\n  - name: seatx\n", "workers:\n  - other\n")
	if rep.Servable {
		t.Fatal("seat absent from the job view must not be reported servable")
	}
	if rep.InJobView {
		t.Error("InJobView should be false")
	}
	if !rep.DosServable {
		t.Error("DosServable should still hold (the registry projection is fine)")
	}
}

func TestVerifySeatServable_NotServableWhenNoCreds(t *testing.T) {
	seat := activeSeat("seatx", "u1")
	seat.Identity.HasCreds = false // credential missing → LoginNeedsLogin, not serveable
	reg := Registry{Homes: []Home{seat}}
	rep := VerifySeatServable(reg, "seatx", "  - name: seatx\n", "  - seatx\n")
	if rep.Servable || rep.DosServable {
		t.Fatalf("a credential-less seat must not be servable: %+v", rep)
	}
	if rep.LoginStatus != LoginNeedsLogin {
		t.Errorf("login status = %q, want %q", rep.LoginStatus, LoginNeedsLogin)
	}
}

func TestVerifySeatServable_SeatNotInRegistry(t *testing.T) {
	reg := Registry{Homes: []Home{activeSeat("seatx", "u1")}}
	rep := VerifySeatServable(reg, "ghost", "x", "y")
	if rep.Servable {
		t.Fatal("a seat absent from the registry cannot be servable")
	}
}
