package sessionimage

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestRehydrateDriveProjection is the #4127 witness: a resume re-attaches the compact drive
// projection and a lean consumer seeds a governor from it, so the resumed child comes up at
// the CARRIED spent budget (200 of a 1000 cap) — not a fresh full one. When the image carries
// no projection (a pre-#4126 image), the present flag is false and the re-seed is a no-op, so
// the resume stays byte-identical to today.
func TestRehydrateDriveProjection(t *testing.T) {
	ctx := context.Background()

	t.Run("re-seeds a governor at the carried spent budget", func(t *testing.T) {
		dir := t.TempDir()
		src := driveState("sess-rh") // TokensLeft 200 (20% of a 1000 cap)
		if _, err := DumpDir(dir, Input{SessionID: "sess-rh", Drive: src, Now: 1}); err != nil {
			t.Fatalf("DumpDir: %v", err)
		}
		img, err := LoadDir(dir)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		r, err := img.Rehydrate(ctx, RehydrateOptions{})
		if err != nil {
			t.Fatalf("Rehydrate: %v", err)
		}
		if !r.DriveProjectionPresent {
			t.Fatalf("DriveProjectionPresent=false for an image carrying a projection")
		}
		if r.DriveProjection.Budget.TokensLeft != 200 {
			t.Fatalf("projection TokensLeft: got %d want 200", r.DriveProjection.Budget.TokensLeft)
		}

		// A lean relaunch consumer seeds a FRESH-full governor from the projection: the spent
		// "left" axes overwrite the full ones, while the base's caps (denominators) survive.
		base := session.State{Budget: session.Budget{TokensLeft: 1000, ContextTokensCap: 8192, SpendMicroCentsCap: 100_000}}
		seeded := r.DriveProjection.SeedState(base)
		if seeded.Budget.TokensLeft != 200 {
			t.Fatalf("re-seed did not carry the spent budget: got TokensLeft=%d want 200 (a fresh child would stay 1000)", seeded.Budget.TokensLeft)
		}
		if seeded.Priority != src.Priority || seeded.Pace != src.Pace {
			t.Fatalf("re-seed lost priority/pace: got prio=%d pace=%+v want prio=%d pace=%+v",
				seeded.Priority, seeded.Pace, src.Priority, src.Pace)
		}
		if seeded.ObjectivePin.PinID != src.ObjectivePin.PinID || seeded.ObjectivePin.Digest != src.ObjectivePin.Digest {
			t.Fatalf("re-seed lost objective-pin identity: got %+v want PinID=%q Digest=%q",
				seeded.ObjectivePin, src.ObjectivePin.PinID, src.ObjectivePin.Digest)
		}
		if seeded.Generation != src.Generation {
			t.Fatalf("re-seed lost generation: got %d want %d", seeded.Generation, src.Generation)
		}
		if seeded.Budget.ContextTokensCap != 8192 || seeded.Budget.SpendMicroCentsCap != 100_000 {
			t.Fatalf("re-seed clobbered the base caps (should re-seed only the remaining axes): got ctxCap=%d spendCap=%d",
				seeded.Budget.ContextTokensCap, seeded.Budget.SpendMicroCentsCap)
		}
	})

	t.Run("a pre-#4126 image re-seeds nothing", func(t *testing.T) {
		dir := t.TempDir()
		meta, err := DumpDir(dir, Input{SessionID: "sess-old", Drive: driveState("sess-old"), Now: 1})
		if err != nil {
			t.Fatalf("DumpDir: %v", err)
		}
		// Simulate a pre-#4126 image: strip drive.json and its integrity Part entry, so the
		// image is byte-shaped exactly like one dumped before this feature existed.
		if err := os.Remove(filepath.Join(dir, DriveFile)); err != nil {
			t.Fatalf("remove %s: %v", DriveFile, err)
		}
		kept := meta.Parts[:0]
		for _, p := range meta.Parts {
			if p.Name != DriveFile {
				kept = append(kept, p)
			}
		}
		meta.Parts = kept
		if err := writeImageJSON(dir, meta); err != nil {
			t.Fatalf("writeImageJSON: %v", err)
		}

		img, err := LoadDir(dir)
		if err != nil {
			t.Fatalf("LoadDir (pre-#4126 image): %v", err)
		}
		r, err := img.Rehydrate(ctx, RehydrateOptions{})
		if err != nil {
			t.Fatalf("Rehydrate: %v", err)
		}
		if r.DriveProjectionPresent {
			t.Fatalf("DriveProjectionPresent=true for an image carrying no drive.json")
		}
		if r.DriveProjection != (DriveProjection{}) {
			t.Fatalf("absent projection decoded non-zero: %+v", r.DriveProjection)
		}
	})
}
