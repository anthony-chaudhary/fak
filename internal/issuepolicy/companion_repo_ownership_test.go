package issuepolicy

import "testing"

func TestOwnCompanionRepo(t *testing.T) {
	for _, tc := range []struct {
		name     string
		classes  []PathRepoClass
		wantRepo CompanionRepo
		wantOK   bool
		wantName string
	}{
		{name: "all private", classes: []PathRepoClass{PathPrivateOnly, PathPrivateOnly}, wantRepo: RepoPrivate, wantOK: true, wantName: "anthony-chaudhary/fak-private"},
		{name: "all public", classes: []PathRepoClass{PathPublicOnly}, wantRepo: RepoPublic, wantOK: true, wantName: "anthony-chaudhary/fak"},
		{name: "private with no-signal paths", classes: []PathRepoClass{PathBoth, PathPrivateOnly, PathNeither, PathUnresolved}, wantRepo: RepoPrivate, wantOK: true, wantName: "anthony-chaudhary/fak-private"},
		{name: "split", classes: []PathRepoClass{PathPublicOnly, PathPrivateOnly}, wantRepo: RepoUnresolved, wantOK: false},
		{name: "split with no-signal", classes: []PathRepoClass{PathPrivateOnly, PathBoth, PathPublicOnly}, wantRepo: RepoUnresolved, wantOK: false},
		{name: "only no-signal", classes: []PathRepoClass{PathBoth, PathNeither, PathUnresolved}, wantRepo: RepoUnresolved, wantOK: false},
		{name: "empty", classes: nil, wantRepo: RepoUnresolved, wantOK: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := OwnCompanionRepo(tc.classes)
			if ok != tc.wantOK || got != tc.wantRepo {
				t.Fatalf("OwnCompanionRepo(%v) = (%v, %v), want (%v, %v)", tc.classes, got, ok, tc.wantRepo, tc.wantOK)
			}
			if tc.wantOK && got.Name() != tc.wantName {
				t.Fatalf("Name() = %q, want %q", got.Name(), tc.wantName)
			}
		})
	}
}

func TestCompanionRepoName(t *testing.T) {
	if got := RepoUnresolved.Name(); got != "" {
		t.Fatalf("RepoUnresolved.Name() = %q, want empty", got)
	}
	if got := RepoPublic.Name(); got != "anthony-chaudhary/fak" {
		t.Fatalf("RepoPublic.Name() = %q", got)
	}
	if got := RepoPrivate.Name(); got != "anthony-chaudhary/fak-private" {
		t.Fatalf("RepoPrivate.Name() = %q", got)
	}
}
