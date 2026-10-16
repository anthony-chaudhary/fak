package childprocess

import (
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestChildPWDWorkdirEnv(t *testing.T) {
	for _, env := range [][]string{nil, {}, {"PWD=untouched", "OLDPWD=old", "KEEP=value"}} {
		dir, got, err := WorkdirEnv("", env)
		if err != nil || dir != "" || !reflect.DeepEqual(got, env) {
			t.Fatalf("no directory: dir=%q env=%v err=%v", dir, got, err)
		}
	}
	t.Setenv("CHILD_PWD_INHERITED", "parent")
	for _, tc := range []struct {
		name string
		env  []string
	}{
		{"inherited", nil},
		{"empty", []string{}},
		{"custom", []string{"PWD=stale", "KEEP=first", "OLDPWD=older", "PWD=duplicate", "pwd=lower", "oldpwd=lower-old", "KEEP=last"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := append([]string(nil), tc.env...)
			dir, got, err := WorkdirEnv(".", tc.env)
			wantDir, absErr := filepath.Abs(".")
			if err != nil || absErr != nil || dir != wantDir {
				t.Fatalf("dir=%q want=%q err=%v absErr=%v", dir, wantDir, err, absErr)
			}
			if len(tc.env) > 0 && !reflect.DeepEqual(tc.env, before) {
				t.Fatal("caller environment changed")
			}
			if got[len(got)-1] != "PWD="+wantDir {
				t.Fatalf("last entry=%q", got[len(got)-1])
			}
			if tc.name == "empty" && !reflect.DeepEqual(got, []string{"PWD=" + wantDir}) {
				t.Fatalf("explicit empty environment inherited entries: %v", got)
			}
			if tc.name == "custom" {
				want := []string{"KEEP=first", "pwd=lower", "oldpwd=lower-old", "KEEP=last", "PWD=" + wantDir}
				if runtime.GOOS == "windows" {
					want = []string{"KEEP=first", "KEEP=last", "PWD=" + wantDir}
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("custom env=%v want=%v", got, want)
				}
			}
			if tc.name == "inherited" {
				found := false
				for _, entry := range got {
					found = found || entry == "CHILD_PWD_INHERITED=parent"
				}
				if !found {
					t.Fatal("nil environment did not inherit parent")
				}
			}
		})
	}
}
