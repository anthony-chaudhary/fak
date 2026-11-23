// routepolicy-external is a SEPARATE Go module (not part of the root module's
// ./...) that proves fak's routing policy is usable and replaceable from outside
// the fak module through pkg/routepolicy alone. The replace directive points fak
// at this checkout; a real consumer would `require` a tagged version instead.
module example.com/routepolicy-external

go 1.26.0

require github.com/anthony-chaudhary/fak v0.0.0

replace github.com/anthony-chaudhary/fak => ../..
