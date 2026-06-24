// extdriver is a SEPARATE Go module (not part of the root module's ./...) that
// proves fak's ABI is importable by an OUT-OF-TREE driver via pkg/abi. The
// replace directive points fak at this checkout so the example builds against
// the local source; a real consumer would `require` a tagged version instead.
module github.com/anthony-chaudhary/fak/examples/extdriver

go 1.26

require github.com/anthony-chaudhary/fak v0.0.0

replace github.com/anthony-chaudhary/fak => ../..
