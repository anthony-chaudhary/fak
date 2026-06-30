// Package discoverfixture is read-only DiscoverDir test input (under testdata, so
// the go tool never compiles it). It carries two annotated tunables and one plain
// const, to prove discovery harvests exactly the annotated ones.
package discoverfixture

// fak:opttarget name=alpha metric=alpha_score dir=higher sweep=1,2,3 measurer=fake
const Alpha = 1

// fak:opttarget name=beta metric=beta_latency dir=lower sweep=10,20 measurer=fake
//
// Beta also carries prose after the annotation line, to prove the scanner reads
// the directive out of a multi-line doc group.
const Beta = 10

// Gamma is a normal const with no annotation — it must NOT be discovered.
const Gamma = 99
