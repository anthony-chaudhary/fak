// Package discoverbad is read-only DiscoverDir test input carrying a MALFORMED
// annotation (an unknown direction), to prove the scanner refuses a typo rather
// than silently dropping or mis-reading a target.
package discoverbad

// fak:opttarget name=bad metric=x dir=sideways sweep=1 measurer=fake
const Bad = 1
