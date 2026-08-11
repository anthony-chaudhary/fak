module github.com/anthony-chaudhary/fak/tools/videogen/terminal

go 1.26.5

require golang.org/x/image v0.44.0

require golang.org/x/text v0.40.0 // indirect

// x/image v0.44.0 asks for x/text v0.40.0, which is not in this box's module
// cache, and this build has no network. v0.38.0 is cached and the only thing
// pulled in is font/sfnt's charmap tables, which did not move between the two.
replace golang.org/x/text => golang.org/x/text v0.38.0
