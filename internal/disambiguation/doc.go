// Package disambiguation defines the canonical machine-readable record shared by
// fak's terminology index generator and readers.
//
// EntrySchemaVersion is the wire contract. Version 1 is intentionally pinned:
// readers reject unknown versions, unknown JSON fields, trailing JSON values, and
// incomplete required groups. Any field addition, rename, removal, or type change
// therefore needs a new schema version instead of being interpreted as v1. Empty
// aliases are represented by [] rather than by an absent or null field.
//
// The package owns only the record, parser, validation, and hermetic SelfTest. It
// performs no filesystem or network writes, so the generator added by later work
// remains the single writer of the derived disambiguation index.
package disambiguation
