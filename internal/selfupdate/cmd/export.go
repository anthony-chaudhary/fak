package selfupdatecmd

// Receipt is the stable fak.self-update.receipt/v1 payload.
type Receipt = selfUpdateReceipt

// ReceiptTarget is one target entry in a self-update receipt.
type ReceiptTarget = selfUpdateReceiptTarget

const ReceiptSchema = selfUpdateReceiptSchema

// RepoRevOf preserves the native command's repository revision helper.
func RepoRevOf(root, ref string) string { return repoRevOf(root, ref) }
