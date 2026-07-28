---
title: "Documentation choice-table template"
description: "How to present two or more valid paths on a public route: one observable criterion per row, its outcome, its limit, its proof, and the stated default."
---

# Documentation choice-table template

**Primary audience:** documentation writers presenting two or more valid modes or paths on a public route.

**Lifecycle:** current writing template. **Generation:** `gen/now`. **Support:** maintained for public documentation. **Authority:** this template owns choice presentation; each product or workflow authority owns which choices are valid and supported.

**Next action:** copy the [ready-to-fill block](#ready-to-fill-block), replace every placeholder, and run the [completion check](#completion-check).

Use a choice table when readers with the same goal can validly select different paths. Give each row one observable criterion, one choice, its outcome, its limit, and its proof. State the default and every condition that changes it immediately below the table.

If only one path is supported, state that path directly. A table is not needed.

## Ready-to-fill block

Copy this block into the reader-facing route.

```markdown
## Choose `<thing being selected>`

| Your condition | Use | What you get | Limit or change condition | Proof |
|---|---|---|---|---|
| `<observable reader condition A>` | **`<mode or path A>`** | `<reader outcome A>` | `<boundary, cost, or condition that moves the reader elsewhere>` | `<authority or witness>: <relative link>` |
| `<observable reader condition B>` | **`<mode or path B>`** | `<reader outcome B>` | `<boundary, cost, or condition that moves the reader elsewhere>` | `<authority or witness>: <relative link>` |

**Default:** use **`<default choice>`** when `<default criterion>`.

**Choose `<other choice>` instead when:** `<observable upgrade or exception condition>`.

**Next action:** `<one link, command, decision, or destination that exercises the selected path>`.
```

Add rows only for currently valid choices. Keep one condition and one choice per row. Split combined conditions when they can lead to different selections.

## Field contract

| Field | Required content | Reader test |
|---|---|---|
| `Your condition` | A fact the reader can observe before choosing | `Can I identify my row without knowing internals?` |
| `Use` | One named supported mode or path | `Can I repeat the selection exactly?` |
| `What you get` | The outcome relevant to this audience | `Does this say why the choice fits my goal?` |
| `Limit or change condition` | A scoped boundary, cost, or trigger to choose another row | `Do I know when this choice stops fitting?` |
| `Proof` | The authority or witness for this row's claim | `Does the link prove only this outcome and scope?` |
| `Default` | One choice plus the criterion that makes it default | `Can I choose without comparing every implementation detail?` |
| `Choose instead when` | Every observable upgrade or exception condition | `Can I tell when to leave the default?` |
| `Next action` | One checkable action after selection | `Can I proceed now?` |

Place lifecycle, generation, and support context above the table or in the affected row when choices differ. A global statement applies to every row; a row-level statement applies only to that choice.

## Common shapes

### Two modes with one default

Use the ready-to-fill block as written. Name the default criterion and the one condition that selects the other mode.

### Several equal choices

When no single default is honest, say so affirmatively:

```markdown
**Selection rule:** choose the row whose `<criterion>` matches your `<observable input>`. These choices have no global default because `<scoped reason>`.
```

Every row still needs a distinct observable criterion. `No global default` is valid only when the product authority confirms that status.

### Progressive modes

For modes that add capability, cost, or evidence, state the base default and each upgrade trigger:

```markdown
**Default:** start with **`<base mode>`** for `<base condition>`.

**Upgrade to `<next mode>` when:** `<observable need>`.

**Upgrade to `<highest mode>` when:** `<observable need>`.
```

Do not imply that the highest-cost or newest mode is universally best. Bind each upgrade to the reader need it satisfies.

## Completion check

Mark each line `pass` or `repair`. One `repair` blocks publication.

- [ ] The route has at least two currently valid choices; otherwise the table is removed.
- [ ] One primary audience and one goal define the choice.
- [ ] Every row starts with an observable, mutually distinguishable reader condition.
- [ ] Every row names one choice, outcome, scoped limit, and supporting authority or witness.
- [ ] Lifecycle, generation, and support status apply unambiguously to the table or individual rows.
- [ ] One default and its criterion are explicit, or a scoped authority-backed reason establishes no global default.
- [ ] Every upgrade and exception condition is visible beside the default.
- [ ] Internal rationale and historical alternatives stay in a deeper contributor route.
- [ ] The next action exercises the selected path.
- [ ] Changed links resolve, and an independent reader selects the expected row without guessing.

Use the [public documentation style standard](../standards/public-documentation-style.md) for affirmative wording and scoped claims. Validate local targets with the [front-door link witness](../quality/frontdoor-link-witness.md), then run the [external-reader dogfood rubric](https://github.com/anthony-chaudhary/fak/blob/main/docs/testing/external-reader-dogfood.md) with a realistic choice goal.

## Publication record

Record the route and revision, expected reader condition and choice, completed checks, changed-link results, independent read-back, and every repaired or open ambiguity. Publish only when the reader restates the choices, selects the expected row, gives the default and change condition, and proceeds to the named next action.
