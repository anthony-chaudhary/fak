---
title: "Frontier infrastructure source register"
description: "This register records source lineage, independence, and refresh triggers for report."
---

# Frontier infrastructure source register

This register records source lineage, independence, and refresh triggers for report
lifecycles that cannot safely be represented by one headline. **Cutoff:** 2026-08-27.

## NVIDIA / Hugging Face acquisition-report lifecycle

The record qualifies for the rumor ledger because two genuinely independent outlets
published original, source-attributed reporting. They did not establish the same transaction
state: Business Insider reported talks without a deal, while The Information later reported
an agreement. That conflict is evidence of a report lifecycle, not proof that signing or
closing occurred.

| Published | Source | Role in record | Supported fragment | Independence and limit |
|---|---|---|---|---|
| 2026-08-27 00:34:46Z (2026-08-26 PDT) | [Business Insider — “Nvidia has been in talks to acquire Hugging Face for more than $13 billion”](https://www.businessinsider.com/nvidia-in-talks-to-buy-hugging-face-13-billion-dollars-2026-8) | Original talks report | Recent acquisition discussions reportedly valued Hugging Face above $13B; the report said no deal had been reached and talks could fail. | Original Business Insider reporting attributed to a person familiar with the matter; independent of later The Information sourcing. It does not establish signing, terms, regulatory clearance, or close. |
| 2026-08-26 18:26 PDT | [The Information — “Nvidia Agrees to Buy Open Source AI Platform Hugging Face For $12.9 Billion”](https://www.theinformation.com/articles/nvidia-agrees-buy-open-source-model-repository-hugging-face-12-9-billion) | Original reported-agreement source and indexed URL | NVIDIA reportedly agreed to buy Hugging Face for $12.9B, attributed to a person with knowledge of the agreement. | Separate outlet and source attribution from Business Insider. It is still a report, not a party announcement, published agreement, regulatory filing, or closing notice. |
| 2026-08-26 23:32 PDT | [TechCrunch — “Nvidia closes in on Hugging Face acquisition”](https://techcrunch.com/2026/08/26/nvidia-closes-in-on-hugging-face-acquisition/) | Conflict and party-response check | Presented the $12.9B agreement report alongside Business Insider's no-signed-agreement account and said neither NVIDIA nor Hugging Face had responded. | Independent publication, but it attributes the transaction claims to the two original reports and is not a third corroborating source for agreement. Its direct contribution is the response-status check. |

An earlier [Business Insider sale-process report published 2026-08-23](https://www.businessinsider.com/hugging-face-could-be-acquired-13-billion-2026-8)
said Hugging Face was evaluating bidder interest around $13B or more and that no deal had
been reached. It supplies prior lifecycle context; it is not counted as an independent
outlet from Business Insider's later NVIDIA-specific report.

### Primary-party confirmation check

| Checked 2026-08-27 | Result | Boundary |
|---|---|---|
| [NVIDIA News Archive](https://nvidianews.nvidia.com/news) | No NVIDIA / Hugging Face acquisition or signing announcement was present; the latest listed items were dated 2026-08-26. | Absence from the reviewed archive is not proof that private signing did not occur. |
| [Hugging Face official blog index](https://huggingface.co/blog) | No NVIDIA acquisition, signing, or closing announcement was present in the reviewed official index through 2026-08-27. | The index includes community content and is not a corporate filing system; this is a bounded party-surface check. |

Primary party confirmation could not be found as of 2026-08-27. The record therefore
remains `reported_agreement_open` with `last_checked_at` 2026-08-27 and `expires_at`
2026-11-30.

### Lifecycle and unresolved fields

| State | Evidence at cutoff | What would resolve it |
|---|---|---|
| Talks / sale process | Independently reported by Business Insider. | Party statement that talks ended, or later transaction evidence. |
| Reported agreement | Reported by The Information; repeated with attribution by other outlets. | NVIDIA or Hugging Face announcement, executed agreement disclosure, or authoritative regulatory filing. |
| Signing / announcement | **Unresolved.** No primary party confirmation found. | Party announcement or disclosed executed agreement. |
| Transaction structure / terms | **Unresolved.** Do not infer cash/stock mix, adjustments, governance, approvals, conditions, termination rights, or operating commitments. | Disclosed agreement, party filing, or authoritative regulatory document. |
| Regulatory conditions | **Unresolved.** No jurisdictions, review status, remedies, or clearance conditions established. | Regulator filing/decision or party disclosure identifying the review and conditions. |
| Close | **Unresolved.** A reported agreement does not transfer ownership. | Party closing announcement, completion filing, or authoritative ownership record. |

Refresh on a NVIDIA or Hugging Face announcement, an executed-agreement or regulatory
filing, a termination report, a closing notice, or 2026-11-30. Rewrites and syndications of
Business Insider, The Information, TechCrunch, or Reuters do not count as independent
corroboration.
