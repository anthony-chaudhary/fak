---
title: "Cross-harness trajectory audit"
description: "Schema: fak-trajectory-audit/1"
---
# Cross-harness trajectory audit

Schema: `fak-trajectory-audit/1`

## Run provenance

- Captured 2026-08-21 from the default Claude and Codex transcript homes with a clean archive of `main` at `d10e2f7283`; the audit leaf was `internal/trajectory@r3+g0e0041e503`.
- Command: `fak trajectory audit --since 7d --jsonl docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.jsonl --md docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md`.
- Outcome: exit 1 (`TRAJECTORY_SCHEMA_REFUSED`). Totals are exact for 33,745 supported usage records; 94 unsupported records were refused and never estimated.
- Privacy review: the versioned rows contain aggregate fields, relative source paths, and session identifiers only; the committed artifacts contain no syntactic absolute paths or transcript content fields.
- Follow-ups: [#8508](https://github.com/anthony-chaudhary/fak/issues/8508) covers 48 Claude records missing `input_tokens`; [#8509](https://github.com/anthony-chaudhary/fak/issues/8509) covers 46 Codex cumulative-usage decreases; [#8512](https://github.com/anthony-chaudhary/fak/issues/8512) covers the deterministic rank-one cache-read concentration.

## Exact totals

- Sources: 2; sessions: 889; files scanned: 889/889; records: 213259.
- Input: 424036274; output: 8989266; cache create: 0; cache read: 3084145664 exact tokens.
- Input:output ratio: 390.263; cache-create burden: 0.00%.
- Repeated failures: 12; mutation churn: 92; hook p95: 1557 ms.

## Source denominator

| Source | Root | Present | Files scanned/discovered | Records | Exact usage | Applied usage | Duplicates | Refused |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| claude | `claude/projects` | true | 395/395 | 22950 | 4748/4796 | 2712 | 2036 | 48 |

claude token semantics: message usage buckets are disjoint; duplicate message ids are counted once.
| codex | `codex/sessions` | true | 494/494 | 190309 | 28997/29043 | 275 | 0 | 46 |

codex token semantics: final cumulative input includes cached/cache-write subsets; fresh input is exact subtraction.

## Token-weighted bottlenecks

| Rank | Source/session | Accounted tokens | Dominant bucket |
|---:|---|---:|---|
| 1 | codex/`01a021aa-dc64-79b3-9e6e-6dd692a5ec81` | 144497502 | cache_read (138797056) |
| 2 | codex/`01a021ab-8054-7dc2-97bc-99032cda9029` | 138849258 | cache_read (110813696) |
| 3 | codex/`01a022ee-1cd6-7123-b1d1-ae152c967ca7` | 135679495 | cache_read (134222592) |
| 4 | codex/`01a022ba-cbda-74b0-8c4a-bbdcebcaaccd` | 129557818 | cache_read (127719424) |
| 5 | codex/`01a022db-c48b-7cd2-ae58-18d74fd56201` | 95816356 | cache_read (94121728) |
| 6 | codex/`01a021fa-8dca-7953-b3d5-e5ecae9cd37d` | 77242414 | cache_read (76012032) |
| 7 | codex/`01a0275d-8e45-7180-ac08-306409efd5bb` | 70366947 | cache_read (49717248) |
| 8 | codex/`01a02327-b2d2-7280-9cba-01a1b625a88a` | 69675672 | cache_read (68653568) |
| 9 | codex/`01a021fa-9283-79f3-a33b-abd71969d049` | 68623543 | cache_read (67437824) |
| 10 | codex/`01a021fa-9b65-77c0-9431-02eb349623e5` | 68404039 | cache_read (67504896) |
| 11 | codex/`01a02760-702c-7521-aaaa-c04a26322f12` | 68129292 | cache_read (63307008) |
| 12 | codex/`01a024b0-98bb-71c2-bc18-9cfd28bcbc76` | 56182236 | cache_read (44625408) |
| 13 | codex/`01a024b0-30cf-7252-b033-6e85c3f3f928` | 52987926 | cache_read (46666240) |
| 14 | codex/`01a02763-3916-7421-b4a4-ec4b3a45d5f5` | 49774096 | cache_read (45678848) |
| 15 | codex/`01a024b1-1b87-7121-8066-723c09328aef` | 48409277 | cache_read (45656576) |
| 16 | codex/`01a02360-1a43-7570-b875-34b91cca49f7` | 46537779 | cache_read (45893632) |
| 17 | codex/`01a022db-c41a-79a2-b633-36c31fb2bf9f` | 45044217 | cache_read (44158976) |
| 18 | codex/`01a022ee-1c02-7830-8d7b-55941be900fa` | 44906584 | cache_read (44232192) |
| 19 | codex/`01a02762-38f0-7093-8817-682e40751873` | 44866813 | input (25896813) |
| 20 | codex/`01a024b1-3652-7053-8c9b-9e782f5c8c0b` | 44161657 | cache_read (42906112) |
| 21 | codex/`01a02764-891b-7ae0-a8ed-665d4ac1a7d6` | 44130804 | cache_read (23995392) |
| 22 | codex/`01a024aa-c24f-7791-b8cb-04e4e6d687fb` | 43548854 | cache_read (42064896) |
| 23 | codex/`01a02237-53fa-7cb2-bf17-8dc90d7ce97e` | 41488641 | cache_read (40706304) |
| 24 | codex/`01a024af-5574-79f2-b519-53d409cc5543` | 39949832 | cache_read (32272128) |
| 25 | codex/`01a024b1-5a95-7fe2-9daf-fe3d49157184` | 39878892 | cache_read (38852864) |
| 26 | codex/`01a02237-50b1-7d43-8c88-1369dfe1bc92` | 39745612 | cache_read (39292672) |
| 27 | codex/`01a02368-0aee-7ca0-8853-008dbadc0d86` | 38879396 | cache_read (38297344) |
| 28 | codex/`01a022de-a96c-78e1-87a0-0d8250da9f0c` | 38498190 | cache_read (37832960) |
| 29 | codex/`01a024b1-49d7-7c50-97f9-354d6dcfc292` | 38064036 | cache_read (37064704) |
| 30 | codex/`01a022db-c3fe-7cc3-afe1-ec2a2d96aca6` | 36740660 | cache_read (35986688) |
| 31 | codex/`01a0275c-b44f-7910-a07d-47d0f425a8c1` | 34972529 | cache_read (29335552) |
| 32 | codex/`01a024d5-e23f-7691-b4c7-b649959043f1` | 34469817 | cache_read (20272896) |
| 33 | codex/`01a022f6-25c4-74c2-99d3-8ae5c7579563` | 32646213 | cache_read (31874048) |
| 34 | codex/`01a027dc-db05-7253-bda8-54eaeb61f481` | 29170531 | cache_read (28589568) |
| 35 | codex/`01a021c0-c6b2-73c3-9df9-6dd8c1f7d126` | 28225532 | cache_read (27634944) |
| 36 | codex/`01a0278a-97ed-7063-8c2c-6d84da58402e` | 26211999 | cache_read (25620992) |
| 37 | codex/`01a021c0-dda4-7710-8f31-e6ee3e2dfd4a` | 22905762 | cache_read (22560000) |
| 38 | codex/`01a02237-4f8c-7110-b696-33bcd857786a` | 22823191 | cache_read (22034560) |
| 39 | codex/`01a022d1-9da8-77e3-b521-4d6e8cdcdfd5` | 22355693 | cache_read (21895424) |
| 40 | codex/`01a027e8-52a3-7a60-a0f8-b64292689d80` | 22045981 | cache_read (21526272) |
| 41 | codex/`01a0275f-6582-7a62-83f7-1b9504d92e3d` | 22038610 | cache_read (21515648) |
| 42 | codex/`01a023b8-f0ec-7743-84c3-656b197e1210` | 21459351 | cache_read (20960000) |
| 43 | codex/`01a021c0-c367-7831-a2ee-93a6b39a7595` | 20774935 | cache_read (20304512) |
| 44 | codex/`01a022ba-2f1d-77e1-b92b-a825ee2a9bc7` | 20690302 | cache_read (20223232) |
| 45 | codex/`01a024eb-ad58-7723-b271-3fe28bb74884` | 20153282 | cache_read (17261568) |
| 46 | codex/`01a027b0-d928-7f50-9874-f28f806779c8` | 19994988 | cache_read (19464192) |
| 47 | codex/`01a022af-cf8f-7bf2-98ee-001ed898064a` | 19675648 | cache_read (19057152) |
| 48 | codex/`01a022af-cf49-7410-93e9-4b97d9b21917` | 19312528 | cache_read (18903808) |
| 49 | codex/`01a027ea-b503-7fe3-9436-e33d248b87e9` | 19017401 | cache_read (18517248) |
| 50 | codex/`01a023b8-f100-77b0-89c8-b1128945acda` | 18371604 | cache_read (17826048) |
| 51 | codex/`01a021bf-9926-7d20-b6a1-ca16638072d8` | 17716137 | cache_read (17052416) |
| 52 | codex/`01a023b8-f3fe-7ef2-b0da-9bb8932a0926` | 17642206 | cache_read (17215488) |
| 53 | codex/`01a02379-4aed-7221-844c-99dec88ff6ca` | 17614751 | cache_read (13116928) |
| 54 | codex/`01a02375-a849-7912-b7f7-839920de6ea1` | 17214984 | cache_read (15091712) |
| 55 | codex/`01a02376-0957-78b1-aa0c-f5892b98ba99` | 16994366 | cache_read (12937216) |
| 56 | codex/`01a0276b-db53-7c82-b338-f24b23f38361` | 16266338 | cache_read (15741312) |
| 57 | codex/`01a027e8-8ce7-77f3-98a7-4ddf18c32f78` | 16173777 | cache_read (15837440) |
| 58 | codex/`01a0236f-b56f-7811-99bb-9302a4f706b1` | 15703655 | cache_read (15279360) |
| 59 | codex/`01a023b8-f3c3-7333-ae8e-f46b7b36d77b` | 15135163 | cache_read (14629120) |
| 60 | claude/`57d7b674-8d34-4621-8905-e930a3376efd` | 15110187 | input (15075909) |
| 61 | codex/`01a022af-cf63-7531-9834-c298c0716e8b` | 14823126 | cache_read (14547456) |
| 62 | codex/`01a027b1-3062-7f72-9f71-75a072791f90` | 14622853 | cache_read (14114304) |
| 63 | codex/`01a02376-8a44-7f31-bc75-45a7546c4a17` | 14079344 | cache_read (8489472) |
| 64 | codex/`01a02804-4088-76c1-b9bc-f64910b464c0` | 13960991 | cache_read (13645312) |
| 65 | codex/`01a02384-0754-74b1-a7bd-66335274c18f` | 13650236 | cache_read (10952704) |
| 66 | codex/`01a02394-f05d-7e72-b236-c361e40b61be` | 13561187 | cache_read (13254144) |
| 67 | codex/`01a0237c-ca59-7b30-baba-3d3fd1701b65` | 13487388 | cache_read (12738304) |
| 68 | codex/`01a02800-6300-7dd1-ac61-8cc22ec59e04` | 13323019 | cache_read (13027072) |
| 69 | codex/`01a027dc-478a-7753-85d8-3ccb4c005897` | 13307588 | cache_read (13019392) |
| 70 | codex/`01a02377-bc01-7450-92e9-fe0f5e9cc819` | 13246353 | cache_read (11782144) |
| 71 | codex/`01a0237a-7ca1-7782-bbc6-8200057ebeb9` | 13238797 | cache_read (11360256) |
| 72 | codex/`01a022cc-dd9c-7fa0-9e73-627c5f30091d` | 12917916 | cache_read (12666368) |
| 73 | codex/`01a023b8-f018-7db3-be96-82c0022422f3` | 12520660 | cache_read (12016640) |
| 74 | codex/`01a0237c-ca5a-7012-b06d-8dba0d6f1c22` | 12130332 | cache_read (11617792) |
| 75 | codex/`01a0275b-9df9-7fd0-8495-222748d53325` | 11880820 | cache_read (10185728) |
| 76 | codex/`01a027dd-0f50-7673-a421-2a90df72a416` | 11817337 | cache_read (11523840) |
| 77 | codex/`01a0237c-cac3-75b2-976a-6f4544480041` | 11735204 | cache_read (11265792) |
| 78 | codex/`01a02380-6ceb-7673-9877-afc008dab804` | 11663531 | cache_read (8756224) |
| 79 | codex/`01a0237c-ca6c-7833-b792-6f56afe81e9a` | 11294906 | cache_read (10698752) |
| 80 | codex/`01a023b8-f0ec-7743-84c3-656b197e1210` | 11203638 | cache_read (10520576) |
| 81 | codex/`01a024d1-3726-72f0-a553-ae58aff1afa6` | 11121334 | cache_read (8583168) |
| 82 | codex/`01a0237c-631c-7fd0-b666-0c1a7668dc21` | 11085461 | cache_read (9359872) |
| 83 | codex/`01a023b8-f018-7db3-be96-82c0022422f3` | 10665122 | cache_read (10214400) |
| 84 | codex/`01a0237c-ca7e-71e2-9f48-03356cca4c57` | 10654782 | cache_read (10071552) |
| 85 | codex/`01a0237c-cac3-75b2-976a-6f4544480041` | 10458805 | cache_read (9977600) |
| 86 | codex/`01a0237c-ca7e-71e2-9f48-03356cca4c57` | 10436533 | cache_read (9957888) |
| 87 | codex/`01a0276b-db50-7f81-aa9b-352b690425ab` | 10272184 | cache_read (10012416) |
| 88 | codex/`01a02804-6fa5-7822-8466-a9afaa589939` | 10218229 | cache_read (9957632) |
| 89 | codex/`01a0237c-ca5a-7012-b06d-8dba0d6f1c22` | 9854298 | cache_read (9427968) |
| 90 | codex/`01a027e8-be3c-7360-bf04-20f7a10f2767` | 9759154 | cache_read (9514496) |
| 91 | codex/`01a02516-c795-7870-9cbf-08cd7eeb404f` | 9552228 | cache_read (6047232) |
| 92 | codex/`01a0237c-cac3-75b2-976a-6f4544480041` | 9451386 | cache_read (8939776) |
| 93 | codex/`01a024a3-b746-7991-bab5-b7379e439dc2` | 9327323 | cache_read (6042624) |
| 94 | claude/`78bb0cad-8d00-4f45-b3de-d43e66e67ed6` | 8768112 | input (8732141) |
| 95 | codex/`01a0237c-ca7e-71e2-9f48-03356cca4c57` | 8618285 | cache_read (8245504) |
| 96 | codex/`01a0237c-cac3-75b2-976a-6f4544480041` | 8365362 | cache_read (7905792) |
| 97 | codex/`01a027b3-1bf3-7493-b3fe-b2534cf27eff` | 8161787 | cache_read (7854848) |
| 98 | codex/`01a023b8-f018-7db3-be96-82c0022422f3` | 8023427 | cache_read (7707904) |
| 99 | codex/`01a024b3-4f6c-7dc2-b522-6f6a6a85b822` | 7916081 | cache_read (7507456) |
| 100 | codex/`01a024ea-df18-7691-b7aa-977c18d221fe` | 7811807 | cache_read (5418496) |
| 101 | codex/`01a024da-1017-73c1-a3c4-114f49219f65` | 7768944 | cache_read (7340288) |
| 102 | codex/`01a023c4-7ea9-75c3-997d-4a1143300b4e` | 7726634 | cache_read (7298304) |
| 103 | codex/`01a02770-8ce0-7810-b4e9-857ecfd43f6b` | 7720942 | cache_read (7368192) |
| 104 | codex/`01a024da-187f-7e13-a4b8-3674e66c3edd` | 7576421 | cache_read (7187968) |
| 105 | codex/`01a02770-8ce0-7810-b4e9-857ecfd43f6b` | 7450654 | cache_read (7168000) |
| 106 | codex/`01a024d6-f6a0-7a42-9dee-66f961b05f9a` | 7277238 | cache_read (5626368) |
| 107 | codex/`01a02770-8ce0-7810-b4e9-857ecfd43f6b` | 7107094 | cache_read (6824448) |
| 108 | codex/`01a02770-8ce0-7810-b4e9-857ecfd43f6b` | 6860582 | cache_read (6499584) |
| 109 | codex/`01a02808-2a70-7c03-853a-7c0ede90da0c` | 6750318 | cache_read (6555904) |
| 110 | codex/`01a024ea-635a-7eb0-ba4e-d3cabeb342d5` | 6605628 | cache_read (5295104) |
| 111 | claude/`394d2e98-d003-44e9-8ee8-8c603a905fae` | 6541776 | input (6528038) |
| 112 | codex/`01a024f7-b7ff-73b1-81e6-0aa81ecc57b3` | 6280368 | cache_read (5927040) |
| 113 | codex/`01a0237c-ca2d-7152-8d05-111a8d073e8f` | 6254309 | cache_read (5543680) |
| 114 | codex/`01a0275f-d718-7b92-a430-168407ad49f4` | 6252978 | cache_read (5994496) |
| 115 | codex/`01a024b5-c2b3-71c0-9d9f-a0aa15f8336b` | 6179625 | cache_read (5683456) |
| 116 | codex/`01a024da-051d-71a1-b419-f01ea5456608` | 6166642 | cache_read (5796608) |
| 117 | codex/`01a02509-6485-79c1-9cfb-ad0ccb219e56` | 6095410 | cache_read (5926144) |
| 118 | codex/`01a0275f-e2cd-7071-8b7c-da0fb89351e2` | 5983386 | cache_read (5757696) |
| 119 | codex/`01a027ac-e5e0-7990-a692-b501e16ae3fd` | 5948882 | cache_read (5721088) |
| 120 | codex/`01a0237c-ca34-70b1-a43b-b822fd55c4f9` | 5925021 | cache_read (5314048) |
| 121 | codex/`01a0238f-0588-7021-bb01-78233ab56260` | 5793889 | cache_read (5553408) |
| 122 | codex/`01a023b8-f0ec-7743-84c3-656b197e1210` | 5787474 | cache_read (5379072) |
| 123 | codex/`01a024cc-eead-7d60-a807-ad80705e0341` | 5547012 | cache_read (5262592) |
| 124 | codex/`01a023b8-f0ec-7743-84c3-656b197e1210` | 5428190 | cache_read (4926208) |
| 125 | codex/`01a024cd-0690-76c1-ace2-7be218fd4087` | 5376639 | cache_read (5072128) |
| 126 | claude/`71753b04-a9a3-4bd7-98ec-b57eabee4366` | 5107596 | input (5098030) |
| 127 | codex/`01a024cc-fcb5-79c2-837c-6245f2a54bae` | 5084080 | cache_read (4824064) |
| 128 | codex/`01a0237c-ca5a-7012-b06d-8dba0d6f1c22` | 5083808 | cache_read (4779264) |
| 129 | codex/`01a0237e-0abf-7752-87c8-1c5bb9c28383` | 5023250 | input (3256049) |
| 130 | claude/`9d957588-ca81-4c62-8768-d3694f91530f` | 4944615 | input (4934214) |
| 131 | claude/`1168ee04-d59a-451f-8b2a-fc4b1fff1ea6` | 4809421 | input (4788820) |
| 132 | claude/`2e238b6a-3ddd-41a1-bb31-f25e0a71f762` | 4787768 | input (4774944) |
| 133 | codex/`01a0237c-ca7e-71e2-9f48-03356cca4c57` | 4686446 | cache_read (4431360) |
| 134 | codex/`01a0275f-cd9b-7f61-bf54-df4b15754821` | 4515053 | cache_read (4246016) |
| 135 | claude/`fd75e387-1c12-481a-af75-63acfa781978` | 4478231 | input (4461719) |
| 136 | codex/`01a024e5-ce55-74a0-8b81-051efacf2f05` | 4421008 | cache_read (2536960) |
| 137 | claude/`40abfdb7-625d-4c1c-845f-dfe78cf74447` | 4348668 | input (4335851) |
| 138 | codex/`01a024d0-d5df-7813-9196-3f7483fda341` | 4282875 | cache_read (4091648) |
| 139 | codex/`01a0237c-ca5a-7012-b06d-8dba0d6f1c22` | 4143662 | cache_read (3847424) |
| 140 | claude/`db4d11c5-9dc2-4e05-8e78-4cc949a77312` | 4128384 | input (4116025) |
| 141 | claude/`fb1d1e1f-cc44-4409-b5eb-894ea928bbf9` | 4084781 | input (4072017) |
| 142 | claude/`c9c0ce6b-b2e6-4c46-bc78-8d55f34afd06` | 4066488 | input (4058413) |
| 143 | codex/`01a02504-65f0-7d73-9ef5-7402010b2ac8` | 3949595 | cache_read (3711104) |
| 144 | claude/`a8a64b4c-605e-450c-86ef-22d674c8cd15` | 3916348 | input (3904549) |
| 145 | claude/`75ebc007-34ff-4b49-9556-b3fd8f4854c1` | 3867151 | input (3850141) |
| 146 | codex/`01a024d0-bdd9-7252-8e4d-a7825a1d2d3d` | 3830850 | cache_read (3597440) |
| 147 | codex/`01a02509-410b-7fb1-81a7-d96da4d93eba` | 3811429 | cache_read (3658496) |
| 148 | claude/`3949540d-9340-495c-877d-6bc5b07b42ca` | 3770885 | input (3759613) |
| 149 | codex/`01a024d2-3e10-74e1-81a5-136b105bc30d` | 3731729 | cache_read (3580160) |
| 150 | codex/`01a024d2-4a04-78e2-9d9f-54659ed0908d` | 3623738 | cache_read (3443200) |
| 151 | codex/`01a02521-fc9c-7ef2-a1f2-bd0443059315` | 3482259 | cache_read (3153664) |
| 152 | codex/`01a024b3-42bf-7723-b20c-a821cd1abb75` | 3468013 | cache_read (3146240) |
| 153 | codex/`01a0236f-b59f-76b2-926f-f59a04ac5da3` | 3409250 | cache_read (3219072) |
| 154 | codex/`01a02522-5baa-7fa0-a377-19557ec3b815` | 3389020 | cache_read (3072000) |
| 155 | codex/`01a022c3-d4ae-72b3-a066-0ba979b11c23` | 3288448 | cache_read (3108608) |
| 156 | codex/`01a0279f-15a8-7bb2-80f6-6811fa90d3ed` | 3288063 | cache_read (3077120) |
| 157 | codex/`01a024b3-44c8-7973-8e67-d999bbf8edcf` | 3140340 | cache_read (2824192) |
| 158 | codex/`01a024f7-c4c6-7d83-8cc0-39045427cafe` | 3103937 | cache_read (2941184) |
| 159 | codex/`01a023c4-4f24-7781-8587-23687aa49262` | 3070927 | cache_read (2761472) |
| 160 | codex/`01a02522-13a2-7301-87f8-093ad212e75e` | 3055849 | cache_read (2764032) |
| 161 | codex/`01a02503-e7e4-7013-a419-56895a4b8726` | 3052660 | cache_read (2946560) |
| 162 | codex/`01a024b5-cbe8-7430-9128-26da0ea0523d` | 3027706 | cache_read (2839808) |
| 163 | codex/`01a02524-90a8-7741-95d1-0d864c3ae552` | 2996080 | cache_read (2783744) |
| 164 | codex/`01a02509-e1f4-7333-9d34-85d8fab4fe70` | 2939544 | cache_read (2781184) |
| 165 | codex/`01a024b5-d613-7eb3-b000-0f4e443bfb55` | 2879722 | cache_read (2673152) |
| 166 | claude/`b5eeb61b-dd61-47dc-9756-47ee60f71be8` | 2842265 | input (2834798) |
| 167 | codex/`01a02521-8e5b-72c3-a634-eb1a4f8bea12` | 2826448 | cache_read (2458880) |
| 168 | claude/`0aa797c4-c629-4f5a-9b5d-04a6e40360d7` | 2811989 | input (2787358) |
| 169 | codex/`01a02761-0de2-7122-80ad-74d0ab956782` | 2798291 | cache_read (2285568) |
| 170 | codex/`01a02521-9766-70e1-9f23-d395d9ef76ad` | 2796564 | cache_read (2533376) |
| 171 | claude/`f3f9b088-45af-41c3-9309-791082badd2d` | 2790604 | input (2779183) |
| 172 | codex/`01a02509-abfb-7a32-bac9-c747400c699f` | 2768343 | cache_read (2616064) |
| 173 | codex/`01a022d9-3d5f-76e0-a66b-8a209a55b83b` | 2748153 | cache_read (2581504) |
| 174 | codex/`01a023b8-f018-7db3-be96-82c0022422f3` | 2728195 | cache_read (2540288) |
| 175 | codex/`01a02509-d5fa-7712-93fa-78b16d28a516` | 2711879 | cache_read (2558720) |
| 176 | codex/`01a02504-30ee-7963-9179-77aa6fb2b368` | 2571008 | cache_read (2371968) |
| 177 | codex/`01a0250c-2d52-7190-9ded-7378292bb858` | 2533553 | cache_read (2376192) |
| 178 | claude/`59ecdca6-baa8-41ca-b84e-85c8e63aadc1` | 2504010 | input (2494566) |
| 179 | codex/`01a022c3-d44a-7d32-8d13-5ad93422a27a` | 2430894 | cache_read (2257408) |
| 180 | claude/`b6ac35a5-6dcc-4ded-9053-fee880f2d8a4` | 2429530 | input (2419317) |
| 181 | codex/`01a02522-79c1-71f3-aa35-a2823be20ba2` | 2416211 | cache_read (2212352) |
| 182 | codex/`01a024f7-f96a-7443-b561-f14541946dbc` | 2413119 | cache_read (2283520) |
| 183 | codex/`01a0250c-a980-7ac1-8a9b-bcc4efe74e3d` | 2398516 | cache_read (2254592) |
| 184 | codex/`01a02524-76b5-7902-9004-df88e9fc68e4` | 2374245 | cache_read (2165248) |
| 185 | claude/`1473e7b6-3068-4b6d-83e3-0c212bcac819` | 2334592 | input (2317184) |
| 186 | codex/`01a02516-91e8-78d3-b038-9399c30fc5e2` | 2268017 | cache_read (2150656) |
| 187 | claude/`66a44d72-94cd-456a-9145-b8d9383feb1d` | 2203280 | input (2197602) |
| 188 | claude/`8da2fd3f-5473-49d6-aef2-6311137cb8ac` | 2184520 | input (2166640) |
| 189 | claude/`f115c608-6d4e-4187-9f8b-76b61fa8afbb` | 2165706 | input (2156373) |
| 190 | codex/`01a024f8-47bf-7021-a7f9-698a620d7c33` | 2138240 | cache_read (1956096) |
| 191 | codex/`01a022ba-cbda-7d50-97bc-b1c3ea09bc0c` | 2113228 | cache_read (1933056) |
| 192 | claude/`23b75576-cc0c-4388-bef9-c7747db0f442` | 2109684 | input (2098542) |
| 193 | codex/`01a024c9-6314-7283-bc1c-1eb4dd6afdb4` | 2046426 | cache_read (1811584) |
| 194 | codex/`01a021aa-3470-7e63-a20b-22d745c8c22d` | 2044169 | cache_read (1280000) |
| 195 | codex/`01a02524-8320-7ac0-94e9-1d535dc390ef` | 2007311 | cache_read (1806848) |
| 196 | codex/`01a0250c-5953-7452-a2cc-4f84bfe2067b` | 1989029 | cache_read (1525632) |
| 197 | codex/`01a024d0-c9d7-7a61-84e2-50145df97143` | 1954605 | cache_read (1818112) |
| 198 | codex/`01a024d2-5530-7e61-8cd0-26837e2d91b4` | 1951535 | cache_read (1839360) |
| 199 | codex/`01a02516-db3e-7781-9dc2-8c8c63af0a63` | 1933067 | cache_read (1842176) |
| 200 | codex/`01a024c0-1dd5-7860-a2f0-be075835cecf` | 1922914 | cache_read (1790976) |
| 201 | codex/`01a024c0-1287-74b3-bef8-899a628b0cb1` | 1919526 | cache_read (1757312) |
| 202 | claude/`09edc97e-ada0-4060-90fe-8eefeaa7dc1e` | 1849847 | input (1842425) |
| 203 | codex/`01a024f8-3668-7c81-a39e-c8eba2bda38d` | 1848035 | cache_read (1677440) |
| 204 | codex/`01a024c9-726b-7f50-a68a-d53238fd8a21` | 1725080 | cache_read (1608192) |
| 205 | codex/`01a02503-db75-7671-9404-c294c69f7c58` | 1715902 | cache_read (1601536) |
| 206 | codex/`01a02522-040c-7b51-b462-d3c329a677aa` | 1689660 | cache_read (1482752) |
| 207 | codex/`01a0250c-22bd-71c3-b60a-03dbf7b0aca1` | 1687491 | cache_read (1569536) |
| 208 | codex/`01a0237e-d07e-7412-85de-37fdf909e2aa` | 1681874 | cache_read (1277952) |
| 209 | codex/`01a02522-6dc3-71e3-a7fd-165d2cae6171` | 1681455 | cache_read (1477632) |
| 210 | codex/`01a02521-d7c5-77b0-949d-049e6ec3ddbf` | 1676716 | cache_read (1102080) |
| 211 | codex/`01a024c9-7c61-7662-9b92-f34d643d808c` | 1666907 | cache_read (1539072) |
| 212 | claude/`67a80318-2449-4283-ad4f-89726df1190b` | 1649952 | input (1641144) |
| 213 | codex/`01a024f8-7143-7253-9456-c3143f8d10c3` | 1648638 | cache_read (1496576) |
| 214 | claude/`ef59fab7-ed74-444d-975e-fb71c9924b41` | 1616166 | input (1605446) |
| 215 | codex/`01a02509-39c8-7622-bad1-959eee761c04` | 1602329 | cache_read (1218688) |
| 216 | claude/`359157fa-0006-4825-9823-ae3626e9ee54` | 1569703 | input (1563968) |
| 217 | codex/`01a0250c-9ed0-78d3-a37f-cac5cfa25cfc` | 1469425 | cache_read (1370112) |
| 218 | claude/`fb1d1e1f-cc44-4409-b5eb-894ea928bbf9` | 1457676 | input (1443126) |
| 219 | codex/`01a02516-a442-7b53-aa4c-203493ee178c` | 1455636 | cache_read (886912) |
| 220 | codex/`01a02509-9e0b-7d13-8480-38f52cb66667` | 1444936 | cache_read (1161088) |
| 221 | codex/`01a022c5-22b1-7871-bdb1-4c3667307563` | 1444660 | cache_read (1296384) |
| 222 | claude/`138fda3b-fd5e-42a3-b3e4-7484734e203b` | 1434018 | input (1424965) |
| 223 | codex/`01a027ba-6569-7ae2-9abe-db3c12e9109e` | 1414169 | cache_read (1308416) |
| 224 | codex/`01a022c3-d4b0-7622-8383-f809d70f9b7d` | 1390442 | cache_read (1239296) |
| 225 | codex/`01a02516-e1db-7a60-9a52-b5ae64101a18` | 1379663 | cache_read (1295872) |
| 226 | codex/`01a022de-a96a-7c93-bdc6-a5fa676f1061` | 1378339 | cache_read (1300736) |
| 227 | claude/`5411b312-09f1-44bb-92c8-eb4f02cbb172` | 1376836 | input (1368659) |
| 228 | codex/`01a02504-78db-7481-8e83-71d8f2f70aee` | 1367663 | cache_read (1274112) |
| 229 | claude/`0aa797c4-c629-4f5a-9b5d-04a6e40360d7` | 1363908 | input (1349392) |
| 230 | claude/`527ec17c-5f4d-46dd-aa39-c50ea3c23d87` | 1342435 | input (1336226) |
| 231 | codex/`01a02368-0ae9-7852-a2cc-60d4886b9954` | 1329084 | cache_read (1209344) |
| 232 | claude/`53621f42-f415-4013-b03a-3d5a9f8f6b67` | 1326765 | input (1322991) |
| 233 | claude/`9c889f11-60be-4166-83a3-f4f2ad71180c` | 1320854 | input (1314277) |
| 234 | claude/`187a9e90-30f1-4bd6-b885-8fc50d1c43ca` | 1311839 | input (1304130) |
| 235 | claude/`7a90c824-89e1-4704-aa74-2d1b49a22373` | 1302326 | input (1296001) |
| 236 | codex/`01a02509-6f96-7aa2-985c-6b6320bbe209` | 1288557 | cache_read (1194496) |
| 237 | claude/`7092ece1-8e8f-4d12-bacc-7fccc509e4dc` | 1276488 | input (1269679) |
| 238 | codex/`01a024c0-296a-7511-a06d-72299e42e3a8` | 1267824 | cache_read (1153536) |
| 239 | codex/`01a022a4-d028-7533-a658-0d8d170c9b00` | 1250813 | cache_read (1136896) |
| 240 | claude/`f74e393e-927a-4bd1-a4ea-41b6a47cd084` | 1217721 | input (1209001) |
| 241 | codex/`01a021fa-8efb-7552-a0bb-3b48a5354ee5` | 1194284 | cache_read (1074176) |
| 242 | claude/`2989d6b2-638a-4348-9176-19031be71e2a` | 1180678 | input (1174727) |
| 243 | codex/`01a021fa-9cdb-7a53-ae56-ca5cda886c3c` | 1174756 | cache_read (1075968) |
| 244 | claude/`cab70c67-250f-4c0f-a2f1-d66611b36b29` | 1158943 | input (1154799) |
| 245 | codex/`01a02818-e995-78d0-846e-ef7435495d2a` | 1147158 | cache_read (1019904) |
| 246 | claude/`9fdb0759-9958-49a8-8554-3bcb182feaeb` | 1136097 | input (1129608) |
| 247 | codex/`01a02368-0b20-7612-a258-35e89f487999` | 1122556 | cache_read (1016832) |
| 248 | claude/`8da2fd3f-5473-49d6-aef2-6311137cb8ac` | 1060317 | input (1052797) |
| 249 | claude/`678ff27c-91dc-42ac-9b9c-db3d90dddc6b` | 1058463 | input (1054261) |
| 250 | codex/`01a022ee-1c4f-73c3-a3dc-a1a7e9b6e1b0` | 998069 | cache_read (839168) |
| 251 | codex/`01a02521-e12f-7c80-ab33-d47b92f378df` | 995948 | cache_read (906752) |
| 252 | claude/`415686a9-bfaa-4202-98c7-dad2d5ac045e` | 995729 | input (993347) |
| 253 | claude/`d9e678c0-9955-4482-a090-8dd88282eb17` | 989210 | input (980541) |
| 254 | claude/`485da193-27cd-4f59-a4ff-0b4831a2c97b` | 986647 | input (982729) |
| 255 | codex/`01a0281b-abcc-7c32-813a-d8c684a2b12b` | 955117 | cache_read (846336) |
| 256 | claude/`9036939f-3129-4be5-9431-7a5a2ee65912` | 951438 | input (947857) |
| 257 | claude/`fe54e69f-55ca-4a52-aa8b-fa0294658cf7` | 927592 | input (922548) |
| 258 | codex/`01a02516-a040-7ee3-a780-132aff493d29` | 919939 | cache_read (683008) |
| 259 | claude/`086456b1-01f7-422b-8769-48dd6983a19a` | 918356 | input (906079) |
| 260 | claude/`d9e678c0-9955-4482-a090-8dd88282eb17` | 905951 | input (902617) |
| 261 | codex/`01a02800-993b-7080-afdf-88c8c3cffde4` | 905715 | cache_read (815616) |
| 262 | codex/`01a02383-be41-7ec2-8785-2070e6a2873d` | 884422 | cache_read (629248) |
| 263 | codex/`01a0250c-6bfe-7da2-9481-76a6b682d249` | 882085 | cache_read (786944) |
| 264 | codex/`01a027e8-be3c-7360-bf04-20f7a10f2767` | 881455 | cache_read (809984) |
| 265 | codex/`01a022ee-1c94-76a1-9385-71474f82b059` | 870711 | cache_read (797184) |
| 266 | codex/`01a0236f-b598-7422-a48a-f65e75575137` | 864416 | cache_read (761344) |
| 267 | codex/`01a02504-3f30-7933-848e-87e352500df3` | 851625 | cache_read (791808) |
| 268 | claude/`07f79cbe-9764-43e8-b86d-b38c0a019f6b` | 846376 | input (840893) |
| 269 | claude/`f24e3269-d9f3-4029-9a24-f032c1b74817` | 836539 | input (832103) |
| 270 | codex/`01a02383-c691-7592-ad8a-d385d6e40a89` | 834984 | cache_read (570368) |
| 271 | codex/`01a02383-bbd2-7090-9309-ffdf3cd78865` | 811937 | input (405771) |
| 272 | claude/`9faa2ee1-c561-4315-9740-69e71698afc1` | 788039 | input (785094) |
| 273 | codex/`01a0277b-6696-7243-a2dd-2e402e87705f` | 785492 | cache_read (686848) |
| 274 | codex/`01a02522-5040-7990-9fa5-5a2e1ea39c86` | 785480 | cache_read (581120) |
| 275 | codex/`01a022ee-1cdf-7362-8929-68f9c21f454d` | 764841 | cache_read (679424) |
| 276 | claude/`2356782a-851b-4b5b-bcd4-306f725dc59d` | 763134 | input (759026) |
| 277 | codex/`01a022ee-1bfd-7c01-b8e7-074108180359` | 757603 | cache_read (666880) |
| 278 | codex/`01a022c6-ed46-75d0-a806-8dbf25397959` | 749746 | cache_read (656640) |
| 279 | codex/`01a024f8-83e9-73b3-8e2a-453cf7e4e581` | 741617 | cache_read (639232) |
| 280 | codex/`01a02383-c6a4-7df2-97bd-ba2183252af4` | 723490 | cache_read (584192) |
| 281 | codex/`01a022e5-c378-7e32-aa6d-91edf1420396` | 703344 | cache_read (657152) |
| 282 | codex/`01a022ee-1cd4-76c3-9c96-93195e25a004` | 699288 | cache_read (613632) |
| 283 | codex/`01a022ba-cc12-7751-9ac8-a9903c4ce8f1` | 697384 | cache_read (623616) |
| 284 | codex/`01a024f7-ead4-7b31-85f6-f89b7167d965` | 693736 | cache_read (630528) |
| 285 | codex/`01a02383-c3d6-7981-b85d-706552b65f01` | 680093 | cache_read (452608) |
| 286 | codex/`01a0236f-b5b9-7561-8172-e96ce63b331e` | 672264 | cache_read (569088) |
| 287 | codex/`01a02383-b8aa-7123-a63e-95ac67b012d5` | 667738 | cache_read (381952) |
| 288 | codex/`01a022c8-e74b-7d90-9f83-4ebfbfe5ba86` | 661378 | cache_read (599040) |
| 289 | codex/`01a021c0-d9f2-7323-9932-5284d4534852` | 660932 | cache_read (578048) |
| 290 | claude/`fb1d1e1f-cc44-4409-b5eb-894ea928bbf9` | 655230 | input (645504) |
| 291 | claude/`e0aa32b3-1b67-4fe1-a983-1484c2f090b8` | 642661 | input (638667) |
| 292 | codex/`01a022db-c466-7b50-b2c3-7ed289c2083b` | 617655 | cache_read (556288) |
| 293 | codex/`01a02766-9b7b-71d2-8146-b14ce3646eac` | 611726 | cache_read (571520) |
| 294 | codex/`01a022a4-d902-7cf0-ab03-785f5449638a` | 611067 | cache_read (510720) |
| 295 | codex/`01a0236f-b581-7913-b168-1f2a97a2041c` | 590634 | cache_read (525312) |
| 296 | codex/`01a022b1-20aa-7483-8924-5603376bea35` | 573256 | cache_read (502528) |
| 297 | codex/`01a022d6-3556-7601-a6e1-7dcd4b10f9c9` | 570311 | cache_read (504320) |
| 298 | codex/`01a022d4-9356-7191-a343-3028eda33f08` | 561299 | cache_read (492288) |
| 299 | codex/`01a0277b-5a05-7680-895e-d564198b5ff7` | 539432 | cache_read (442368) |
| 300 | codex/`01a022d1-9e03-7b50-ae62-823929d25750` | 534380 | cache_read (460544) |
| 301 | claude/`fb1d1e1f-cc44-4409-b5eb-894ea928bbf9` | 513425 | input (500675) |
| 302 | claude/`7f702ee8-c7f3-4e4b-89e7-b1918c811b88` | 502397 | input (497020) |
| 303 | codex/`01a022d4-935b-79d0-8fd3-83df801ba43c` | 484093 | cache_read (412672) |
| 304 | codex/`01a02766-90ad-7603-8bc9-df331ca74bc8` | 474716 | cache_read (387456) |
| 305 | codex/`01a0281c-fcf4-7852-96b3-5badf1abeccf` | 470891 | cache_read (382976) |
| 306 | codex/`01a02528-89fc-72a3-aedb-bd1ea38bf695` | 450391 | cache_read (368640) |
| 307 | codex/`01a022b1-2049-7042-9c9f-3afcff0da226` | 443303 | cache_read (386048) |
| 308 | codex/`01a02528-880a-75c2-ad48-a38351fb95b5` | 436499 | cache_read (350208) |
| 309 | codex/`01a0277b-6e5f-7de1-a1e8-7ababb7888cb` | 432716 | cache_read (350464) |
| 310 | codex/`01a024d0-f171-7f50-b6b4-a7ab1ee50399` | 405998 | cache_read (294400) |
| 311 | codex/`01a022e5-c333-7883-8013-7c36650d6203` | 396390 | cache_read (336128) |
| 312 | codex/`01a022c0-ac31-7bc1-a1d7-1541fe01f786` | 389384 | cache_read (327936) |
| 313 | claude/`89a4b83f-ffd1-40c6-97f1-126a3c4e016e` | 367032 | input (365953) |
| 314 | codex/`01a022ee-1c29-7732-aade-b8bd3499fc78` | 356883 | cache_read (316416) |
| 315 | codex/`01a022af-d0a1-7501-b686-9c34b675b568` | 354754 | cache_read (306432) |
| 316 | codex/`01a02766-a7db-7560-ba78-f182a2113129` | 348389 | cache_read (255232) |
| 317 | claude/`0ada9f09-21be-4fb6-b3a0-d86ac9e270f2` | 332301 | input (330994) |
| 318 | codex/`01a02528-84f1-7002-a4f6-dffb05757415` | 317503 | cache_read (252416) |
| 319 | claude/`8f7eec95-de44-4453-9083-23655d34ac68` | 307187 | input (302939) |
| 320 | codex/`01a022d9-3d60-78f3-84be-b58a96631280` | 299933 | cache_read (257536) |
| 321 | codex/`01a02528-91a6-7080-9e33-2a85ca6cd0da` | 280517 | cache_read (224768) |
| 322 | codex/`01a02528-8c28-7f20-9d4d-5d65666f79c7` | 277881 | cache_read (221440) |
| 323 | claude/`39e04088-2146-49f9-b3ab-a23ee2624efc` | 276521 | input (275670) |
| 324 | codex/`01a02528-84ed-7f60-962f-d64111fecad8` | 271191 | cache_read (216576) |
| 325 | claude/`530ba588-e506-426b-8cc4-b16f831148f5` | 267701 | input (266705) |
| 326 | codex/`01a022cc-dda5-7fe0-be85-295377d626e0` | 255499 | cache_read (202496) |
| 327 | codex/`01a022c6-b893-7c30-89ed-88238f6bf61e` | 245096 | cache_read (202496) |
| 328 | codex/`01a02528-9118-7d33-acf3-a25b4a452b41` | 241844 | cache_read (185856) |
| 329 | codex/`01a02521-f081-7de3-b5a8-787f35974f88` | 226409 | cache_read (187136) |
| 330 | codex/`01a022c0-ac66-7650-9914-d2290e32dfda` | 212675 | cache_read (164608) |
| 331 | codex/`01a02528-8039-7f71-94dc-0f344bb583f4` | 184893 | cache_read (142080) |
| 332 | codex/`01a023c5-d10e-7ca0-a3e4-14ddfda37510` | 170520 | cache_read (127744) |
| 333 | codex/`01a027b9-c72e-76a3-81b7-f95f6e391079` | 118615 | cache_read (96256) |
| 334 | claude/`5decc542-72eb-4e2f-a389-1971fc760474` | 80584 | input (80165) |
| 335 | codex/`01a027b9-b00d-7d62-a8dc-6a91ff33f21f` | 59630 | cache_read (38400) |
| 336 | codex/`01a027b9-bb99-7123-8332-e0cd325633f7` | 59479 | cache_read (38400) |
| 337 | codex/`01a027b9-257b-7ac3-b975-80efa4c8cbd9` | 59371 | cache_read (38400) |
| 338 | claude/`c0c08850-bdd6-48f0-a6a8-03dd16bb567c` | 51220 | input (50716) |
| 339 | claude/`b65c72d9-21d6-4b4c-8984-319c1533715b` | 37597 | input (37472) |
| 340 | claude/`cdc9ccbb-8b94-4401-b753-ecaf832ad44a` | 37589 | input (37472) |
| 341 | claude/`84d46478-3736-448d-9548-f42dc58bfcce` | 37586 | input (37472) |
| 342 | claude/`7dec35fb-d0a1-4547-9e0a-22c2b0852267` | 37572 | input (37472) |
| 343 | claude/`55c7fe78-751e-451b-98cb-031f2e44a804` | 37561 | input (37472) |
| 344 | claude/`5be1236b-1014-4cc9-a2ae-7b0b7357adac` | 37561 | input (37472) |
| 345 | claude/`c1f61033-e51c-4723-b0a7-e5b3660ba207` | 37540 | input (37472) |
| 346 | claude/`ab20cdbf-6019-4a98-9565-6a245267d687` | 34367 | input (34349) |
| 347 | claude/`4ba85449-2b4e-4e17-aea1-51d63b3712fb` | 31796 | input (31791) |
| 348 | claude/`eaf8ddd2-2191-466e-9dc9-b3035e96f7e8` | 31303 | input (31277) |
| 349 | claude/`0940f1f4-a8d8-436c-a5ac-df22791726ba` | 31300 | input (31294) |
| 350 | claude/`eef54f5e-c093-459a-8711-d9a90a0c79fa` | 31296 | input (31290) |
| 351 | claude/`3c97ef08-cbd3-46ac-b68a-ddb1ff995521` | 31208 | input (31195) |
| 352 | codex/`01a02778-cc41-7422-b9bd-6cbd51afc85f` | 28238 | input (17225) |
| 353 | claude/`fda6d8c9-0953-4f5f-b5ac-37ce88dfd5d3` | 25923 | input (25917) |
| 354 | claude/`8369dc77-befa-433b-a9d6-618ea62cfe09` | 25921 | input (25915) |
| 355 | claude/`0020ae62-b07d-44bf-a926-dbb392be0bd0` | 25908 | input (25895) |
| 356 | claude/`162fa36d-50cc-4e7f-8194-c22e731b6918` | 25908 | input (25895) |
| 357 | claude/`1f614179-11d7-4acf-b3d5-7b3470836ec2` | 25908 | input (25895) |
| 358 | claude/`364a41f2-85f9-4463-bc0b-d63091410619` | 25908 | input (25895) |
| 359 | claude/`45381b28-aec6-4429-ae18-de086e36b85a` | 25908 | input (25895) |
| 360 | claude/`5553ac30-0628-46c7-bf4b-d6288e170f57` | 25908 | input (25895) |
| 361 | claude/`5b8ce153-5781-42f4-8b89-cc0271a7efac` | 25908 | input (25895) |
| 362 | claude/`657fb86c-efb0-486f-908c-d8a80e28913e` | 25908 | input (25895) |
| 363 | claude/`cc5c5da0-b23e-4772-9521-2467f359aa29` | 25908 | input (25895) |
| 364 | claude/`db2d6e2e-dc94-40dd-bdf2-a869ff7a607d` | 25908 | input (25895) |
| 365 | claude/`1d865da1-06c8-4db5-a557-f10a0c3801a9` | 25905 | input (25892) |
| 366 | claude/`371d43b2-71eb-47ce-a77e-6fd453ce69fa` | 25905 | input (25892) |
| 367 | claude/`8e2f33c8-7b86-4344-83c7-5681a42bc016` | 25905 | input (25892) |
| 368 | claude/`de0e5d66-e413-4e4d-b856-f285089e764e` | 25905 | input (25892) |
| 369 | claude/`e0c2aa49-1c82-48d0-96a6-a281b3a2a06a` | 25905 | input (25892) |
| 370 | claude/`fe6d9641-c39a-4e24-832d-1728b3f1436f` | 25905 | input (25892) |
| 371 | claude/`331ebb5f-d6ef-41a7-877c-d739a2566adc` | 25901 | input (25895) |
| 372 | claude/`90dd1483-5394-4db8-bc75-7daf7557c0f5` | 25901 | input (25895) |
| 373 | claude/`a9dbfc2f-a227-46fd-ade5-bfa625e9bf96` | 25901 | input (25895) |
| 374 | claude/`fac520a0-d636-45c8-8c8c-94ed48d76e6e` | 25901 | input (25895) |
| 375 | claude/`1a6fe1d4-fb9a-4b17-a5a0-55781703b755` | 25898 | input (25892) |
| 376 | claude/`4b82612e-58ae-4afa-b2f5-85e261498813` | 25898 | input (25892) |
| 377 | claude/`9b11aef9-7b07-4d50-a76c-c3520b98958f` | 25898 | input (25892) |
| 378 | claude/`bbb59682-323b-4dd6-b41f-ebebae4f8a56` | 25898 | input (25892) |
| 379 | claude/`cbcd574a-5c81-4677-b8fb-500e33dc720b` | 25898 | input (25892) |
| 380 | claude/`e826de13-d4d0-4856-852e-2a1fbf5db689` | 25898 | input (25892) |
| 381 | claude/`17f33877-3c8b-41c1-918f-675d5da04ff8` | 25896 | input (25883) |
| 382 | claude/`24585cfe-f34f-47db-acf9-19571007350d` | 25896 | input (25883) |
| 383 | claude/`35e9b2f9-33c0-40af-bc71-7c8af8c9efde` | 25744 | input (25738) |
| 384 | claude/`37b46488-bd5f-45dd-bcb9-1fb24cc0b035` | 25744 | input (25738) |
| 385 | claude/`52d5abc0-aebd-42a6-b847-0fb2f76ec7fc` | 25744 | input (25738) |
| 386 | claude/`640f09ea-d56e-4c5f-8fe6-5d74ae30ca3a` | 25744 | input (25738) |
| 387 | claude/`5abeacea-218f-4c41-bff3-a33d53b3d7cc` | 25726 | input (25713) |
| 388 | claude/`0a25e88b-9c0a-4822-9d17-4d4c89394655` | 25696 | input (25690) |
| 389 | claude/`2f8efdc1-f3f6-43b7-9f35-323b93d1d94d` | 25696 | input (25690) |
| 390 | claude/`1a333fd3-15f7-4203-a9fc-57c061e640e1` | 25674 | input (25668) |
| 391 | claude/`8e5affcb-2ea0-46f3-afe6-868c3b54e6cd` | 25660 | input (25647) |
| 392 | claude/`8436ee9b-8ad9-42b4-a034-597b7b69a6df` | 25653 | input (25647) |
| 393 | claude/`5e04ecf3-cfbf-4109-b4ee-d26315a29b2b` | 25154 | input (25141) |
| 394 | claude/`21344b9c-f37f-4db6-a792-8991cdaaad3e` | 24615 | input (24557) |
| 395 | claude/`7a90c824-89e1-4704-aa74-2d1b49a22373` | 21286 | input (20913) |
| 396 | claude/`7a90c824-89e1-4704-aa74-2d1b49a22373` | 20815 | input (20160) |
| 397 | claude/`138fda3b-fd5e-42a3-b3e4-7484734e203b` | 20711 | input (20445) |
| 398 | claude/`7a90c824-89e1-4704-aa74-2d1b49a22373` | 20596 | input (20159) |
| 399 | claude/`7a90c824-89e1-4704-aa74-2d1b49a22373` | 20490 | input (20157) |
| 400 | claude/`138fda3b-fd5e-42a3-b3e4-7484734e203b` | 20444 | input (20132) |
| 401 | claude/`138fda3b-fd5e-42a3-b3e4-7484734e203b` | 20405 | input (20117) |
| 402 | claude/`138fda3b-fd5e-42a3-b3e4-7484734e203b` | 20338 | input (20152) |
| 403 | codex/`01a021ad-eb54-77e2-88ff-0b4b620cd268` | 17769 | input (17757) |
| 404 | codex/`01a0221f-81dc-77f0-b827-9872006e4c6e` | 17768 | input (17757) |
| 405 | claude/`39ef9d96-33ab-44de-bdd4-0db15605da99` | 17340 | input (17327) |
| 406 | claude/`ac0aef17-3c58-4463-86f5-d0b67908e4b5` | 17340 | input (17327) |
| 407 | claude/`d985bfd6-b7a4-4474-9102-7be4e2c6263f` | 17340 | input (17327) |
| 408 | claude/`c3a7c948-c2a3-4e87-b627-e800dea5fa7d` | 17245 | input (17233) |
| 409 | claude/`1e6b3477-8ef8-4907-ac04-b1f84b4614e8` | 17125 | input (17112) |
| 410 | claude/`bf7f5501-a062-4dcf-8004-3eeaf99609a2` | 17089 | input (17076) |
| 411 | codex/`01a021a4-a1b7-72f1-8008-15d79d2ad0bd` | 16551 | cache_read (11008) |
| 412 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 413 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 414 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 415 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 416 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 417 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 418 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 419 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 420 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 421 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 422 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 423 | claude/`00000000-aaaa-bbbb-cccc-000000000000` | 0 | input (0) |
| 424 | claude/`008dc62f-8220-4480-a16c-0879b465e9c0` | 0 | input (0) |
| 425 | claude/`01b6a652-e7a3-43be-81bb-12b3f2c3a325` | 0 | input (0) |
| 426 | claude/`04950001-c23a-4534-b9c7-910a73078708` | 0 | input (0) |
| 427 | claude/`05a502ea-f9cc-4769-8d42-0450262e6060` | 0 | input (0) |
| 428 | claude/`060af08b-3870-48da-a743-d4666d37cb59` | 0 | input (0) |
| 429 | claude/`06a1f84d-ac10-4c16-af10-5b335b3c392c` | 0 | input (0) |
| 430 | claude/`07e56cd9-99e3-4363-8819-9bffea3ae955` | 0 | input (0) |
| 431 | claude/`081f20b5-5368-4985-bf35-dac7bdee5142` | 0 | input (0) |
| 432 | claude/`082be084-a1ab-4fc0-8cb6-e0aa08c166b8` | 0 | input (0) |
| 433 | claude/`08cb02c4-c77c-4241-8cc1-afc850608cc5` | 0 | input (0) |
| 434 | claude/`0982a17f-0024-484a-b0a4-b6eb0d4bfeb6` | 0 | input (0) |
| 435 | claude/`0b71cb31-0781-4e03-8dd7-c8d638bfa7ec` | 0 | input (0) |
| 436 | claude/`0c2205dc-936b-4f2a-b319-9b372a05d15d` | 0 | input (0) |
| 437 | claude/`0e72c261-3122-44eb-9452-ee41e7d73c8b` | 0 | input (0) |
| 438 | claude/`0f9fcb23-b565-460f-bea2-24134dff8476` | 0 | input (0) |
| 439 | claude/`109a175a-4e3c-481a-8234-bafe6e476f70` | 0 | input (0) |
| 440 | claude/`114d122b-bbd7-4e1a-a885-7bb939960cfd` | 0 | input (0) |
| 441 | claude/`115d3842-ea62-43db-b676-a33c16601ad5` | 0 | input (0) |
| 442 | claude/`11c4db2f-9394-41ac-bc72-2654eed1d2e7` | 0 | input (0) |
| 443 | claude/`1240bd39-1bf2-4e52-9b98-f38033ee0a63` | 0 | input (0) |
| 444 | claude/`13e28bc3-b5d2-4e17-9981-c2f7aab1fa10` | 0 | input (0) |
| 445 | claude/`14733188-3887-4aa1-9361-c2518b7fd3cd` | 0 | input (0) |
| 446 | claude/`151d4b8b-cb02-48b1-918f-6950447f28d2` | 0 | input (0) |
| 447 | claude/`171ccf91-cd42-4ddb-933a-1d4f0cf70b39` | 0 | input (0) |
| 448 | claude/`1762d7dd-0983-4cba-80d3-5ac8d60a7074` | 0 | input (0) |
| 449 | claude/`17788e51-b25b-4b72-afa1-2cc4bd79f868` | 0 | input (0) |
| 450 | claude/`17d60ab0-69a6-4bf2-97cd-ec70457e989b` | 0 | input (0) |
| 451 | claude/`1959e8b4-4522-4cee-919a-02cd68716425` | 0 | input (0) |
| 452 | claude/`1a27b996-6570-4c32-9966-8e1fe846db00` | 0 | input (0) |
| 453 | claude/`1bb0d6d9-da00-4277-b99c-e9f44169368a` | 0 | input (0) |
| 454 | claude/`1d735f59-e201-4677-813a-863f8513fbe4` | 0 | input (0) |
| 455 | claude/`1e1a6c35-d3d7-49e5-a060-313e8efdeea6` | 0 | input (0) |
| 456 | claude/`227cd7a8-2d98-490e-a638-7050ad168df3` | 0 | input (0) |
| 457 | claude/`2404e980-3a2d-46a6-8bab-003d263da14c` | 0 | input (0) |
| 458 | claude/`2522cbcb-9bcc-4fe5-a09a-0e744096275d` | 0 | input (0) |
| 459 | claude/`256ce533-b688-4a92-bf58-400778def2cb` | 0 | input (0) |
| 460 | claude/`26af0e7f-ea40-435e-be4d-d6151fe2f571` | 0 | input (0) |
| 461 | claude/`276c2179-8b64-4016-92be-c1ae03c0a6ca` | 0 | input (0) |
| 462 | claude/`27c209ce-f3a7-4d80-97aa-7c9b2dd0069d` | 0 | input (0) |
| 463 | claude/`2938dbc2-be89-4be7-a162-e68a89aa1867` | 0 | input (0) |
| 464 | claude/`2960c5f9-f832-475a-b7aa-799528c3b776` | 0 | input (0) |
| 465 | claude/`29efbf84-c636-4653-b6fe-d9cd2ba305c0` | 0 | input (0) |
| 466 | claude/`2b017f6b-71d8-4688-9223-a3782970de6a` | 0 | input (0) |
| 467 | claude/`2e0511aa-bea5-4282-8761-2ac11b97546e` | 0 | input (0) |
| 468 | claude/`3057f6b7-d1cb-49e8-8fb3-5229d42ee79a` | 0 | input (0) |
| 469 | claude/`307b0aa5-99e6-4fab-b904-e58c44bb113f` | 0 | input (0) |
| 470 | claude/`30a0a591-1224-4582-9048-d1a7ecf786ac` | 0 | input (0) |
| 471 | claude/`30e425ec-83e0-482b-bad2-15fa7f731838` | 0 | input (0) |
| 472 | claude/`32b97bb3-1bc4-416a-af90-5f6c257a069e` | 0 | input (0) |
| 473 | claude/`3390dfbc-b816-4a57-96a3-113e4597dcef` | 0 | input (0) |
| 474 | claude/`348e43b0-d83c-45b6-b36f-68b76239dc63` | 0 | input (0) |
| 475 | claude/`35641d0b-a5f8-43c9-a206-82a474ee30d3` | 0 | input (0) |
| 476 | claude/`3719854d-5d10-456b-94e5-a3e24f475543` | 0 | input (0) |
| 477 | claude/`37dcb865-2f21-4038-ab30-d1b4412cb3ea` | 0 | input (0) |
| 478 | claude/`38bfc791-343f-49ae-b09e-ec9fa530fe68` | 0 | input (0) |
| 479 | claude/`3ad20a8b-c8dd-4184-9cc3-b9636aabfb98` | 0 | input (0) |
| 480 | claude/`3bd12b50-9c61-4b2e-9fc7-635cd6cda6f5` | 0 | input (0) |
| 481 | claude/`3be0b587-d920-4e89-b081-5b434516bceb` | 0 | input (0) |
| 482 | claude/`3c5fe416-6d24-4122-a3d1-d088a0b42eb3` | 0 | input (0) |
| 483 | claude/`3c7c0c38-4262-4130-a50a-638b2951ad4a` | 0 | input (0) |
| 484 | claude/`3db7d5f7-4220-4e9b-a041-e609d36745e1` | 0 | input (0) |
| 485 | claude/`404732f7-2963-4359-b547-50561a5b608e` | 0 | input (0) |
| 486 | claude/`437a6463-7d34-484d-beea-d2b3c4c510e2` | 0 | input (0) |
| 487 | claude/`438eecb7-1a74-4268-a87a-bbabe491b522` | 0 | input (0) |
| 488 | claude/`446878aa-08ef-46dd-9e52-c4adbeb535d8` | 0 | input (0) |
| 489 | claude/`44e985a7-599e-4996-b8ad-1ecbea5c2b39` | 0 | input (0) |
| 490 | claude/`4519cb8f-2ea5-4d5d-b322-9edef0a402b0` | 0 | input (0) |
| 491 | claude/`4631c7b9-8990-4073-8792-049b9156b6e9` | 0 | input (0) |
| 492 | claude/`46958393-47b3-4daf-946f-ba14a9f0ca32` | 0 | input (0) |
| 493 | claude/`47aa7ad0-c33e-4871-8c8c-fe554818d957` | 0 | input (0) |
| 494 | claude/`494afb7a-eb4d-452d-b217-4838700c2a2b` | 0 | input (0) |
| 495 | claude/`495d523d-351d-492d-aae0-f4d98aee0f46` | 0 | input (0) |
| 496 | claude/`49a9ec50-a8e5-4752-acce-b9ac3311d2d2` | 0 | input (0) |
| 497 | claude/`4cd42a0b-9615-473d-a15c-2358998ab7ca` | 0 | input (0) |
| 498 | claude/`4d63888b-6362-4c82-97c0-8b56bcadaad3` | 0 | input (0) |
| 499 | claude/`4ecd3bdb-fa40-4afe-a47a-976a200df332` | 0 | input (0) |
| 500 | claude/`4eefd7aa-5205-4691-a2a9-f3b220fc8eac` | 0 | input (0) |
| 501 | claude/`506635f8-df29-4061-8037-821c3dc8c66d` | 0 | input (0) |
| 502 | claude/`50a8b731-8200-4914-964d-529cb70b9099` | 0 | input (0) |
| 503 | claude/`50f0b475-1cbe-4174-a7c9-0950f444c4d6` | 0 | input (0) |
| 504 | claude/`54108a25-062b-43e4-828d-2aadb18bddd3` | 0 | input (0) |
| 505 | claude/`5439fef8-97d1-473e-b933-2325b9550113` | 0 | input (0) |
| 506 | claude/`54f5ae5c-d196-4b63-ac36-042d807daea3` | 0 | input (0) |
| 507 | claude/`57f05178-8c8e-4275-b83c-5be216b07592` | 0 | input (0) |
| 508 | claude/`5925cf8e-a1ab-48e7-bf6d-b6a9fc0610d9` | 0 | input (0) |
| 509 | claude/`5a0ce06a-f4a2-43e3-9f60-a22ee5bae380` | 0 | input (0) |
| 510 | claude/`5bdbeacb-6cba-429d-bc3b-cee018ed20ca` | 0 | input (0) |
| 511 | claude/`5c694f88-458a-405f-a677-509a2ec351d6` | 0 | input (0) |
| 512 | claude/`5f7015e2-7bcc-4f53-902a-4c71a5470c02` | 0 | input (0) |
| 513 | claude/`610bed5a-6d2d-4a94-af34-3046c18cdd87` | 0 | input (0) |
| 514 | claude/`613ad633-4489-42f2-8a13-8c812adb432f` | 0 | input (0) |
| 515 | claude/`61521f2a-2296-43f6-8a48-59708554b9df` | 0 | input (0) |
| 516 | claude/`627028f5-0ee5-45a1-81df-39384f889019` | 0 | input (0) |
| 517 | claude/`634c35d0-27ce-4aec-9202-7ff0c00fc827` | 0 | input (0) |
| 518 | claude/`63db2292-c817-49cc-a052-ea4a169ed49a` | 0 | input (0) |
| 519 | claude/`6461ec7a-6a8c-43d1-95a9-3c885e917421` | 0 | input (0) |
| 520 | claude/`6482a0b9-2bd0-4182-beec-347ee622f1fa` | 0 | input (0) |
| 521 | claude/`6554a193-d0f9-46fc-a325-42346cbfa59b` | 0 | input (0) |
| 522 | claude/`65dae71d-a4c3-4e2f-8002-66b49b5a9622` | 0 | input (0) |
| 523 | claude/`665ecfee-80d9-489f-802a-ded06862c59e` | 0 | input (0) |
| 524 | claude/`668aad29-818e-46b3-9401-848f9afd7ea6` | 0 | input (0) |
| 525 | claude/`6781aaf9-64c3-44ac-89fd-9452a440c129` | 0 | input (0) |
| 526 | claude/`679843d0-c08f-477f-864e-4aca92727a8d` | 0 | input (0) |
| 527 | claude/`687b3a4c-a5ad-4839-85f7-81a454f9f5a5` | 0 | input (0) |
| 528 | claude/`69fec3af-860d-48f9-b2f4-093519165074` | 0 | input (0) |
| 529 | claude/`6b9457ff-9266-44f9-968e-a0755d3013e4` | 0 | input (0) |
| 530 | claude/`6ce831cb-8457-406b-b017-4b90c871e55d` | 0 | input (0) |
| 531 | claude/`6fd2b645-6446-47b3-bf8b-af386e4d6e08` | 0 | input (0) |
| 532 | claude/`71dbd98f-d718-4488-9246-54e50b778978` | 0 | input (0) |
| 533 | claude/`75180f18-39e3-4f5e-a068-d470509c39f4` | 0 | input (0) |
| 534 | claude/`7525b40e-232b-4f27-b6dc-bc7b1f25d88c` | 0 | input (0) |
| 535 | claude/`752ede4b-1ffc-4f3d-88bc-febedc210a9b` | 0 | input (0) |
| 536 | claude/`75a34c0b-a1ae-4943-82a6-dc59c8fc14cb` | 0 | input (0) |
| 537 | claude/`76ef959c-d124-4cf2-bb05-bd7b8b3352ed` | 0 | input (0) |
| 538 | claude/`77a3ae0e-5826-47fb-a356-097e7981fb30` | 0 | input (0) |
| 539 | claude/`786ac1cf-a5ec-40c5-be16-9d38b7ef8daa` | 0 | input (0) |
| 540 | claude/`79aba768-2e73-4c3f-b4cf-77deab96c267` | 0 | input (0) |
| 541 | claude/`79c423cd-7077-404e-8b3f-417b759c5dc7` | 0 | input (0) |
| 542 | claude/`7a2fd515-370e-4f67-8e3c-fef8e5127122` | 0 | input (0) |
| 543 | claude/`7d8c7d84-c6ba-4eb9-a7d3-f992c034464d` | 0 | input (0) |
| 544 | claude/`7dcf8a24-b83b-427b-aeb2-1e433fe37bb7` | 0 | input (0) |
| 545 | claude/`7ff1e460-871b-4580-821d-985d9856dce2` | 0 | input (0) |
| 546 | claude/`84a2083d-d9ad-44f1-8a01-5007c1531463` | 0 | input (0) |
| 547 | claude/`8513caa6-05ff-4d4e-adb8-f9165898b1e5` | 0 | input (0) |
| 548 | claude/`854322f2-dd34-43ea-8ed1-093e09773a5a` | 0 | input (0) |
| 549 | claude/`859685dd-c249-41d6-8745-734589decf13` | 0 | input (0) |
| 550 | claude/`866f7f5f-fc10-47d3-90b2-d262460855ef` | 0 | input (0) |
| 551 | claude/`86da076a-9c0b-4d49-90c3-46b2ff830e6b` | 0 | input (0) |
| 552 | claude/`86fe1362-a92a-4209-9483-be766d7b3903` | 0 | input (0) |
| 553 | claude/`89637049-47bb-40b1-bb36-2ae99b78ead5` | 0 | input (0) |
| 554 | claude/`897426d0-8202-4e0d-8fad-4816f635307e` | 0 | input (0) |
| 555 | claude/`89a8dee7-e900-4cf3-8759-86fc6b672be0` | 0 | input (0) |
| 556 | claude/`8bc4d34b-5eae-444a-a7df-c5c2ca7f799c` | 0 | input (0) |
| 557 | claude/`8c062d2d-73ae-4bf6-95d0-19f145b42830` | 0 | input (0) |
| 558 | claude/`8c4882ac-9286-4991-87bc-be27b491e3f5` | 0 | input (0) |
| 559 | claude/`8deb85c5-a1b2-4965-aca7-f38f8a021358` | 0 | input (0) |
| 560 | claude/`906ebd82-f9e7-48e9-a773-f36657f3d28a` | 0 | input (0) |
| 561 | claude/`91594d02-faaa-4716-a5b6-0845fef1b8ba` | 0 | input (0) |
| 562 | claude/`92c5d87b-3627-4d8a-bdac-da98b71be726` | 0 | input (0) |
| 563 | claude/`94d416b7-8de7-4629-a8aa-55cc910ca962` | 0 | input (0) |
| 564 | claude/`96862836-461a-48d0-b0e2-25ad54868ea6` | 0 | input (0) |
| 565 | claude/`96e949a6-3b87-41ca-a5c1-b4a57982406b` | 0 | input (0) |
| 566 | claude/`975ec9cf-849e-471b-a9e6-3adc40b053b1` | 0 | input (0) |
| 567 | claude/`97b4bbde-d392-4349-971e-3161e8b81ccf` | 0 | input (0) |
| 568 | claude/`97e1b05b-b293-41d9-a0b1-d27a8df458d7` | 0 | input (0) |
| 569 | claude/`9b31b43b-ad4e-4742-baf2-17b4fb69505a` | 0 | input (0) |
| 570 | claude/`9b908451-5698-4973-b690-f1ba9167df43` | 0 | input (0) |
| 571 | claude/`9dd19660-d47e-4684-b03a-4c634f84ab27` | 0 | input (0) |
| 572 | claude/`9e4017bc-ad2a-4bdb-9b1c-f7218cca1db1` | 0 | input (0) |
| 573 | claude/`a073658a-3bac-43bb-a84d-f18ba00571f4` | 0 | input (0) |
| 574 | claude/`a2f8f4f5-4fd1-4848-b473-8ad795f4193b` | 0 | input (0) |
| 575 | claude/`a34f8d7e-9a7e-4a34-9e88-43b39c61e87b` | 0 | input (0) |
| 576 | claude/`a39ad324-48b3-4502-b211-754a39626edf` | 0 | input (0) |
| 577 | claude/`a48ee427-0108-4fe4-b6e4-c4ef40ebdff3` | 0 | input (0) |
| 578 | claude/`a76d3adc-9f59-4dd7-a6e5-a622d76afb6b` | 0 | input (0) |
| 579 | claude/`a9d1edb7-70fc-4f14-a118-0b43d2e4e664` | 0 | input (0) |
| 580 | claude/`aa33cba9-97c5-4763-ba39-18b6e4c0913f` | 0 | input (0) |
| 581 | claude/`ad1f2ec3-dee7-4020-aaec-c5ef41e01b4e` | 0 | input (0) |
| 582 | claude/`ae35e806-dade-41a8-bf30-40aee1b6a414` | 0 | input (0) |
| 583 | claude/`b129378e-fde4-491f-81d9-d717538ac644` | 0 | input (0) |
| 584 | claude/`b205f7a4-f789-48b6-b8bf-d993b5a8bae2` | 0 | input (0) |
| 585 | claude/`b2a6ba9d-1a62-4e28-a585-4d27f7f85f90` | 0 | input (0) |
| 586 | claude/`b3253a1d-704f-46f5-9d3f-650f0189452d` | 0 | input (0) |
| 587 | claude/`b35b43f6-7db8-488b-a707-0f796e252865` | 0 | input (0) |
| 588 | claude/`b3f2c419-28bf-4dfe-97b6-863ae262db65` | 0 | input (0) |
| 589 | claude/`b45d4856-c339-4786-804b-778e7edb9a69` | 0 | input (0) |
| 590 | claude/`b487b5a3-b2f5-4f63-979a-68b8a5453f62` | 0 | input (0) |
| 591 | claude/`b4b63bab-6494-4464-b1fa-98e963857895` | 0 | input (0) |
| 592 | claude/`b5d2efa2-409e-4695-a521-8f50afbfab68` | 0 | input (0) |
| 593 | claude/`b6ce6ee7-1f26-499f-94e0-cb54ffa622e1` | 0 | input (0) |
| 594 | claude/`b83ec8d8-4b79-49ad-b183-750ac96490fd` | 0 | input (0) |
| 595 | claude/`b9c0fb9f-b389-49b2-a6c4-95f87db3f985` | 0 | input (0) |
| 596 | claude/`b9ddadf2-b532-4581-a5a8-3ba1e54101fa` | 0 | input (0) |
| 597 | claude/`bac39f7a-1ccc-4d16-ace3-e3b6e6b109ac` | 0 | input (0) |
| 598 | claude/`bc12be33-d996-456d-bc4a-cf34a1997134` | 0 | input (0) |
| 599 | claude/`bc45a028-0998-4d4e-898a-87c117934cde` | 0 | input (0) |
| 600 | claude/`bc50f12a-64bf-4045-9266-fd5751418e9d` | 0 | input (0) |
| 601 | claude/`bc9b6392-0daa-4758-866a-86ba0dcf3b20` | 0 | input (0) |
| 602 | claude/`bcd6d4a4-c77c-4727-a523-e49467600f41` | 0 | input (0) |
| 603 | claude/`bd27091d-6a04-44d1-8d4c-669cce66f8d6` | 0 | input (0) |
| 604 | claude/`bf6822e2-bce7-474d-8846-c04dfb00939d` | 0 | input (0) |
| 605 | claude/`c26417df-08aa-40f4-8f94-28f9f28e34df` | 0 | input (0) |
| 606 | claude/`c2d78454-2f2e-4bf4-a136-da27e9a9767c` | 0 | input (0) |
| 607 | claude/`c34c3c53-1b70-48f6-b507-9f7831892567` | 0 | input (0) |
| 608 | claude/`c73f2292-42b5-4342-aa55-fe58b3b4ee38` | 0 | input (0) |
| 609 | claude/`c756e4f0-df69-42fa-957c-f3c876fdde7d` | 0 | input (0) |
| 610 | claude/`c77a2eef-ce10-4525-9a7a-c4eeba169aac` | 0 | input (0) |
| 611 | claude/`c841fc06-9e49-4fb8-a7e5-e01774a8691e` | 0 | input (0) |
| 612 | claude/`c8d4f472-338e-401d-9ea8-7d21ca8a6030` | 0 | input (0) |
| 613 | claude/`c90f1b50-595e-4ed7-865b-967916447a27` | 0 | input (0) |
| 614 | claude/`c9f32687-75eb-4b7c-bb3f-76b69f44186b` | 0 | input (0) |
| 615 | claude/`ca32c212-02b3-4729-8ab0-2eab007cd236` | 0 | input (0) |
| 616 | claude/`ca9baa26-01f2-4dc4-8e95-45e69de095e5` | 0 | input (0) |
| 617 | claude/`cada42e1-806e-41e6-a539-b61099e89f97` | 0 | input (0) |
| 618 | claude/`caee6e07-55e3-42a2-b50b-db8492536964` | 0 | input (0) |
| 619 | claude/`cd65a59b-09a2-4511-96a9-4d63d58b44d4` | 0 | input (0) |
| 620 | claude/`cd7c693d-ef4e-4c6b-9671-f935426e5c3a` | 0 | input (0) |
| 621 | claude/`ce611bca-9928-4f47-bb67-f62d79580ac2` | 0 | input (0) |
| 622 | claude/`cedd96fd-d1b6-433e-a59e-397408d89161` | 0 | input (0) |
| 623 | claude/`cf0114f4-0f9f-459d-85a0-bd43dac11001` | 0 | input (0) |
| 624 | claude/`cfa9f65d-67a8-405b-902a-f903e8a2cfb9` | 0 | input (0) |
| 625 | claude/`d0b068ad-3513-45a3-8935-fea8da5890a1` | 0 | input (0) |
| 626 | claude/`d2fba7dc-1bdc-45a7-ad7c-70ab2e683a82` | 0 | input (0) |
| 627 | claude/`d37014bb-4d27-4955-b802-1d28b88dbfbf` | 0 | input (0) |
| 628 | claude/`d3b45838-1ab7-46c7-8e15-a9a9770fd861` | 0 | input (0) |
| 629 | claude/`d4618aa4-36ef-4bbf-82c0-0fc1d14d7c7c` | 0 | input (0) |
| 630 | claude/`d5832b55-d469-45fe-a9b8-c86f2e103cf5` | 0 | input (0) |
| 631 | claude/`d925a340-80a3-4d96-a4c8-ff5785a5e7b9` | 0 | input (0) |
| 632 | claude/`da98da39-fb1c-4b03-af59-08d7b4d7eaf6` | 0 | input (0) |
| 633 | claude/`db51f0e3-0659-47e1-9439-2c0f73381236` | 0 | input (0) |
| 634 | claude/`dc1239f0-a05f-4edf-9008-d7de8711ec2f` | 0 | input (0) |
| 635 | claude/`dce36a50-25d6-4e05-934f-39bcd2b8db0e` | 0 | input (0) |
| 636 | claude/`dd987b5f-2285-4f57-8f3f-93377444c98d` | 0 | input (0) |
| 637 | claude/`dda52db0-aa00-4435-aa72-4c21994245d8` | 0 | input (0) |
| 638 | claude/`de0a6c81-ea7a-4f7d-99d4-b294be8d6c13` | 0 | input (0) |
| 639 | claude/`deae9487-1414-4583-a7ea-68d16c459b53` | 0 | input (0) |
| 640 | claude/`e1f47c7e-38bd-4faf-8590-86fbc1030c0b` | 0 | input (0) |
| 641 | claude/`e2d2408e-0c43-47b7-b05b-366b5a1da064` | 0 | input (0) |
| 642 | claude/`e395efed-6623-439a-a6ec-3e0860585dc1` | 0 | input (0) |
| 643 | claude/`e3bc339f-5c98-4829-9eed-3e12f2a0e4a6` | 0 | input (0) |
| 644 | claude/`e5c12f82-716a-4c26-b336-bf3a386f1f0d` | 0 | input (0) |
| 645 | claude/`e6b5af71-ed9d-4ea1-81c4-907b38bb932a` | 0 | input (0) |
| 646 | claude/`e93eb253-d6ab-469c-9e4b-c3c4e1d40501` | 0 | input (0) |
| 647 | claude/`e9d09195-71b2-4fa5-abf5-e41f7cc58295` | 0 | input (0) |
| 648 | claude/`ea60bcfa-df89-4d63-939c-0b1305eb8657` | 0 | input (0) |
| 649 | claude/`ebc998fb-c418-4de8-a628-aef512166a56` | 0 | input (0) |
| 650 | claude/`ebd70e40-79b2-4161-b2fe-77ac9a44a53d` | 0 | input (0) |
| 651 | claude/`eca2a21d-99f8-4ec8-a40f-73332470e3ca` | 0 | input (0) |
| 652 | claude/`ee1b21dc-ac1a-4b8f-8109-aece9e167383` | 0 | input (0) |
| 653 | claude/`ee8ff48c-e7ab-425e-9e07-73ab7bd389a2` | 0 | input (0) |
| 654 | claude/`f041a6cf-129b-4c23-a48e-7eabadf27fdc` | 0 | input (0) |
| 655 | claude/`f1829675-fefb-4d2e-b0df-07dea3e7e809` | 0 | input (0) |
| 656 | claude/`f2072f1b-951b-4046-8541-0d3d9f4859b5` | 0 | input (0) |
| 657 | claude/`f289c1c2-78eb-4ca3-b8cc-9af38aaaebd0` | 0 | input (0) |
| 658 | claude/`f56cb92d-69ec-49c1-9913-b1ce10c1f654` | 0 | input (0) |
| 659 | claude/`f613e3cf-e961-443a-a68d-3e6ba8cb53ee` | 0 | input (0) |
| 660 | claude/`f87dfc6a-18e4-434c-a4ac-1539928995b0` | 0 | input (0) |
| 661 | claude/`f98ad6a5-fc4b-4bfe-846b-4c4f5fe35304` | 0 | input (0) |
| 662 | claude/`fb826b65-cca0-4d52-87b7-1f262a04cca0` | 0 | input (0) |
| 663 | claude/`fc2ff068-19d7-4e2f-8efb-c5a476c4f370` | 0 | input (0) |
| 664 | claude/`fc864af3-a32e-495d-8e24-7e2b6610e6a5` | 0 | input (0) |
| 665 | claude/`fd3477e8-d0ee-44bf-9c4c-c0e1917e6414` | 0 | input (0) |
| 666 | claude/`ff9cbcbb-7f94-4fd4-86a6-41ba28f2d5ed` | 0 | input (0) |
| 667 | claude/`ffbbce6a-c949-499b-96ed-3f9d1036abc2` | 0 | input (0) |
| 668 | claude/`ffea0d41-0b49-4240-8bc4-96b1420cfbee` | 0 | input (0) |
| 669 | claude/`journal` | 0 | input (0) |
| 670 | claude/`journal` | 0 | input (0) |
| 671 | codex/`01a021a2-1cd5-7ce0-a153-94de332f83aa` | 0 | input (0) |
| 672 | codex/`01a021a6-6d16-7060-9cd8-a7c811d53552` | 0 | input (0) |
| 673 | codex/`01a021dc-a86e-7e33-8f7b-332931739a87` | 0 | input (0) |
| 674 | codex/`01a0252b-082f-7aa3-8ebc-48b67553aa9b` | 0 | input (0) |
| 675 | codex/`01a0252b-090b-7213-b1cd-27b57e819d88` | 0 | input (0) |
| 676 | codex/`01a0252b-0b77-7b63-bf14-fd505dd7ad22` | 0 | input (0) |
| 677 | codex/`01a0252b-0d7b-7861-b866-fa269d6e551b` | 0 | input (0) |
| 678 | codex/`01a0252b-1272-71e3-a7ea-21b44c1407fd` | 0 | input (0) |
| 679 | codex/`01a0252b-1349-7a30-a34f-7fcf80e3b9ea` | 0 | input (0) |
| 680 | codex/`01a0252b-1693-7470-a3a0-695a02570efe` | 0 | input (0) |
| 681 | codex/`01a0252b-17fb-7ed0-9696-3c2c84b5b48b` | 0 | input (0) |
| 682 | codex/`01a0252d-e9b0-7ed2-808b-d714108c877d` | 0 | input (0) |
| 683 | codex/`01a0252d-ea37-7282-9b58-8f37acee9654` | 0 | input (0) |
| 684 | codex/`01a0252d-ec50-7101-9001-9cd6c5079132` | 0 | input (0) |
| 685 | codex/`01a0252d-f0f7-7f52-b652-413f30130fff` | 0 | input (0) |
| 686 | codex/`01a0252d-f7fc-7b63-b3fb-135a42b3ac9c` | 0 | input (0) |
| 687 | codex/`01a0252d-f8bf-78f0-a9b1-47f7f7d2efd5` | 0 | input (0) |
| 688 | codex/`01a0252d-f940-70d0-b6d8-4f0db1a99a60` | 0 | input (0) |
| 689 | codex/`01a0252d-fb59-7750-8811-06061a4495ca` | 0 | input (0) |
| 690 | codex/`01a0252f-9db8-70c3-b22b-b43ae22c391b` | 0 | input (0) |
| 691 | codex/`01a0252f-9fca-7f93-af46-7a9267504601` | 0 | input (0) |
| 692 | codex/`01a0252f-a17c-7032-8cb5-aa3d0cacac7a` | 0 | input (0) |
| 693 | codex/`01a0252f-a989-74c1-af67-a2214c47aee2` | 0 | input (0) |
| 694 | codex/`01a0252f-a9df-7043-bcbc-39073b3f81f1` | 0 | input (0) |
| 695 | codex/`01a0252f-ac5c-79c3-9da3-a06b900d3197` | 0 | input (0) |
| 696 | codex/`01a02532-19bc-7b01-be95-872e0a2a5252` | 0 | input (0) |
| 697 | codex/`01a02532-1a2d-7153-9e9b-c9bb32161beb` | 0 | input (0) |
| 698 | codex/`01a02532-1c3e-7331-8c16-849979c7c5c3` | 0 | input (0) |
| 699 | codex/`01a02532-1c70-7210-9510-61b35447beba` | 0 | input (0) |
| 700 | codex/`01a02532-255c-7863-9dbc-16c21fc3d704` | 0 | input (0) |
| 701 | codex/`01a02532-276a-7dd3-aa70-32e8fb9f9f8d` | 0 | input (0) |
| 702 | codex/`01a02532-2a6b-7462-bc7b-0429780f2f8b` | 0 | input (0) |
| 703 | codex/`01a02532-2c12-7d51-ba77-7a021d8bedc2` | 0 | input (0) |
| 704 | codex/`01a02533-7bc7-7291-ab1a-2c4d69278a1d` | 0 | input (0) |
| 705 | codex/`01a02533-87d2-7100-83c3-cdbeff2f840f` | 0 | input (0) |
| 706 | codex/`01a02533-939b-73d0-b96b-73097fad2570` | 0 | input (0) |
| 707 | codex/`01a02534-0cde-7ec2-8bff-43ece1c0a59a` | 0 | input (0) |
| 708 | codex/`01a02534-0ef1-7f22-b1d3-ebfbbf419f53` | 0 | input (0) |
| 709 | codex/`01a02534-10e2-7c22-9044-40c244f4bdfa` | 0 | input (0) |
| 710 | codex/`01a02534-1848-7050-b0ad-34a671de50ca` | 0 | input (0) |
| 711 | codex/`01a02534-1892-7bc3-824b-2cbc15159dc7` | 0 | input (0) |
| 712 | codex/`01a02534-1ab8-7111-b5fb-29e1773b8225` | 0 | input (0) |
| 713 | codex/`01a02535-c2b1-75a1-a2e3-bb1cf2edaace` | 0 | input (0) |
| 714 | codex/`01a02535-ced0-7cd2-9bde-65dd5c16897e` | 0 | input (0) |
| 715 | codex/`01a02535-daa9-76d2-80a8-bdebd9885ae3` | 0 | input (0) |
| 716 | codex/`01a02535-e6fc-7f51-a57c-e4fcaedf2cc6` | 0 | input (0) |
| 717 | codex/`01a02535-f225-7800-ba4c-413dd4acc95e` | 0 | input (0) |
| 718 | codex/`01a02535-fe16-70a3-931a-45384048eaff` | 0 | input (0) |
| 719 | codex/`01a02536-0d6d-7cd0-b9f7-cdc4a8bdf921` | 0 | input (0) |
| 720 | codex/`01a02536-183f-77d0-af52-b998253dcdfb` | 0 | input (0) |
| 721 | codex/`01a02536-235b-7f12-8c43-af497fbaa7cd` | 0 | input (0) |
| 722 | codex/`01a02536-2e29-76a2-b356-36374d492e11` | 0 | input (0) |
| 723 | codex/`01a02536-3ac2-7bb1-8768-986e1539f239` | 0 | input (0) |
| 724 | codex/`01a02537-b685-7892-9540-48ca69da540a` | 0 | input (0) |
| 725 | codex/`01a02537-c242-7e03-b5d6-244a71c0a4a4` | 0 | input (0) |
| 726 | codex/`01a02537-cf4d-77e0-86b4-a37c8555c30e` | 0 | input (0) |
| 727 | codex/`01a02537-da8e-7381-bb28-51384c5cb1f1` | 0 | input (0) |
| 728 | codex/`01a02537-eaaf-7052-ab0f-7c51e070329c` | 0 | input (0) |
| 729 | codex/`01a02537-f241-7533-aa8b-d91e062490f9` | 0 | input (0) |
| 730 | codex/`01a02537-fb66-7120-95f0-b51951ac2a7c` | 0 | input (0) |
| 731 | codex/`01a02538-078d-7be3-a829-b9e0af0c66ac` | 0 | input (0) |
| 732 | codex/`01a02538-138d-7da2-a4b8-683422cc1ef8` | 0 | input (0) |
| 733 | codex/`01a02538-208f-7022-b03c-3b2607cd978f` | 0 | input (0) |
| 734 | codex/`01a02538-2d60-7bb2-815c-e330318d0730` | 0 | input (0) |
| 735 | codex/`01a02539-214f-7913-843d-d12e41769783` | 0 | input (0) |
| 736 | codex/`01a02539-253b-79e1-9f23-daf1de5a7ce3` | 0 | input (0) |
| 737 | codex/`01a02539-2a31-7b50-8adc-6954fb3a3bf0` | 0 | input (0) |
| 738 | codex/`01a02539-2c33-7232-8442-55a57e2d68c6` | 0 | input (0) |
| 739 | codex/`01a02539-35e8-79a0-b137-e3a3fcf9058d` | 0 | input (0) |
| 740 | codex/`01a02539-3615-7b71-bdb3-fac9130e7f93` | 0 | input (0) |
| 741 | codex/`01a02539-4187-7d72-b883-b7dc0d460e42` | 0 | input (0) |
| 742 | codex/`01a02539-41f4-7511-bb04-41b83cdf822b` | 0 | input (0) |
| 743 | codex/`01a02539-4f10-70c1-93e2-ffad9b028515` | 0 | input (0) |
| 744 | codex/`01a02539-509c-70e3-8720-52f2c1ded8be` | 0 | input (0) |
| 745 | codex/`01a02539-5b43-7932-84f1-40e56709b5c8` | 0 | input (0) |
| 746 | codex/`01a02539-5d19-7523-97d3-a978ca4ac5e0` | 0 | input (0) |
| 747 | codex/`01a02539-6842-70c2-80c0-cd0d3b94ce63` | 0 | input (0) |
| 748 | codex/`01a02539-6991-7ae0-92f5-c4c997087e42` | 0 | input (0) |
| 749 | codex/`01a02539-7719-7841-9b9c-daba7be7d9b3` | 0 | input (0) |
| 750 | codex/`01a02539-772c-7ce2-bcfc-d220ddf16466` | 0 | input (0) |
| 751 | codex/`01a02539-7d15-7d83-b583-a240a4b4dab9` | 0 | input (0) |
| 752 | codex/`01a02539-8002-7db3-b119-1f48f734cda8` | 0 | input (0) |
| 753 | codex/`01a02539-8eda-7ff0-9567-0a91d366a6c9` | 0 | input (0) |
| 754 | codex/`01a02539-902d-7f50-9a19-8f8095c55c3a` | 0 | input (0) |
| 755 | codex/`01a02539-9b74-7fe2-8090-60b1926c805d` | 0 | input (0) |
| 756 | codex/`01a02539-9cb5-72f1-8d7c-ac0d3f1f0a9b` | 0 | input (0) |
| 757 | codex/`01a0253b-d175-71f2-8fee-13d125c7547d` | 0 | input (0) |
| 758 | codex/`01a0253b-d37a-7df2-9061-d5d8e31f4a73` | 0 | input (0) |
| 759 | codex/`01a0253b-d582-7490-ada2-09f5bd5397f3` | 0 | input (0) |
| 760 | codex/`01a0253b-dd23-7db3-b006-401cb44f66dc` | 0 | input (0) |
| 761 | codex/`01a0253b-df09-71a2-b9e5-e0cfa63c4312` | 0 | input (0) |
| 762 | codex/`01a0253b-e10e-7c12-a6ba-f946ac1ed543` | 0 | input (0) |
| 763 | codex/`01a0253b-e979-7532-a94c-6539d0b454e6` | 0 | input (0) |
| 764 | codex/`01a0253b-e9ab-7823-97c6-ab09d5335fa8` | 0 | input (0) |
| 765 | codex/`01a0253b-ebdb-7270-89d2-6d1646fab00e` | 0 | input (0) |
| 766 | codex/`01a0253b-f53f-7692-bd6a-009c2eb5ce37` | 0 | input (0) |
| 767 | codex/`01a0253b-f77c-7c80-a61a-de2477e7eb73` | 0 | input (0) |
| 768 | codex/`01a0253b-f964-7f42-8da8-c8effabf6671` | 0 | input (0) |
| 769 | codex/`01a0253c-01c4-76c1-9ddd-8a1704bf609f` | 0 | input (0) |
| 770 | codex/`01a0253c-029b-7ac1-8fbf-ed4ad66a172d` | 0 | input (0) |
| 771 | codex/`01a0253c-047e-70b3-999f-2118a1b4494a` | 0 | input (0) |
| 772 | codex/`01a0253c-0be5-7e22-96e2-94f11b95e3af` | 0 | input (0) |
| 773 | codex/`01a0253c-0d2a-71e0-9d12-80ea5fd32976` | 0 | input (0) |
| 774 | codex/`01a0253c-0ece-7471-b349-8bdce636db2b` | 0 | input (0) |
| 775 | codex/`01a0253c-18d2-7a01-9bb3-41981ba263d3` | 0 | input (0) |
| 776 | codex/`01a0253c-1aaf-7de1-beeb-5bb490c2ae70` | 0 | input (0) |
| 777 | codex/`01a0253c-1cb8-7481-8211-5a1316a9d3f7` | 0 | input (0) |
| 778 | codex/`01a0253c-2667-7190-aa2c-1be7809cd53c` | 0 | input (0) |
| 779 | codex/`01a0253c-28bc-7af1-a892-b3a6066a7cd8` | 0 | input (0) |
| 780 | codex/`01a0253c-2ad6-76e0-8fdd-728f83bb9b66` | 0 | input (0) |
| 781 | codex/`01a0253c-3053-7141-8f56-d56e2c7a4734` | 0 | input (0) |
| 782 | codex/`01a0253c-311b-7ae0-8d2a-d104ea7d93bd` | 0 | input (0) |
| 783 | codex/`01a0253c-32fa-7d90-a03c-46c30f80ed52` | 0 | input (0) |
| 784 | codex/`01a0253c-3b41-74e3-8ca3-532263e97e65` | 0 | input (0) |
| 785 | codex/`01a0253c-3bff-74a1-a482-0282a786b778` | 0 | input (0) |
| 786 | codex/`01a0253c-3e19-7781-a843-4fc40a0b51a9` | 0 | input (0) |
| 787 | codex/`01a0253c-476d-76e2-a0f7-de406a85f1f9` | 0 | input (0) |
| 788 | codex/`01a0253c-47f5-7343-b49d-af36f8ce16af` | 0 | input (0) |
| 789 | codex/`01a0253c-4a1e-7870-87af-d961fe565d33` | 0 | input (0) |
| 790 | codex/`01a02543-d885-7b80-8df0-74e9318bd5b0` | 0 | input (0) |
| 791 | codex/`01a02543-e418-7722-be86-bf3e7706c402` | 0 | input (0) |
| 792 | codex/`01a02543-f01f-7191-9a4f-e38956d70c7a` | 0 | input (0) |
| 793 | codex/`01a02543-fba5-76e0-a419-7b18f7e10352` | 0 | input (0) |
| 794 | codex/`01a02544-080c-79c1-b959-5d8283f45c56` | 0 | input (0) |
| 795 | codex/`01a02544-1376-7582-b138-f7ce6950f802` | 0 | input (0) |
| 796 | codex/`01a02544-1f17-7140-811a-776153809123` | 0 | input (0) |
| 797 | codex/`01a02544-2aab-7ec0-b8eb-337f33a28785` | 0 | input (0) |
| 798 | codex/`01a02544-36c8-7f13-b2f8-f77eb2528789` | 0 | input (0) |
| 799 | codex/`01a02544-4218-7962-8c22-71f441b473bd` | 0 | input (0) |
| 800 | codex/`01a02544-4df2-75d3-8247-95a7e48faa24` | 0 | input (0) |
| 801 | codex/`01a02544-5992-7e13-b567-56f4b0d426a8` | 0 | input (0) |
| 802 | codex/`01a02544-654b-7bd1-a4e1-1de0b68255b7` | 0 | input (0) |
| 803 | codex/`01a02544-715d-7290-9536-c95ecebd613c` | 0 | input (0) |
| 804 | codex/`01a02544-7d10-7e13-a398-2d4e247d086f` | 0 | input (0) |
| 805 | codex/`01a02544-8895-7ac1-86e1-4d3246ce0524` | 0 | input (0) |
| 806 | codex/`01a02544-9444-7cc1-a295-428c8b516a89` | 0 | input (0) |
| 807 | codex/`01a02544-a05f-7b51-be9b-deb14eb1cdf8` | 0 | input (0) |
| 808 | codex/`01a02544-abfc-7e22-ad99-3f8253b7e5e0` | 0 | input (0) |
| 809 | codex/`01a02544-b7e2-7ca3-b9db-7bca1db8125e` | 0 | input (0) |
| 810 | codex/`01a02544-c355-7950-8453-846a8a5cbd35` | 0 | input (0) |
| 811 | codex/`01a02544-cf16-7ca2-9372-dce1882e75c4` | 0 | input (0) |
| 812 | codex/`01a02544-dabc-7792-bf80-bd5871a12b64` | 0 | input (0) |
| 813 | codex/`01a02544-e68e-7021-beb2-16a01ec52f12` | 0 | input (0) |
| 814 | codex/`01a02544-f25f-76c3-8138-952dd264c594` | 0 | input (0) |
| 815 | codex/`01a02544-fe2d-79d1-b319-9bc8adc7753d` | 0 | input (0) |
| 816 | codex/`01a02545-09e4-7be1-a736-220d18524c19` | 0 | input (0) |
| 817 | codex/`01a02545-1587-7421-ad48-440d787c762d` | 0 | input (0) |
| 818 | codex/`01a02545-2147-7eb0-b75f-86fb829ba1e8` | 0 | input (0) |
| 819 | codex/`01a02545-946d-7bd2-a01a-983ccb822e17` | 0 | input (0) |
| 820 | codex/`01a02545-9650-7211-9c18-cb3221101223` | 0 | input (0) |
| 821 | codex/`01a02545-986f-7d33-8ab4-a37f3ddb6450` | 0 | input (0) |
| 822 | codex/`01a02545-a029-71c2-a9ae-c06745f8e594` | 0 | input (0) |
| 823 | codex/`01a02545-a223-7400-8d92-35fa510289fc` | 0 | input (0) |
| 824 | codex/`01a02545-a428-7251-b4b2-e6a2c759426f` | 0 | input (0) |
| 825 | codex/`01a02545-ad3d-7681-a217-d5640ed54d9c` | 0 | input (0) |
| 826 | codex/`01a02545-af5a-7d81-8f87-a80ac0e4d565` | 0 | input (0) |
| 827 | codex/`01a02545-b133-7480-bb67-120f992ad312` | 0 | input (0) |
| 828 | codex/`01a02545-b89b-7ec0-9d99-fb374362c5a7` | 0 | input (0) |
| 829 | codex/`01a02545-b8ff-7fe0-b4c8-3c9e1b7a53ef` | 0 | input (0) |
| 830 | codex/`01a02545-baff-7f41-bc65-f3720ae96626` | 0 | input (0) |
| 831 | codex/`01a02545-c48b-73a0-8eee-53788dbc589a` | 0 | input (0) |
| 832 | codex/`01a02545-c62a-7b91-8da7-ac3f61d5f5b7` | 0 | input (0) |
| 833 | codex/`01a02545-c841-7733-991b-85915b67c45f` | 0 | input (0) |
| 834 | codex/`01a02545-d0d8-74a2-ba4e-254a58e922fb` | 0 | input (0) |
| 835 | codex/`01a02545-d125-7913-ae67-5430c7f73772` | 0 | input (0) |
| 836 | codex/`01a02545-d356-7fb1-9320-8f6ad810ae4d` | 0 | input (0) |
| 837 | codex/`01a02545-dc56-7500-8d64-e3f499b91cbd` | 0 | input (0) |
| 838 | codex/`01a02545-dcdc-71d2-94d7-e7c343b11cfd` | 0 | input (0) |
| 839 | codex/`01a02545-deb0-79e0-acb9-76fb625a442d` | 0 | input (0) |
| 840 | codex/`01a02550-566f-7262-bcdd-0f001983cf21` | 0 | input (0) |
| 841 | codex/`01a02550-5690-7b32-b54c-155a6bb559bd` | 0 | input (0) |
| 842 | codex/`01a02550-586a-7193-8e1f-aa18be9d074f` | 0 | input (0) |
| 843 | codex/`01a02550-62e6-7b81-99fb-1c0c56f8b73b` | 0 | input (0) |
| 844 | codex/`01a02550-6395-7d41-8b70-89dceaac2f50` | 0 | input (0) |
| 845 | codex/`01a02550-657e-7c03-a2e0-a3d9cbbbfd47` | 0 | input (0) |
| 846 | codex/`01a02550-6f2a-7a72-9e94-687e9376d230` | 0 | input (0) |
| 847 | codex/`01a02550-6fcc-7201-8c13-e18f67533c0c` | 0 | input (0) |
| 848 | codex/`01a02550-71a5-7823-b1e2-0d85b170af83` | 0 | input (0) |
| 849 | codex/`01a02550-7bb0-75a2-9ede-13eb01e27bec` | 0 | input (0) |
| 850 | codex/`01a02550-7c09-7230-a343-0c60c639b3c7` | 0 | input (0) |
| 851 | codex/`01a02550-7e15-73b3-bf7c-3720ad10aee1` | 0 | input (0) |
| 852 | codex/`01a02550-872f-7a53-be6a-803e5f45521e` | 0 | input (0) |
| 853 | codex/`01a02550-892d-7b33-bb3d-aeca4eadbd9e` | 0 | input (0) |
| 854 | codex/`01a02550-8b2f-7062-8920-df473cf7ebba` | 0 | input (0) |
| 855 | codex/`01a02550-92bf-7642-a908-8ce3d301bddd` | 0 | input (0) |
| 856 | codex/`01a02550-94ea-7921-98b7-21fae63622ad` | 0 | input (0) |
| 857 | codex/`01a02550-96df-7763-85b6-3acf00fbbf42` | 0 | input (0) |
| 858 | codex/`01a02550-a046-70a1-90eb-980ac128492c` | 0 | input (0) |
| 859 | codex/`01a02550-a240-77d3-97c3-644e8de9a5c9` | 0 | input (0) |
| 860 | codex/`01a02550-a47f-7c42-81bc-90e81f156da5` | 0 | input (0) |
| 861 | codex/`01a02550-ab2b-7690-8349-9cb494d039b0` | 0 | input (0) |
| 862 | codex/`01a02550-abb9-7eb3-b8ab-8b791c641c4a` | 0 | input (0) |
| 863 | codex/`01a02550-ad7b-7412-bb02-cf9c4f2d50c1` | 0 | input (0) |
| 864 | codex/`01a02550-b6a1-75c2-a97a-fb55fb994d6e` | 0 | input (0) |
| 865 | codex/`01a02550-b6e1-7d33-8a68-4915e8eff9ac` | 0 | input (0) |
| 866 | codex/`01a02550-b8db-7491-b1ce-b067bbf17d7f` | 0 | input (0) |
| 867 | codex/`01a02550-c307-73e2-aa27-3645520a8531` | 0 | input (0) |
| 868 | codex/`01a02550-c357-7573-a11c-323254573785` | 0 | input (0) |
| 869 | codex/`01a02550-c558-7023-9df2-98da15df66f4` | 0 | input (0) |
| 870 | codex/`01a02550-ce05-7b83-a3ce-2c5e8e041999` | 0 | input (0) |
| 871 | codex/`01a02550-ce9c-7290-a599-8f1fe97cd141` | 0 | input (0) |
| 872 | codex/`01a02550-d093-7571-9e97-64e35350cf3e` | 0 | input (0) |
| 873 | codex/`01a02550-d8ef-7112-b734-25421c95ccbe` | 0 | input (0) |
| 874 | codex/`01a02550-d966-7753-9ee9-d8b2d43ba3a3` | 0 | input (0) |
| 875 | codex/`01a02550-dbd5-70d1-a514-6a4917ea4b7e` | 0 | input (0) |
| 876 | codex/`01a02550-e466-7c72-8bf5-af22af3a4d7a` | 0 | input (0) |
| 877 | codex/`01a02550-e4ca-76a3-bebc-11debdb20b8f` | 0 | input (0) |
| 878 | codex/`01a02550-e6ca-7691-9ab3-fb1226214008` | 0 | input (0) |
| 879 | codex/`01a02550-f02c-7c71-b2e6-118c70f9790e` | 0 | input (0) |
| 880 | codex/`01a02550-f229-7e51-98d3-a188b75d35d6` | 0 | input (0) |
| 881 | codex/`01a02550-f41b-76d3-a6d5-efdc5d88697e` | 0 | input (0) |
| 882 | codex/`01a02550-fbdd-7770-b8d7-9ad0ed05362f` | 0 | input (0) |
| 883 | codex/`01a02550-fde8-7ba3-9aa3-3ee9c3cddec9` | 0 | input (0) |
| 884 | codex/`01a02551-001d-7ee1-80ed-3e6d4a0ffb19` | 0 | input (0) |
| 885 | codex/`01a02551-0707-7fb0-bdc3-ffe86261ef8c` | 0 | input (0) |
| 886 | codex/`01a02762-2ab7-7ba2-b615-6b2e26d794f6` | 0 | input (0) |
| 887 | codex/`01a027dd-6b48-7473-9a74-f1953cfd3f5b` | 0 | input (0) |
| 888 | codex/`01a027dd-b506-7a52-b72e-186d0cdcc4f5` | 0 | input (0) |
| 889 | codex/`01a027dd-f91c-7990-8893-f256ea1dd35d` | 0 | input (0) |

Highest-cost bottleneck: codex/`01a021aa-dc64-79b3-9e6e-6dd692a5ec81` with 144497502 accounted tokens; deterministic ties sort by source, transcript, then relative path.

## Refused transcript shapes

The audit refused to estimate these records and must exit non-zero.

| Source | Relative path | Line | Code | Detail |
|---|---|---:|---|---|
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10095-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10277-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-10370-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_main_json_runs_and_emits_0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_off_writes0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 2 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 3 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 4 | `claude_input_tokens` | input_tokens is missing |
| claude | `C--Users-USER-AppData-Local-Temp-pytest-of-USER-pytest-9932-test_route_findings_on_appends0-ws/00000000-aaaa-bbbb-cccc-000000000000.jsonl` | 5 | `claude_input_tokens` | input_tokens is missing |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 554 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 567 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 574 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 581 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 587 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 594 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 600 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 607 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 614 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 621 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 628 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 637 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 642 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 647 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 652 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 659 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 664 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 671 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 676 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 681 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 688 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 693 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 700 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 705 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 712 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 718 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 725 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 730 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 735 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 740 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 745 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 752 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 760 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 766 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 773 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 779 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 784 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 789 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 794 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 801 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 807 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 816 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 823 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 829 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 835 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
| codex | `2026/08/21/rollout-2026-08-21T20-08-02-01a02770-8ce0-7810-b4e9-857ecfd43f6b.jsonl` | 840 | `codex_total_usage_decreased` | cumulative total_token_usage decreased |
