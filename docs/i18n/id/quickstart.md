---
title: "fak — mulai cepat (10 menit menuju model lokal)"
description: "Titik masuk Bahasa Indonesia untuk fak: dari nol ke AI lokal yang tergovernansi dalam ~10 menit — offline, tanpa key, tanpa tagihan cloud, data tetap di mesin Anda. Satu binary Go yang meninjau setiap tool call sebelum berjalan."
---

# fak — mulai cepat (10 menit menuju model lokal)

> Ini adalah **titik masuk terlokalisasi (entry point)**, bukan terjemahan lengkap.
> Dokumentasi lengkap tersedia dalam bahasa Inggris — halaman ini memberi Anda inti
> fak, bukti 60 detik, dan jalur pemasangan, lalu mengarahkan Anda ke
> [dokumentasi bahasa Inggris](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Catatan:** halaman ini dihasilkan oleh mesin dan masih menunggu tinjauan penutur
> asli — silakan buka issue/PR untuk perbaikan.
> **Bahasa lain:** lihat [i18n hub](../README.md).

## Janjinya

Dari nol ke **AI lokal yang tergovernansi dalam sekitar 10 menit**: berjalan offline,
tanpa API key, tanpa tagihan cloud, dan data tetap di mesin Anda. CPU sudah cukup untuk
model kecil — GPU tidak wajib.

## Jalur tercepat

Bungkus agen yang sudah Anda pakai dengan fak, dalam satu perintah:

```bash
fak manage claude
```

Anda tidak menulis ulang agen apa pun. Cukup arahkan satu base URL ke `fak serve`, atau
bungkus agen yang ada seperti di atas — setiap tool call lalu melewati capability floor
lebih dulu.

## Bukti 60 detik (tanpa key, tanpa model, tanpa GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

`refund_payment` ditolak karena tidak pernah ada di allow-list; `search_kb` diizinkan.
Keputusan itu dicek **di dalam kernel**, di jalur panggilan yang sama, dan **fail-closed**:
tindakan yang tidak ada di allow-list tidak bisa berjalan, sekeras apa pun model dibujuk.
Pada perintah terakhir, prompt injection diblokir sementara task tetap tuntas — hasil tool
yang mencurigakan di-*quarantine* penuh dari konteks model (ini soal struktur, bukan
detektor). Dalam uji langsung, injection menembus baseline tanpa proteksi 5/5, dan fak
menembokinya 5/5.

## Apa itu fak

**fak adalah satu binary Go statis** yang duduk di antara agen AI Anda dan tool yang
dipanggilnya. Ia meninjau setiap tool call *sebelum* dijalankan, dan memakai ulang kerja
bersama yang berulang sepanjang sesi panjang. Hasilnya: loop agen yang sama menjadi
**lebih aman, lebih murah, dan lebih cepat**, tanpa ditulis ulang.

fak meng-*govern* dan meng-*cache* model Anda — ia tidak menggantinya. **Qwen2/Qwen3 dan
GLM-MoE** terbukti bit-exact di reference engine in-kernel; selebihnya (DeepSeek, Mistral,
model open-weights apa pun) terhubung lewat wire yang OpenAI-compatible — Ollama / vLLM /
SGLang / llama.cpp / LM Studio atau API OpenAI-compatible mana pun.

**Untuk pasar Indonesia:** biaya token terasa dalam rupiah (IDR), jadi memakai ulang kerja
bersama menekan biaya per-sesi dan melindungi margin. Karena fak mengutamakan self-host —
sebuah binary statis di depan **model lokal** — data tidak keluar dari mesin Anda, sejalan
dengan **UU PDP 2022 (UU Perlindungan Data Pribadi)**. Diskon prompt-cache milik provider
pun tetap hidup: fak menjamin prefix yang di-cache tetap identik byte-per-byte sambil
melepas turn-turn lama di tengah; apakah provider benar-benar memakai ulang cache itu
adalah keputusan provider yang fak teruskan, bukan fak klaim.

## Seberapa cepat (angka yang jujur)

Pada sesi terukur **50 turn × 5 agen**, keunggulan jujur atas stack warm-cache yang
*sudah dituning* adalah sekitar **~4,1× lebih sedikit kerja**. Angka ~60× (kira-kira
~19 jam menjadi ~19 menit) **hanya benar terhadap loop naif yang mengirim ulang semuanya**
tiap turn — jangan pernah menyajikannya sebagai headline. Kemenangan pemakaian-ulang ini
**hanya untuk self-host** dan berlaku pada fleet yang berat-baca (read-heavy).

Setiap angka dilacak ke commit dan artefaknya di
[BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).

## Lisensi & biaya

**Apache-2.0**, gratis, self-host. Tanpa kartu kredit, tanpa invoice lintas-batas, tanpa
badan hukum. `git clone` lalu `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`
adalah keseluruhan jalurnya.

## Ke mana selanjutnya

- [README (gambaran lengkap)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 menit ke model lokal](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — pasang binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — hubungkan agen Anda](../../integrations/README.md)
- [Data residency & kepatuhan — untuk UU PDP 2022](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — sumber setiap angka](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — mana yang shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
