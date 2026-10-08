<div align="center">

# 🎙️ EgySpeech

**Build the best open Egyptian-Arabic speech dataset for production TTS — from a list of YouTube links.**

Download → remove music → diarize → cut at pauses → quality-filter → single-speaker check →
transcribe (with code-switching and optional paralinguistic tags) → forced-align → balance speakers → analyze → publish.

![python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![cuda](https://img.shields.io/badge/CUDA-12.6%20%7C%2013.0-76B900?logo=nvidia&logoColor=white)
![nemo](https://img.shields.io/badge/NVIDIA-NeMo%20Sortformer-76B900)
![hf](https://img.shields.io/badge/🤗-datasets-FFD21E)
![resumable](https://img.shields.io/badge/every%20step-resumable-2563eb)

</div>

---

## ✨ What you get

| | |
|---|---|
| 🎧 **Clean audio** | 24 kHz mono FLAC, loudness-normalized (−20 LUFS) per clip; optional music removal (`separation.mode: auto`) for sources with music beds |
| 🗣️ **One speaker per clip** | Sortformer diarization with a guard distance from every other voice **+** a TitaNet check that every 3-second window matches the clip's own voice |
| ✂️ **Clean cuts** | 5–30 s clips cut **only inside silences** (VAD + energy; a dynamic program over pause candidates); the silence at both clip edges is measured and checked by the filter — never inside a word |
| 🏅 **Quality-gated** | DNSMOS (SIG/BAK/OVRL) + UTMOS thresholds; only clean clips are sent to ASR (saves GPU hours) |
| 📝 **Egyptian transcripts** | QwenCleo-ASR (default, Egyptian + English code-switching), Cohere Transcribe Arabic, NVIDIA Parakeet/FastConformer, or an **audio LLM** (Qwen3-Omni on vLLM) with a configurable prompt |
| 😄 **Paralinguistic tags** | with the LLM backend: `[laughs] [sighs] [breath] [whispering] … [/whispering]`, emotion/style tags — validated against a fixed tag set |
| ⏱️ **Word alignment** | MMS forced alignment on romanized text → word timestamps for Arabic **and** English words; mismatching transcripts removed |
| ⚖️ **Balanced** | speakers merged across episodes by voice, each capped (hours and share) so no host dominates; unseen-speaker test split |
| 📊 **Analysis** | speakers, gender, pure-Arabic vs code-switching, tags, topics (semantic clusters), quality, funnel — Plotly figures in `graphs/` |
| 🔁 **Resumable, observable** | every step skips finished work (stop / restart any time) and shows a progress bar with hours of audio, speed (× real time) and ETA |
| 🏠 **Two stages** | prepare clean clips on a laptop with a small GPU → push them to a private Hub dataset → transcribe on a big GPU (`pull_chunks`) |
| 🤗 **Publishing** | one command to the Hugging Face Hub — **private by default** — with an auto-generated dataset card |

---

## 🧭 Pipeline

```mermaid
flowchart LR
    L[links.txt] --> D[download<br/>FLAC · 24 kHz] --> I[index] --> S[separate<br/>optional]
    S --> Z[diarize<br/>Sortformer] --> G[segment<br/>cut in silences]
    G --> Q[quality<br/>DNSMOS · UTMOS] --> K[speaker_check<br/>TitaNet] --> F[filter] --> R[review<br/>listen]
    R --> PU[push_chunks<br/>🤗 private] -.GPU machine.-> PL[pull_chunks]
    PL --> T[transcribe<br/>QwenCleo · Cohere · Parakeet · LLM+tags] --> V[verify<br/>optional 2nd ASR]
    V --> A[align<br/>MMS words] --> CL[cluster<br/>global speakers · gender]
    CL --> B[balance<br/>caps · splits] --> AN[analysis<br/>graphs] --> P[publish<br/>🤗 private]
```

| # | step | env | what it does | output |
|---|---|---|---|---|
| 1 | `download` | main | expand playlists / channels / videos into **unique** videos → best audio as FLAC 24 kHz mono (yt-dlp archive: resumable) | `raw_download/<playlist>/<title> [<id>].flac` |
| 2 | `index` | main | list the episodes on disk (any folder of `<title> [<id>].flac` works) with durations | `meta/videos.jsonl` |
| 3 | `separate` | main | *optional* vocal stem (`separation.mode: auto / always`; default `never`) | `audio/vocals/` |
| 4 | `diarize` | nemo | NVIDIA Streaming Sortformer v2.1, frame probabilities (next episode decoded while the GPU works) | `diar/` |
| 5 | `segment` | main | one chunk per single-speaker **diarization turn** (2–30 s, overlaps left out, long turns split at quiet moments; `method: vad` = pause planner), loudness norm | `chunks/`, `meta/chunks/` |
| 6 | `quality` | main | DNSMOS (CPU pool) + UTMOS (GPU), concurrently | `meta/quality/` |
| 7 | `speaker_check` | nemo | TitaNet window embeddings → single-speaker score, voice embedding | `meta/speakers/` |
| 8 | `filter` | main | quality thresholds, then **speaker cap** (`--max-hours N`, every speaker kept); prints speakers / hours stats (re-run freely, `--dry-run`) | `meta/filtered.jsonl` |
| 9 | `review` | main | listening page: random kept clips + examples of every rejection reason | `review/index.html` |
| 10 | `push_chunks` | main | filtered clips (no transcripts) → private Hub dataset, resumable shards | 🤗 |
| 11 | `pull_chunks` | main | (GPU machine) Hub dataset → clips + metadata on disk | `chunks/`, `meta/` |
| 12 | `transcribe` | per backend | ASR on filtered clips + hallucination / truncation checks | `meta/transcripts/<backend>/` |
| 13 | `verify` | per backend | *optional* second ASR → agreement (CER) filter | `meta/transcripts/<backend2>/` |
| 14 | `align` | main | word timestamps, alignment score, no-word-cut edge check | `meta/aligned/` |
| 15 | `cluster` | main | global speaker IDs across episodes (`MALE_00001`, `FEMALE_00002`, …) from TitaNet voice prints + gender | `meta/speakers.json` |
| 16 | `balance` | main | final selection, per-speaker caps, train/validation/test | `final/` |
| 17 | `analysis` | main | statistics + figures | `graphs/`, `final/stats.json` |
| 18 | `publish` | main | push the final dataset to the Hub (private) + dataset card | 🤗 |

`run --stage local` = index → diarize → segment → speaker_check → cluster, then `push_chunks`;
`run --stage gpu` (after `pull_chunks`) = quality → filter → transcribe → verify → align → balance → analysis → publish.

Steps run in different conda envs because their libraries conflict (QwenCleo needs `transformers==4.57.6`, Cohere/vLLM need `transformers>=5`); the CLI starts each step in the right env for you.

---

## 🚀 Quick start on RunPod

### 1 · Create the pod

| setting | value |
|---|---|
| GPU | **1× RTX PRO 6000 (96 GB)** or **1× H100 80 GB** (needed for the Qwen3-Omni LLM backend in BF16). Without the LLM backend, an RTX 4090 / 5090 / L40S is enough. |
| vCPU | **16+** (segmentation and DNSMOS run on CPU workers) |
| Volume | **300–600 GB** at `/workspace` (see *Disk* below) |
| Template | any RunPod PyTorch / CUDA 12+ template with JupyterLab |
| HTTP ports | `8888` (Jupyter) |

### 2 · Install (≈15–25 min)

```bash
cd /workspace
git clone https://github.com/MohammedAly22/EgySpeech.git
cd EgySpeech
bash scripts/install.sh            # add --vllm for the audio-LLM backend (Qwen3-Omni)
```

This installs Miniforge into `/workspace/miniforge3` and creates `egyspeech`, `egyspeech-nemo`, `egyspeech-qwen` (and `egyspeech-vllm`). The right PyTorch/CUDA build is chosen from your GPU (Blackwell included) and every env ends with a GPU check. Re-running is safe.

### 3 · Configure

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh && conda activate egyspeech
export EGYSPEECH_DATA=/workspace/egyspeech_data HF_HOME=/workspace/hf_cache
huggingface-cli login                    # to publish (and for any gated model)
nano links.txt                           # your YouTube links, one per line
nano configs/config.yaml                 # backend, thresholds, prompt, publishing …
```

`links.txt` accepts playlists, channels and single videos — duplicates are removed automatically:

```text
# podcasts
https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx
https://www.youtube.com/@SomeEgyptianChannel/videos
https://www.youtube.com/watch?v=xxxxxxxxxxx
```

### 4 · Run

```bash
python -m egyspeech.cli download          # links.txt -> $EGYSPEECH_DATA/raw_download/*.flac
python -m egyspeech.cli run --limit 5     # pilot: 5 videos end-to-end (check quality + timing)
python -m egyspeech.cli review            # listen: $EGYSPEECH_DATA/review/index.html
python -m egyspeech.cli run               # everything (resumes where it stopped)
python -m egyspeech.cli status            # progress of every step
```

Run it inside `tmux` so it survives a closed terminal (`tmux new -s egy`, detach with `Ctrl-b d`, back with `tmux attach -t egy`).

Other useful forms:

```bash
python -m egyspeech.cli run --from segment           # from a step on
python -m egyspeech.cli run --steps quality,filter   # selected steps
python -m egyspeech.cli filter                       # one step (after changing thresholds)
python -m egyspeech.cli transcribe --limit 3         # try a backend on 3 videos
python -m egyspeech.cli segment --watch              # segment while diarize runs in another terminal
```

### Two stages: chunks on a small machine, quality + transcripts on a big GPU (EgySpeech v1)

```bash
# 1. machine with the episodes (laptop; configs/local.yaml)
python -m egyspeech.cli run --stage local          # diarization chunks + global speaker IDs + gender
python -m egyspeech.cli push_chunks                # all chunks -> chunks/ of mohammedaly22/EgySpeech-V1 (private)

# 2. GPU machine
python -m egyspeech.cli pull_chunks
python -m egyspeech.cli quality                    # DNSMOS + UTMOS for every chunk
python -m egyspeech.cli filter --max-hours 15 --dry-run   # compare caps: speakers, mean / min / max hours
python -m egyspeech.cli filter --max-hours 15      # write the selection
python -m egyspeech.cli run --stage gpu --from transcribe  # transcribe -> align -> balance -> analysis -> publish
```

The chunk dataset (`load_dataset("mohammedaly22/EgySpeech-V1", "chunks")`) has every chunk with its speaker ID,
gender, TitaNet embedding and segmentation info; `publish` adds the final transcribed dataset as the default
config (`data/`) of the same private repository. Set `EGYSPEECH_CONFIG=configs/local.yaml` (or pass `--config`)
to use another config file.

### 5 · Explore every step in Jupyter

Open `notebooks/` with the kernel **Python (egyspeech)** — each notebook runs its step and shows the result (tables, plots, audio players):

| notebook | shows |
|---|---|
| `00_setup` | envs, GPU, HF login, config, links |
| `01_download_index` | episodes per playlist, lengths, failures, first listen |
| `02_separate` | original vs vocal stem |
| `03_diarize` | speaker activity timeline |
| `04_segment` | clip lengths, cut types, music level, listen |
| `05_quality_filter` | score distributions with thresholds, worst/best clips, filter report |
| `06_transcribe` | sanity flags, transcripts with audio, tags |
| `07_align` | alignment scores, word boundaries on the waveform |
| `08_cluster_balance` | speakers, gender, per-speaker hours before/after balancing |
| `09_publish` | push to the Hub |
| `analysis` | **all dataset figures** |

---

## 🤖 Transcription backends

Set `transcription.backend` in `configs/config.yaml`:

| backend | model | notes |
|---|---|---|
| `qwencleo` **(default)** | [QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr) | Egyptian dialect, keeps English in Latin script |
| `cohere` | [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) | multi-dialect Arabic, Arabic–English |
| `parakeet` | any NeMo ASR checkpoint (default `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`) | NVIDIA's Arabic model is MSA-oriented |
| `llm` | any OpenAI-compatible audio LLM (default **Qwen3-Omni-30B-A3B-Instruct** on vLLM) | configurable system prompt, **inline paralinguistic tags** |

### Audio LLM with tags

```bash
bash scripts/install.sh --vllm                # once
tmux new -s llm "bash scripts/serve_llm.sh"   # BF16; QUANT=fp8 for 48 GB GPUs
```

```yaml
transcription:
  backend: llm
  llm:
    url: http://localhost:8000/v1
    model: Qwen/Qwen3-Omni-30B-A3B-Instruct
    use_tags: true                 # false = plain verbatim transcripts
    system_prompt: |               # edit freely
      You are an expert transcriber of Egyptian Arabic (Masri) speech. ...
  tags:
    events: [laughs, chuckles, sighs, breath, gasps, coughs, clears_throat, sniffs, lip_smack, pause]
    spans:  [whispering, laughing_speech, shouting]
    styles: [excited, happy, sad, angry, calm, surprised, sarcastic, friendly, serious]
```

The model's output is validated: unknown tags are removed, unclosed spans are closed, and `text` (no tags) and `text_tagged` are both stored.

**Agreement filter (recommended for the highest precision):** set `transcription.verify.backend` (e.g. `cohere`) — clips where the two ASR systems disagree by more than `verify.max_cer` are dropped.

---

## ⚙️ Key settings

| setting | default | meaning |
|---|---|---|
| `download.dir` | `<work_dir>/raw_download` | episodes folder (any folder of `<title> [<id>].flac` files) |
| `download.audio_format / sample_rate / channels` | `flac`, `24000`, `1` | stored episode format |
| `separation.mode` | `never` | `auto` = separate only episodes where a probe finds music; `always` |
| `separation.use_original_below_music_db` | `-35` | keep the original audio when music is this quiet |
| `segmentation.min_sec / max_sec / target_sec` | `5 / 30 / 14` | clip lengths |
| `segmentation.min_pause_sec / guard_sec` | `0.2 / 0.15` | shortest pause used as a cut; distance from other voices |
| `segmentation.allow_weak_cuts` | `false` | `true` = also cut long pause-less turns at energy dips (not silence) |
| `segmentation.workers / memory_gb` | `8 / 24` | parallel episodes and the RAM they may use together |
| `filter.min_dnsmos_bak` | `3.6` | background noise / music / effects (DNSMOS BAK) |
| `filter.min_window_similarity` | `0.78` | single speaker: every 3 s window matches the clip's voice |
| `filter.<other metrics>` | `null` | SIG, OVRL, P.808, UTMOS, speech ratio, clipping, edge loudness: `null` = off |
| `alignment.min_score` | `-3.0` | transcript/audio agreement |
| `speakers.merge_similarity` | `0.65` | merge per-episode speakers into one person |
| `balance.max_hours_per_speaker / max_speaker_share` | `8 h / 3 %` | no dominant speaker |
| `hub.chunks_repo_id / private` | – / `true` | the untranscribed chunk dataset (`push_chunks` / `pull_chunks`) |
| `publish.enabled / private` | `false / true` | push to the Hub, private |

Tune the filter by ear with `python -m egyspeech.cli tune` → `review/tuner.html`: switch metrics on / off, move thresholds, listen to accepted and rejected clips, copy the `filter:` block it shows. Scores are cached, so `python -m egyspeech.cli filter` re-applies new thresholds in seconds.

---

## ⏱️ Hardware, time and disk

Measured on the first ~1,800 h (1,331 episodes) and a 27-episode pilot:

| | per hour of episode audio | ~1,800 h |
|---|---|---|
| episodes (FLAC, 24 kHz mono) | ~170 MB | ~305 GB |
| single-speaker clips (all, before the filter) | ~0.48 h of clips, ~44 MB | ~860 h, ~77 GB |
| clips kept by the filter (BAK ≥ 3.6, voice ≥ 0.78) | ~0.32 h | ~570 h, ~52 GB |
| conda envs + models (fixed) | | ~45 GB |

**Volume:**

| plan | volume |
|---|---|
| everything on one pod (download → publish) | **500 GB** for ~1,800 h (600 GB for ~2,500 h) |
| clips made elsewhere, pod only for `pull_chunks` → `publish` | **150 GB** |

Set `storage.delete_rejected_clips: true` to drop rejected clip audio once the filter is final.

**Hardware per stage:**

| stage | steps | needs |
|---|---|---|
| download | `download` | CPU only (4+ vCPU), network-bound — a cheap CPU pod on the same volume |
| clips | `index` … `filter` | GPU ≥ 16 GB (RTX 4090 / L4 / A5000; a 6 GB laptop GPU works, slower), **16–32 vCPU**, **64 GB RAM** (segmentation + DNSMOS run on CPU workers) |
| v1 transcripts | `transcribe` (QwenCleo / Cohere / Parakeet), `align`, `cluster` … `publish` | one 24 GB GPU (RTX 4090 / L4 / A10; A100 / H100 is faster), 8+ vCPU, 32 GB RAM |
| v2 tags | `transcribe` with `backend: llm` (Qwen3-Omni-30B-A3B on vLLM) | **H100 80 GB or RTX PRO 6000 96 GB** (BF16); 48 GB with `QUANT=fp8` |

Diarization runs long episodes in 30-minute windows (`diarization.window_min`), so GPU memory does not grow with episode
length: a GTX 1660 Ti (6 GB) diarizes at ~90–100× real time (~18 h for 1,800 h); data-center GPUs are several times
faster. Every bar shows the measured speed (× real time) and ETA after the first videos.

---

## 📦 Output

```
/workspace/egyspeech_data/
├── final/train.jsonl · validation.jsonl · test.jsonl   ← the dataset (+ stats.json)
├── chunks/<video>/<clip>.flac                          ← audio
├── graphs/*.html|png                                   ← analysis figures
└── meta/ …                                             ← every intermediate result (resumable)
```

Each row: `id, path, text, text_tagged, tags, words[{word,start,end}], duration, speaker_id, gender,
code_switched, dnsmos_*, utmos, align_score, channel, video_id, video_url, start, end, source`.

On the Hub the same columns are published with an `audio` column (24 kHz), splits `train / validation / test`
(test = speakers never seen in training), a generated dataset card and the figures.

---

## 🧯 Troubleshooting

| symptom | fix |
|---|---|
| a video failed in a step | it is listed in `meta/failures/<step>.jsonl` and retried on the next run; the step keeps going |
| WSL: killed / out of memory | lower `segmentation.memory_gb` / `workers`, or give WSL more RAM in `%UserProfile%\.wslconfig` |
| a step crashed | fix the cause and re-run the same command — finished videos are skipped |
| `no kernel image is available` | re-run `bash scripts/install.sh` (it picks the CUDA build for your GPU) |
| LLM backend: connection refused | start `bash scripts/serve_llm.sh` and check `transcription.llm.url` |
| too few clips pass | open `review/tuner.html` (`cli tune`), relax thresholds, then `cli filter` |
| many videos failed in `diarize` | the step restarts itself after a GPU error; lower `diarization.window_min` on small GPUs |
| PNG figures missing | Plotly's PNG export needs Chrome (`python -c "import kaleido; kaleido.get_chrome_sync()"`); HTML figures are always saved |

---

## ⚖️ Responsible use

YouTube content remains the property of its creators, and the voices belong to real people. Keep the dataset
private unless you have the rights to publish it, never use it to impersonate or clone identifiable speakers
without consent, and label synthetic audio. The pipeline publishes **private** by default for this reason.

## 🙏 Built with

[yt-dlp](https://github.com/yt-dlp/yt-dlp) · [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) ·
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Streaming Sortformer, TitaNet, FastConformer) · [Silero VAD](https://github.com/snakers4/silero-vad) ·
[DNSMOS](https://github.com/microsoft/DNS-Challenge) via torchmetrics · [UTMOS](https://github.com/tarepan/SpeechMOS) ·
[QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr) · [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) ·
[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) + [vLLM](https://github.com/vllm-project/vllm) ·
[MMS forced aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) + [uroman](https://github.com/isi-nlp/uroman) ·
[Plotly](https://plotly.com/python/) · 🤗 [datasets](https://github.com/huggingface/datasets)
