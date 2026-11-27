"""Download a local benchmark clip set from Tatoeba (https://tatoeba.org).

Each clip is a short spoken sentence with a human English translation, which is close to
voice-chat callouts and gives us a reference to score translations against.

The clips are downloaded for LOCAL testing only. Audio licenses on Tatoeba are set per
contributor and the API doesn't report them, so clips must never be committed or
redistributed (benchmarks/clips/ is git-ignored).

Usage:
    python scripts/fetch_clips.py                      # default languages, 12 clips each
    python scripts/fetch_clips.py --langs spa,rus --per-lang 20
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://tatoeba.org/en/api_v0/search"
AUDIO = "https://tatoeba.org/en/audio/download/{}"
UA = {"User-Agent": "lingo-overlay-benchmark/0.1"}

# Tatoeba (ISO 639-3) -> Whisper language code
LANGS = {
    "spa": "es", "por": "pt", "rus": "ru", "cmn": "zh", "deu": "de", "fra": "fr",
    "jpn": "ja", "ara": "ar", "kor": "ko", "ita": "it", "tur": "tr", "pol": "pl",
    "ukr": "uk", "nld": "nl", "eng": "en",
}
DEFAULT = "spa,por,rus,cmn,deu,fra,jpn,ara,kor,tur,eng"


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # network hiccup or non-JSON error page
            if i == tries - 1:
                raise
            print(f"   retry ({e.__class__.__name__})", flush=True)
            time.sleep(2 + 2 * i)


def download(url, dest, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            if len(data) < 1000:
                raise ValueError("audio too small")
            dest.write_bytes(data)
            return True
        except Exception:
            if i == tries - 1:
                return False
            time.sleep(2 + 2 * i)


def english_of(sentence, lang):
    if lang == "eng":
        return sentence["text"]
    for group in sentence.get("translations") or []:  # [direct, indirect]
        for t in group or []:
            if t.get("lang") == "eng":
                return t["text"]
    return None


def fetch_lang(lang, per_lang, out_dir, min_words, max_words):
    got, page, clips = 0, 1, []
    while got < per_lang and page <= 10:
        q = urllib.parse.urlencode({
            "from": lang, "to": "eng", "has_audio": "yes", "orphans": "no", "unapproved": "no",
            "word_count_min": min_words, "word_count_max": max_words,
            "sort": "random", "page": page,
        })
        try:
            res = get_json(f"{API}?{q}")
        except Exception as e:
            print(f"   {lang}: API failed ({e}); skipping language")
            break
        for s in res.get("results", []):
            if got >= per_lang:
                break
            eng = english_of(s, lang)
            if not eng or not s.get("audios"):
                continue
            audio = s["audios"][0]
            fname = f"{lang}_{s['id']}.mp3"
            if not (out_dir / fname).exists() and not download(AUDIO.format(audio["id"]), out_dir / fname):
                continue
            clips.append({
                "file": fname, "lang": LANGS[lang], "tatoeba_lang": lang,
                "text": s["text"], "english": eng,
                "sentence_id": s["id"], "audio_id": audio["id"], "audio_author": audio.get("author"),
            })
            got += 1
            time.sleep(0.3)  # be polite
        page += 1
        if not res.get("results"):
            break
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=DEFAULT, help=f"Tatoeba codes (default {DEFAULT})")
    ap.add_argument("--per-lang", type=int, default=12)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "clips"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else []
    have = {c["file"] for c in manifest}

    for lang in args.langs.split(","):
        lang = lang.strip()
        if lang not in LANGS:
            sys.exit(f"Unknown language '{lang}'. Known: {', '.join(LANGS)}")
        existing = sum(1 for c in manifest if c["tatoeba_lang"] == lang)
        need = args.per_lang - existing
        if need <= 0:
            print(f"{lang}: already have {existing}")
            continue
        print(f"{lang}: fetching {need} clips...", flush=True)
        new = [c for c in fetch_lang(lang, need, out_dir, args.min_words, args.max_words) if c["file"] not in have]
        manifest += new
        have |= {c["file"] for c in new}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
        print(f"   got {len(new)}")

    counts = {}
    for c in manifest:
        counts[c["lang"]] = counts.get(c["lang"], 0) + 1
    print(f"\nManifest: {len(manifest)} clips -> {manifest_path}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
