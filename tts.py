#!/usr/bin/env python3
"""Qwen3-TTS on CPU (GitHub Actions).

  python tts.py design  --voices voices/voices.json --only a,b --out out/   # VoiceDesign 1.7B -> voices/<id>/ref.wav
  python tts.py render  --job jobs/x.json --shard 0 --nshards 2 --model 1.7B --out out/  # clone from voices/<id>/
  python tts.py join    --job jobs/x.json --parts parts/ --out final/

Job JSON:
  {"language": "English",
   "lines": [ {"speaker": "a", "text": "...", "style": "", "pause": 0.2}, ... ]}
  A line may also be a list [speaker, text, pause].
"style" is honoured only by the VoiceDesign model (mode=design-lines); the clone (Base) model
has no instruction input, so in render mode style is recorded but not applied.
"""
import argparse, json, os, sys, time, glob
import numpy as np
import soundfile as sf

MODELS = {
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
MODEL_ROOT = os.environ.get("QWEN_MODELS", os.path.expanduser("~/qwen_models"))


def local_model(key):
    mid = MODELS[key]
    d = os.path.join(MODEL_ROOT, mid.split("/")[1])
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        from huggingface_hub import snapshot_download
        snapshot_download(mid, local_dir=d)
    return d


def load(key):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    from qwen_tts import Qwen3TTSModel
    t = time.time()
    m = Qwen3TTSModel.from_pretrained(local_model(key), device_map="cpu", dtype=torch.float32)
    print(f"[load] {MODELS[key]} {time.time()-t:.1f}s", flush=True)
    return m


def seed(s):
    import torch, random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def speaker_seeds():
    """per-speaker clone seed lock from voices/voices.json ("clone_seed"), e.g. c=39 so a voice never drifts between jobs"""
    try:
        return {k: int(v["clone_seed"]) for k, v in json.load(open("voices/voices.json")).items() if "clone_seed" in v}
    except Exception:
        return {}


def norm_lines(job):
    lock = speaker_seeds()
    out = []
    for i, l in enumerate(job["lines"]):
        if isinstance(l, list):
            l = {"speaker": l[0], "text": l[1], "pause": l[2] if len(l) > 2 else 0.2}
        out.append({"i": i, "speaker": l["speaker"].lower(), "text": l["text"],
                    "style": l.get("style", ""), "pause": float(l.get("pause", 0.2)),
                    "seed": int(l.get("seed", lock.get(l["speaker"].lower(), job.get("seed", 1234))))})
    return out


def trim(w, sr, thr_db=-45, pad=0.04):
    a = np.abs(w)
    thr = 10 ** (thr_db / 20) * max(a.max(), 1e-6)
    idx = np.where(a > thr)[0]
    if len(idx) == 0:
        return w
    p = int(pad * sr)
    return w[max(0, idx[0] - p): min(len(w), idx[-1] + p)]


def cmd_design(a):
    spec = json.load(open(a.voices))
    only = set(x.strip() for x in a.only.split(",")) if a.only else None
    m = load("design")
    os.makedirs(a.out, exist_ok=True)
    stats = []
    for vid, v in spec.items():
        if only and vid not in only:
            continue
        for k in range(a.takes):
            s = v.get("seed", 7) + k
            seed(s)
            t = time.time()
            wavs, sr = m.generate_voice_design(text=v["ref_text"], language=v.get("language", "English"),
                                               instruct=v["description"])
            el = time.time() - t
            w = trim(wavs[0], sr)
            d = os.path.join(a.out, vid if a.takes == 1 else f"{vid}_take{k}")
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, "ref.wav"), w, sr)
            open(os.path.join(d, "ref.txt"), "w").write(v["ref_text"] + "\n")
            st = dict(voice=vid, model=MODELS["design"], seed=s, description=v["description"],
                      language=v.get("language", "English"), sr=sr, dur=round(len(w) / sr, 2),
                      gen_s=round(el, 1), rtf=round(el / max(len(w) / sr, 0.01), 2))
            json.dump(st, open(os.path.join(d, "settings.json"), "w"), indent=1)
            stats.append(st); print("[design]", st, flush=True)
    json.dump(stats, open(os.path.join(a.out, "design_stats.json"), "w"), indent=1)


def cmd_render(a):
    job = json.load(open(a.job))
    lines = norm_lines(job)[a.shard::a.nshards]
    lang = job.get("language", "English")
    m = load(a.model)
    os.makedirs(a.out, exist_ok=True)
    prompts = {}
    stats = []
    for l in lines:
        sp = l["speaker"]
        if sp not in prompts:
            vd = os.path.join("voices", sp)
            ref_text = open(os.path.join(vd, "ref.txt")).read().strip()
            prompts[sp] = m.create_voice_clone_prompt(ref_audio=os.path.join(vd, "ref.wav"), ref_text=ref_text)
        seed(l["seed"])
        t = time.time()
        wavs, sr = m.generate_voice_clone(text=l["text"], language=lang, voice_clone_prompt=prompts[sp])
        el = time.time() - t
        w = trim(wavs[0], sr)
        fn = f"{l['i']:03d}_{sp}.wav"
        sf.write(os.path.join(a.out, fn), w, sr)
        st = dict(l, file=fn, sr=sr, dur=round(len(w) / sr, 2), gen_s=round(el, 1),
                  rtf=round(el / max(len(w) / sr, 0.01), 2), model=MODELS[a.model],
                  style_applied=False if l["style"] else None)
        stats.append(st); print("[render]", st["i"], st["dur"], st["gen_s"], flush=True)
    json.dump(stats, open(os.path.join(a.out, f"stats_{a.shard}.json"), "w"), indent=1)


def cmd_join(a):
    job = json.load(open(a.job))
    lines = norm_lines(job)
    stats = []
    for f in glob.glob(os.path.join(a.parts, "**", "stats_*.json"), recursive=True):
        stats += json.load(open(f))
    by = {s["i"]: s for s in stats}
    files = {os.path.basename(f): f for f in glob.glob(os.path.join(a.parts, "**", "*.wav"), recursive=True)}
    os.makedirs(os.path.join(a.out, "lines"), exist_ok=True)
    seg, sr0, t, timeline = [], None, 0.0, []
    for l in lines:
        s = by[l["i"]]
        w, sr = sf.read(files[s["file"]], dtype="float32")
        sr0 = sr0 or sr
        sf.write(os.path.join(a.out, "lines", s["file"]), w, sr)
        timeline.append(dict(i=l["i"], speaker=l["speaker"], text=l["text"], start=round(t, 3), end=round(t + len(w) / sr, 3)))
        seg += [w, np.zeros(int(l["pause"] * sr), dtype=np.float32)]
        t += len(w) / sr + l["pause"]
    sf.write(os.path.join(a.out, "joined.wav"), np.concatenate(seg), sr0)
    speech = sum(s["dur"] for s in stats); gen = sum(s["gen_s"] for s in stats)
    summ = dict(lines=len(lines), speech_s=round(speech, 2), gen_s=round(gen, 1),
                s_per_speech_s=round(gen / max(speech, 0.01), 2), model=stats[0]["model"] if stats else None)
    json.dump(dict(summary=summ, timeline=timeline, per_line=sorted(stats, key=lambda x: x["i"])),
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("design"); d.add_argument("--voices", default="voices/voices.json"); d.add_argument("--only", default="")
    d.add_argument("--takes", type=int, default=1); d.add_argument("--out", required=True)
    r = sp.add_parser("render"); r.add_argument("--job", required=True); r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1); r.add_argument("--model", default="1.7B", choices=["1.7B", "0.6B"])
    r.add_argument("--out", required=True)
    j = sp.add_parser("join"); j.add_argument("--job", required=True); j.add_argument("--parts", required=True); j.add_argument("--out", required=True)
    a = p.parse_args()
    dict(design=cmd_design, render=cmd_render, join=cmd_join)[a.cmd](a)
