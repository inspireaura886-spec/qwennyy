#!/usr/bin/env python3
"""Listen-check + mastering for TTS output.
  python qa.py --list list.json --out qa.json [--master DIR]
list.json: [[wav_path, reference_text], ...]
Per file: duration, peak, clipped samples, silence gaps > 0.6 s, median f0 (pYIN), integrated LUFS,
faster-whisper small.en transcript and word error rate vs the reference text.
--master DIR writes a -14 LUFS copy of every file (true-peak limited to -1 dBTP by a simple peak clamp).
"""
import argparse, json, os, re
import numpy as np, soundfile as sf, librosa, pyloudnorm as pyln

NUM = {"1979": "nineteen seventy nine"}


def norm(t):
    t = t.lower()
    for k, v in NUM.items():
        t = t.replace(k, v)
    t = t.replace("-", " ").replace("’", "'")
    return re.findall(r"[a-z']+", t)


def wer(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        nd = [i]
        for j, y in enumerate(h, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (x != y)))
        d = nd
    return d[-1] / max(len(r), 1)


def master(w, sr, target=-14.0):
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(w)
    g = 10 ** ((target - lufs) / 20)
    y = w * g
    lim = 10 ** (-1 / 20)
    pk = np.abs(y).max()
    if pk > lim:  # soft-knee limiter: tanh above 0.7*lim
        k = 0.7 * lim
        a = np.abs(y)
        over = a > k
        y[over] = np.sign(y[over]) * (k + (lim - k) * np.tanh((a[over] - k) / (lim - k)))
    return y, lufs, meter.integrated_loudness(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True); p.add_argument("--out", required=True); p.add_argument("--master", default="")
    a = p.parse_args()
    from faster_whisper import WhisperModel
    M = WhisperModel("small.en", device="cpu", compute_type="int8")
    res = []
    for path, ref in json.load(open(a.list)):
        w, sr = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        fr = int(0.02 * sr)
        e = np.array([np.sqrt(np.mean(w[i:i + fr] ** 2)) for i in range(0, max(len(w) - fr, 1), fr)])
        sil = e < 10 ** (-40 / 20) * max(e.max(), 1e-6)
        gaps, run = [], 0
        for s in list(sil) + [False]:
            if s:
                run += 1
            else:
                if run * 0.02 > 0.6:
                    gaps.append(round(run * 0.02, 2))
                run = 0
        y16 = librosa.resample(w, orig_sr=sr, target_sr=16000)
        f0, _, _ = librosa.pyin(y16, fmin=60, fmax=700, sr=16000, frame_length=1024)
        f0 = f0[~np.isnan(f0)]
        segs, _ = M.transcribe(y16, beam_size=5, language="en")
        hyp = " ".join(s.text for s in segs).strip()
        r = dict(file=path, dur=round(len(w) / sr, 2), sr=sr, peak=round(float(np.abs(w).max()), 3),
                 clipped=int((np.abs(w) > 0.999).sum()), gaps_over_0_6s=gaps,
                 f0_median=round(float(np.median(f0)), 1) if len(f0) else None,
                 wer=round(wer(norm(ref), norm(hyp)), 3), asr=hyp, ref=ref)
        if a.master:
            os.makedirs(a.master, exist_ok=True)
            y, l0, l1 = master(w.astype(np.float64), sr)
            mp = os.path.join(a.master, os.path.basename(os.path.dirname(path)) + "__" + os.path.basename(path)
                              if os.path.basename(path) == "ref.wav" else os.path.basename(path))
            sf.write(mp, y.astype(np.float32), sr, subtype="PCM_16")
            r.update(lufs_in=round(l0, 2), lufs_master=round(l1, 2), master=mp, master_peak=round(float(np.abs(y).max()), 3))
        print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
