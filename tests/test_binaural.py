"""Les sons purs binauraux affaiblis par la séparation sont remis au niveau de l'original."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.abus_binaural import restore_binaural, _write_wav, _read

def tone_level(x, f, sr=4000):
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), 1 / sr)
    return 20 * np.log10(s[np.argmin(np.abs(fr - f))] + 1e-12)

def test_restauration():
    sr, dur = 16000, 40
    t = np.arange(sr * dur) / sr
    left = 0.1 * np.sin(2 * np.pi * 200 * t)
    right = 0.1 * np.sin(2 * np.pi * 204 * t)                      # battement 4 Hz
    noise = 0.02 * np.random.default_rng(0).standard_normal((len(t), 2))
    orig = np.stack([left, right], 1) + noise
    bed = orig.copy()
    bed[sr * 20:, 0] -= 0.9 * left[sr * 20:]; bed[sr * 20:, 1] -= 0.9 * right[sr * 20:]   # Demucs a "mangé" les sons purs
    d = tempfile.mkdtemp()
    _write_wav(os.path.join(d, "o.wav"), (orig * 32767).astype(np.int16), sr)
    _write_wav(os.path.join(d, "b.wav"), (bed * 32767).astype(np.int16), sr)
    out = os.path.join(d, "r.wav")
    assert restore_binaural(os.path.join(d, "b.wav"), os.path.join(d, "o.wav"), out)
    o, b, r = _read(os.path.join(d, "o.wav")), _read(os.path.join(d, "b.wav")), _read(out)
    seg = slice(4000 * 25, 4000 * 35)
    assert tone_level(b[seg, 0], 200) < tone_level(o[seg, 0], 200) - 15
    assert abs(tone_level(r[seg, 0], 200) - tone_level(o[seg, 0], 200)) < 3
    assert abs(tone_level(r[seg, 1], 204) - tone_level(o[seg, 1], 204)) < 3
    early = slice(4000 * 3, 4000 * 15)                              # partie déjà correcte : inchangée
    assert abs(tone_level(r[early, 0], 200) - tone_level(b[early, 0], 200)) < 1
