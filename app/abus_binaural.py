"""
Restauration des sons binauraux après la séparation Demucs.

Les battements binauraux sont des sons purs et stables (par ex. 298,8 Hz à gauche et 302,65 Hz à
droite). Demucs en range parfois une partie avec la voix, surtout quand quelqu'un parle par-dessus :
le fond garde alors les bonnes fréquences mais plus faiblement par endroits.

Principe : on repère dans le morceau d'origine les sons purs stables (pics étroits qui durent
plusieurs secondes, entre 25 et 1000 Hz), oreille par oreille, et on remet dans le fond exactement
ces fréquences-là, prises dans l'original, au niveau général du fond. Une voix ne reste jamais
plusieurs secondes sur la même fréquence, elle n'est donc pas réinjectée.
On travaille à 4000 Hz (largement suffisant pour des porteuses < 1000 Hz) : peu de mémoire,
puis la correction est ajoutée au fond à sa fréquence d'échantillonnage d'origine.
"""
import os
import subprocess

import numpy as np

import structlog
logger = structlog.get_logger()

RATE = 4000
NPERSEG = 8192          # 2,05 s -> résolution 0,49 Hz
HOP = 2048
F_MIN, F_MAX = 25.0, 1000.0
PEAK_OVER_MEDIAN_DB = 18.0
STABLE_FRAMES = 4        # présent sur au moins 4 trames voisines sur 7 (~3 s)


def _read(path, rate=RATE):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-f", "f32le",
                          "-ac", "2", "-ar", str(rate), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def _stft(x):
    win = np.hanning(NPERSEG)
    n = 1 + max(len(x) - NPERSEG, 0) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, NPERSEG)[::HOP][:n] * win
    return np.fft.rfft(frames, axis=1)


def _istft(spec, length):
    win = np.hanning(NPERSEG)
    frames = np.fft.irfft(spec, n=NPERSEG, axis=1) * win
    out = np.zeros(length)
    norm = np.zeros(length)
    for i, frame in enumerate(frames):
        a = i * HOP
        b = min(a + NPERSEG, length)
        out[a:b] += frame[:b - a]
        norm[a:b] += win[:b - a] ** 2
    return out / np.maximum(norm, 1e-8)


def _tonal_mask(mag):
    """Pics étroits et stables : les porteuses binaurales (et les drones), pas la voix."""
    freqs = np.fft.rfftfreq(NPERSEG, 1 / RATE)
    band = (freqs >= F_MIN) & (freqs <= F_MAX)
    db = 20 * np.log10(mag + 1e-12)
    floor = np.median(db[:, band], axis=1, keepdims=True)
    peak = np.ones_like(db, dtype=bool)
    for k in (1, 2, 3):
        peak &= db >= np.roll(db, k, axis=1)
        peak &= db >= np.roll(db, -k, axis=1)
    peak &= db > floor + PEAK_OVER_MEDIAN_DB
    peak[:, ~band] = False
    # stabilité dans le temps (à ±1 case de fréquence près)
    near = peak | np.roll(peak, 1, axis=1) | np.roll(peak, -1, axis=1)
    count = np.zeros(peak.shape, dtype=np.int16)
    for dt in range(-3, 4):
        count += np.roll(near, dt, axis=0)
    stable = peak & (count >= STABLE_FRAMES)
    # on prend aussi les 2 cases voisines pour capter toute l'énergie du son pur
    return stable | np.roll(stable, 1, axis=1) | np.roll(stable, -1, axis=1) \
        | np.roll(stable, 2, axis=1) | np.roll(stable, -2, axis=1)


def restore_binaural(bed_file: str, original_file: str, output_file: str, report: dict = None) -> bool:
    """
    Écrit dans output_file le fond sonore (bed_file) dont les sons purs stables ont été remis au
    niveau de l'original. Renvoie True si une correction a été appliquée.
    """
    original = _read(original_file)
    bed = _read(bed_file)
    length = min(len(original), len(bed))
    if length < NPERSEG * 2:
        return False
    original, bed = original[:length], bed[:length]

    delta = np.zeros((length, 2))
    stats = {}
    for ch, name in ((0, "G"), (1, "D")):
        O = _stft(original[:, ch])
        B = _stft(bed[:, ch])
        mask = _tonal_mask(np.abs(O))
        if not mask.any():
            continue
        ratio = np.abs(B[mask]) / (np.abs(O[mask]) + 1e-12)
        # niveau général du fond par rapport à l'original (là où Demucs a bien gardé les sons purs)
        gain = float(np.clip(np.percentile(ratio, 75), 0.05, 1.0))
        # seulement les manques nets (> 6 dB sous le niveau du fond) et durables (~3 s) :
        # on évite de toucher aux passages déjà corrects et aux variations passagères
        deficit = mask & (np.abs(B) < 0.5 * gain * np.abs(O))
        count = np.zeros(deficit.shape, dtype=np.int16)
        for dt in range(-3, 4):
            count += np.roll(deficit, dt, axis=0)
        missing = deficit & (count >= 6)
        corr = np.zeros_like(O)
        corr[missing] = O[missing] * gain - B[missing]
        delta[:, ch] = _istft(corr, length)
        stats[name] = {"gain_db": round(20 * np.log10(gain), 1),
                       "tonal_bins": int(mask.sum()), "restored_bins": int(missing.sum())}
    if report is not None:
        report.update(stats)
    logger.info(f"[abus_binaural.py] restore_binaural - {stats}")
    if not np.any(delta):
        return False

    delta_file = os.path.splitext(output_file)[0] + "_delta.wav"
    pcm = (np.clip(delta, -1, 1) * 32767).astype(np.int16)
    _write_wav(delta_file, pcm, RATE)
    ext = os.path.splitext(output_file)[1].lstrip(".").lower()
    codec = {"flac": ["-c:a", "flac", "-sample_fmt", "s32"], "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"]}.get(
        ext, ["-c:a", "pcm_s24le"])
    # le fond garde sa fréquence d'échantillonnage ; la correction (4 kHz) est rééchantillonnée dessus
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                            "-of", "csv=p=0", bed_file], capture_output=True, text=True).stdout.strip() or "48000"
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", bed_file, "-i", delta_file, "-filter_complex",
           f"[1:a]aresample={probe}:resampler=soxr[d];[0:a][d]amix=inputs=2:duration=first:normalize=0[out]",
           "-map", "[out]", "-ac", "2", *codec, output_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        cmd[cmd.index(f"[1:a]aresample={probe}:resampler=soxr[d];[0:a][d]amix=inputs=2:duration=first:normalize=0[out]")] = \
            f"[1:a]aresample={probe}[d];[0:a][d]amix=inputs=2:duration=first:normalize=0[out]"
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        try:
            os.remove(delta_file)
        except OSError:
            pass
    return os.path.exists(output_file)


def _write_wav(path, pcm, rate):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
