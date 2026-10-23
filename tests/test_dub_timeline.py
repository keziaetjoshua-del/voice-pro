"""
Tests du placement des phrases (app/abus_dub_timeline.py) avec des sons de synthèse.
Lancer depuis la racine du projet :  python -m pytest tests  (ou)  python tests/test_dub_timeline.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydub import AudioSegment
from pydub.generators import Sine

from app.abus_dub_timeline import build_dub_timeline, format_report


def make_subs(spec):
    """spec = [(début_ms, fin_ms, durée_voix_fr_ms), ...] ; la durée FR est codée dans le texte."""
    return [SimpleNamespace(start=s, end=e, text=f"len={d}") for s, e, d in spec]


def fake_synth(text, path):
    duration = int(text.split("=")[1])
    Sine(440).to_audio_segment(duration=duration).set_frame_rate(24000).export(path, format="wav")
    return True


def onsets(path, threshold=-40.0, step=10):
    """Débuts des zones sonores (ms) dans le fichier produit."""
    audio = AudioSegment.from_file(path)
    result, inside = [], False
    for pos in range(0, len(audio), step):
        loud = audio[pos:pos + step].dBFS > threshold
        if loud and not inside:
            result.append(pos)
        inside = loud
    return result


def run(spec, mode="meditation", **opts):
    folder = tempfile.mkdtemp()
    out = os.path.join(folder, "voice.wav")
    report = build_dub_timeline(make_subs(spec), fake_synth, os.path.join(folder, "seg"), out, "wav",
                                options={"timing_mode": mode, **opts})
    return out, report


def test_phrases_restent_ancrees():
    # Les phrases FR tiennent dans leurs créneaux : elles démarrent pile à l'heure.
    spec = [(1000, 4000, 2500), (10000, 13000, 2800), (30000, 33000, 2000)]
    out, report = run(spec)
    starts = onsets(out)
    assert [abs(a - b) <= 20 for a, b in zip(starts, [1000, 10000, 30000])] == [True] * 3, starts
    assert report["late"] == [] and report["stretched"] == 0


def test_debordement_absorbe_dans_la_pause():
    # Phrase FR de 6 s dans un créneau de 3 s, mais longue pause ensuite : rien ne bouge.
    spec = [(1000, 4000, 6000), (20000, 23000, 2000)]
    out, report = run(spec)
    starts = onsets(out)
    assert abs(starts[1] - 20000) <= 20, starts
    assert report["stretched"] == 0


def test_acceleration_plafonnee():
    # 4 s de voix pour 3,75 s disponibles (4 s - respiration 250 ms) : accélération ~1,07.
    spec = [(0, 4000, 4000), (4000, 7000, 2000)]
    out, report = run(spec, max_speedup=1.10)
    assert report["stretched"] == 1
    assert report["late"] == []
    starts = onsets(out)
    assert abs(starts[1] - 4000) <= 20, starts


def test_pas_de_derive_cumulee():
    # Trois phrases trop longues d'affilée, puis une pause : la 5e phrase revient à l'heure.
    spec = [(0, 2000, 4000), (2000, 4000, 4000), (4000, 6000, 4000), (6000, 8000, 1000), (60000, 62000, 1500)]
    out, report = run(spec)
    starts = onsets(out)
    assert abs(starts[-1] - 60000) <= 20, starts
    assert len(report["late"]) >= 1
    assert "trop longues" in format_report(report)

    # Avec l'ancien comportement, la dernière phrase est aussi à l'heure ici (grande pause),
    # mais les phrases 2 à 4 sont repoussées sans accélération : on vérifie juste que le mode existe.
    out_legacy, report_legacy = run(spec, mode="legacy")
    assert report_legacy["mode"] == "legacy"


def test_ancien_mode_derive():
    # Sans pause pour rattraper, l'ancien mode accumule le retard ; le nouveau le limite.
    spec = [(i * 3000, i * 3000 + 3000, 3600) for i in range(10)]
    out_new, _ = run(spec)
    out_old, _ = run(spec, mode="legacy")
    end_new = len(AudioSegment.from_file(out_new))
    end_old = len(AudioSegment.from_file(out_old))
    assert end_new < end_old, (end_new, end_old)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
