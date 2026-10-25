"""La voix traduite reprend la position gauche/droite de la voix d'origine, phrase par phrase."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydub import AudioSegment
from pydub.generators import Sine
from app.abus_dub_timeline import apply_source_panning


def test_voix_droite_puis_centre():
    d = tempfile.mkdtemp()
    s = lambda ms, vol=-12: Sine(300).to_audio_segment(duration=ms, volume=vol)
    silent = AudioSegment.silent
    # Voix d'origine : 0-3 s oreille droite seule, 5-8 s au centre
    left = silent(duration=5000) + s(3000)
    right = s(3000) + silent(duration=2000) + s(3000)
    AudioSegment.from_mono_audiosegments(left, right).export(os.path.join(d, "ref.wav"), format="wav")
    voice = s(2500, -6) + silent(duration=2500) + s(2500, -6) + silent(duration=500)   # voix FR, mono
    voice.export(os.path.join(d, "fr.wav"), format="wav")
    out = os.path.join(d, "fr_spatial.wav")
    gains = apply_source_panning(os.path.join(d, "fr.wav"), os.path.join(d, "ref.wav"),
                                 [(0, 2500, 0, 3000), (5000, 2500, 5000, 8000)], out)
    assert gains[0][0] < 0.05 and gains[0][1] > 0.95          # 1re phrase : droite
    assert gains[1][0] > 0.9 and gains[1][1] > 0.9            # 2e phrase : centre
    l, r = AudioSegment.from_file(out).split_to_mono()
    assert r[200:2300].dBFS - l[200:2300].dBFS > 25
    assert abs(r[5200:7300].dBFS - l[5200:7300].dBFS) < 1
