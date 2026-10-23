"""Test du mixage voix + fond (app/abus_ffmpeg.ffmpeg_mix_dub) : stéréo binaurale préservée, pas de -6 dB, durée."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_gr = types.ModuleType("gradio"); _gr.Progress = lambda *a, **k: None
sys.modules.setdefault("gradio", _gr)   # abus_ffmpeg importe gradio pour la progression
sys.modules.setdefault("ffmpeg", types.ModuleType("ffmpeg"))
from pydub import AudioSegment
from pydub.generators import Sine
from app.abus_ffmpeg import ffmpeg_mix_dub


def test_mix_binaural():
    d = tempfile.mkdtemp()
    left = Sine(200).to_audio_segment(duration=10000, volume=-12)
    right = Sine(210).to_audio_segment(duration=10000, volume=-12)
    bed = AudioSegment.from_mono_audiosegments(left, right)
    bed.export(os.path.join(d, "bed.wav"), format="wav")
    voice = AudioSegment.silent(duration=2000, frame_rate=24000) + Sine(440).to_audio_segment(duration=3000, volume=-12).set_frame_rate(24000)
    voice.export(os.path.join(d, "voice.wav"), format="wav")
    out = os.path.join(d, "mix.flac")
    assert ffmpeg_mix_dub(os.path.join(d, "voice.wav"), os.path.join(d, "bed.wav"), out, 0, 0, "flac")
    mix = AudioSegment.from_file(out)
    assert mix.channels == 2
    assert abs(len(mix) - 10000) < 50, len(mix)              # durée = morceau d'origine
    l, r = mix.split_to_mono()
    quiet = slice(0, 1500)
    # Fond seul (avant la voix) : même niveau qu'à l'origine (ancien mixage : -6 dB)
    assert abs(l[quiet].dBFS - left[quiet].dBFS) < 0.5, (l[quiet].dBFS, left[quiet].dBFS)
    # Binaural préservé : gauche et droite restent différentes
    assert l[quiet].get_array_of_samples()[:2000] != r[quiet].get_array_of_samples()[:2000]
    # Gain du fond appliqué
    out2 = os.path.join(d, "mix2.wav")
    assert ffmpeg_mix_dub(os.path.join(d, "voice.wav"), os.path.join(d, "bed.wav"), out2, 0, -10, "wav")
    l2 = AudioSegment.from_file(out2).split_to_mono()[0]
    assert abs((l[quiet].dBFS - l2[quiet].dBFS) - 10) < 0.5




def test_pochette_pas_video():
    """Un FLAC avec pochette intégrée doit être traité comme de l'audio seul."""
    import subprocess
    from app.abus_ffmpeg import ffmpeg_codec_type
    d = tempfile.mkdtemp()
    wav, png, flac = (os.path.join(d, n) for n in ("a.wav", "c.png", "a.flac"))
    Sine(220).to_audio_segment(duration=500).export(wav, format="wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", png], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-i", png, "-map", "0:a", "-map", "1:v",
                    "-c:a", "flac", "-c:v", "png", "-disposition:v", "attached_pic", flac], check=True)
    assert ffmpeg_codec_type(flac) == (True, False)


if __name__ == "__main__":
    test_mix_binaural(); print("OK test_mix_binaural")
    test_pochette_pas_video(); print("OK test_pochette_pas_video")
