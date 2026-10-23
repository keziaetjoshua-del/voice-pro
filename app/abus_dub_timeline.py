"""
Placement des phrases doublées sur la ligne de temps (timeline).

Ce module remplace la boucle qui était recopiée dans les 5 moteurs de voix
(Edge, Azure, F5, CosyVoice, kokoro). L'ancienne boucle repoussait toutes les
phrases suivantes dès qu'une phrase traduite était plus longue que l'originale :
sur une méditation de 20 minutes, le décalage s'accumulait et les pauses
perdaient leur place.

Mode "meditation" (par défaut) :
  * chaque phrase reste ancrée sur son début d'origine ;
  * si elle déborde, le débordement est d'abord absorbé dans la pause suivante
    (en gardant un petit temps de respiration) ;
  * si ça ne suffit pas, la phrase est accélérée légèrement (sans changer la
    hauteur de la voix), avec un plafond réglable (10 % par défaut) ;
  * si elle déborde encore, la phrase suivante démarre juste après, puis la
    ligne de temps revient sur les timings d'origine dès la pause suivante :
    le retard ne s'accumule jamais.

Mode "legacy" : comportement d'origine de Voice-Pro.

Aucune dépendance à Gradio : la barre de progression est passée en paramètre.
"""

import os
import subprocess

from pydub import AudioSegment

import structlog
logger = structlog.get_logger()


DEFAULT_DUB_OPTIONS = {
    "timing_mode": "meditation",   # "meditation" | "legacy"
    "max_speedup": 1.10,           # accélération maximale d'une phrase (1.0 = jamais)
    "breath_gap_ms": 250,          # silence minimal gardé avant la phrase suivante
    "late_warning_ms": 500,        # au-delà, la phrase est signalée comme trop longue
}

_options = dict(DEFAULT_DUB_OPTIONS)
_last_report = {}


def set_dub_options(**kwargs):
    """Met à jour les réglages (appelé par l'interface avant chaque doublage)."""
    global _last_report
    _last_report = {}
    for key, value in kwargs.items():
        if key in DEFAULT_DUB_OPTIONS and value is not None:
            _options[key] = value
    _options["max_speedup"] = min(max(float(_options["max_speedup"]), 1.0), 1.5)
    _options["breath_gap_ms"] = max(int(_options["breath_gap_ms"]), 0)
    logger.debug(f"[abus_dub_timeline.py] set_dub_options - {_options}")


def get_dub_options():
    return dict(_options)


def get_last_report():
    """Résumé du dernier doublage : phrases accélérées, phrases en retard, etc."""
    return dict(_last_report)


def format_report(report: dict) -> str:
    """Texte court en français pour l'afficher dans l'interface."""
    if not report:
        return ""
    if report.get("mode") == "legacy":
        return ""
    parts = [f"{report['lines']} phrases placées"]
    if report["stretched"]:
        parts.append(f"{report['stretched']} légèrement accélérées")
    if report["failed"]:
        parts.append(f"{len(report['failed'])} en échec (n° {', '.join(map(str, report['failed'][:10]))})")
    text = " · ".join(parts)
    if report["late"]:
        lines = ", ".join(f"n° {n} (+{ms / 1000:.1f} s)" for n, ms in report["late"][:10])
        text += f"\nPhrases encore trop longues, à raccourcir dans la traduction : {lines}"
    return text


def time_stretch_file(input_file: str, output_file: str, ratio: float) -> bool:
    """Accélère un fichier audio d'un facteur `ratio` sans changer la hauteur (ffmpeg atempo)."""
    ratio = min(max(ratio, 0.5), 2.0)
    command = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", input_file, "-filter:a", f"atempo={ratio:.4f}", output_file,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return os.path.exists(output_file)
    except Exception as e:
        logger.error(f"[abus_dub_timeline.py] time_stretch_file - error: {e}")
        return False


def _iterate(n, progress):
    if progress is None:
        return range(n)
    try:
        return progress.tqdm(range(n), desc="Generating...")
    except Exception:
        return range(n)


def build_dub_timeline(subs, synth, segments_folder: str, output_file: str, audio_format: str,
                       progress=None, options: dict = None) -> dict:
    """
    subs   : liste d'événements pysubs2 (start/end en millisecondes, text)
    synth  : fonction (texte, chemin_fichier) -> bool qui génère la voix d'une phrase
    Écrit la piste voix complète dans `output_file` et renvoie un rapport.
    """
    global _last_report
    opts = dict(_options)
    if options:
        opts.update({k: v for k, v in options.items() if v is not None})

    os.makedirs(segments_folder, exist_ok=True)

    if opts["timing_mode"] == "legacy":
        report = _build_legacy(subs, synth, segments_folder, output_file, audio_format, progress)
    else:
        report = _build_meditation(subs, synth, segments_folder, output_file, audio_format, progress, opts)

    _last_report = report
    logger.info(f"[abus_dub_timeline.py] build_dub_timeline - {report}")
    return report


def _build_meditation(subs, synth, segments_folder, output_file, audio_format, progress, opts):
    max_speedup = float(opts["max_speedup"])
    breath_gap = int(opts["breath_gap_ms"])
    late_warning = int(opts["late_warning_ms"])

    placements = []          # (position_ms, AudioSegment)
    cursor = 0               # fin de la dernière phrase posée
    report = {"mode": "meditation", "lines": 0, "stretched": 0, "late": [], "failed": [], "max_late_ms": 0}

    for i in _iterate(len(subs), progress):
        line = subs[i]
        next_line = subs[i + 1] if i < len(subs) - 1 else None
        number = i + 1

        segment_file = os.path.join(segments_folder, f"tts_{number}.{audio_format}")
        if not line.text.strip() or not synth(line.text, segment_file) or not os.path.exists(segment_file):
            report["failed"].append(number)
            continue

        segment = AudioSegment.from_file(segment_file)
        start = max(line.start, cursor)

        # Place disponible avant la phrase suivante (en gardant une respiration)
        if next_line is not None:
            available = next_line.start - start - breath_gap
            if len(segment) > available:
                needed = len(segment) / available if available > 0 else float("inf")
                ratio = min(needed, max_speedup)
                if ratio > 1.01:
                    fitted_file = os.path.join(segments_folder, f"tts_{number}_fit.{audio_format}")
                    if time_stretch_file(segment_file, fitted_file, ratio):
                        segment = AudioSegment.from_file(fitted_file)
                        report["stretched"] += 1

        late_ms = start - line.start
        if late_ms > late_warning:
            report["late"].append((number, late_ms))
        report["max_late_ms"] = max(report["max_late_ms"], late_ms)

        placements.append((start, segment))
        cursor = start + len(segment)
        report["lines"] += 1

    if not placements:
        AudioSegment.silent(duration=1000).export(output_file, format=audio_format)
        return report

    # Toutes les phrases au même format que la première
    ref = placements[0][1]
    total = max(pos + len(seg) for pos, seg in placements)
    canvas = AudioSegment.silent(duration=total, frame_rate=ref.frame_rate)
    canvas = canvas.set_channels(ref.channels).set_sample_width(ref.sample_width)
    for pos, seg in placements:
        seg = seg.set_frame_rate(ref.frame_rate).set_channels(ref.channels).set_sample_width(ref.sample_width)
        canvas = canvas.overlay(seg, position=pos)

    canvas.export(output_file, format=audio_format)
    return report


def _build_legacy(subs, synth, segments_folder, output_file, audio_format, progress):
    """Comportement d'origine de Voice-Pro (le retard s'accumule)."""
    report = {"mode": "legacy", "lines": 0, "stretched": 0, "late": [], "failed": [], "max_late_ms": 0}
    combined_audio = AudioSegment.empty()
    for i in _iterate(len(subs), progress):
        line = subs[i]
        next_line = subs[i + 1] if i < len(subs) - 1 else None

        if i == 0:
            combined_audio += AudioSegment.silent(duration=line.start)

        segment_file = os.path.join(segments_folder, f"tts_{i + 1}.{audio_format}")
        if synth(line.text, segment_file) == False:
            report["failed"].append(i + 1)
            if next_line:
                combined_audio += AudioSegment.silent(duration=next_line.start - line.start)
            continue

        combined_audio += AudioSegment.from_file(segment_file)
        report["lines"] += 1

        if next_line and len(combined_audio) < next_line.start:
            combined_audio += AudioSegment.silent(duration=next_line.start - len(combined_audio))
        elif next_line:
            duration = next_line.end - next_line.start
            next_line.start = len(combined_audio)
            next_line.end = next_line.start + duration

    combined_audio.export(output_file, format=audio_format)
    return report
