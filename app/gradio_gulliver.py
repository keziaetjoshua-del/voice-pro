
import json
import asyncio
import shutil

from src.config import UserConfig
from app.abus_downloader import *
from app.abus_path import *
from app.abus_ffmpeg import *
from app.abus_demucs import *
from app.abus_dub_timeline import *
from app.abus_genuine import *
from app.abus_files import *

from app.abus_asr_parameters import *
from app.abus_asr_faster_whisper import *
from app.abus_asr_whisper import *
from app.abus_asr_whisper_timestamped import *

from app.abus_translate_deep import *
from app.abus_translate_deepl import set_deepl_key
from app.abus_translate_azure import *

from app.abus_voice_ms import *
from app.abus_voice_celeb import *
from app.abus_voice_kokoro import *

from app.abus_tts_azure import *
from app.abus_tts_edge import *
from app.abus_tts_f5 import *
from app.abus_tts_cosyvoice import *
from app.abus_tts_kokoro import *


import src.ui as ui
from src.i18n.i18n import I18nAuto
i18n = I18nAuto()

import structlog
logger = structlog.get_logger()


class GradioGulliver:
    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
        
        self.fm = FileManager()

        self.downloader = YoutubeDownloader()
        self.ms_voice_manager = MSVoiceManager(self.user_config.get('ms_language', "English"))
        self.edge_tts = AzureTTS() if azure_text_api_working() else EdgeTTS()
        
        self.f5_tts = F5TTS()
        self.cosy_tts = CosyVoiceInference(self.user_config.get("cosy_model", "CosyVoice2-0.5B"))
        
        self.kokoro_tts = KokoroTTS()
        self.kokoro_vm = KokoroVoiceManager()

        self.translator = AzureTranslator() if azure_text_api_working() == True else DeepTranslator()
        
        asr_engine = self.user_config.get("asr_engine", 'faster-whisper')
        self.whisper_inf = self.switch_case(asr_engine)   

        
        # self.mdxnet_models_dir = os.path.join(os.getcwd(), 'model', 'mdxnet-model')
        # with open(os.path.join(self.mdxnet_models_dir, 'model_data.json')) as infile:
        #     self.mdx_model_params = json.load(infile)

    def switch_case(self, case):
        switch_dict = {
            'faster-whisper': lambda: FasterWhisperInference(),
            'whisper': lambda: WhisperInference(),
            'whisper-timestamped': lambda: WhisperTimestampedInference()
            }
        return switch_dict.get(case, lambda: FasterWhisperInference())()    
            
            
    def gradio_workspace_folder(self):
        cmd_open_explorer(path_workspace_folder())
    
    def gradio_temp_folder(self):
        cmd_open_explorer(path_gradio_folder())
    
    
    def get_asr_engines(self):
        return ['faster-whisper', 'whisper', 'whisper-timestamped']
    
    def update_whisper_models(self, asr_engine):
        whisper_inf = self.switch_case(asr_engine)       
        model_list = whisper_inf.available_models()
        if len(model_list) > 0:
            model_name = self.user_config.get(f'{asr_engine.replace("-", "_")}_model', 'large')
            return gr.update(choices=model_list, value=model_name)
        
        return gr.update(choices=[], value=None)

                    
    def get_whisper_models(self):
        return self.whisper_inf.available_models()
        
    def get_whisper_languages(self):
        return self.whisper_inf.available_langs()

    def get_whisper_compute_types(self):
        return FasterWhisperInference.available_compute_types()
            
    # return Video, Audio, File    
    def gradio_upload_source(self, 
                      file_obj, mic_file, youtube_url: str, video_quality: str, audio_format: str):
        
        self.user_config.set("video_quality", video_quality)
        self.user_config.set("audio_format", audio_format)

        try:
            logger.debug(f'upload_source: file_obj={file_obj}, mic_file={mic_file}, youtube_url={youtube_url}')
            # ffmpeg는 다운로드 병합/오디오 추출에 필수 — 없으면 즉시 명확한 안내
            if shutil.which("ffmpeg") is None:
                raise gr.Error(i18n("ffmpeg is not installed. Run configure.bat (or configure.sh) as administrator to install it."), duration=None)

            self.fm = FileManager()
            if self._upload(file_obj, mic_file, youtube_url, video_quality, audio_format) == False:
                raise gr.Error(i18n("Please provide a media file, a microphone recording, or a YouTube URL."), duration=None)

            source_audio = self.fm.get_split("Source.audio")
            if(self.has_video and ffmpeg_browser_compatible(self.source_file)):
                return self.source_file, source_audio, self.fm.get_all_files()
            else:
                return None, source_audio, self.fm.get_all_files()
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] gradio_upload_source - Error transcribing file: {e}")
            # duration=None: 사용자가 닫기 전까지 유지되는 오류 토스트 (자동 소멸로 놓치는 문제 방지)
            raise gr.Error(f'{e}', duration=None)
        

    def _upload(self,
                file_obj, mic_file, youtube_url: str, video_quality: str, audio_format: str):
        if (file_obj is not None):
            uploaded_file = cmd_copy_file_to(file_obj.name, path_workspace_subfolder(file_obj.name))
        elif mic_file and mic_file.strip():
            uploaded_file = cmd_copy_file_to(mic_file, path_workspace_subfolder(mic_file))
        elif youtube_url and youtube_url.strip():
            youtube_file = self.downloader.yt_download(youtube_url, path_youtube_folder(), video_quality)
            uploaded_file = cmd_copy_file_to(youtube_file, path_workspace_subfolder(youtube_file))
        else:
            return False
        
        self.source_file = uploaded_file
        
        self.has_audio, self.has_video = ffmpeg_codec_type(self.source_file)
        logger.debug(f'upload_source: source_file={self.source_file}, has_audio={self.has_audio}, has_video={self.has_video}')
        if self.has_audio == False:     # error
            raise gr.Error(i18n("The selected media has no audio track."), duration=None)
        elif self.has_video == False:   # audio-only
            self.fm.set_split("Source.video", None)
            self.fm.set_split("Source.audio", self.source_file)   
        else:
            input_audio_file = path_change_ext(self.source_file, f'.{audio_format}')
            ffmpeg_extract_audio(self.source_file, input_audio_file, audio_format)    
            self.fm.set_split("Source.video", self.source_file)
            self.fm.set_split("Source.audio", input_audio_file)
        return True     
    

    def gradio_whisper_default(self):
        return ["large", "english", "float16", 0]

    # return Video, Audio, File    
    def gradio_whisper(self, 
                      asr_engine, modelName, whisper_language, compute_type, denoise_level):
        
        self.user_config.set("asr_engine", asr_engine)
        self.user_config.set(f'{asr_engine.replace("-", "_")}_model', modelName)
        self.user_config.set("whisper_language", whisper_language)
        self.user_config.set("whisper_compute_type", compute_type)
        self.user_config.set("denoise_level", denoise_level)          
        
        try:
            source_audio = self.fm.get_split("Source.audio")
            # 미디어 미등록(또는 등록 진행 중) 상태에서 자막생성을 누른 경우
            if not source_audio:
                raise gr.Error(i18n("No media source registered. Please upload media and click Submit first."), duration=None)
            denoise_inst_path, denoise_vocal_path = self._denoise(source_audio, denoise_level)
            input_path = denoise_vocal_path if os.path.exists(denoise_vocal_path) else source_audio

            params = WhisperParameters(model_size=modelName, 
                                       lang=whisper_language.lower(), 
                                       compute_type=compute_type)   
            
            self.whisper_inf = self.switch_case(asr_engine)                            
            subtitles = self.whisper_inf.transcribe_file(input_path, params, False, gr.Progress())  # highlight_words=False
            self.fm.set_subtitles(subtitles, whisper_language, source_audio) 
            srt_file = self.fm.get_subtitle('.srt')
            srt_string = self._read_file(srt_file)
            try:
                self._last_transcription_end_ms = max((e.end for e in pysubs2.SSAFile.from_string(srt_string) if e.text.strip()), default=0)
            except Exception:
                self._last_transcription_end_ms = 0
            logger.debug(f'srt_file = {srt_file}, self.source_file = {self.source_file}')
            
            if(self.has_video and ffmpeg_browser_compatible(self.source_file)):
                if srt_file:
                    # gradio 6: 자막은 (video, srt) 튜플 대신 Video 컴포넌트의 subtitles 속성으로 전달
                    return gr.Video(value=self.source_file, subtitles=srt_file), source_audio, srt_string, self.fm.get_all_files()
                else:
                    return self.source_file, source_audio, srt_string, self.fm.get_all_files()
            else:
                return None, source_audio, srt_string, self.fm.get_all_files()
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] gradio_whisper - Error transcribing file: {e}")
            raise gr.Error(f'{e}', duration=None)

    
    # return inst, vocal    
    def _denoise(self, source_audio, denoise_level=2):
        if denoise_level == 1:
            return self._demucs_htdemucs(source_audio)
        elif denoise_level ==2:
            return self._demucs_htdemucs_ft(source_audio)
        else:
            return "", ""
                
        
            
    def _demucs_htdemucs(self, source_audio):
        _, extension = os.path.splitext(os.path.basename(source_audio))
        output_dir = os.path.dirname(source_audio)
        
        inst_audio_file, vocal_audio_file = demucs_split_file(source_audio, output_dir, 'htdemucs', extension[1:])
        self.fm.set_split("Instrumental.audio", inst_audio_file)
        self.fm.set_split("Vocals.audio", vocal_audio_file)

        return inst_audio_file, vocal_audio_file
    
    def _demucs_htdemucs_ft(self, source_audio):
        _, extension = os.path.splitext(os.path.basename(source_audio))
        output_dir = os.path.dirname(source_audio)
        
        inst_audio_file, vocal_audio_file = demucs_split_file(source_audio, output_dir, 'htdemucs_ft', extension[1:])
        self.fm.set_split("Instrumental.audio", inst_audio_file)
        self.fm.set_split("Vocals.audio", vocal_audio_file)

        return inst_audio_file, vocal_audio_file   


    # Doublage « méditation » : réglages, fond sonore, mixage
    def _check_translation_covers_source(self, translation_text):
        """Refuse une traduction qui s'arrête bien avant la fin de la transcription en cours (ancien essai resté dans la case)."""
        source_end = getattr(self, "_last_transcription_end_ms", 0)
        if not source_end or not self._is_subtitle_format(translation_text or ""):
            return
        try:
            events = [e for e in pysubs2.SSAFile.from_string(translation_text) if e.text.strip()]
        except Exception:
            return
        translated_end = max((e.end for e in events), default=0)
        if translated_end < 0.6 * source_end:
            raise gr.Error(f"{i18n('The translated subtitles stop at')} {translated_end // 60000}:{translated_end // 1000 % 60:02d}, "
                           f"{i18n('but the transcription goes to')} {source_end // 60000}:{source_end // 1000 % 60:02d}. "
                           + i18n("Click Translate again for this file before generating the voice."), duration=None)

    def _apply_dub_options(self):
        set_dub_options(
            timing_mode=self.user_config.get("dub_timing_mode", "meditation"),
            max_speedup=self.user_config.get("dub_max_speedup", 1.10),
            breath_gap_ms=self.user_config.get("dub_breath_gap_ms", 250),
        )

    def _dub_bed(self, source_audio_file):
        """Fond sonore à placer sous la voix : piste fournie, sinon séparation Demucs (mise en cache)."""
        bed_file = getattr(self, "dub_bed_file", None)
        if bed_file and os.path.exists(bed_file):
            logger.debug(f"[gradio_gulliver.py] _dub_bed - piste de fond fournie : {bed_file}")
            return bed_file

        model = self.user_config.get("dub_demucs_model", "htdemucs_ft")
        if model == "none":
            return None

        inst_audio_file, _ = self._demucs_cached(source_audio_file, model)
        return inst_audio_file

    def _demucs_cached(self, source_audio_file, model):
        """Séparation fond / voix d'origine, calculée une seule fois par morceau et par modèle."""
        if not hasattr(self, "_bed_cache"):
            self._bed_cache = {}
        key = (source_audio_file, model)
        cached = self._bed_cache.get(key)
        if cached and all(os.path.exists(f) for f in cached):
            return cached
        found = self._find_existing_stems(source_audio_file, model)
        if found:
            logger.info(f"[gradio_gulliver.py] _demucs_cached - séparation déjà faite, réutilisée : {found[0]}")
            gr.Info(i18n("Demucs separation already done for this track: reused (no new calculation)."), duration=10)
            self._bed_cache[key] = found
            return found
        if model == "htdemucs":
            result = self._demucs_htdemucs(source_audio_file)
        else:
            result = self._demucs_htdemucs_ft(source_audio_file)
        self._bed_cache[key] = result
        return result

    @staticmethod
    def _fingerprint(path):
        import hashlib
        size = os.path.getsize(path)
        h = hashlib.sha1(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(1 << 20))
            if size > (1 << 20):
                f.seek(max(size - (1 << 20), 0))
                h.update(f.read(1 << 20))
        return h.hexdigest()

    def _find_existing_stems(self, source_audio_file, model):
        """Cherche dans workspace/ une séparation Demucs déjà faite pour le même morceau (même contenu)."""
        try:
            target = self._fingerprint(source_audio_file)
            root = path_workspace_folder()
            for folder in os.listdir(root):
                folder_path = os.path.join(root, folder)
                if not os.path.isdir(folder_path):
                    continue
                for name in os.listdir(folder_path):
                    marker = f"_{model}_inst."
                    if marker not in name:
                        continue
                    base, ext = name.split(marker)
                    inst = os.path.join(folder_path, name)
                    vocal = os.path.join(folder_path, f"{base}_{model}_vocal.{ext}")
                    if not os.path.exists(vocal):
                        continue
                    for candidate in os.listdir(folder_path):
                        cand_path = os.path.join(folder_path, candidate)
                        if os.path.splitext(candidate)[0] == base and os.path.isfile(cand_path) \
                                and os.path.getsize(cand_path) == os.path.getsize(source_audio_file) \
                                and self._fingerprint(cand_path) == target:
                            return inst, vocal
        except Exception as e:
            logger.warning(f"[gradio_gulliver.py] _find_existing_stems - {e}")
        return None

    def _pan_reference(self, source_audio_file):
        """Voix d'origine isolée (Demucs) si on l'a déjà, sinon le morceau d'origine."""
        model = self.user_config.get("dub_demucs_model", "htdemucs_ft")
        for (src, _model), files in getattr(self, "_bed_cache", {}).items():
            if src == source_audio_file and files and os.path.exists(files[1]):
                return files[1]
        if model in ("htdemucs", "htdemucs_ft") and not getattr(self, "dub_bed_file", None):
            return self._demucs_cached(source_audio_file, model)[1]
        return source_audio_file

    def gradio_set_deepl_key(self, key):
        key = (key or "").strip()
        self.user_config.set("deepl_api_key", key)
        set_deepl_key(key)

    def gradio_set_deepl_formality(self, formality):
        self.user_config.set("deepl_formality", formality)
        DeepTranslator.formality = formality

    def gradio_set_bed_file(self, bed_file):
        """Piste de fond sans voix fournie par l'utilisateur (non mémorisée entre deux sessions)."""
        self.dub_bed_file = bed_file if bed_file else None

    def gradio_extract_source_lines(self, source_srt):
        """Texte d'origine sans les timings, une phrase (un sous-titre) par ligne, à traduire ailleurs (DeepL...)."""
        source_srt = (source_srt or "").strip()
        if not source_srt or not self._is_subtitle_format(source_srt):
            raise gr.Error(i18n("First transcribe the audio: the subtitles on the left must contain timings."), duration=None)
        subs = pysubs2.SSAFile.from_string(source_srt)
        return "\n".join(" ".join(e.plaintext.split()) for e in subs if e.text.strip())

    def gradio_align_translation(self, source_srt, translated_lines):
        """Pose un texte traduit (une phrase par ligne) sur les timings des sous-titres d'origine."""
        source_srt = (source_srt or "").strip()
        lines = [l.strip() for l in (translated_lines or "").splitlines() if l.strip()]
        if not source_srt or not self._is_subtitle_format(source_srt):
            raise gr.Error(i18n("First transcribe the audio: the subtitles on the left must contain timings."), duration=None)
        if not lines:
            raise gr.Error(i18n("Paste the translated text first, one sentence per line."), duration=None)

        # Une traduction collée au format sous-titres : on ne garde que le texte
        pasted = "\n".join(lines)
        if self._is_subtitle_format(pasted):
            lines = [" ".join(e.plaintext.split()) for e in pysubs2.SSAFile.from_string(pasted) if e.text.strip()]

        subs = pysubs2.SSAFile.from_string(source_srt)
        events = [e for e in subs if e.text.strip()]
        if len(lines) != len(events):
            # Nombre différent (DeepL a fusionné ou coupé des lignes) : on répartit automatiquement
            units = [sentence for line in lines for sentence in split_sentences(line)]
            texts = distribute_translation([e.plaintext for e in events], units)
            gr.Info(f"{len(lines)} {i18n('translated lines')} → {len(events)} {i18n('subtitles')} : "
                    + i18n("distributed automatically by position in the text. Check the result."), duration=15)
        else:
            texts = lines
        for event, text in zip(events, texts):
            event.text = text
        # Les sous-titres restés vides deviennent des silences
        subs.events = [e for e in subs.events if e.text.strip() or e not in events]
        return subs.to_string("srt")

    def _mix_dub(self, aidub_audio_file, source_audio_file, mixed_audio_file, audio_format):
        bed = self._dub_bed(source_audio_file)

        # Spatialisation : la voix française suit la position gauche/droite de la voix d'origine
        placements = get_last_report().get("placements") or []
        if self.user_config.get("dub_follow_pan", True) and placements:
            try:
                panned_file = path_add_postfix(aidub_audio_file, "_spatial")
                applied = apply_source_panning(aidub_audio_file, self._pan_reference(source_audio_file),
                                               placements, panned_file,
                                               match_level=bool(self.user_config.get("dub_match_level", True)))
                aidub_audio_file = panned_file
                off_center = sum(1 for gl, gr_ in applied if abs(gl - gr_) > 0.3 * max(gl, gr_, 1e-6))
                logger.info(f"[gradio_gulliver.py] _mix_dub - spatialisation : {off_center}/{len(applied)} phrases hors centre")
            except Exception as e:
                logger.error(f"[gradio_gulliver.py] _mix_dub - spatialisation impossible : {e}")
                gr.Warning(i18n("Could not follow the original left/right placement; the voice stays centered."), duration=10)

        voice_gain = float(self.user_config.get("dub_voice_gain_db", 0.0))
        bed_gain = float(self.user_config.get("dub_bed_gain_db", 0.0))
        if bed:
            ok = ffmpeg_mix_dub(aidub_audio_file, bed, mixed_audio_file, voice_gain, bed_gain, audio_format)
        else:
            ok = False
        if not ok:
            # Pas de fond (ou échec du mixage) : on livre la voix seule, au bon format
            ffmpeg_convert_audio(aidub_audio_file, mixed_audio_file, audio_format)

        report = format_report(get_last_report())
        if report:
            gr.Info(report.replace("\n", " — "), duration=15)


    # Translate
    def gradio_translate_languages(self):
        return self.translator.get_languages()
    
    
    def gradio_language_detection(self,
                                  transcription_text):
        languageName = AbusText.detect_language_name(transcription_text[:200])
        return languageName
                                        
    
    def gradio_translate(self,
                        source_lang, transcription_text, target_lang):
        if len(transcription_text) < 1:
            logger.warning(f"[gradio_gulliver.py] gradio_translate - no actions")
            return None, None, None, self.fm.get_all_files() 
        
        set_deepl_key(self.user_config.get("deepl_api_key", ""))
        DeepTranslator.formality = self.user_config.get("deepl_formality", "prefer_more")
        self.user_config.set("translate_source_language", source_lang)            
        self.user_config.set("translate_target_language", target_lang)    

        transcription_file = None
        if self._is_subtitle_format(transcription_text):
            subs = pysubs2.SSAFile.from_string(transcription_text)
            transcription_file = os.path.join(path_translate_folder(), path_new_filename(f".{subs.format}"))
            subs.save(transcription_file)

        translation_file = None
        translation_text = None
        if transcription_file:
            translation_file = self._translate_subtitle(transcription_file, source_lang, target_lang)
        else:
            translation_text = self._translate_text(transcription_text, source_lang, target_lang)

        source_video_file = self.fm.get_split("Source.video")
        source_audio_file = self.fm.get_split("Source.audio")
        
        output_video_path = gr.Video(value=source_video_file, subtitles=translation_file) if source_video_file and translation_file else source_video_file
        output_audio_path = source_audio_file
        output_translation_text = self._read_file(translation_file) if translation_file else translation_text
        return output_video_path, output_audio_path, output_translation_text, self.fm.get_all_files()    
                           

    def _read_file(self, filepath):    
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            return "Error: The file is not a valid text file or uses an unsupported encoding."
        except IOError:
            return "Error: Unable to read the file."
        
        
    def _translate_subtitle(self, subtitle_file, source_lang, target_lang):
        try:
            translation_file = path_add_postfix(subtitle_file, f"_{target_lang}")
            self.translator.translate_file(source_lang, target_lang, subtitle_file, translation_file)
            
            target_lang_code = self.translator.get_language_code(target_lang)
            self.fm.set_translation(target_lang_code, translation_file)            
            return translation_file
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _translate_subtitle - error: {e}")
            raise gr.Error(f'{e}', duration=None)
            return None           

    def _translate_text(self, text, source_lang, target_lang):
        try:
            translated = self.translator.translate_text(source_lang, target_lang, text)
            return translated   
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _translate_text - error: {e}")
            raise gr.Error(f'{e}', duration=None)
            return None            


    # TTS
    
    def _is_subtitle_format(self, text):
        try:
            pysubs2.SSAFile.from_string(text)
            return True
        except Exception as e:
            return False     


           

    # Edge-TTS or Azure-TTS                             
    def gradio_edge_default(self):
        return [0, 0, 0]        
    
    def gradio_edge_dubbing(self, 
                            translation_text, voice_name: str, 
                            semitones, speed_factor, volume_factor, audio_format: str):
                         
        logger.debug(f"[gradio_gulliver.py] gradio_edge_dubbing - \
                    voice_name = {voice_name}, \
                    semitones = {semitones}, speed_factor = {speed_factor}, volume_factor = {volume_factor}, audio_format = {audio_format}")
        
        self._apply_dub_options()
        self._check_translation_covers_source(translation_text)
        if len(translation_text) < 1:
            logger.warning(f"[gradio_gulliver.py] gradio_edge_dubbing - no actions")
            return None, None, self.fm.get_all_files() 
        
        self.user_config.set("edge_tts_pitch", semitones)     
        self.user_config.set("edge_tts_rate", speed_factor)
        self.user_config.set("edge_tts_volume", volume_factor)             
        self.user_config.set("audio_format", audio_format)
        
        
        aidub_video_file = None
        mixed_audio_file = None                

        translation_file = None 
        if len(translation_text) > 0 and self._is_subtitle_format(translation_text):
            subs = pysubs2.SSAFile.from_string(translation_text)
            translation_file = os.path.join(path_translate_folder(), path_new_filename(f".{subs.format}"))
            subs.save(translation_file)   


        if translation_file:
            aidub_video_file, mixed_audio_file = self._edge_tts_subtitle(translation_file, 
                                                voice_name, semitones, speed_factor, volume_factor, audio_format)  
        elif translation_text:
            aidub_video_file, mixed_audio_file = self._edge_tts_text(translation_text, 
                                                voice_name, semitones, speed_factor, volume_factor, audio_format)

        output_video_path = gr.Video(value=aidub_video_file, subtitles=translation_file) if aidub_video_file and translation_file else aidub_video_file
        output_audio_path = mixed_audio_file
        return output_video_path, output_audio_path, self.fm.get_all_files()      
                            

    def _edge_tts_text(self, text: str, voice_name: str, semitones, speed_factor, volume_factor, audio_format: str):
        try:
            # TTS voice
            ms_voice = self.ms_voice_manager.get_voice(voice_name)
            target_language = ms_voice.getLanguageName()
                        
            # TTS
            source_audio_file = self.fm.get_split("Source.audio")                        
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{target_language}")        
            
            self.edge_tts.text_to_voice(text, aidub_audio_file, ms_voice.name, semitones, speed_factor, volume_factor, audio_format)

            self.fm.set_dubbing(f'{voice_name}.audio', aidub_audio_file)           

            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{target_language}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{target_language}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                
                self.fm.set_dubbing(f'{voice_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file, 
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _edge_tts_text - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None    

    
    
    def _edge_tts_subtitle(self, 
                           subtitle_file, voice_name: str, semitones, speed_factor, volume_factor, audio_format: str):
        logger.debug(f"[gradio_gulliver.py] _edge_tts_subtitle - subtitle_file = {subtitle_file}, voice_name = {voice_name}, \
                     semitones = {semitones}, speed_factor = {speed_factor}, volume_factor = {volume_factor}, audio_format = {audio_format}")
        
        try:
            # TTS voice
            ms_voice = self.ms_voice_manager.get_voice(voice_name)
            target_language = ms_voice.getLanguageName()
                        
            # TTS
            source_audio_file = self.fm.get_split("Source.audio")                        
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{target_language}")        
            
            self.edge_tts.srt_to_voice(subtitle_file, aidub_audio_file, ms_voice.name, semitones, speed_factor, volume_factor, audio_format)

            self.fm.set_dubbing(f'{voice_name}.audio', aidub_audio_file)           
        
            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{target_language}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{target_language}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                
                self.fm.set_dubbing(f'{voice_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _edge_tts_subtitle - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None 


    # F5-TTS

    def gradio_f5_default(self):
        return ["SWivid/F5-TTS_v1", 1.0]
    
    def gradio_f5_available_models(self):
        return self.f5_tts.available_models()


    def gradio_f5_dubbing_single(self, 
                                 translation_text, 
                                 celeb_name, celeb_audio, celeb_transcript, 
                                 model_choice, speed_factor, audio_format: str):
        
        logger.debug(f"[gradio_gulliver.py] gradio_f5_dubbing_single - \
                    celeb_name = {celeb_name}, celeb_audio = {celeb_audio}, \
                    model_choice = {model_choice}, speed_factor = {speed_factor}, audio_format = {audio_format}")
        
        self._apply_dub_options()
        self._check_translation_covers_source(translation_text)
        if len(translation_text) < 1:
            logger.warning(f"[gradio_gulliver.py] gradio_f5_dubbing_single - no actions")
            return None, None, self.fm.get_all_files() 
        
            
        translation_file = None 
        if len(translation_text) > 0 and self._is_subtitle_format(translation_text):
            subs = pysubs2.SSAFile.from_string(translation_text)
            translation_file = os.path.join(path_translate_folder(), path_new_filename(f".{subs.format}"))
            subs.save(translation_file)                    
                

        aidub_video_file = None
        mixed_audio_file = None                

        if translation_file:
            aidub_video_file, mixed_audio_file = self._f5_tts_single(self._read_file(translation_file), 
                                                celeb_name, celeb_audio, celeb_transcript, model_choice, speed_factor, audio_format)
        elif translation_text:
            aidub_video_file, mixed_audio_file = self._f5_tts_single(translation_text, 
                                                celeb_name, celeb_audio, celeb_transcript, model_choice, speed_factor, audio_format)

        output_video_path = gr.Video(value=aidub_video_file, subtitles=translation_file) if aidub_video_file and translation_file else aidub_video_file
        output_audio_path = mixed_audio_file
        return output_video_path, output_audio_path, self.fm.get_all_files()                   
                  
        
       
    def _f5_tts_single(self, text:str, celeb_name, celeb_audio, celeb_transcript, model_choice, speed_factor, audio_format: str):
        logger.debug(f"[gradio_gulliver.py] _f5_tts_single - text = {text}, \
                    celeb_name = {celeb_name}, celeb_audio = {celeb_audio}, \
                    model_choice = {model_choice}, speed_factor = {speed_factor}, audio_format = {audio_format}")     
                
        try:
            # F5-TTS
            source_audio_file = self.fm.get_split("Source.audio")           
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{celeb_name}")                               
            self.f5_tts.infer_single(text.strip(), aidub_audio_file, celeb_audio, celeb_transcript, model_choice, speed_factor, audio_format)
            
            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{celeb_name}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
            self.fm.set_dubbing(f'{celeb_name}.audio', aidub_audio_file)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{celeb_name}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                self.fm.set_dubbing(f'{celeb_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _f5_tts_single - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None    
    
    
    # Cosy-Voice
    
    def gradio_cosy_default(self):
        return ["Zero-Shot", 1.0]    
        
    def gradio_cosy_dubbing(self, 
                        translation_text, 
                        celeb_name, celeb_audio, celeb_transcript, 
                        mode_choice, speed_factor, audio_format: str):
        self._apply_dub_options()
        self._check_translation_covers_source(translation_text)
        if len(translation_text) < 1:
            logger.warning(f"[gradio_gulliver.py] gradio_f5_dubbing_single - no actions")
            return None, None, self.fm.get_all_files() 
        
            
        translation_file = None 
        if len(translation_text) > 0 and self._is_subtitle_format(translation_text):
            subs = pysubs2.SSAFile.from_string(translation_text)
            translation_file = os.path.join(path_translate_folder(), path_new_filename(f".{subs.format}"))
            subs.save(translation_file)                    
                

        aidub_video_file = None
        mixed_audio_file = None   
                
        if translation_file:
            aidub_video_file, mixed_audio_file = self._cosy_tts_single(self._read_file(translation_file), 
                                                celeb_name, celeb_audio, celeb_transcript, mode_choice, speed_factor, audio_format)
        elif translation_text:
            aidub_video_file, mixed_audio_file = self._cosy_tts_single(translation_text, 
                                                celeb_name, celeb_audio, celeb_transcript, mode_choice, speed_factor, audio_format)

        output_video_path = gr.Video(value=aidub_video_file, subtitles=translation_file) if aidub_video_file and translation_file else aidub_video_file
        output_audio_path = mixed_audio_file
        return output_video_path, output_audio_path, self.fm.get_all_files()        

   
    
    def _cosy_tts_single(self, text:str, celeb_name, celeb_audio, celeb_transcript, mode_choice, speed_factor, audio_format: str):
        logger.debug(f"[gradio_gulliver.py] _cosy_tts_single - text = {text}, \
                    celeb_name = {celeb_name}, celeb_audio = {celeb_audio}, \
                    mode_choice = {mode_choice}, speed_factor = {speed_factor}, audio_format = {audio_format}")     
                
        try:
            # F5-TTS
            source_audio_file = self.fm.get_split("Source.audio")           
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{celeb_name}")                               
            self.cosy_tts.infer_single(text.strip(), aidub_audio_file, celeb_audio, celeb_transcript, mode_choice, speed_factor, audio_format)
            
            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{celeb_name}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
            self.fm.set_dubbing(f'{celeb_name}.audio', aidub_audio_file)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{celeb_name}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                self.fm.set_dubbing(f'{celeb_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _cosy_tts_single - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None    
       
       
       
    # kokoro    
        
    def gradio_kokoro_default(self):
        return [1, "mp3"]
    
    def gradio_kokoro_dubbing(self, 
                            translation_text, language_name, voice_name: str, 
                            speed_factor, audio_format: str):
                         
        logger.debug(f"[gradio_gulliver.py] gradio_kokoro_dubbing - \
                    language_name = {language_name}, voice_name = {voice_name}, \
                    speed_factor = {speed_factor}, audio_format = {audio_format}")
        
        self._apply_dub_options()
        self._check_translation_covers_source(translation_text)
        if len(translation_text) < 1:
            logger.warning(f"[gradio_gulliver.py] gradio_kokoro_dubbing - no actions")
            return None, None, self.fm.get_all_files() 
        
        # self.user_config.set("edge_tts_rate", speed_factor)  
        self.user_config.set("audio_format", audio_format)
        
        
        aidub_video_file = None
        mixed_audio_file = None                

        translation_file = None 
        if len(translation_text) > 0 and self._is_subtitle_format(translation_text):
            subs = pysubs2.SSAFile.from_string(translation_text)
            translation_file = os.path.join(path_translate_folder(), path_new_filename(f".{subs.format}"))
            subs.save(translation_file)   


        if translation_file:
            aidub_video_file, mixed_audio_file = self._kokoro_tts_subtitle(translation_file, 
                                                language_name, voice_name, speed_factor, audio_format)  
        elif translation_text:
            aidub_video_file, mixed_audio_file = self._kokoro_tts_text(translation_text, 
                                                language_name, voice_name, speed_factor, audio_format)
        
        
        output_video_path = gr.Video(value=aidub_video_file, subtitles=translation_file) if aidub_video_file and translation_file else aidub_video_file
        output_audio_path = mixed_audio_file
        return output_video_path, output_audio_path, self.fm.get_all_files()        
                     
        
    def _kokoro_tts_text(self, text: str, language_name, voice_name: str, speed_factor, audio_format: str):
        try:
            # TTS voice
            kokoro_voice = self.kokoro_vm.find_voice(language_name, voice_name)
                        
            # TTS
            source_audio_file = self.fm.get_split("Source.audio")                        
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{voice_name}")        
            
            self.kokoro_tts.text_to_voice(text, aidub_audio_file, kokoro_voice, speed_factor, audio_format)

            self.fm.set_dubbing(f'{voice_name}.audio', aidub_audio_file)           

            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{voice_name}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{voice_name}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                
                self.fm.set_dubbing(f'{voice_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file, 
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _kokoro_tts_text - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None    

    
    
    def _kokoro_tts_subtitle(self, 
                           subtitle_file, language_name, voice_name: str, speed_factor, audio_format: str):
        logger.debug(f"[gradio_gulliver.py] _kokoro_tts_subtitle - subtitle_file = {subtitle_file}, \
                    voice_name = {voice_name}, language_name = {language_name}, \
                    speed_factor = {speed_factor}, audio_format = {audio_format}")
        
        try:
            # TTS voice
            kokoro_voice = self.kokoro_vm.find_voice(language_name, voice_name)
                        
            # TTS
            source_audio_file = self.fm.get_split("Source.audio")                        
            aidub_audio_file = path_add_postfix(source_audio_file, f"_{voice_name}")        
            
            self.kokoro_tts.srt_to_voice(subtitle_file, aidub_audio_file, kokoro_voice, speed_factor, audio_format)

            self.fm.set_dubbing(f'{voice_name}.audio', aidub_audio_file)           
        
            # Mix
            mixed_audio_file = path_add_postfix(source_audio_file, f"_mixed_{voice_name}")            
            self._mix_dub(aidub_audio_file, source_audio_file, mixed_audio_file, audio_format)
                                
            if self.has_video:
                source_video_file = self.fm.get_split("Source.video")
                aidub_video_file = path_add_postfix(source_video_file, f"_{voice_name}")
                
                ffmpeg_replace_audio(source_video_file, mixed_audio_file, aidub_video_file)
                
                self.fm.set_dubbing(f'{voice_name}.video', aidub_video_file)
                return aidub_video_file, mixed_audio_file
            else:
                return None, mixed_audio_file
        except Exception as e:
            logger.error(f"[gradio_gulliver.py] _kokoro_tts_subtitle - Error : {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None         