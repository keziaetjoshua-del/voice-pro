import gradio as gr
import pysubs2
import re
import time
from deep_translator import GoogleTranslator, MyMemoryTranslator
from app.abus_translate_deepl import DeepLTranslator, DeepLError, deepl_available

from app.abus_genuine import *
from app.abus_path import *
from app.abus_text import *
from app.abus_nlp_spacy import *

import structlog
logger = structlog.get_logger()

# 회사망 등의 보안장비가 translate.google.com 요청을 간헐적으로 거부(레이트리밋)하는
# 환경 대응: 요청 간 소량 간격 + 거부 시 지수 백오프 재시도.
REQUEST_INTERVAL = 0.2   # 각 요청 사이 간격(초) — 레이트리밋 유발 완화
MAX_RETRIES = 2          # 라인당 재시도 횟수 (Google 차단 시 오래 기다리지 않고 MyMemory로 전환)
GOOGLE_COOLDOWN = 600    # Google이 차단한 뒤 다시 시도하기까지 (초)
MAX_CONSECUTIVE_FAILURES = 3   # 두 서비스 모두 연속 실패 시 중단


class TranslationUnavailable(Exception):
    pass


class _FallbackTranslator:
    """
    Google (gratuit) d'abord ; s'il refuse (trop de requêtes, réseau d'entreprise...), on passe à
    MyMemory pour le reste de la session et on arrête vite si les deux échouent, au lieu
    de réessayer pendant des minutes.
    """
    google_blocked_until = 0.0

    def __init__(self, source_code: str, target_code: str, code_to_name: dict):
        self.google = GoogleTranslator(source=source_code, target=target_code)
        self.mymemory = None
        src_name = code_to_name.get(source_code)
        tgt_name = code_to_name.get(target_code)
        if src_name and tgt_name:
            try:
                self.mymemory = MyMemoryTranslator(source=src_name, target=tgt_name)
            except Exception as e:
                logger.warning(f"[abus_translate_deep.py] MyMemory unavailable for {src_name}->{tgt_name}: {e}")
        self.consecutive_failures = 0
        self.used_mymemory = False

    def translate(self, text: str) -> str:
        errors = []
        if time.time() >= _FallbackTranslator.google_blocked_until:
            try:
                result = DeepTranslator._translate_with_retry(self.google, text)
                self.consecutive_failures = 0
                return result
            except Exception as e:
                errors.append(f"Google: {e}")
                _FallbackTranslator.google_blocked_until = time.time() + GOOGLE_COOLDOWN
                logger.warning(f"[abus_translate_deep.py] Google Translate refused, switching to MyMemory for {GOOGLE_COOLDOWN}s - {e}")
        if self.mymemory is not None:
            try:
                result = self.mymemory.translate(text)
                time.sleep(REQUEST_INTERVAL)
                self.consecutive_failures = 0
                self.used_mymemory = True
                return result
            except Exception as e:
                errors.append(f"MyMemory: {e}")
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise TranslationUnavailable(" | ".join(errors))
        raise Exception(" | ".join(errors))



class DeepTranslator:
    def __init__(self) -> None:
        self.translator = GoogleTranslator(source='auto', target='en')
        self.languages_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        
   
    def get_languages(self) -> list:
        capitalized_keys = [key.capitalize() for key in self.languages_dict.keys()]
        return capitalized_keys
    
    def get_language_code(self, language_name) -> str:
        search_name = language_name.lower()
        for key, value in self.languages_dict.items():
            if key.lower() == search_name:
                return value
        return "en"
    
    def get_language_value(self, language_name):
        search_name = language_name.lower()
        for key, value in self.languages_dict.items():
            if key.lower() == search_name:
                return key
        return None


    formality = "prefer_more"   # vouvoiement par défaut (méditations guidées)

    def _translate_events_deepl(self, subs, source_code, target_code) -> bool:
        """Traduit tous les sous-titres avec DeepL (si une clé est configurée). Renvoie False pour passer au plan B."""
        if not deepl_available():
            return False
        events = [e for e in subs if e.text]
        try:
            translated = DeepLTranslator(source_code, target_code, DeepTranslator.formality).translate_many(
                [e.plaintext for e in events])
        except DeepLError as e:
            logger.error(f"[abus_translate_deep.py] DeepL failed: {e}")
            gr.Warning(f"DeepL : {e}. " + i18n("Falling back to the free services."), duration=15)
            return False
        for event, text in zip(events, translated):
            event.plaintext = text
        gr.Info(i18n("Translated with DeepL."), duration=8)
        return True

    def _code_to_name(self) -> dict:
        return {code: name for name, code in self.languages_dict.items()}

    @staticmethod
    def _translate_with_retry(translator: GoogleTranslator, text: str) -> str:
        """레이트리밋으로 인한 간헐적 연결 거부를 지수 백오프로 흡수한다."""
        delay = 2
        for attempt in range(MAX_RETRIES):
            try:
                result = translator.translate(text)
                time.sleep(REQUEST_INTERVAL)
                return result
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(f"[abus_translate_deep.py] translate retry {attempt + 1}/{MAX_RETRIES - 1} in {delay}s - {e}")
                time.sleep(delay)
                delay *= 2
    
  
    
    def translate_text(self, source_lang: str, target_lang: str, text: str, progress=gr.Progress()) -> str:
        source_code = self.get_language_code(source_lang)
        target_code = self.get_language_code(target_lang)
        
        translator = _FallbackTranslator(source_code, target_code, self._code_to_name())
        
        # line 끝 마침표 확인인
        use_punctuation = AbusText.has_ending_marks([text])
        
        # 텍스트를 문장 단위로 분리
        sentences = AbusText.split_into_sentences(text, use_punctuation)
        sentences = sentences
        
        if deepl_available():
            try:
                translated = DeepLTranslator(source_code, target_code, DeepTranslator.formality).translate_many(sentences)
                gr.Info(i18n("Translated with DeepL."), duration=8)
                return ' '.join(translated)
            except DeepLError as e:
                logger.error(f"[abus_translate_deep.py] translate_text - DeepL failed: {e}")
                gr.Warning(f"DeepL : {e}. " + i18n("Falling back to the free services."), duration=15)

        translated_sentences = []
        failed_count = 0

        # 각 문장을 번역
        for sentence in progress.tqdm(sentences, desc="Translating sentences..."):
            try:
                translated = translator.translate(sentence)
                translated_sentences.append(translated)
                logger.debug(f"[abus_translate_deep.py] translate_text - {source_code}: {sentence} -> {target_code}: {translated}")
            except TranslationUnavailable as e:
                logger.error(f"[abus_translate_deep.py] translate_text - no translation service reachable: {e}")
                raise gr.Error(i18n("No free translation service is reachable from this network (Google and MyMemory refused). Use the DeepL route instead: button 1, translate in DeepL, paste, button 2."), duration=None)
            except Exception as e:
                logger.error(f"Translation error: {e}")
                translated_sentences.append(sentence)  # 에러 발생 시 원본 문장 사용
                failed_count += 1

        if failed_count > 0:
            gr.Warning(f"({failed_count}/{len(sentences)}) " + i18n("Some lines could not be translated and were kept in the original language. The translation service may be rate-limited by your network - please try again later."), duration=None)

        # 번역된 문장들을 다시 하나의 텍스트로 결합
        final_text = ' '.join(translated_sentences)
        return final_text

    def translate_file(self, source_lang: str, target_lang: str, subtitle_file_path: str, output_file_path: str, progress=gr.Progress()):
        tts_source_file = path_add_postfix(subtitle_file_path, f"-{source_lang}", ".srt")
        
        # AbusText.process_subtitle_for_tts(subtitle_file_path, tts_source_file)
        AbusSpacy.process_subtitle_for_tts(subtitle_file_path, tts_source_file)
        
        source_code = self.get_language_code(source_lang)
        target_code = self.get_language_code(target_lang)

        
        # Load subtitles using pysubs2
        full_subs = pysubs2.load(tts_source_file)
        if self._translate_events_deepl(full_subs, source_code, target_code):
            full_subs.save(output_file_path)
            cmd_delete_file(tts_source_file)
            return

        translator = _FallbackTranslator(source_code, target_code, self._code_to_name())
        logger.debug(f"[abus_translate_deep.py] translate_file {source_code}: {subtitle_file_path} -> {target_code}: {output_file_path}")

        # Load subtitles using pysubs2
        full_subs = pysubs2.load(tts_source_file)
        subs = full_subs
        
        # 구두점이 없는 언어의 경우 각 자막을 개별적으로 번역
        failed_count = 0
        for event in progress.tqdm(subs, desc='Translate...'):
            if not event.text:
                continue

            text = event.plaintext
            try:
                translated_text = translator.translate(text)
                if translated_text:
                    event.text = translated_text
                    logger.debug(f"[abus_translate_deep.py] translate_file : text       - {text}")
                    logger.debug(f"[abus_translate_deep.py] translate_file : translated - {translated_text}")
                else:
                    logger.warning(f"[abus_translate_deep.py] translate_file - Empty translation for: {text}")
            except TranslationUnavailable as e:
                logger.error(f"[abus_translate_deep.py] translate_file - no translation service reachable: {e}")
                cmd_delete_file(tts_source_file)
                raise gr.Error(i18n("No free translation service is reachable from this network (Google and MyMemory refused). Use the DeepL route instead: button 1, translate in DeepL, paste, button 2."), duration=None)
            except Exception as e:
                logger.error(f"Translation error for text '{text}': {e}")
                # 에러 발생 시 원본 텍스트 유지
                failed_count += 1

        if failed_count > 0:
            gr.Warning(f"({failed_count}/{len(subs)}) " + i18n("Some lines could not be translated and were kept in the original language. The translation service may be rate-limited by your network - please try again later."), duration=None)

        if translator.used_mymemory:
            gr.Info(i18n("Google Translate refused the requests: MyMemory was used instead. Proofread the translation."), duration=15)

        # Save the translated subtitles
        subs.save(output_file_path)
        cmd_delete_file(tts_source_file)

            
