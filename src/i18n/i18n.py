import json
import locale
import os
from pathlib import Path

def load_language_list(language):
    json_path = os.path.join(Path(__file__).resolve().parent, f"locale/{language}.json")

    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/de_DE.json")  
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/en_US.json")
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/es_ES.json")                   
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/ja_JP.json")
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/ko_KR.json")    
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/pt_BR.json")    
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/zh_CN.json")
    # json_path = os.path.join(Path(__file__).resolve().parent, f"locale/zh_TW.json")    
  

    with open(json_path, "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
            
        locale_dir = os.path.join(Path(__file__).resolve().parent, "locale")
        json_path = os.path.join(locale_dir, f"{language}.json")
        if not os.path.exists(json_path):
            # fr_BE, fr_CH... -> fr_FR : même langue, autre pays
            prefix = (language or "").split("_")[0]
            same_language = sorted(f[:-5] for f in os.listdir(locale_dir) if f.startswith(prefix + "_") and f.endswith(".json")) if prefix else []
            language = same_language[0] if same_language else "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
