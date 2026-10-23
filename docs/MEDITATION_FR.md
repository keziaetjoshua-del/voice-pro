# Doubler une méditation en français avec Voice-Pro

Ce fork ajoute un mode « méditation » au **Studio de doublage** : les phrases françaises restent à leur place,
les pauses sont respectées et le fond sonore (nappes, battements binauraux) garde sa stéréo et son volume.

## Ce qui a changé par rapport à Voice-Pro d'origine

| Problème d'origine | Ce que fait le mode méditation |
|---|---|
| Une phrase française trop longue repoussait **toutes** les suivantes : le décalage s'accumulait. | Chaque phrase reste ancrée sur son moment d'origine. Le débordement va d'abord dans la pause qui suit, puis la phrase est accélérée légèrement (10 % max par défaut, sans changer la hauteur de la voix). Le retard ne s'accumule jamais. |
| Le mixage baissait voix **et** fond de 6 dB. | Plus de normalisation : volumes réglables en dB, limiteur doux contre la saturation. |
| Demucs recalculé à chaque essai de voix, modèle rapide uniquement. | Choix du modèle (htdemucs_ft meilleur, htdemucs plus rapide) et séparation mise en cache. Possibilité de fournir le fond d'origine sans voix. |
| Un FLAC/MP3 avec pochette était pris pour une vidéo. | La pochette est ignorée, le fichier est traité comme audio. |
| Traduction automatique Google seulement. | Bouton « Poser sur les timings d'origine » : tu colles ta propre traduction (une phrase par ligne) et elle est placée sur les timings anglais. |

Le code est dans `app/abus_dub_timeline.py` (placement), `app/abus_ffmpeg.py` (`ffmpeg_mix_dub`)
et `app/gradio_gulliver.py` (`_mix_dub`, `_dub_bed`, `gradio_align_translation`).

## Marche à suivre

1. **Média** : dépose ton `.flac` ou `.mp3`, clique **Envoyer**.
2. **Sous-titres Whisper** : langue du média `english`, puis **Transcrire**. Sur un PC sans carte NVIDIA,
   choisis le type de calcul `int8` (plus rapide sur processeur).
3. **Traduction** :
   - soit **Traduire** (Google, rapide mais plat),
   - soit colle ta traduction dans « Texte traduit, une phrase par ligne » (autant de lignes que de sous-titres)
     puis **Poser sur les timings d'origine**.
   Relis le résultat dans « Sous-titres traduits » : tu peux le modifier directement.
4. **🧘 Méditation : rythme et mixage** :
   - Placement : *Méditation (sans dérive)* ;
   - Fond sonore : commence par *htdemucs (plus rapide)* pour un essai, *htdemucs_ft* pour la version finale ;
   - Volume du fond : souvent entre -3 et -6 dB pour que la voix se détache.
5. **Génération de voix** : onglet **kokoro** (voix française, licence Apache 2.0) ou **Edge-TTS**,
   puis lance le doublage. Un message indique les phrases encore trop longues : raccourcis-les
   dans la traduction et relance seulement la voix (Demucs n'est pas recalculé).

## Tests

```
installer_files\env\Scripts\python.exe -m pytest tests
```

## Droits

Ne publie que des doublages d'enregistrements dont tu détiens les droits (tes propres créations, ou une
licence qui l'autorise). Vérifie aussi la licence de la voix de synthèse avant un usage commercial.
