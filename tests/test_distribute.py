"""Répartition d'une traduction sur les sous-titres quand le nombre de lignes diffère."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.abus_dub_timeline import distribute_translation, split_sentences

SRC = ["This is the first step along your path to a gateway.",
       "A gateway beyond which is discovery.",
       "Your own discovery of reality, of truth, of who and what you are.",
       "Now a simple preparation."]


def test_meme_nombre_inchange():
    tr = ["Un.", "Deux.", "Trois.", "Quatre."]
    assert distribute_translation(SRC, tr) == tr


def test_lignes_fusionnees():
    # DeepL a fusionné les phrases 2 et 3 sur une seule ligne
    tr = ["Voici le premier pas sur le chemin vers une porte.",
          "Une porte au-delà de laquelle se trouve la découverte. Votre propre découverte de la réalité, de la vérité, de qui et de ce que vous êtes.",
          "Maintenant, une simple préparation."]
    units = [s for line in tr for s in split_sentences(line)]
    out = distribute_translation(SRC, units)
    assert len(out) == 4
    assert out[0].startswith("Voici") and out[1].startswith("Une porte")
    assert out[2].startswith("Votre propre") and out[3].startswith("Maintenant")


def test_plus_de_lignes_que_de_sous_titres():
    tr = ["Voici le premier pas", "sur le chemin vers une porte.", "Une porte au-delà de laquelle se trouve la découverte.",
          "Votre propre découverte de la réalité,", "de la vérité, de qui et de ce que vous êtes.", "Maintenant, une simple préparation."]
    out = distribute_translation(SRC, tr)
    assert len(out) == 4
    assert " ".join(out).split() == " ".join(tr).split()      # rien de perdu, ordre gardé
    assert out[3].startswith("Maintenant")


def test_moins_de_lignes():
    out = distribute_translation(SRC, ["Tout en une seule phrase."])
    assert len(out) == 4 and sum(1 for o in out if o) == 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("OK", n)
