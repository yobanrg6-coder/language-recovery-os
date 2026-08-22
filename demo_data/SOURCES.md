# Demo data — sources and licenses

Every file under `demo_data/` used by the one-click "Load Mapudungun demo
archive" button (`web_app/app.py::DEMO_SOURCES`) is public-domain or
openly licensed academic/community material, cited below. None of it was
supplied under an informal community permission — this is deliberate:
governance and provenance are exactly what Language Recovery OS promises
to demonstrate, so the demo data has to hold up to the same standard.

## Dictionary — `dictionary_augusta_1916.pdf` / `dictionary_augusta_1916_ocr.txt`

Félix José de Augusta, *Diccionario Araucano-Español y Español-Araucano*,
1916. Source: [archive.org](https://archive.org/details/diccionarioarauc01fluoft).
**License: public domain** (published 1916, well past any copyright term).

## Grammar — `grammar_augusta_1903_ocr.txt`

Félix José de Augusta, *Gramática Araucana*, 1903. Source:
[archive.org](https://archive.org/details/gramticaaraucan00augugoog).
**License: public domain**.

## Corpus — `corpus_repo/translation-clean/*.txt`

Source: [github.com/mingjund/mapudungun-corpus](https://github.com/mingjund/mapudungun-corpus)
(AVENUE project — CMU, Chilean Ministry of Education, Instituto de
Estudios Indígenas at Universidad de La Frontera). **License: CC
BY-NC-SA 3.0 — non-commercial use only.**

Curated down from the full corpus (3,106 files / 97MB across
`TRANSCRIPTION/`, `TRANSLATION/`, `transcription-clean/`,
`translation-clean/`, `transcription-2align/`, `dataset_splits/`) to 3
representative aligned Mapudungun/Spanish transcript pairs, trimmed
2026-08-19:

| File | Notes |
|---|---|
| `nmlch-pmope1.txt` | Used by `DEMO_SOURCES` today — the file the one-click demo actually loads as the corpus source. |
| `nmlch-pmope2.txt` | Companion file, same recorded conversation as pmope1. Kept for future corpus expansion, not currently wired into `DEMO_SOURCES`. |
| `nfmcp-nfemm1.txt` | Different conversation (remedios/enfermedad vocabulary — `l'awen`, `kutran`). Kept for content diversity / future evidence cross-referencing, not currently wired into `DEMO_SOURCES`. |

The rest of the cloned corpus (including its own nested `.git/`) was
moved out of the repo, not needed for the demo, and would have shipped
the full CC BY-NC-SA dataset to a public GitHub repo without cause.

**Citations required by the corpus license** (from `corpus_repo/README.md`):

Raw data:
```
@dataset{mapudungun,
    title={Mapudungun Speech Corpus},
    author={Luis Caniupil, Flor Caniupil; Héctor Painequeo; Rosendo Huisca; Hugo Carrasco; Rodolfo M Vega; Lori Levin; Jaime Carbonell}
}
```

Cleaned dataset (what `translation-clean/` is):
```
@misc{duan2019mapudungun,
    author={Mingjun Duan, Carlos Fasola, Sai Krishna Rallabandi, Rodolfo M. Vega, Antonios Anastasopoulos, Lori Levin, and Alan W Black}
    title={A Resource for Computational Experiments on Mapudungun},
    note={preprint},
    year={2019}
}
```

Full license text kept at `corpus_repo/License.txt` per license terms
(4a — a copy of the license must ship with any distributed copy).

## Audio — `audio_victor_wikitongues_mapudungun.webm`

Wikitongues, real speaker of Mapudungun ("Victor"), ~61s. Source:
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Victor_speaking_Mapudungun.webm).
**License: CC BY 3.0** (commercial use permitted, attribution required).

## Known data note (non-blocking)

The original CMU audio archive referenced in `corpus_repo/README.md`
(`http://tts.speech.cs.cmu.edu/mapudungun/AUDIO.zip`) no longer resolves
— confirmed dead via DNS, WebFetch, and no Wayback Machine snapshot. The
demo's real audio therefore comes from a different source (Wikitongues)
than its corpus text (CMU/AVENUE). This is intentional and arguably a
better demo: it's exactly the product's real use case — cross-referencing
a newly digitized recording against a pre-existing but unrelated corpus.
