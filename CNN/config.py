import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA = ROOT.parent / "data" / "nutrition5k"
DATA_ROOT = Path(os.environ.get("FUSION_DATA_ROOT", str(_DEFAULT_DATA)))
SPLITS_DIR = DATA_ROOT / "dish_ids" / "splits"
METADATA_CAFE1 = DATA_ROOT / "metadata" / "dish_metadata_cafe1.csv"
IMAGERY_DIR = DATA_ROOT / "realsense_overhead"
DISHES_SKIP_FILE = ROOT / "dishes_to_skip.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"
PRETRAINED_DIR = Path(os.environ.get("FUSION_MIDAS_PRETRAINED", str(CHECKPOINT_DIR / "pretrained")))
MIDAS_WEIGHTS = PRETRAINED_DIR / "dpt_hybrid_midas.pt"
TORCH_HUB_DIR = PRETRAINED_DIR / "torch_hub"
DEPTH_MIDAS_DIR = ROOT / "depth" / "midas"
DEPTH_TRAIN_SPLIT = SPLITS_DIR / "depth_train_ids.txt"
DEPTH_TEST_SPLIT = SPLITS_DIR / "depth_test_ids.txt"
NUTRIENT_NAMES = ("calories", "mass", "fat", "carb", "protein")
IMAGE_W = 640
IMAGE_H = 480
FUSION_EPOCHS = 150
FUSION_BATCH_SIZE = 32
FUSION_LR = 5e-5
FUSION_LR_DECAY = 0.98
FUSION_VAL_RATIO = 0.1
DEPTH_MAX_M = 0.4
MIDAS_MODEL_TYPE = "DPT_Hybrid"
def fusion_ckpt_path(depth_source: str, flava_txt: bool = False, flava_vis: bool = False,
                     no_skip: bool = False, density: bool = False) -> Path:
    """Checkpoint distinct par variante — aucun écrasement.

    Suffixe : ``_f`` + ``t`` (texte) et/ou ``v`` (image) pour FLAVA ; ``_all`` si
    entraîné sans écarter les images « non convenables » (NO_SKIP) ; ``_d`` en mode
    densité (predict masse + densités /100g, total = densité×masse à l'éval, au lieu
    des totaux directs). Ex. fusion_gt_best.pt, fusion_gt_ft_best.pt,
    fusion_gt_d_best.pt, fusion_gt_ft_all_d_best.pt.
    """
    suffix = ""
    if flava_txt or flava_vis:
        suffix = "_f" + ("t" if flava_txt else "") + ("v" if flava_vis else "")
    if no_skip:
        suffix += "_all"
    if density:
        suffix += "_d"
    return CHECKPOINT_DIR / f"fusion_{depth_source}{suffix}_best.pt"
FLAVA_MODEL = "facebook/flava-full"
FLAVA_ALIGN_LAMBDA = 10.0       # poids de l'alignement vision <-> texte (FLAVA_TXT)
FLAVA_VIS_LAMBDA = 10.0         # poids de l'alignement vision <-> image FLAVA (FLAVA_VISUEL)
FLAVA_ALIGN_TEMPERATURE = 0.4
FLAVA_PROJ_DIM = 256
# Négatifs texte supplémentaires par step pour le contrastif (cache FLAVA gelé).
# Batch=16 + 48 => pool de 64 textes côté image->texte. Voir losses/align.py.
FLAVA_ALIGN_NEG = 48

# ---- Recipe1M : pré-entraînement densité /100g -------------------------------
# Gros dataset externe, situé dans le dossier data FRÈRE du projet (ROOT.parent.parent
# = Stage_2026/ en local, gluci/ sur le cluster), pas dans le projet (voir to_cluster.sh).
_DEFAULT_RECIPE1M = ROOT.parent.parent / "data"
RECIPE1M_ROOT = Path(os.environ.get("RECIPE1M_DATA_ROOT", str(_DEFAULT_RECIPE1M)))
RECIPE1M_IMAGES_DIR = RECIPE1M_ROOT / "recipe1M"
RECIPE1M_NUTR_JSON = RECIPE1M_ROOT / "Extraction_Recipes1M" / "recipes_with_nutritional_info.json"
RECIPE1M_IMAGES_CSV = RECIPE1M_ROOT / "Extraction_Recipes1M" / "nutritional_images.csv"
# Profondeur MiDaS (gelé) pré-calculée pour les photos Recipe1M — générée une fois
# (generate_recipe1m_depth.py), stockée à côté des données, réutilisée à chaque époque.
RECIPE1M_DEPTH_DIR = RECIPE1M_ROOT / "depth_midas"

# Densités prédites au pré-entraînement = nutriments Nutrition5k SAUF la masse
# (Recipe1M ne donne pas la portion de la photo → aucune masse à prédire ici).
DENSITY_NUTRIENTS = tuple(n for n in NUTRIENT_NAMES if n != "mass")  # calories, fat, carb, protein
# Correspondance densité Nutrition5k -> champ /100g de Recipe1M.
# ATTENTION : carb <- sugars est un PROXY (Recipe1M n'a pas les glucides totaux,
# seulement les sucres) ; energy/fat/protein sont directs.
RECIPE1M_DENSITY_MAP = {"calories": "energy", "fat": "fat", "carb": "sugars", "protein": "protein"}

PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_LR = 1e-4
PRETRAIN_LR_DECAY = 0.98
PRETRAIN_IMG_SIZE = 224


def pretrain_ckpt_path(flava_txt: bool = False, flava_vis: bool = False,
                       tag: str = "best") -> Path:
    """Poids pré-entraînés (backbone RGB) — un fichier par variante FLAVA.
    Suffixe ``_f`` + ``t``/``v`` comme pour la fusion. ``tag`` = "best" (meilleur
    val, à transférer) ou "last" (dernier état complet, pour REPRISE après un job
    coupé). Ex. pretrain_rgb_best.pt, pretrain_rgb_ft_last.pt."""
    suffix = ""
    if flava_txt or flava_vis:
        suffix = "_f" + ("t" if flava_txt else "") + ("v" if flava_vis else "")
    return CHECKPOINT_DIR / f"pretrain_rgb{suffix}_{tag}.pt"
