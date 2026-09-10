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
# Backbone Swin (timm). swin_base ~ analogue capacité à ResNet-101 ;
# swin_tiny_patch4_window7_224 pour une version plus légère (mémoire/batch).
SWIN_MODEL = "swin_base_patch4_window7_224"
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
# Mode densité : nutriments prédits comme densités /100g = tous SAUF la masse.
DENSITY_NUTRIENTS = tuple(n for n in NUTRIENT_NAMES if n != "mass")  # calories, fat, carb, protein
FLAVA_MODEL = "facebook/flava-full"
FLAVA_ALIGN_LAMBDA = 10.0       # poids de l'alignement vision <-> texte (FLAVA_TXT)
FLAVA_VIS_LAMBDA = 10.0         # poids de l'alignement vision <-> image FLAVA (FLAVA_VISUEL)
FLAVA_ALIGN_TEMPERATURE = 0.4
FLAVA_PROJ_DIM = 256
# Négatifs texte supplémentaires par step pour le contrastif (cache FLAVA gelé).
# Batch=16 + 48 => pool de 64 textes côté image->texte. Voir losses/align.py.
FLAVA_ALIGN_NEG = 48
