import gc

import torch
import torch.nn as nn
from transformers import AutoProcessor, FlavaModel


class FlavaBranch(nn.Module):
    """Branche(s) d'alignement FLAVA, gelées, avec projection(s) entraînable(s).

    Deux voies indépendantes, activables séparément :
      - texte (``use_text``)  : encodeur texte FLAVA -> features de la recette ;
      - image (``use_image``) : encodeur image FLAVA -> features de la photo du plat.

    Dans les deux cas l'encodeur FLAVA est gelé ; seules les têtes de projection
    (``text_proj`` / ``img_proj``) vers l'espace visuel ``vis_dim`` sont apprises.
    Les features FLAVA d'un plat ne changeant jamais (encodeur gelé, image/texte de
    base), on les pré-calcule une fois (cache) puis on peut libérer les encodeurs
    lourds via :meth:`free_encoders`.
    """

    def __init__(self, model_name: str, vis_dim: int = 256,
                 use_text: bool = True, use_image: bool = False):
        super().__init__()
        self.use_text = use_text
        self.use_image = use_image
        self.processor = AutoProcessor.from_pretrained(model_name)
        full = FlavaModel.from_pretrained(model_name)
        flava_dim = full.config.projection_dim

        self.text_model = self.text_projection = None
        self.image_model = self.image_projection = None
        if use_text:
            self.text_model = full.text_model
            self.text_projection = full.text_projection
            self.text_proj = nn.Linear(flava_dim, vis_dim)
        if use_image:
            self.image_model = full.image_model
            self.image_projection = full.image_projection
            self.img_proj = nn.Linear(flava_dim, vis_dim)

        # On ne garde que ce qui sert ; le reste est libéré.
        del full.multimodal_model
        if not use_text:
            del full.text_model, full.text_projection
        if not use_image:
            del full.image_model, full.image_projection
        del full
        gc.collect()

        for m in (self.text_model, self.text_projection, self.image_model, self.image_projection):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def _device(self):
        return next(self.parameters()).device

    # ----- Texte -----
    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        keys = ("input_ids", "attention_mask", "token_type_ids")
        model_inputs = {k: v.to(self._device()) for k, v in inputs.items() if k in keys}
        out = self.text_model(**model_inputs)
        feats = self.text_projection(out.last_hidden_state)
        return feats[:, 0] if feats.dim() == 3 else feats

    @torch.no_grad()
    def encode_text_corpus(self, texts: list[str], chunk: int = 64) -> torch.Tensor:
        feats = [self._encode_text(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=self._device())

    def project_text(self, flava_feats: torch.Tensor) -> torch.Tensor:
        return self.text_proj(flava_feats)

    # ----- Image -----
    @torch.no_grad()
    def _encode_image(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device())
        out = self.image_model(pixel_values=pixel_values)
        feats = self.image_projection(out.last_hidden_state)
        return feats[:, 0] if feats.dim() == 3 else feats

    @torch.no_grad()
    def encode_image_corpus(self, images: list, chunk: int = 32) -> torch.Tensor:
        feats = [self._encode_image(images[i:i + chunk]) for i in range(0, len(images), chunk)]
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=self._device())

    def project_image(self, flava_feats: torch.Tensor) -> torch.Tensor:
        return self.img_proj(flava_feats)

    # ----- Libération mémoire après mise en cache -----
    def free_encoders(self):
        """Supprime les encodeurs FLAVA gelés (features déjà cachées) ; garde les
        têtes de projection entraînables."""
        self.text_model = self.text_projection = None
        self.image_model = self.image_projection = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def trainable_parameters(self):
        params = []
        if self.use_text:
            params += list(self.text_proj.parameters())
        if self.use_image:
            params += list(self.img_proj.parameters())
        return params
