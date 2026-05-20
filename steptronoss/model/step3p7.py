from __future__ import annotations

from steptronoss.model.common.image_insert_decoder import ImageInsertDecoderMixin
from steptronoss.model.step3p5 import Step3p5Model


class Step3p7Model(ImageInsertDecoderMixin, Step3p5Model):
    """Step3.7 decoder with a decoupled vision encoder and image insertion."""

    def build(self, layer_map):
        super().build(layer_map)
        self.build_multimodal_modules()
