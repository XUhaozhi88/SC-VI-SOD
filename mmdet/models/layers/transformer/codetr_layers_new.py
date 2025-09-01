from mmdet.models.layers.transformer import DinoTransformerDecoder
from mmdet.registry import MODELS

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

# In order to save the cost and effort of reproduction,
# I did not refactor it into the style of mmdet 3.x DETR.


class DinoTransformerDecoderNew(DinoTransformerDecoder):
    """Transformer decoder of DINO."""
    def __init__(self,
                 *args,
                num_cp: int = -1,
                **kwargs) -> None:
        assert num_cp <= kwargs.get('num_layers')
        self.num_cp = num_cp
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
