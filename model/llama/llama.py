"""LLaMA models' APIs."""
import copy

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore import mint
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid
from mindformers.modules.layers import Linear, FreqsMgr
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_disable_custom_fa, get_predict_run_mode, get_use_rope_self_define

from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.models.llama.llama_interleave import LLamaDecodeLayerInterleave

from mindformers.models.llama.llama import LlamaModel, LlamaForCausalLM

from mindformers.models.utils import lazy_inline
# from ...tools.logger import logger

__all__ = ['LlamaModel', 'LlamaForCausalLM', "LlamaForCausalLMSptMoBA"]

from model.llama.llama_config import LlamaConfigSupMoba


class LlamaModelSptMoBA(LlamaModel):
    def __init__(self, config:LlamaConfigSupMoba=None):
        # 1.调用父类初始化，此时的 layer 为父类中定义的 DecoderLayer
        super().__init__(config)

        # 2. 根据配置决定是否替换父类中 Attention 的实现
        if config.use_moba_attn:
            if config.use_flash_attention:
                # TODO：清空layers列表，并重新添加层
                pass
            # TODO： 是否使用 FA 优化进行

class LlamaForCausalLMSptMoBA(LlamaForCausalLM):
    """
        LlamaForCausalLM 的子类，为支持MoBA计算而进行相应的适配改造
    """
    def __init__(self, config:LlamaConfigSupMoba=None):
        # 1. 调用父类初始化，此时的 self.model 是 LlamaModel实例
        super().__init__(config)

        # 2. 根据配置替换父类中的model
        if config.use_moba_attn:
            if config.use_flash_attention:
                pass
            else:
                pass