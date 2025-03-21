from mindformers import LlamaConfig
import inspect
import logging
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ["LlamaConfigSupMoba"]


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlamaConfigSupMoba(LlamaConfig):
    """
        添加使用 MoBA 的配置
    """

    def __init__(self,
                 use_moba_attn=False,
                 moba_chunk_size: int = 1024,   # 分块大小，默认为 LlamaConfig 默认的序列长度
                 moba_top_k: int = 3,            # topK
                 moba_attn_impl: str = 'moba',  # moba 的实现方式
                 **kwargs):
        # 1. 获取父类 LlamaConfig.__init__ 的参数列表（排除 self 和 **kwargs）
        parent_signature = inspect.signature(LlamaConfig.__init__)
        parent_params = [
            p.name
            for p in parent_signature.parameters.values()
            if p.name not in ['self', 'kwargs']
        ]

        # print(parent_params)

        # 2. 从 kwargs 中提取出父类所需的参数
        parent_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in parent_params}

        # 3. 调用父类初始化，传递提取的参数和剩余 kwargs
        super().__init__(**parent_args, **kwargs)

        # 4. 添加自定义参数
        self.use_moba_attn = use_moba_attn
        self.moba_chunk_size = moba_chunk_size
        self.moba_top_k = moba_top_k
        self.moba_attn_impl = moba_attn_impl

        # 5. 校验参数是否合理
        if self.use_moba_attn:
            if self.moba_chunk_size > self.seq_length:
                logging.warning("MoBA Chunck Size is larger than Sequence Length, MoBA might lose efficiency！")


class MoBAConfig:
    moba_chunk_size: int
    moba_topK: int


if __name__ == '__main__':
    cfg = LlamaConfigSupMoba(num_heads=1000)
    print(cfg)
