from trellis.modules.sparse import BACKEND


SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'


def __from_env():
    import os

    global SPCONV_ALGO
    env_spconv_algo = os.environ.get("SPCONV_ALGO")
    if env_spconv_algo is not None and env_spconv_algo in [
        "auto",
        "implicit_gemm",
        "native",
    ]:
        SPCONV_ALGO = env_spconv_algo


__from_env()

if BACKEND == "torchsparse":
    from trellis.modules.sparse.conv.conv_torchsparse import *
elif BACKEND == "spconv":
    from trellis.modules.sparse.conv.conv_spconv import *
