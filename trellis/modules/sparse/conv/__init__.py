from trellis.modules.sparse import BACKEND


SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'

if BACKEND == "torchsparse":
    from trellis.modules.sparse.conv.conv_torchsparse import *
elif BACKEND == "spconv":
    from trellis.modules.sparse.conv.conv_spconv import *
