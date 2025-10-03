import importlib

BACKEND = "spconv"  # "spconv" or "torchsparse"
DEBUG = False
ATTN = "flash_attn"  # "xformers" or "flash_attn"


__attributes = {
    "SparseTensor": "basic",
    "sparse_batch_broadcast": "basic",
    "sparse_batch_op": "basic",
    "sparse_cat": "basic",
    "sparse_unbind": "basic",
    "SparseGroupNorm": "norm",
    "SparseLayerNorm": "norm",
    "SparseGroupNorm32": "norm",
    "SparseLayerNorm32": "norm",
    "SparseReLU": "nonlinearity",
    "SparseSiLU": "nonlinearity",
    "SparseGELU": "nonlinearity",
    "SparseActivation": "nonlinearity",
    "SparseLinear": "linear",
    "sparse_scaled_dot_product_attention": "attention",
    "SerializeMode": "attention",
    "sparse_serialized_scaled_dot_product_self_attention": "attention",
    "sparse_windowed_scaled_dot_product_self_attention": "attention",
    "SparseMultiHeadAttention": "attention",
    "SparseConv3d": "conv",
    "SparseInverseConv3d": "conv",
    "SparseDownsample": "spatial",
    "SparseUpsample": "spatial",
    "SparseSubdivide": "spatial",
}

__submodules = ["transformer"]

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
