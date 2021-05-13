import tvm
from tvm import relay

class ConvertLayoutMixin:
    def convert_layout(self, mod: tvm.IRModule) -> tvm.IRModule:
        seq = tvm.transform.Sequential([
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default'],
                                           'nn.conv3d': ['NCDHW', 'default'],
                                           'qnn.conv2d': ['NCHW', 'default'],
                                           'qnn.conv3d': ['NCDHW', 'default']})
        ])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

class DynamicToStaticMixin:
    def dynamic_to_static(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Clean up unnecessary dynamic ops."""
        seq = tvm.transform.Sequential([relay.transform.DynamicToStatic()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod
