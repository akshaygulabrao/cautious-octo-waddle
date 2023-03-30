import numpy as np
from boxes import *
import tf2onnx
import tensorflow as tf
import onnx
import tensorrt as trt
import skimage
import pycuda.driver as cuda
import pycuda.autoinit
from common_cuda import *

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

model = tf.keras.models.load_model('ssd_model',compile=False)

onnx_model,_ = tf2onnx.convert.from_keras(model)
onnx.checker.check_model(onnx_model)


# Modify the ONNX model to add NMS
graph = onnx_model.graph
graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
graph.input[0].type.tensor_type.shape.dim[1].dim_value=128
graph.input[0].type.tensor_type.shape.dim[2].dim_value=128
graph.input[0].type.tensor_type.shape.dim[3].dim_value=1

# Save ONNX model as sanity check baselines
onnx.save(onnx_model, 'model.onnx')

 # the build phase
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network,logger)
success = parser.parse_from_file('model.onnx')
print(f"Success:{success}")
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
profile = builder.create_optimization_profile();
profile.set_shape("input_1", (1,128, 128,1), (1,128, 128,1), (1,128,128,1)) 
config.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
# Allocate memory for input and output tensors
image = skimage.img_as_float32(skimage.io.imread("0104.pgm"))
print(image.shape)
if len(image.shape) == 2:
    image = image[:,:, np.newaxis]
inputs, outputs, bindings, stream = allocate_buffers(engine)
inputs[0].host = image
# cuda.memcpy_htod(inputs[0]['allocation'], image)
do_inference(context, bindings, inputs, outputs, stream, batch_size=1)
"""
classes = outputs[0].host.reshape(-1,2)
offsets = outputs[1].host.reshape(-1,8)
print(offsets)
"""
classes = outputs[0].host.reshape(-1,2)
offsets = outputs[1].host.reshape(-1,8)
class_names, rects, class_ids, boxes = show_boxes(args,
                                          image,
                                          classes,
                                          offsets,
                                          ssd.feature_shapes,
                                          show=True)


