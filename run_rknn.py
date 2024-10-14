import os
import time
import numpy as np
from rkllm_binding import *
from rknnlite.api.rknn_lite import RKNNLite
import signal
import cv2

MODEL_PATH = "qwen.rkllm"
VISION_ENCODER_PATH = "vision_transformer.rknn"
handle = None
img_size = 448

# exit on ctrl-c
def signal_handler(signal, frame):
    print("Ctrl-C pressed, exiting...")
    global handle
    if handle:
        abort(handle)
        destroy(handle)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# export RKLLM_LOG_LEVEL=1
os.environ["RKLLM_LOG_LEVEL"] = "1"

inference_count = 0
inference_start_time = 0
def result_callback(result, userdata, state):
    global inference_start_time
    global inference_count
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        if inference_count == 0:
            first_token_time = time.time()
            print(f"Time to first token: {first_token_time - inference_start_time:.2f} seconds")
        inference_count += 1
        print(result.contents.text.decode(), end="", flush=True)
    elif state == LLMCallState.RKLLM_RUN_FINISH:
        print("\n\n(finished)")
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        print("\nError occurred during LLM call")

# Initialize vision encoder
vision_encoder = RKNNLite(verbose=False)
model_size = os.path.getsize(VISION_ENCODER_PATH)
print(f"Start loading vision encoder model (size: {model_size / 1024 / 1024:.2f} MB)")
start_time = time.time()
vision_encoder.load_rknn(VISION_ENCODER_PATH)
end_time = time.time()
print(f"Vision encoder loaded in {end_time - start_time:.2f} seconds (speed: {model_size / (end_time - start_time) / 1024 / 1024:.2f} MB/s)")
vision_encoder.init_runtime()

# image embedding
img_path = "test.jpg"

normalize_mean = 0.5
normalize_std = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (img_size, img_size))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
# img = img / 255.0
# img = (img - normalize_mean) / normalize_std
img = img[np.newaxis, :, :, :]
print(img.shape)
start_time = time.time()
image_embeddings = vision_encoder.inference(inputs=[img.astype(np.float32)], data_type="float32", data_format="nhwc")[0]
end_time = time.time()
print(f"Vision encoder inference time: {end_time - start_time:.2f} seconds")
print(image_embeddings.flags)
print(image_embeddings.shape)
np.save("image_embeddings_rknn.npy", image_embeddings)


vision_encoder.release() # free memory, rockchip plz fix this

# Initialize RKLLM
param = create_default_param()
param.model_path = MODEL_PATH.encode()
param.img_start = "<image>".encode()
param.img_end = "</image>".encode()
param.img_content = "<unk>".encode()
extend_param = RKLLMExtendParam()
extend_param.base_domain_id = 0  # iommu domain 0 for vision encoder
param.extend_param = extend_param
model_size = os.path.getsize(MODEL_PATH)
print(f"Start loading language model (size: {model_size / 1024 / 1024:.2f} MB)")
start_time = time.time()
handle = init(param, result_callback)
end_time = time.time()
print(f"Language model loaded in {end_time - start_time:.2f} seconds (speed: {model_size / (end_time - start_time) / 1024 / 1024:.2f} MB/s)")

# Create input
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
详细介绍一下这张图片: <|im_end|>
<|im_start|>assistant

"""

# image_embeddings = np.load("image_embeddings_pth_orig.npy")
# print(image_embeddings.shape)
# rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_EMBED, embed=image_embeddings.astype(np.float32))

rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_MULTIMODAL, prompt=prompt, image_embed=image_embeddings.astype(np.float32))

# Create inference parameters
infer_param = RKLLMInferParam()
infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value

# Run RKLLM
print("Start inference...")
inference_start_time = time.time()
run(handle, rkllm_input, infer_param, None)

# Clean up
destroy(handle)