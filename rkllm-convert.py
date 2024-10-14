from rkllm.api import RKLLM

modelpath = '.'
llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cpu')
if ret != 0:
    print('Load model failed!')
    exit(ret)

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8_g128',
                quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3, extra_qparams=qparams)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm("./qwen.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
