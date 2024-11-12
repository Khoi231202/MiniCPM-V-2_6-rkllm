import os
import time
import signal
from multiprocessing import Process, Queue, Event
import cv2
import numpy as np
from rkllm_binding import *
from rknnlite.api.rknn_lite import RKNNLite

# 视觉编码器进程
def vision_encoder_process(load_ready_queue, embedding_queue, img_path_queue, start_event):
    
    VISION_ENCODER_PATH = "vision_transformer.rknn"
    img_size = 448
    
    # 初始化视觉编码器
    vision_encoder = RKNNLite(verbose=False)
    model_size = os.path.getsize(VISION_ENCODER_PATH)
    print(f"Start loading vision encoder model (size: {model_size / 1024 / 1024:.2f} MB)")
    start_time = time.time()
    vision_encoder.load_rknn(VISION_ENCODER_PATH)
    end_time = time.time()
    print(f"Vision encoder loaded in {end_time - start_time:.2f} seconds")
    vision_encoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    
    # 通知主进程加载完成
    load_ready_queue.put("vision_ready")
    
    # 等待开始信号
    start_event.wait()
    
    def process_image(img_path, vision_encoder):
        img = cv2.imread(img_path)
        if img is None:
            return None
        print("Start vision inference...")
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]
        
        start_time = time.time()
        image_embeddings = vision_encoder.inference(inputs=[img], data_format="nhwc")[0].astype(np.float32)
        end_time = time.time()
        print(f"Vision encoder inference time: {end_time - start_time:.2f} seconds")
        return image_embeddings

    while True:
        img_path = img_path_queue.get()
        if img_path == "STOP":
            break
        embeddings = process_image(img_path, vision_encoder)
        if embeddings is not None:
            embedding_queue.put(embeddings)
        else:
            embedding_queue.put("ERROR")

# LLM进程
def llm_process(load_ready_queue, embedding_queue, prompt_queue, inference_done_queue, start_event):

    
    MODEL_PATH = "qwen.rkllm"
    handle = None
    
    def signal_handler(signal, frame):
        print("Ctrl-C pressed, exiting...")
        global handle
        if handle:
            abort(handle)
            destroy(handle)
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    os.environ["RKLLM_LOG_LEVEL"] = "1"
    
    inference_count = 0
    inference_start_time = 0
    def result_callback(result, userdata, state):
        nonlocal inference_start_time, inference_count
        if state == LLMCallState.RKLLM_RUN_NORMAL:
            if inference_count == 0:
                first_token_time = time.time()
                print(f"Time to first token: {first_token_time - inference_start_time:.2f} seconds")
            inference_count += 1
            print(result.contents.text.decode(), end="", flush=True)
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            print("\n\n(finished)")
            inference_done_queue.put("DONE")
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            print("\nError occurred during LLM call")
            inference_done_queue.put("ERROR")
    
    # 初始化LLM
    param = create_default_param()
    param.model_path = MODEL_PATH.encode()
    param.img_start = "<image>".encode()
    param.img_end = "</image>".encode()
    param.img_content = "<unk>".encode()
    extend_param = RKLLMExtendParam()
    extend_param.base_domain_id = 1
    param.extend_param = extend_param
    
    model_size = os.path.getsize(MODEL_PATH)
    print(f"Start loading language model (size: {model_size / 1024 / 1024:.2f} MB)")
    start_time = time.time()
    handle = init(param, result_callback)
    end_time = time.time()
    print(f"Language model loaded in {end_time - start_time:.2f} seconds")
    
    # 通知主进程加载完成
    load_ready_queue.put("llm_ready")
    
    # 创建推理参数
    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value
    
    while True:
        prompt = prompt_queue.get()
        # print(f"Received prompt: ====\n{prompt}\n====")
        if prompt == "STOP":
            break
            
        image_embeddings = embedding_queue.get()
        if isinstance(image_embeddings, str) and image_embeddings == "ERROR":
            print("Error processing image")
            continue
            
        rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_MULTIMODAL,
                                        prompt=prompt,
                                        image_embed=image_embeddings)
        
        inference_start_time = time.time()
        run(handle, rkllm_input, infer_param, None)
    
    # 清理
    destroy(handle)

def main():
    load_ready_queue = Queue()
    embedding_queue = Queue()
    img_path_queue = Queue()
    prompt_queue = Queue()
    inference_done_queue = Queue()
    start_event = Event()
    
    vision_process = Process(target=vision_encoder_process,
                           args=(load_ready_queue, embedding_queue, img_path_queue, start_event))
    lm_process = Process(target=llm_process,
                        args=(load_ready_queue, embedding_queue, prompt_queue, inference_done_queue, start_event))
    
    vision_process.start()
    lm_process.start()
    
    # 等待模型加载
    ready_count = 0
    while ready_count < 2:
        status = load_ready_queue.get()
        print(f"Received ready signal: {status}")
        ready_count += 1
    
    print("All models loaded, starting interactive mode...")
    start_event.set()
    
    # 交互循环
    try:
        while True:
            print("""
Enter your input (3 empty lines to start inference, Ctrl+C to exit, for example: 
详细描述一下{{./test.jpg}}这张图片
What is the weather in {{./test.jpg}}?
How many people are in {{./test.jpg}}?
):
""")
            user_input = []
            empty_lines = 0
            
            while empty_lines < 3:
                line = input()
                if line.strip() == "":
                    empty_lines += 1
                else:
                    empty_lines = 0
                user_input.append(line)
            
            # 解析输入
            full_input = "\n".join(user_input[:-3])  # 去掉最后3个空行
            import re
            img_match = re.search(r'\{\{(.+?)\}\}', full_input)
            if not img_match:
                print("No image path found in input")
                continue
                
            img_path = img_match.group(1)
            # 将图片标记替换为<image>标记
            prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{full_input.replace(img_match.group(0), '<image>')}<|im_end|>
<|im_start|>assistant

"""
            img_path_queue.put(img_path)
            prompt_queue.put(prompt)
            
            # 等待推理完成
            status = inference_done_queue.get()
            if status == "ERROR":
                print("Inference failed")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        img_path_queue.put("STOP")
        prompt_queue.put("STOP")
    
    vision_process.join()
    lm_process.join()

if __name__ == "__main__":
    main()