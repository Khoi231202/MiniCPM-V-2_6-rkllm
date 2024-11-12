---
base_model:
- openbmb/MiniCPM-V-2_6
tags:
- rknn
- rkllm
---
# MiniCPM-V-2_6-rkllm

## (English README see below)

在RK3588上运行强大的MiniCPM-V-2.6 视觉大模型!

- 推理速度(RK3588): 视觉编码器 3.2s(三核并行) + LLM 填充 1.7s (92 tokens / 53 tps) + 解码 4.03 tps
- 内存占用(RK3588, 默认上下文长度): 视觉编码器 1.9GB + LLM 7.8GB = 9.7GB

## 使用方法

1. 克隆或者下载此仓库到本地. 模型较大, 请确保有足够的磁盘空间.
   
2. 开发板的RKNPU2内核驱动版本必须>=0.9.6才能运行这么大的模型. 
   使用root权限运行以下命令检查驱动版本:
   ```bash
   > cat /sys/kernel/debug/rknpu/version 
   RKNPU driver: v0.9.8
   ```
   如果版本过低, 请更新驱动. 你可能需要更新内核, 或查找官方文档以获取帮助.
   
3. 安装依赖

```bash
pip install numpy<2 opencv-python 
```
你还需要手动安装rknn-toolkit2-lite2. 

4. 运行
   
```bash
python multiprocess_inference.py
```

如果实测性能不理想, 可以调整CPU调度器让CPU始终运行在最高频率, 并把推理程序绑定到大核(`taskset -c 4-7 python multiprocess_inference.py`)

test.jpg:
![test.jpg](./test.jpg)

>```
>Start loading language model (size: 7810.02 MB)
>
>I rkllm: rkllm-runtime version: 1.1.2, rknpu driver version: 0.9.8, platform: RK3588
>
>W rknn-toolkit-lite2 version: 2.2.0
>Start loading vision encoder model (size: 942.29 MB)
>Vision encoder loaded in 10.22 seconds
>I RKNN: [02:28:20.939] RKNN Runtime Information, librknnrt version: 2.1.0 (967d001cc8@2024-08-07T19:28:19)
>I RKNN: [02:28:20.939] RKNN Driver Information, version: 0.9.8
>I RKNN: [02:28:20.940] RKNN Model Information, version: 6, toolkit version: 2.2.0(compiler version: 2.2.0 (c195366594@2024-09-14T12:24:14)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: dynamic_shape
>W RKNN: [02:28:20.940] RKNN Model version: 2.2.0 not match with rknn runtime version: 2.1.0
>Received ready signal: vision_ready
>Language model loaded in 29.21 seconds
>Received ready signal: llm_ready
>All models loaded, starting interactive mode...
>
>Enter your input (3 empty lines to start inference, Ctrl+C to exit, for example: 
>详细描述一下{{./test.jpg}}这张图片
>What is the weather in {{./test.jpg}}?
>How many people are in {{./test.jpg}}?
>):
>
>以猫猫的身份描述一下{{test.jpg}}吧喵~
>
>
>
>Start vision inference...
>
>Vision encoder inference time: 3.28 seconds
>
>Time to first token: 1.74 seconds
>
>观察到一个人正走在街道上，旁边是一条繁忙的道路。他手里撑着一把蓝白相间的伞保护自己免受阳光直射的侵袭，并正在过马路横穿斑马线。
>
>附近停泊和行驶着几辆汽车，显示出这是一个熙攘的城市环境。在人行道的一侧可以看到各种树木和建筑物的存在，进一步增强了都市感。
>
>从猫的角度看，这个人穿着米色外套、黑色裤子和蓝色鞋子，走在繁忙的街道上让人感觉很酷炫。同时这个人的行为也表明了他正在享受一个阳光明媚的日子，利用伞来保护自己免受直射阳光的影响。
>总的来说这是一个宁静的城市环境，有一个人在过马路，周围停着汽车和各种树木建筑物的存在，营造出一种熙攘的城市氛围。
>
>(finished)
>
>--------------------------------------------------------------------------------------
> Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second       
>--------------------------------------------------------------------------------------
> Prefill       1708.63          94        18.18                    55.01                   
> Generate      40668.17         164       248.97                   4.02                    
>--------------------------------------------------------------------------------------
>```

## 模型转换

#### 准备工作

1. 安装rknn-toolkit2 v2.1.0或更高版本, 以及rkllm-toolkit v1.1.2或更高版本.
2. 下载此仓库到本地, 但不需要下载`.rkllm`和`.rknn`结尾的模型文件.
3. 下载MiniCPM-V-2.6的huggingface模型仓库到本地. (https://huggingface.co/openbmb/MiniCPM-V-2_6)
  
#### 转换LLM

1. 将此仓库中的`rename_tensors.py`文件复制到MiniCPM-V-2.6的huggingface模型仓库根目录并运行. 稍等片刻, 会生成`model-renamed-00001-of-00004.safetensors`等4个safetensors文件和一个json文件.
2. 不用管那个json文件, 将那4个safetensors文件移动到此仓库根目录下.
3. 执行`rkllm-convert.py`. 等一会, 会生成`qwen.rkllm`, 就是转换后的模型.

#### 转换视觉编码器

1. 将此仓库中的`patched_modeling_navit_siglip.py`和`patched_resampler.py`复制到MiniCPM-V-2.6的huggingface模型仓库根目录下, 重命名为`modeling_navit_siglip.py`和`resampler.py`, 替换掉原来的文件.

2. 打开`vision_export_onnx.py`, 修改其中的`MODEL_PATH`为MiniCPM-V-2.6模型文件夹的路径. 然后执行. 等一会, 会生成`vision_encoder.onnx`.
3. 执行`vision_convert_rknn.py`. 等一会, 会生成`vision_encoder.rknn`, 这就是转换后的视觉编码器.

## 已知问题

- ~~由于疑似RKLLM中存在的问题, 目前此模型无法正常推理.~~ (已修复)
- ~~由于RKLLM中存在的问题, 目前视觉编码器和LLM无法同时被加载, 必须先卸载掉视觉编码器, 再重新加载LLM. 如果要推理多次, 必须重复执行卸载和加载操作, 速度非常慢.~~ (已修复)
- 由于疑似RKLLM中存在的问题, 如果视觉编码器和LLM加载进同一个Python进程, 会导致LLM推理时报错段错误. 可以使用多进程来解决. 参考`multiprocess_inference.py`.
- 由于RKLLM的多模态输入的限制, 在整个对话中只能加载一张图片. 可以通过Embedding输入的方式来解决, 但我没有实现.
- 没有实现多轮对话.
- RKLLM的w8a8量化貌似存在不小的精度损失.
- 视觉编码器转换ONNX的代码取自 https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6 , 感谢Sophgo提供的代码. 但是这个转换方法似乎将原模型中的自适应图像分块算法删除了, 可能会导致精度下降.

## 参考

- [sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
- [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)


## English README

Run the Powerful MiniCPM-V-2.6 Visual Language Model on RK3588!

- Inference speed (RK3588): Visual encoder 3.2s (triple core parallel) + LLM prefill 1.7s (92 tokens / 53 tps) + decoding 4.03 tps
- Memory usage (RK3588, default context length): Visual encoder 1.9GB + LLM 7.8GB = 9.7GB

## Usage

1. Clone or download this repository locally. The model is large, so make sure you have enough disk space.
   
2. The RKNPU2 kernel driver version on the development board must be >=0.9.6 to run such a large model. 
   Use the following command with root privileges to check the driver version:
   ```bash
   > cat /sys/kernel/debug/rknpu/version 
   RKNPU driver: v0.9.8
   ```
   If the version is too low, please update the driver. You may need to update the kernel or refer to official documentation for help.
   
3. Install dependencies

```bash
pip install numpy<2 opencv-python 
```
You also need to manually install rknn-toolkit2-lite2.

4. Run
   
```bash
python multiprocess_inference.py
```

If the performance is not satisfactory, you can change the CPU scheduler to keep the CPU running at the highest frequency, and bind the inference program to the big core cluster (`taskset -c 4-7 python multiprocess_inference.py`).

test.jpg:
![test.jpg](./test.jpg)

>```
>Start loading language model (size: 7810.02 MB)
>
>I rkllm: rkllm-runtime version: 1.1.2, rknpu driver version: 0.9.8, platform: RK3588
>
>W rknn-toolkit-lite2 version: 2.2.0
>Start loading vision encoder model (size: 942.29 MB)
>Vision encoder loaded in 10.22 seconds
>I RKNN: [02:28:20.939] RKNN Runtime Information, librknnrt version: 2.1.0 (967d001cc8@2024-08-07T19:28:19)
>I RKNN: [02:28:20.939] RKNN Driver Information, version: 0.9.8
>I RKNN: [02:28:20.940] RKNN Model Information, version: 6, toolkit version: 2.2.0(compiler version: 2.2.0 (c195366594@2024-09-14T12:24:14)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: dynamic_shape
>W RKNN: [02:28:20.940] RKNN Model version: 2.2.0 not match with rknn runtime version: 2.1.0
>Received ready signal: vision_ready
>Language model loaded in 29.21 seconds
>Received ready signal: llm_ready
>All models loaded, starting interactive mode...
>
>Enter your input (3 empty lines to start inference, Ctrl+C to exit, for example: 
>详细描述一下{{./test.jpg}}这张图片
>What is the weather in {{./test.jpg}}?
>How many people are in {{./test.jpg}}?
>):
>
>Describe the image: {{test.jpg}} in every detail.
>
>
>
>Start vision inference...
>
>Vision encoder inference time: 3.26 seconds
>
>Time to first token: 1.72 seconds
>
>The image depicts an urban street scene with various elements that contribute to its bustling atmosphere.
>
>A person, likely male based on appearance, is walking across the crosswalk carrying a blue and white checked umbrella. He's dressed casually yet stylishly, wearing a beige jacket over what appears to be dark pants or leggings paired with patterned slip-on shoes in shades of gray, black, and yellow.
>
>The street itself features multiple lanes filled with vehicles; there are cars visible on both sides, including a prominent SUV that is parked by the roadside. The presence of these automobiles adds to the sense of movement and activity within this urban setting.
>
>In terms of infrastructure, the crosswalk has clear pedestrian markings for safety, and an adjacent railing provides support or boundary along one side of the street. Beyond the immediate foreground where pedestrians traverse, there's a sidewalk lined with lush green trees which add natural beauty to the otherwise concrete-dominated environment.
>
>The sky is visible in parts through breaks in clouds above, indicating fair weather conditions that contribute positively to outdoor activities like walking down this cityscape path.
>
>Overall, it appears as though an ordinary day unfolds within this urban setting, capturing moments of daily life and movement.
>
>(finished)
>
>--------------------------------------------------------------------------------------
> Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second       
>--------------------------------------------------------------------------------------
> Prefill       1714.78          94        18.24                    54.82                   
> Generate      58689.71         236       249.75                   4.00                    
>--------------------------------------------------------------------------------------
>```

## Model Conversion

#### Preparation

1. Install rknn-toolkit2 v2.1.0 or higher, and rkllm-toolkit v1.1.2 or higher.
2. Download this repository locally, but you don't need to download the model files ending with `.rkllm` and `.rknn`.
3. Download the MiniCPM-V-2.6 Hugging Face model repository locally. (https://huggingface.co/openbmb/MiniCPM-V-2_6)
  
#### Converting LLM

1. Copy the `rename_tensors.py` file from this repository to the root directory of the MiniCPM-V-2.6 Hugging Face model repository and run it. Wait for a moment, it will generate 4 safetensors files like `model-renamed-00001-of-00004.safetensors` and a json file.
2. Ignore the json file, move those 4 safetensors files to the root directory of this repository.
3. Execute `rkllm-convert.py`. After a while, it will generate `qwen.rkllm`, which is the converted model.

#### Converting Visual Encoder

1. Copy `patched_modeling_navit_siglip.py` and `patched_resampler.py` from this repository to the root directory of the MiniCPM-V-2.6 Hugging Face model repository, rename them to `modeling_navit_siglip.py` and `resampler.py`, replacing the original files.

2. Open `vision_export_onnx.py`, modify the `MODEL_PATH` to the path of the MiniCPM-V-2.6 model folder. Then execute it. After a while, it will generate `vision_encoder.onnx`.
3. Execute `vision_convert_rknn.py`. After a while, it will generate `vision_encoder.rknn`, which is the converted visual encoder.

## Known Issues

- ~~Due to a suspected issue in RKLLM, this model currently cannot perform inference normally.~~ (Fixed)
- ~~Due to an issue in RKLLM, the visual encoder and LLM cannot be loaded simultaneously at present. The visual encoder must be unloaded first, then the LLM reloaded. If multiple inferences are required, the unloading and loading operations must be repeated, which is very slow.~~ (Fixed)
- Due to a suspected issue in RKLLM, if the visual encoder and LLM are loaded into the same Python process, the LLM inference will segmentation fault. You can use multiprocessing to solve this problem. See `multiprocess_inference.py`.
- Due to the limitation of RKLLM's multimodal input, only one image can be loaded in the entire conversation. This can be solved by using embedding input, but I haven't implemented it yet.
- I don't implement multi-turn chat.
- There is a significant precision loss in RKLLM's w8a8 quantization.
- The code for converting the visual encoder to ONNX is taken from https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6, thanks to Sophgo for providing the code. However, this conversion method seems to have removed the adaptive image partitioning algorithm from the original model, which may lead to a decrease in accuracy.

## References

- [sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
- [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)