注意: 由于疑似RKLLM那边的问题, 目前此模型的推理输出结果不正常 (https://github.com/airockchip/rknn-llm/issues/101), 未来修复后这个repo会更新.

NOTE: Due to suspected issues in RKLLM(https://github.com/airockchip/rknn-llm/issues/101) , the model cannot be used normally for inference at the moment. Once fixed, this repo will be updated.

# MiniCPM-V-2_6-rkllm

## (English README see below)

在RK3588上运行强大的MiniCPM-V-2.6 视觉大模型!

- 推理速度(RK3588): 视觉编码器 4.8s(单核) + LLM 填充 2.2s (92 tokens / 42.5 tps) + 解码 3.25 tps
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
python run_rknn.py
```

你可以修改`run_rknn.py`中的内容来测试不同的输入.

## 模型转换

#### 准备工作

1. 安装rknn-toolkit2 v2.1.0或更高版本, 以及rkllm-toolkit v1.1.0或更高版本.
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

- 由于疑似RKLLM中存在的问题, 目前此模型无法正常推理. 
- 由于RKLLM中存在的问题, 目前视觉编码器和LLM无法同时被加载, 必须先卸载掉视觉编码器, 再重新加载LLM. 如果要推理多次, 必须重复执行卸载和加载操作, 速度非常慢.
- 视觉编码器转换ONNX的代码取自 https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6 , 感谢Sophgo提供的代码. 但是这个转换方法似乎将原模型中的自适应图像分块算法删除了, 可能会导致精度下降.

## 参考

[sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
[openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
[Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)


## English README

Run the Powerful MiniCPM-V-2.6 Visual Language Model on RK3588!

- Inference speed (RK3588): Visual encoder 4.8s (single core) + LLM filling 2.2s (92 tokens / 42.5 tps) + decoding 3.25 tps
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
python run_rknn.py
```

You can modify the content in `run_rknn.py` to test different inputs.

## Model Conversion

#### Preparation

1. Install rknn-toolkit2 v2.1.0 or higher, and rkllm-toolkit v1.1.0 or higher.
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

- Due to a suspected issue in RKLLM, this model currently cannot perform inference normally.
- Due to an issue in RKLLM, the visual encoder and LLM cannot be loaded simultaneously at present. The visual encoder must be unloaded first, then the LLM reloaded. If multiple inferences are required, the unloading and loading operations must be repeated, which is very slow.
- The code for converting the visual encoder to ONNX is taken from https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6, thanks to Sophgo for providing the code. However, this conversion method seems to have removed the adaptive image partitioning algorithm from the original model, which may lead to a decrease in accuracy.

## References

[sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
[openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
[Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)