import json
import os
import shutil
import mmap
import re

def rename_tensors():
    # 读取JSON文件
    with open('model.safetensors.index.json', 'r') as f:
        data = json.load(f)

    # 获取所有唯一的safetensors文件名
    safetensor_files = set(data['weight_map'].values())

    # 复制并重命名safetensors文件
    for file in safetensor_files:
        new_file = file.replace('model-', 'model-renamed-')
        shutil.copy(file, new_file)

        # 在新文件的前1MB范围内替换字符串
        with open(new_file, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 1024*1024)  # 映射前1MB
            content = mm.read()
            # 使用字节字符串进行替换
            content = content.replace(b'"llm.', b'    "')
            mm.seek(0)
            mm.write(content)
            mm.close()

    # 更新JSON数据
    new_weight_map = {}
    for key, value in data['weight_map'].items():
        new_key = re.sub(r'^llm\.', '', key)
        new_value = value.replace('model-', 'model-renamed-')
        new_weight_map[new_key] = new_value

    data['weight_map'] = new_weight_map

    # 写入新的JSON文件
    with open('model-renamed.safetensors.index.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("处理完成。新的JSON文件已生成：model-renamed.safetensors.index.json")

if __name__ == "__main__":
    rename_tensors()