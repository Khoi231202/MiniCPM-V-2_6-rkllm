import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../MiniCPM-V-2_6/"
DEVICE_MAP = "cpu"

origin_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, attn_implementation='eager', device_map=DEVICE_MAP).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

for param in origin_model.parameters():
    param.requires_grad = False

class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vpm = origin_model.vpm
        self.resampler = origin_model.resampler
        self.tgt_sizes = torch.Tensor([[32, 32]]).type(torch.int32)

    def forward(self, pixel_values):
        vit_embeds = self.vpm(pixel_values).last_hidden_state
        vit_embeds = self.resampler(vit_embeds, self.tgt_sizes)
        return vit_embeds


def convert_vision_transformer():
    model = VisionTransformer()
    IMAGE_SIZE = 448
    pixel_values = torch.randn(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    
    # test first
    vit_embeds = model(pixel_values)
    print(vit_embeds.shape)  #1x64x3584
    if vit_embeds.shape != (1, 64, 3584):
        raise ValueError("vit_embeds shape is not correct, something is wrong")


    torch.onnx.export(model, pixel_values,
                      f'vision_transformer.onnx',
                      verbose=False,
                      input_names=['pixel_values'],
                      output_names=['vit_embeds'],
                      dynamic_axes={'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
                                    'vit_embeds': {0: 'batch_size', 1: 'seq_len'}},
                      do_constant_folding=True,
                      opset_version=17)

if __name__ == "__main__":
    convert_vision_transformer()
