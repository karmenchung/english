import os
import sys

# ======================================================================
# 第一步：强制锁定环境变量 (必须在最顶层)
# ======================================================================
os.environ['HF_HOME'] = r"C:\Users\Administrator\.cache\huggingface"
os.environ['HF_HUB_CACHE'] = r"C:\Users\Administrator\.cache\huggingface\hub"
os.environ['HF_HUB_OFFLINE'] = "1"

# ======================================================================
# 第二步：路径注入
# ======================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
index_tts_root = os.path.join(current_dir, "TTS", "index-tts")
if index_tts_root not in sys.path:
    sys.path.append(index_tts_root)
    sys.path.append(os.path.join(index_tts_root, "indextts"))

import torch
import argparse

# ======================================================================
# 第三步：劫持 Transformers (防止下载)
# ======================================================================
try:
    from transformers import PreTrainedModel
    _orig_from_pretrained = PreTrainedModel.from_pretrained
    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs['local_files_only'] = True
        return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    PreTrainedModel.from_pretrained = patched_from_pretrained
except:
    pass

# ======================================================================
# 第四步：模型封装
# ======================================================================
_model_instance = None

def get_model(model_dir):
    global _model_instance
    if _model_instance is None:
        model_dir = os.path.abspath(model_dir)
        cfg_path = os.path.join(model_dir, "config.yaml")
        from indextts.infer_v2 import IndexTTS2
        print(f">> Loading IndexTTS2 model from {model_dir}...")
        _model_instance = IndexTTS2(
            model_dir=model_dir,
            cfg_path=cfg_path,
            use_fp16=torch.cuda.is_available()
        )
    return _model_instance

def run_tts_with_model(tts, prompt_wav, text, output_path):
    prompt_wav = os.path.abspath(prompt_wav)
    output_path = os.path.abspath(output_path)
    
    kwargs = {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 50,
        "temperature": 1.0,
        "length_penalty": 1.0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_mel_tokens": 1024,
    }

    try:
        result_path = tts.infer(
            spk_audio_prompt=prompt_wav,
            text=text,
            output_path=output_path,
            verbose=True,
            **kwargs
        )
        return result_path
    except Exception as e:
        print(f"Inference failed: {e}")
        return None

def run_tts(model_dir, prompt_wav, text, output_path):
    """旧接口兼容"""
    tts = get_model(model_dir)
    return run_tts_with_model(tts, prompt_wav, text, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--text", type=str, default="你好，这是一个测试。")
    parser.add_argument("--output", type=str, default="toy_output.wav")
    args = parser.parse_args()
    run_tts(args.model_dir, args.prompt_wav, args.text, args.output)
