import os
import sys

# ======================================================================
# 第一步：在任何其他 import 之前，强制锁定环境变量
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

# ======================================================================
# 第三步：导入其他库
# ======================================================================
import torch
import argparse

# 强行给 transformers 的 from_pretrained 加上 local_files_only=True
# 这样即便代码里没写，它也会被迫只读取本地文件
try:
    import transformers
    from transformers import PreTrainedModel, PretrainedConfig, FeatureExtractionMixin
    
    # 记录原始方法
    _orig_from_pretrained = PreTrainedModel.from_pretrained
    
    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs['local_files_only'] = True
        return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    
    # 替换
    PreTrainedModel.from_pretrained = patched_from_pretrained
    print(">> Transformers patched: local_files_only=True is now FORCED.")
except Exception as e:
    print(f">> Patching failed (non-critical): {e}")

# ======================================================================
# 业务逻辑
# ======================================================================
def run_tts(model_dir, prompt_wav, text, output_path):
    model_dir = os.path.abspath(model_dir)
    prompt_wav = os.path.abspath(prompt_wav)
    output_path = os.path.abspath(output_path)
    cfg_path = os.path.join(model_dir, "config.yaml")

    try:
        from indextts.infer_v2 import IndexTTS2
        print("Successfully imported IndexTTS2")
    except ImportError as e:
        print(f"Error importing IndexTTS2: {e}")
        sys.exit(1)

    print(f"Initializing IndexTTS2...")
    
    # 此时初始化，IndexTTS2 内部调用 from_pretrained 时会被我们拦截
    tts = IndexTTS2(
        model_dir=model_dir,
        cfg_path=cfg_path,
        use_fp16=torch.cuda.is_available()
    )

    print(f"Generating audio for text: '{text}'")
    
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
        print(f"TTS result saved to: {result_path}")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--text", type=str, default="你好，这是一个测试。")
    parser.add_argument("--output", type=str, default="toy_output.wav")

    args = parser.parse_args()
    run_tts(args.model_dir, args.prompt_wav, args.text, args.output)