import os
import sys
import wave

# Keep cache/offline behavior aligned with existing script
os.environ['HF_HOME'] = r"C:\Users\Administrator\.cache\huggingface"
os.environ['HF_HUB_CACHE'] = r"C:\Users\Administrator\.cache\huggingface\hub"
os.environ['HF_HUB_OFFLINE'] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
index_tts_root = os.path.join(current_dir, "TTS", "index-tts")
if index_tts_root not in sys.path:
    sys.path.append(index_tts_root)
    sys.path.append(os.path.join(index_tts_root, "indextts"))

import argparse
import numpy as np
import torch

try:
    from transformers import PreTrainedModel

    _orig_from_pretrained = PreTrainedModel.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs['local_files_only'] = True
        kwargs.setdefault('low_cpu_mem_usage', True)
        return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = patched_from_pretrained
except Exception:
    pass

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
            use_fp16=torch.cuda.is_available(),
        )
    return _model_instance


def _build_infer_kwargs(stable_mode=True):
    if stable_mode:
        return {
            "do_sample": False,
            "top_p": 1.0,
            "top_k": 0,
            "temperature": 1.0,
            "length_penalty": 1.0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "max_mel_tokens": 1024,
        }

    return {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 50,
        "temperature": 1.0,
        "length_penalty": 1.0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_mel_tokens": 1024,
    }


def _load_mono_float(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        frames = wf.getnframes()
        data = wf.readframes(frames)

    if sample_width != 2:
        return None, None

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, frame_rate


def _has_obvious_glitch(wav_path):
    samples, sr = _load_mono_float(wav_path)
    if samples is None or sr is None or len(samples) == 0:
        return True

    if len(samples) < int(0.12 * sr):
        return True

    peak = float(np.max(np.abs(samples)))
    if peak < 1e-4:
        return True

    clip_ratio = float(np.mean(np.abs(samples) >= 0.999))
    if clip_ratio > 0.02:
        return True

    d = np.abs(np.diff(samples))
    if len(d) < 100:
        return False

    p999 = float(np.percentile(d, 99.9))
    med = float(np.median(d)) + 1e-9
    if p999 > 0.45 and (p999 / med) > 120.0:
        return True

    return False


def run_tts_with_model(
    tts,
    prompt_wav,
    text,
    output_path,
    *,
    stable_mode=True,
    max_retries=2,
    seed=None,
    quality_check=True,
):
    prompt_wav = os.path.abspath(prompt_wav)
    output_path = os.path.abspath(output_path)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    kwargs = _build_infer_kwargs(stable_mode=stable_mode)
    attempts = max(1, int(max_retries) + 1)

    for attempt in range(1, attempts + 1):
        try:
            result_path = tts.infer(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=output_path,
                verbose=True,
                **kwargs,
            )

            if not result_path or not os.path.exists(result_path):
                raise RuntimeError("TTS returned empty output path")

            if quality_check and _has_obvious_glitch(result_path):
                print(f"Inference quality check failed on attempt {attempt}/{attempts}, retrying...")
                continue

            return result_path
        except Exception as e:
            print(f"Inference failed on attempt {attempt}/{attempts}: {e}")

    return None


def run_tts(model_dir, prompt_wav, text, output_path):
    tts = get_model(model_dir)
    return run_tts_with_model(tts, prompt_wav, text, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--text", type=str, default="Hello, this is a test.")
    parser.add_argument("--output", type=str, default="toy_output.wav")
    parser.add_argument("--non_stable", action="store_true", help="Use stochastic decoding")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_quality_check", action="store_true")
    args = parser.parse_args()

    tts_model = get_model(args.model_dir)
    run_tts_with_model(
        tts_model,
        args.prompt_wav,
        args.text,
        args.output,
        stable_mode=not args.non_stable,
        max_retries=args.max_retries,
        seed=args.seed,
        quality_check=not args.no_quality_check,
    )
