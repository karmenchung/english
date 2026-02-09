import os
import sys
import argparse
import pysbd
from simple_tts import get_model, run_tts_with_model

def split_long_text(text, target_len=50):
    """
    智能切分长文本。
    """
    seg_en = pysbd.Segmenter(language="en", clean=False)
    sentences = seg_en.segment(text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        if len(current_chunk) + len(sent) > target_len and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent + " "
        else:
            current_chunk += sent + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Efficient Batch TTS")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="batch_out")
    parser.add_argument("--target_len", type=int, default=150)

    args = parser.parse_args()

    print(">> Initializing system...")
    try:
        tts_model = get_model(args.model_dir)
    except Exception as e:
        print("CRITICAL: Failed to load model:", e)
        sys.exit(1)

    print(">> Splitting text...")
    segments = split_long_text(args.text, target_len=args.target_len)
    total = len(segments)
    print(">> Total segments:", total)

    for i, seg_text in enumerate(segments):
        idx = i + 1
        output_name = "{}_{:03d}.wav".format(args.output_prefix, idx)
        print("\n--- [Segment {}/{}] ---".format(idx, total))
        
        # 预先处理特殊字符，避免 f-string 渲染出错
        safe_text = seg_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        
        preview = safe_text[:60] + "..." if len(safe_text) > 60 else safe_text
        print("Synthesizing:", preview)
        
        success_path = run_tts_with_model(
            tts=tts_model,
            prompt_wav=args.prompt_wav,
            text=safe_text,
            output_path=output_name
        )
        
        if success_path:
            print("Saved:", output_name)
        else:
            print("FAILED:", output_name)

if __name__ == "__main__":
    main()
