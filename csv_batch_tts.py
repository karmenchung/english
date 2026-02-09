import os
import sys
import csv
import argparse
from pydub import AudioSegment
from simple_tts import get_model, run_tts_with_model

def main():
    parser = argparse.ArgumentParser(description="CSV Dialog TTS with Merged Output")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--prompt_wav", type=str, required=True, help="Reference audio")
    parser.add_argument("--model_dir", type=str, required=True, help="Model checkpoints directory")
    parser.add_argument("--output", type=str, default="merged_dialog.wav", help="Final merged wav file")
    parser.add_argument("--temp_dir", type=str, default="temp_tts", help="Directory for temporary wav segments")

    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # 1. 初始化模型 (单例)
    print(">> Initializing TTS system...")
    try:
        tts_model = get_model(args.model_dir)
    except Exception as e:
        print("CRITICAL: Failed to load model:", e)
        sys.exit(1)

    # 2. 读取 CSV 并生成音频片段
    combined_audio = AudioSegment.empty()
    
    # 增加一点静音间隔 (500ms)
    silence = AudioSegment.silent(duration=500)
    # 长一点的静音间隔 (1000ms) 用于句子之间
    sentence_gap = AudioSegment.silent(duration=1000)

    print(">> Reading CSV: {}".format(args.csv))
    
    try:
        # 使用 utf-8-sig 以便更好地处理带 BOM 的 Excel CSV
        with open(args.csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            rows = list(reader)
            total_rows = len(rows)
            print(">> Total rows found:", total_rows)
            
            for i, row in enumerate(rows):
                english_text = row.get('english', '').strip()
                chinese_text = row.get('chinese', '').strip()
                
                if not english_text and not chinese_text:
                    continue
                
                idx = i + 1
                print("\n--- [Row {}/{}] ---".format(idx, total_rows))
                
                # 依次合成英文和中文
                parts = [
                    ("en", english_text),
                    ("zh", chinese_text)
                ]
                
                for lang, text in parts:
                    if not text:
                        continue
                        
                    # 构造临时文件名
                    temp_wav = os.path.join(args.temp_dir, "temp_{}_{}.wav".format(idx, lang))
                    
                    # 清理并预览文本
                    safe_text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                    preview = safe_text[:50] + "..." if len(safe_text) > 50 else safe_text
                    
                    print("Synthesizing ({}): {}".format(lang, preview))
                    
                    success_path = run_tts_with_model(
                        tts=tts_model,
                        prompt_wav=args.prompt_wav,
                        text=safe_text,
                        output_path=temp_wav
                    )
                    
                    if success_path and os.path.exists(temp_wav):
                        # 加载生成的音频并拼接到主音频
                        segment = AudioSegment.from_wav(temp_wav)
                        combined_audio += segment + silence
                    else:
                        print("FAILED to synthesize segment:", lang)

                # 每一行结束后加一个长一点的停顿
                combined_audio += sentence_gap

        # 3. 导出合并后的音频
        if len(combined_audio) > 0:
            print("\n>> Exporting final merged audio to: {}".format(args.output))
            combined_audio.export(args.output, format="wav")
            print(">> Done!")
        else:
            print(">> No audio was generated.")

    except Exception as e:
        print("Error during CSV processing:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()