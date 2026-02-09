import os
import sys
import csv
import argparse
from pydub import AudioSegment
from simple_tts import get_model, run_tts_with_model

def main():
    parser = argparse.ArgumentParser(description="Advanced CSV Dialog TTS with Dual-Voice Support")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--en_prompt", type=str, required=True, help="Reference audio for English")
    parser.add_argument("--zh_prompt", type=str, required=True, help="Reference audio for Chinese")
    parser.add_argument("--model_dir", type=str, required=True, help="Model checkpoints directory")
    parser.add_argument("--output", type=str, default="study_loop_dual.wav", help="Final merged wav file")
    parser.add_argument("--temp_dir", type=str, default="temp_tts", help="Directory for temporary segments")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process")
    parser.add_argument("--ding", type=str, default="ding.mp3", help="Path to notification sound")

    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # 1. 初始化模型与加载提示音
    print(">> Initializing TTS system...")
    try:
        tts_model = get_model(args.model_dir)
    except Exception as e:
        print("CRITICAL: Failed to load model:", e)
        sys.exit(1)

    ding_sound = None
    if os.path.exists(args.ding):
        try:
            ding_sound = AudioSegment.from_file(args.ding)
        except Exception as e:
            print("Warning: Could not load ding sound:", e)

    # 2. 准备拼接
    combined_audio = AudioSegment.empty()
    silence_short = AudioSegment.silent(duration=600)
    silence_long = AudioSegment.silent(duration=1200)

    print(">> Reading CSV: {}".format(args.csv))
    
    try:
        with open(args.csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if args.limit:
                rows = rows[:args.limit]
            
            total_rows = len(rows)
            
            for i, row in enumerate(rows):
                en_text = row.get('english', '').strip()
                zh_text = row.get('chinese', '').strip()
                
                if not en_text and not zh_text:
                    continue
                
                idx = i + 1
                print("\n--- [Row {}/{}] ---".format(idx, total_rows))
                
                en_wav = os.path.join(args.temp_dir, "row_{}_en.wav".format(idx))
                zh_wav = os.path.join(args.temp_dir, "row_{}_zh.wav".format(idx))
                
                en_audio = None
                zh_audio = None

                # 合成英文 - 使用 en_prompt
                if en_text:
                    safe_en = en_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                    print("Synthesizing English (Voice: {})...".format(os.path.basename(args.en_prompt)))
                    if run_tts_with_model(tts_model, args.en_prompt, safe_en, en_wav):
                        en_audio = AudioSegment.from_wav(en_wav)
                
                # 合成中文 - 使用 zh_prompt
                if zh_text:
                    safe_zh = zh_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                    print("Synthesizing Chinese (Voice: {})...".format(os.path.basename(args.zh_prompt)))
                    if run_tts_with_model(tts_model, args.zh_prompt, safe_zh, zh_wav):
                        zh_audio = AudioSegment.from_wav(zh_wav)

                # --- 拼接逻辑 ---
                if en_audio:
                    for _ in range(3):
                        combined_audio += en_audio + silence_short
                
                if zh_audio:
                    combined_audio += zh_audio + silence_short
                
                if en_audio:
                    combined_audio += en_audio + silence_short
                
                if ding_sound:
                    combined_audio += ding_sound
                
                combined_audio += silence_long

        # 3. 导出
        if len(combined_audio) > 0:
            print("\n>> Exporting final audio to: {}".format(args.output))
            combined_audio.export(args.output, format="wav")
            print(">> Done!")
        else:
            print(">> No audio was generated.")

    except Exception as e:
        print("Error during process:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()