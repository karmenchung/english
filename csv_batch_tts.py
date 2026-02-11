import os
import sys
import csv
import argparse
from pydub import AudioSegment
from simple_tts import get_model, run_tts_with_model

def format_srt_time(ms):
    """将毫秒转换为 SRT 时间格式 HH:MM:SS,mmm"""
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)

def main():
    parser = argparse.ArgumentParser(description="CSV TTS with Merged SRT Subtitles")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--en_prompt", type=str, required=True, help="Reference audio for English")
    parser.add_argument("--zh_prompt", type=str, required=True, help="Reference audio for Chinese")
    parser.add_argument("--model_dir", type=str, required=True, help="Model checkpoints directory")
    parser.add_argument("--output", type=str, default="study_loop_merged_srt.wav", help="Final merged wav file")
    parser.add_argument("--temp_dir", type=str, default="temp_tts", help="Directory for temporary segments")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process")
    parser.add_argument("--ding", type=str, default="ding.mp3", help="Path to notification sound")

    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # 1. 初始化
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

    # 2. 准备拼接与时间轴
    combined_audio = AudioSegment.empty()
    srt_entries = []
    current_time_ms = 0
    srt_counter = 1

    silence_short_ms = 800
    silence_long_ms = 1500
    silence_short = AudioSegment.silent(duration=silence_short_ms)
    silence_long = AudioSegment.silent(duration=silence_long_ms)

    print(">> Reading CSV and calculating merged timestamps...")
    
    try:
        with open(args.csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if args.limit:
                rows = rows[:args.limit]
            
            for i, row in enumerate(rows):
                en_text = row.get('english', '').strip()
                zh_text = row.get('chinese', '').strip()
                if not en_text and not zh_text:
                    continue
                
                idx = i + 1
                print("\n--- [Row {}/{}] ---".format(idx, len(rows)))
                
                en_wav = os.path.join(args.temp_dir, "row_{}_en.wav".format(idx))
                zh_wav = os.path.join(args.temp_dir, "row_{}_zh.wav".format(idx))
                en_audio, zh_audio = None, None

                # 推理音频
                safe_en = en_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                if run_tts_with_model(tts_model, args.en_prompt, safe_en, en_wav):
                    en_audio = AudioSegment.from_wav(en_wav)
                
                safe_zh = zh_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                if run_tts_with_model(tts_model, args.zh_prompt, safe_zh, zh_wav):
                    zh_audio = AudioSegment.from_wav(zh_wav)

                # --- 拼接与合并字幕逻辑 ---
                
                # 1. 英文阶段：合并前 3 遍朗读
                if en_audio:
                    en_duration = len(en_audio)
                    start_ms = current_time_ms
                    
                    # 拼接音频：英(1) + 停 + 英(2) + 停 + 英(3)
                    for _ in range(3):
                        combined_audio += en_audio + silence_short
                        current_time_ms += en_duration + silence_short_ms
                    
                    # 生成合并后的英文字幕（包含最后一遍英文后的静音时间）
                    end_ms = current_time_ms
                    srt_entries.append("{}\n{} --> {}\n{}\n".format(srt_counter, format_srt_time(start_ms), format_srt_time(end_ms), en_text))
                    srt_counter += 1

                # 2. 双语阶段：合并 中文(1) + 英文(1)
                if zh_audio and en_audio:
                    zh_duration = len(zh_audio)
                    en_duration = len(en_audio)
                    start_ms = current_time_ms
                    
                    # 拼接音频：中(1) + 停 + 英(1) + 停
                    combined_audio += zh_audio + silence_short
                    current_time_ms += zh_duration + silence_short_ms
                    
                    combined_audio += en_audio + silence_short
                    current_time_ms += en_duration + silence_short_ms
                    
                    # 生成合并后的双语字幕
                    end_ms = current_time_ms
                    combined_text = "{}\n{}".format(en_text, zh_text)
                    srt_entries.append("{}\n{} --> {}\n{}\n".format(srt_counter, format_srt_time(start_ms), format_srt_time(end_ms), combined_text))
                    srt_counter += 1
                
                # 3. 循环结束：Ding 提示音
                if ding_sound:
                    combined_audio += ding_sound
                    current_time_ms += len(ding_sound)
                
                combined_audio += silence_long
                current_time_ms += silence_long_ms

        # 3. 导出
        if len(combined_audio) > 0:
            print("\n>> Exporting audio: {}".format(args.output))
            combined_audio.export(args.output, format="wav")
            
            srt_path = os.path.splitext(args.output)[0] + ".srt"
            print(">> Exporting merged subtitles: {}".format(srt_path))
            with open(srt_path, "w", encoding="utf-8") as f_srt:
                f_srt.write("\n".join(srt_entries))
            
            print(">> All done! Subtitles are now continuous within blocks.")
        else:
            print(">> No audio generated.")

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()