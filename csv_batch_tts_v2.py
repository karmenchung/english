import os
import sys
import csv
import argparse
from pydub import AudioSegment
from simple_tts_v2 import get_model, run_tts_with_model


def format_srt_time(ms):
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def normalize_audio(seg, frame_rate, channels, sample_width):
    if seg.frame_rate != frame_rate:
        seg = seg.set_frame_rate(frame_rate)
    if seg.channels != channels:
        seg = seg.set_channels(channels)
    if seg.sample_width != sample_width:
        seg = seg.set_sample_width(sample_width)
    return seg


def make_silence(ms, frame_rate, channels, sample_width):
    sil = AudioSegment.silent(duration=ms, frame_rate=frame_rate)
    sil = sil.set_channels(channels)
    sil = sil.set_sample_width(sample_width)
    return sil


def clean_quotes(text):
    return text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def load_csv_rows_with_fallback(csv_path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            with open(csv_path, mode="r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(">> CSV encoding detected: {}".format(enc))
            return rows
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Failed to read CSV: {}".format(csv_path))


def maybe_edge_fade(seg, edge_fade_ms):
    if edge_fade_ms <= 0:
        return seg
    if len(seg) <= edge_fade_ms * 2:
        return seg
    return seg.fade_in(edge_fade_ms).fade_out(edge_fade_ms)


def main():
    parser = argparse.ArgumentParser(description="CSV TTS v2 with smoother transitions and robust output")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--en_prompt", type=str, required=True, help="Reference audio for English")
    parser.add_argument("--zh_prompt", type=str, required=True, help="Reference audio for Chinese")
    parser.add_argument("--model_dir", type=str, required=True, help="Model checkpoints directory")
    parser.add_argument("--output", type=str, default="study_loop_merged_srt_v2.wav", help="Final merged wav file")
    parser.add_argument("--temp_dir", type=str, default="temp_tts", help="Directory for temporary segments")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to process")
    parser.add_argument("--ding", type=str, default="ding.mp3", help="Path to notification sound")

    parser.add_argument("--target_sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--target_channels", type=int, default=1, help="Target channels")
    parser.add_argument("--target_sample_width", type=int, default=2, help="Target sample width in bytes")

    parser.add_argument("--silence_short_ms", type=int, default=800)
    parser.add_argument("--silence_long_ms", type=int, default=1500)
    parser.add_argument("--edge_fade_ms", type=int, default=8, help="Fade in/out for each TTS segment")

    parser.add_argument("--no_ding", action="store_true", help="Disable ding sound")
    parser.add_argument("--ding_gain_db", type=float, default=-6.0, help="Ding gain adjustment in dB")
    parser.add_argument("--ding_fade_in_ms", type=int, default=100)
    parser.add_argument("--ding_fade_out_ms", type=int, default=120)

    parser.add_argument("--non_stable", action="store_true", help="Use stochastic decoding")
    parser.add_argument("--max_retries", type=int, default=2, help="Retry count per sentence")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_quality_check", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    print(">> Initializing TTS system...")
    try:
        tts_model = get_model(args.model_dir)
    except Exception as e:
        print("CRITICAL: Failed to load model:", e)
        sys.exit(1)

    target_sr = int(args.target_sr)
    target_channels = int(args.target_channels)
    target_sw = int(args.target_sample_width)

    ding_sound = None
    if (not args.no_ding) and os.path.exists(args.ding):
        try:
            ding_sound = AudioSegment.from_file(args.ding)
            ding_sound = normalize_audio(ding_sound, target_sr, target_channels, target_sw)
            if args.ding_gain_db:
                ding_sound = ding_sound + float(args.ding_gain_db)
            if args.ding_fade_in_ms > 0:
                ding_sound = ding_sound.fade_in(args.ding_fade_in_ms)
            if args.ding_fade_out_ms > 0:
                ding_sound = ding_sound.fade_out(args.ding_fade_out_ms)
        except Exception as e:
            print("Warning: Could not load or process ding sound:", e)
            ding_sound = None

    combined_audio = AudioSegment.silent(duration=0, frame_rate=target_sr)
    combined_audio = combined_audio.set_channels(target_channels).set_sample_width(target_sw)

    srt_entries = []
    current_time_ms = 0
    srt_counter = 1

    silence_short = make_silence(args.silence_short_ms, target_sr, target_channels, target_sw)
    silence_long = make_silence(args.silence_long_ms, target_sr, target_channels, target_sw)

    print(">> Reading CSV and generating audio...")

    try:
        rows = load_csv_rows_with_fallback(args.csv)
        if args.limit:
            rows = rows[:args.limit]

        for i, row in enumerate(rows):
            en_text = row.get("english", "").strip()
            zh_text = row.get("chinese", "").strip()
            if not en_text and not zh_text:
                continue

            idx = i + 1
            print("\n--- [Row {}/{}] ---".format(idx, len(rows)))

            en_wav = os.path.join(args.temp_dir, "row_{}_en.wav".format(idx))
            zh_wav = os.path.join(args.temp_dir, "row_{}_zh.wav".format(idx))
            en_audio, zh_audio = None, None

            safe_en = clean_quotes(en_text)
            if run_tts_with_model(
                tts_model,
                args.en_prompt,
                safe_en,
                en_wav,
                stable_mode=not args.non_stable,
                max_retries=args.max_retries,
                seed=args.seed,
                quality_check=not args.no_quality_check,
            ):
                en_audio = AudioSegment.from_wav(en_wav)
                en_audio = normalize_audio(en_audio, target_sr, target_channels, target_sw)
                en_audio = maybe_edge_fade(en_audio, args.edge_fade_ms)

            safe_zh = clean_quotes(zh_text)
            if run_tts_with_model(
                tts_model,
                args.zh_prompt,
                safe_zh,
                zh_wav,
                stable_mode=not args.non_stable,
                max_retries=args.max_retries,
                seed=args.seed,
                quality_check=not args.no_quality_check,
            ):
                zh_audio = AudioSegment.from_wav(zh_wav)
                zh_audio = normalize_audio(zh_audio, target_sr, target_channels, target_sw)
                zh_audio = maybe_edge_fade(zh_audio, args.edge_fade_ms)

            if en_audio:
                en_duration = len(en_audio)
                start_ms = current_time_ms

                for _ in range(3):
                    combined_audio += en_audio + silence_short
                    current_time_ms += en_duration + args.silence_short_ms

                end_ms = current_time_ms
                srt_entries.append(
                    "{}\n{} --> {}\n{}\n".format(
                        srt_counter,
                        format_srt_time(start_ms),
                        format_srt_time(end_ms),
                        en_text,
                    )
                )
                srt_counter += 1

            if zh_audio and en_audio:
                zh_duration = len(zh_audio)
                en_duration = len(en_audio)
                start_ms = current_time_ms

                combined_audio += zh_audio + silence_short
                current_time_ms += zh_duration + args.silence_short_ms

                combined_audio += en_audio + silence_short
                current_time_ms += en_duration + args.silence_short_ms

                end_ms = current_time_ms
                combined_text = "{}\n{}".format(en_text, zh_text)
                srt_entries.append(
                    "{}\n{} --> {}\n{}\n".format(
                        srt_counter,
                        format_srt_time(start_ms),
                        format_srt_time(end_ms),
                        combined_text,
                    )
                )
                srt_counter += 1

            if ding_sound:
                combined_audio += ding_sound
                current_time_ms += len(ding_sound)

            combined_audio += silence_long
            current_time_ms += args.silence_long_ms

        if len(combined_audio) > 0:
            print("\n>> Exporting audio: {}".format(args.output))
            combined_audio.export(args.output, format="wav")

            srt_path = os.path.splitext(args.output)[0] + ".srt"
            print(">> Exporting subtitles: {}".format(srt_path))
            with open(srt_path, "w", encoding="utf-8") as f_srt:
                f_srt.write("\n".join(srt_entries))

            print(">> All done (v2).")
        else:
            print(">> No audio generated.")

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
