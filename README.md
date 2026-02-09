## Install
```
uv pip install -e .\TTS\index-tts
```

## Test
```
uv run python .\TTS\index-tts\webui.py --model_dir .\TTS\index-tts\checkpoints\
```

```
uv run python .\simple_tts.py  --prompt_wav ".\TTS\voice_01.wav" --text "你好，我测试一下哈。" --output out.wav --model_dir .\TTS\index-tts\checkpoints
```

``` 
uv run python batch_tts.py --prompt_wav ".\TTS\voice_01.wav" --model_dir ".\TTS\index-tts\checkpoints" --target_len 100 --text "The chocolate cake does not contain nuts, but it’s made in a kitchen where nuts are handled." 
```

```
uv run python csv_batch_tts.py --csv "dialogs.csv" --prompt_wav ".\TTS\voice_01.wav" --model_dir ".\TTS\index-tts\checkpoints" --output "dialog_study.wav"
```

```
uv run python csv_batch_tts.py --csv "dialogs.csv" --prompt_wav ".\TTS\voice_01.wav"--model_dir ".\TTS\index-tts\checkpoints" --limit 2 --output "test_loop.wav"
```
最终版本
```
uv run python csv_batch_tts.py --csv "dialogs.csv" --en_prompt ".\TTS\voice_01.wav" --zh_prompt  ".\TTS\voice_02.wav" --model_dir ".\TTS\index-tts\checkpoints" --limit 2 --output "dual_voice_test.wav"
```