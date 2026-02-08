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