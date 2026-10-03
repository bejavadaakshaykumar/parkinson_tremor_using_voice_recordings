# Floating-point WAV upload fix

The screenshot error was:

```
unknown extended format: 00000003-0000-0010-8000-00aa00389b71
```

The previous implementation called Python's `wave.open()` before acoustic analysis. That reader supports PCM WAV but rejects the IEEE-float subtype of WAVE_FORMAT_EXTENSIBLE used by the uploaded recording. This is a decoder compatibility problem; it does not mean the file is not a WAV.

The revised single-file app uses SoundFile to validate and decode WAV, WAVEX and RF64 in memory, then passes the decoded floating-point samples to Praat. Integer PCM and floating-point WAV recordings are supported. No temporary audio file or manual conversion is required. No audio is stretched, padded or normalized.

The screenshot also shows approximately two seconds of audio. The old three-second minimum would have caused a second validation error. The revised app accepts 1–30 seconds and displays a warning for clips shorter than three seconds. A steady 5–10 second vowel is preferable. The existing 20 MB, mono/stereo and 16–96 kHz limits remain, and empty, silent, non-finite and invalid files still produce useful errors. When one of several files fails, the error identifies its filename.

## Update your hosted app

1. Extract the fixed package.
2. Replace your repository's `app.py` and `requirements.txt` with the versions in `voice-analytics-rewrite/`. Leave your training CSV in place beside `app.py`.
3. Reboot the Streamlit app so it installs the added `soundfile==0.14.0` dependency.
4. Upload the WAV again and click **Analyze & Generate Report**.

Updating `app.py` alone is not sufficient if SoundFile is not installed. No changes have been pushed to your GitHub account or public deployment by this task.

WAV analysis without an acoustic CSV shows measured descriptors. Disease-class prediction, SHAP and the model DOCX report still require a matched, complete acoustic feature row; this patch does not fabricate the missing features.

The exact reported exception was reproduced with a generated two-second WAVEX FLOAT file, then verified to be fixed. The actual user recording was not attached, so the regression check validates the matching format rather than that specific file.

References: [Python wave format support](https://docs.python.org/3.12/library/wave.html), [SoundFile decoding and installation](https://python-soundfile.readthedocs.io/en/latest/).
