# LRC Audio Aligner

Force-aligns the English text in each LRC line to the audio with the torchaudio Wav2Vec2 CTC acoustic model. Between adjacent captions, it detects low-energy regions in 10 ms RMS waveform frames and places the next caption at the midpoint of the longest detected silence. If no silence is detected, it uses the CTC-aligned first-word onset. The default behavior is a preview; add `--write` to update the LRC file and remove `[ar]`/`[by]` attribution tags.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r tools/lrc-align/requirements.txt
python3 tools/lrc-align/align_lrc.py "uk/NCE2/86.Out of Control.mp3" "uk/NCE2/86.Out of Control.lrc"
python3 tools/lrc-align/align_lrc.py "uk/NCE2/86.Out of Control.mp3" "uk/NCE2/86.Out of Control.lrc" --write
python3 tools/lrc-align/align_lrc.py --batch-root uk --limit 1
python3 tools/lrc-align/align_lrc.py --batch-root uk --write
python3 tools/lrc-align/align_lrc.py --batch-root uk --start-at 'NCE3/41.Illusions of Pastoral Peace.lrc' --write
```

Batch mode prints periodic heartbeats while a longer audio file is being processed and continues after per-file errors, reporting them in the final summary. It also normalizes legacy time tags with omitted minutes or extra leading zeros when writing aligned captions.

The first run downloads the pretrained English acoustic model. The script expands numeric text (for example, `15`) to spoken words before alignment. The preview lists the three weakest first-word alignment scores for relative review; these scores are not calibrated probabilities and do not shift timestamps. Listen to low-scoring lines before using `--write`; forced alignment depends on the supplied transcript matching the recording.