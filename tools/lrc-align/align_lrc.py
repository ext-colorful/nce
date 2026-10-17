import argparse
import math
import re
import statistics
from pathlib import Path

import numpy as np


TIME_TAG = re.compile(r"\[(\d+):(\d{2})\.(\d{1,3})\]")
ATTRIBUTION_TAG = re.compile(r"^\s*\[(?:ar|by):", re.IGNORECASE)
TEXT_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+")
RMS_FRAME_SECONDS = 0.01
MIN_SILENCE_FRAMES = 2


def normalize_english(text, number_to_words):
    words = []
    for token in TEXT_TOKEN.findall(text):
        expanded = number_to_words(int(token)) if token.isdigit() else token
        words.extend(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", expanded.lower()))
    return words


def format_tag(seconds, match):
    precision = len(match.group(3))
    scale = 10 ** (3 - precision)
    milliseconds = round(seconds * 1000 / scale) * scale
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, fractional = divmod(remainder, 1000)
    fraction = f"{fractional:03d}"[:precision]
    return f"[{minutes:02d}:{whole_seconds:02d}.{fraction}]"


def calculate_rms_frames(samples, sample_rate):
    frame_samples = round(sample_rate * RMS_FRAME_SECONDS)
    frame_count = len(samples) // frame_samples
    if frame_count == 0:
        raise ValueError("Audio is too short for silence detection.")
    framed = samples[:frame_count * frame_samples].reshape(frame_count, frame_samples)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def longest_silence_midpoint(rms_frames, frame_seconds, start, end, threshold):
    if end <= start:
        return None

    first_frame = max(0, int(start / frame_seconds))
    after_last_frame = min(len(rms_frames), math.ceil(end / frame_seconds))
    quiet_frames = rms_frames[first_frame:after_last_frame] <= threshold
    edges = np.diff(np.concatenate(([False], quiet_frames, [False])).astype(np.int8))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    runs = [
        (first_frame + run_start, first_frame + run_end)
        for run_start, run_end in zip(run_starts, run_ends)
        if run_end - run_start >= MIN_SILENCE_FRAMES
    ]
    if not runs:
        return None

    run_start, run_end = max(runs, key=lambda run: run[1] - run[0])
    midpoint = (run_start + run_end) * frame_seconds / 2
    return min(max(midpoint, start), end)


def load_audio(audio_path, sample_rate, torch, av):
    chunks = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    with av.open(str(audio_path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {audio_path}")
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(torch.from_numpy(resampled.to_ndarray()).reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(torch.from_numpy(resampled.to_ndarray()).reshape(-1))
    if not chunks:
        raise ValueError(f"No audio samples decoded from {audio_path}")
    return torch.cat(chunks).unsqueeze(0)


def align_transcript(audio_path, transcript, device):
    import av
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    sample_rate = bundle.sample_rate
    waveform = load_audio(audio_path, sample_rate, torch, av).to(device)
    model = bundle.get_model().to(device).eval()

    with torch.inference_mode():
        emissions, _ = model(waveform)
        log_probs = torch.log_softmax(emissions, dim=-1)

    labels = bundle.get_labels()
    label_ids = {label: index for index, label in enumerate(labels)}
    try:
        target_ids = [label_ids[char] for char in transcript]
    except KeyError as error:
        raise ValueError(f"Character not supported by acoustic model: {error.args[0]!r}") from error

    if log_probs.shape[1] < len(target_ids):
        raise ValueError("Transcript is too long to align to this audio.")

    targets = torch.tensor([target_ids], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.int32, device=device)
    paths, scores = torchaudio.functional.forced_align(
        log_probs,
        targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=label_ids["-"],
    )
    spans = torchaudio.functional.merge_tokens(paths[0], scores[0], blank=label_ids["-"])
    if len(spans) != len(target_ids):
        raise RuntimeError(f"Alignment returned {len(spans)} character spans for {len(target_ids)} transcript characters.")

    samples_per_frame = waveform.shape[1] / log_probs.shape[1]
    return (
        spans,
        samples_per_frame / sample_rate,
        waveform.shape[1] / sample_rate,
        waveform[0].detach().cpu().numpy(),
        sample_rate,
    )


def main():
    parser = argparse.ArgumentParser(description="Force-align English LRC captions to their audio.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("lrc", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--write", action="store_true", help="Write aligned timestamps back to the LRC file.")
    args = parser.parse_args()

    from num2words import num2words

    lines = args.lrc.read_text(encoding="utf-8").splitlines(keepends=True)
    transcript_parts = []
    caption_rows = []
    for line_number, line in enumerate(lines, start=1):
        tag = TIME_TAG.search(line)
        if not tag or "|" not in line:
            continue
        english = line[tag.end():].split("|", 1)[0].strip()
        words = normalize_english(english, num2words)
        if not words:
            continue
        if transcript_parts:
            transcript_parts.append("|")
        char_start = sum(len(part) for part in transcript_parts)
        phrase = "|".join(words).upper()
        transcript_parts.append(phrase)
        caption_rows.append((line_number, tag, char_start, english, len(words[0]), len(phrase)))

    transcript = "".join(transcript_parts)
    spans, seconds_per_frame, audio_duration, audio_samples, sample_rate = align_transcript(
        args.audio, transcript, args.device
    )
    rms_frames = calculate_rms_frames(audio_samples, sample_rate)
    noise_level = float(np.quantile(rms_frames, 0.1))
    speech_level = float(np.quantile(rms_frames, 0.9))
    silence_threshold = max(noise_level * 3, speech_level * 0.02, 1e-5)
    aligned_lines = []
    updates = []
    confidence_rows = []
    timing_sources = {}
    output = []
    rows_by_line = {row[0]: row for row in caption_rows}
    previous_row = None

    for line_number, line in enumerate(lines, start=1):
        if ATTRIBUTION_TAG.match(line):
            print(f"Remove metadata line {line_number}: {line.strip()}")
            continue
        tag = TIME_TAG.search(line)
        row = next((item for item in caption_rows if item[0] == line_number), None)
        if row is None:
            output.append(line)
            continue
        _, _, char_start, english, first_word_length, phrase_length = row
        word_onset = spans[char_start].start * seconds_per_frame
        if previous_row is None:
            new_seconds = word_onset
            timing_source = "first-word onset"
        else:
            previous_char_start = previous_row[2]
            previous_phrase_length = previous_row[5]
            previous_phrase_end = spans[previous_char_start + previous_phrase_length - 1].end * seconds_per_frame
            silence_midpoint = longest_silence_midpoint(
                rms_frames,
                RMS_FRAME_SECONDS,
                previous_phrase_end,
                word_onset,
                silence_threshold,
            )
            if silence_midpoint is None:
                new_seconds = word_onset
                timing_source = "word onset (no silence)"
            else:
                new_seconds = silence_midpoint
                timing_source = "silence midpoint"
        timing_sources[line_number] = timing_source
        replacement = format_tag(new_seconds, tag)
        output.append(line[:tag.start()] + replacement + line[tag.end():])
        aligned_lines.append(english)
        first_word_score = statistics.mean(
            float(spans[index].score) for index in range(char_start, char_start + first_word_length)
        )
        line_score = statistics.mean(
            float(spans[index].score) for index in range(char_start, char_start + phrase_length)
        )
        confidence_rows.append((first_word_score, line_score, line_number, english))
        if replacement != tag.group(0):
            updates.append((line_number, tag.group(0), replacement, english, timing_source))
        previous_row = row

    for line_number, old_tag, new_tag, english, timing_source in updates:
        print(f"{line_number}: {old_tag} -> {new_tag} [{timing_source}]  {english}")
    centered_count = sum(source == "silence midpoint" for source in timing_sources.values())
    onset_count = len(timing_sources) - centered_count
    print(
        f"CTC force-aligned {len(aligned_lines)} caption lines; "
        f"frame resolution about {seconds_per_frame:.3f}s; audio {audio_duration:.2f}s. "
        f"Starts: {centered_count} silence midpoints, {onset_count} word onsets."
    )
    print("Lowest first-word alignment scores (relative log scores; more negative is weaker):")
    for first_word_score, line_score, line_number, english in sorted(confidence_rows)[:3]:
        print(
            f"{line_number}: first word {first_word_score:.3f}, line {line_score:.3f}  "
            f"{english}"
        )

    if args.write:
        args.lrc.write_text("".join(output), encoding="utf-8")
        print(f"Wrote {args.lrc}")


if __name__ == "__main__":
    main()