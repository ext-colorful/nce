import argparse
import math
import re
import statistics
import threading
from pathlib import Path

import numpy as np


TIME_TAG = re.compile(r"\[(?:(\d+):)?(\d{1,3})\.(\d{1,3})\]")
ATTRIBUTION_TAG = re.compile(r"^\s*\[(?:ar|by):", re.IGNORECASE)
TEXT_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+")
RMS_FRAME_SECONDS = 0.01
MIN_SILENCE_FRAMES = 2
INFERENCE_CHUNK_SECONDS = 24
INFERENCE_CONTEXT_SECONDS = 2


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


def infer_emissions_chunked(model, waveform, sample_rate, torch):
    total_samples = waveform.shape[1]
    chunk_samples = INFERENCE_CHUNK_SECONDS * sample_rate
    if total_samples <= chunk_samples:
        with torch.inference_mode():
            emissions, _ = model(waveform)
        return emissions

    context_samples = INFERENCE_CONTEXT_SECONDS * sample_rate
    core_samples = chunk_samples
    emission_chunks = []
    for core_start in range(0, total_samples, core_samples):
        core_end = min(total_samples, core_start + core_samples)
        input_start = max(0, core_start - context_samples)
        input_end = min(total_samples, core_end + context_samples)
        audio_chunk = waveform[:, input_start:input_end]
        with torch.inference_mode():
            chunk_emissions, _ = model(audio_chunk)

        seconds_per_frame = audio_chunk.shape[1] / chunk_emissions.shape[1] / sample_rate
        keep_start = round((core_start - input_start) / sample_rate / seconds_per_frame)
        keep_end = round((core_end - input_start) / sample_rate / seconds_per_frame)
        keep_start = max(0, min(keep_start, chunk_emissions.shape[1]))
        keep_end = max(keep_start, min(keep_end, chunk_emissions.shape[1]))
        if keep_end == keep_start:
            raise RuntimeError(f"Chunk {core_start}:{core_end} produced no retained acoustic frames.")
        emission_chunks.append(chunk_emissions[:, keep_start:keep_end, :])

    return torch.cat(emission_chunks, dim=1)


def load_alignment_model(device):
    import torchaudio

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    return model, bundle


def align_transcript(audio_path, transcript, device, model, bundle):
    import av
    import torch
    import torchaudio

    sample_rate = bundle.sample_rate
    waveform = load_audio(audio_path, sample_rate, torch, av).to(device)

    emissions = infer_emissions_chunked(model, waveform, sample_rate, torch)
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


def align_file(audio_path, lrc_path, device, model, bundle, write, verbose=True):
    from num2words import num2words

    lines = lrc_path.read_text(encoding="utf-8").splitlines(keepends=True)
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
        audio_path, transcript, device, model, bundle
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
    previous_timestamp = None
    removed_tags = 0

    for line_number, line in enumerate(lines, start=1):
        if ATTRIBUTION_TAG.match(line):
            removed_tags += 1
            if verbose:
                print(f"Remove metadata line {line_number}: {line.strip()}")
            continue
        tag = TIME_TAG.search(line)
        row = rows_by_line.get(line_number)
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
        if previous_timestamp is not None and new_seconds < previous_timestamp:
            raise ValueError(
                f"non-monotonic timestamp at line {line_number}: "
                f"{new_seconds:.3f}s follows {previous_timestamp:.3f}s"
            )
        previous_timestamp = new_seconds
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

    centered_count = sum(source == "silence midpoint" for source in timing_sources.values())
    onset_count = len(timing_sources) - centered_count
    if verbose:
        for line_number, old_tag, new_tag, english, timing_source in updates:
            print(f"{line_number}: {old_tag} -> {new_tag} [{timing_source}]  {english}")
        print(
            f"CTC force-aligned {len(aligned_lines)} caption lines; "
            f"frame resolution about {seconds_per_frame:.3f}s; audio {audio_duration:.2f}s. "
            f"Starts: {centered_count} silence midpoints, {onset_count} word onsets."
        )
        if confidence_rows:
            print("Lowest first-word alignment scores (relative log scores; more negative is weaker):")
            for first_word_score, line_score, line_number, english in sorted(confidence_rows)[:3]:
                print(
                    f"{line_number}: first word {first_word_score:.3f}, line {line_score:.3f}  "
                    f"{english}"
                )

    if write:
        lrc_path.write_text("".join(output), encoding="utf-8")
        if verbose:
            print(f"Wrote {lrc_path}")

    return {
        "aligned_lines": len(aligned_lines),
        "silence_midpoints": centered_count,
        "word_onsets": onset_count,
        "updates": len(updates),
        "removed_tags": removed_tags,
        "confidence_rows": confidence_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Force-align English LRC captions to their audio.")
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("lrc", type=Path, nargs="?")
    parser.add_argument("--batch-root", type=Path, help="Process every LRC and matching MP3 below this directory.")
    parser.add_argument("--start-at", type=Path, help="Resume batch mode at this path relative to --batch-root.")
    parser.add_argument("--limit", type=int, help="Limit batch mode to the first N LRC files (useful for a smoke test).")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--write", action="store_true", help="Write aligned timestamps and remove [ar]/[by] tags.")
    args = parser.parse_args()

    if args.batch_root:
        if args.audio or args.lrc:
            parser.error("positional audio/lrc arguments cannot be combined with --batch-root")
        if not args.batch_root.is_dir():
            parser.error(f"batch root is not a directory: {args.batch_root}")
        lrc_paths = sorted(args.batch_root.rglob("*.lrc"))
        if not lrc_paths:
            parser.error(f"no LRC files found under {args.batch_root}")
        if args.start_at:
            start_path = args.start_at.as_posix()
            start_index = next(
                (
                    index
                    for index, path in enumerate(lrc_paths)
                    if path.relative_to(args.batch_root).as_posix() == start_path
                ),
                None,
            )
            if start_index is None:
                parser.error(f"--start-at path not found under {args.batch_root}: {args.start_at}")
            lrc_paths = lrc_paths[start_index:]
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be greater than zero")
            lrc_paths = lrc_paths[:args.limit]
        missing_audio = [path for path in lrc_paths if not path.with_suffix(".mp3").is_file()]
        jobs = [(path.with_suffix(".mp3"), path) for path in lrc_paths if path.with_suffix(".mp3").is_file()]
        if not jobs:
            parser.error(f"no LRC files with matching MP3 found under {args.batch_root}")
    else:
        if not args.audio or not args.lrc:
            parser.error("provide audio and lrc paths, or use --batch-root")
        jobs = [(args.audio, args.lrc)]
        missing_audio = []

    model, bundle = load_alignment_model(args.device)
    if not args.batch_root:
        align_file(*jobs[0], args.device, model, bundle, args.write)
        return

    print(f"Batch: {len(jobs)} files under {args.batch_root}; write={args.write}", flush=True)
    failures = []
    low_confidence = []
    totals = {"aligned_lines": 0, "silence_midpoints": 0, "word_onsets": 0, "updates": 0, "removed_tags": 0}

    heartbeat_stop = threading.Event()

    def report_heartbeat():
        while not heartbeat_stop.wait(20):
            print("[batch] alignment still running", flush=True)

    heartbeat = threading.Thread(target=report_heartbeat, daemon=True)
    heartbeat.start()
    for index, (audio_path, lrc_path) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {lrc_path}", end=" ", flush=True)
        try:
            result = align_file(audio_path, lrc_path, args.device, model, bundle, args.write, verbose=False)
        except Exception as error:
            failures.append((lrc_path, str(error)))
            print(f"FAILED: {error}", flush=True)
            continue
        for key in totals:
            totals[key] += result[key]
        low_confidence.extend(
            (first_score, line_score, lrc_path, line_number, english)
            for first_score, line_score, line_number, english in result["confidence_rows"]
        )
        print(
            f"OK ({result['aligned_lines']} lines, {result['silence_midpoints']} silence midpoints, "
            f"{result['word_onsets']} word onsets, {result['updates']} timestamp changes, "
            f"{result['removed_tags']} tags removed)",
            flush=True,
        )
    heartbeat_stop.set()
    heartbeat.join()

    print(
        f"\nBatch complete: {len(jobs) - len(failures)}/{len(jobs)} succeeded; "
        f"{len(missing_audio)} missing audio; {len(failures)} failures. "
        f"{totals['aligned_lines']} caption lines; {totals['silence_midpoints']} silence midpoints; "
        f"{totals['word_onsets']} word onsets; {totals['updates']} timestamp changes; "
        f"{totals['removed_tags']} attribution tags removed."
    )
    if low_confidence:
        print("Lowest first-word alignment scores (relative log scores; more negative is weaker):")
        for first_score, line_score, path, line_number, english in sorted(low_confidence)[:20]:
            print(f"{path}:{line_number}: first word {first_score:.3f}, line {line_score:.3f}  {english}")
    if missing_audio or failures:
        print("Files needing attention:")
        for path in missing_audio:
            print(f"MISSING AUDIO: {path}")
        for path, error in failures:
            print(f"FAILED: {path}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()