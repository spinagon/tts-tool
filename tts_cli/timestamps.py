"""Timestamps processing and subtitle formatting for video/audio alignment."""

import csv
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp string: HH:MM:SS,mmm"""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder %= 60000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format seconds into WebVTT timestamp string: HH:MM:SS.mmm"""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder %= 60000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
        }


@dataclass
class SentenceTimestamp:
    sentence: str
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class TimestampResult:
    sentences: List[SentenceTimestamp] = field(default_factory=list)
    words: List[WordTimestamp] = field(default_factory=list)
    total_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration": round(self.total_duration, 3),
            "sentences": [s.to_dict() for s in self.sentences],
            "words": [w.to_dict() for w in self.words],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_srt(self, level: Literal["sentence", "word"] = "sentence") -> str:
        """Export as standard SubRip (.srt) subtitle format."""
        lines = []
        index = 1
        items = self.words if level == "word" else self.sentences

        for item in items:
            text = item.word if isinstance(item, WordTimestamp) else item.sentence
            start_str = format_srt_time(item.start)
            end_str = format_srt_time(item.end)
            lines.append(f"{index}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"{text.strip()}")
            lines.append("")
            index += 1

        return "\n".join(lines)

    def to_vtt(self, level: Literal["sentence", "word"] = "sentence") -> str:
        """Export as WebVTT (.vtt) subtitle format."""
        lines = ["WEBVTT", ""]
        index = 1
        items = self.words if level == "word" else self.sentences

        for item in items:
            text = item.word if isinstance(item, WordTimestamp) else item.sentence
            start_str = format_vtt_time(item.start)
            end_str = format_vtt_time(item.end)
            lines.append(f"{index}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"{text.strip()}")
            lines.append("")
            index += 1

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export as CSV with columns: type,start,end,duration,text."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["type", "start", "end", "duration", "text"])

        for s in self.sentences:
            writer.writerow(["sentence", f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}", s.sentence])

        for w in self.words:
            writer.writerow(["word", f"{w.start:.3f}", f"{w.end:.3f}", f"{w.duration:.3f}", w.word])

        return output.getvalue()

    def save(
        self,
        base_path: Union[str, Path],
        fmt: str = "json",
        level: Literal["sentence", "word", "both"] = "both",
    ) -> List[str]:
        """Save timestamps to one or more formats.

        Args:
            base_path: Output audio file path or target file path base.
            fmt: 'json', 'srt', 'vtt', 'csv', or 'all'.
            level: 'sentence', 'word', or 'both'.

        Returns:
            List of generated file paths.
        """
        base = Path(base_path)
        stem = base.parent / base.stem
        saved_files = []

        formats = ["json", "srt", "vtt", "csv"] if fmt.lower() == "all" else [fmt.lower()]

        for f in formats:
            if f == "json":
                out_file = stem.with_suffix(".json")
                out_file.write_text(self.to_json(), encoding="utf-8")
                saved_files.append(str(out_file))

            elif f == "srt":
                if level in ("sentence", "both"):
                    out_file = stem.with_suffix(".srt")
                    out_file.write_text(self.to_srt(level="sentence"), encoding="utf-8")
                    saved_files.append(str(out_file))
                if level in ("word", "both"):
                    out_file = stem.parent / f"{stem.name}.words.srt"
                    out_file.write_text(self.to_srt(level="word"), encoding="utf-8")
                    saved_files.append(str(out_file))

            elif f == "vtt":
                if level in ("sentence", "both"):
                    out_file = stem.with_suffix(".vtt")
                    out_file.write_text(self.to_vtt(level="sentence"), encoding="utf-8")
                    saved_files.append(str(out_file))
                if level in ("word", "both"):
                    out_file = stem.parent / f"{stem.name}.words.vtt"
                    out_file.write_text(self.to_vtt(level="word"), encoding="utf-8")
                    saved_files.append(str(out_file))

            elif f == "csv":
                out_file = stem.with_suffix(".csv")
                out_file.write_text(self.to_csv(), encoding="utf-8")
                saved_files.append(str(out_file))

        return saved_files


def align_phonemes_to_words(
    text: str,
    spoken_timings: List[Any],
    sentence_offset: float = 0.0,
) -> List[WordTimestamp]:
    """Group phoneme-level timings into word-level timestamps."""
    if not spoken_timings:
        return []

    # Filter and group contiguous non-space phonemes
    phoneme_groups = []
    curr_phonemes = []
    curr_start = None
    curr_end = None

    for item in spoken_timings:
        p = getattr(item, "phoneme", "")
        start = getattr(item, "start", 0.0) + sentence_offset
        end = getattr(item, "end", 0.0) + sentence_offset

        if p.strip() and p not in (" ", "\t", "\n"):
            if curr_start is None:
                curr_start = start
            curr_end = end
            curr_phonemes.append(p)
        else:
            if curr_phonemes and curr_start is not None and curr_end is not None:
                phoneme_groups.append((curr_start, curr_end))
                curr_phonemes = []
                curr_start = None
                curr_end = None

    if curr_phonemes and curr_start is not None and curr_end is not None:
        phoneme_groups.append((curr_start, curr_end))

    # Match original words to phoneme groups
    orig_words = [w for w in text.split() if w.strip()]
    results = []

    for i, w in enumerate(orig_words):
        if i < len(phoneme_groups):
            start, end = phoneme_groups[i]
            results.append(WordTimestamp(word=w, start=start, end=end))
        elif results:
            # Fallback anchor if phoneme group count differs
            last_end = results[-1].end
            results.append(WordTimestamp(word=w, start=last_end, end=last_end + 0.2))

    return results


def interpolate_word_timestamps(
    text: str,
    start_time: float,
    end_time: float,
) -> List[WordTimestamp]:
    """Proportionally distribute words within a sentence span based on character lengths."""
    words = [w for w in text.split() if w.strip()]
    if not words:
        return []

    duration = max(0.01, end_time - start_time)
    # Estimate relative weight of each word (characters + small base)
    weights = [len(w) + 2 for w in words]
    total_weight = sum(weights)

    results = []
    current_time = start_time
    for w, weight in zip(words, weights):
        w_dur = (weight / total_weight) * duration
        w_end = min(end_time, current_time + w_dur)
        results.append(WordTimestamp(word=w, start=current_time, end=w_end))
        current_time = w_end

    return results
