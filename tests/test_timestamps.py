"""Tests for timestamps and subtitle generation utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
import pytest

from tts_cli.timestamps import (
    SentenceTimestamp,
    TimestampResult,
    WordTimestamp,
    align_phonemes_to_words,
    format_srt_time,
    format_vtt_time,
    interpolate_word_timestamps,
)


@dataclass
class MockTiming:
    phoneme: str
    start: float
    end: float


def test_time_formatters():
    """Test SRT and VTT timestamp formatting."""
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.234) == "00:00:01,234"
    assert format_srt_time(65.5) == "00:01:05,500"
    assert format_srt_time(3661.05) == "01:01:01,050"

    assert format_vtt_time(0.0) == "00:00:00.000"
    assert format_vtt_time(1.234) == "00:00:01.234"
    assert format_vtt_time(65.5) == "00:01:05.500"


def test_timestamp_result_serialization(tmp_path):
    """Test TimestampResult export to JSON, SRT, VTT, and CSV."""
    words = [
        WordTimestamp(word="Hello", start=0.1, end=0.5),
        WordTimestamp(word="world!", start=0.6, end=1.2),
    ]
    sentence = SentenceTimestamp(
        sentence="Hello world!",
        start=0.0,
        end=1.3,
        words=words,
    )
    result = TimestampResult(sentences=[sentence], words=words, total_duration=1.3)

    # 1. JSON
    json_str = result.to_json()
    data = json.loads(json_str)
    assert data["total_duration"] == 1.3
    assert len(data["sentences"]) == 1
    assert data["sentences"][0]["sentence"] == "Hello world!"
    assert len(data["words"]) == 2

    # 2. SRT Sentence
    srt_sentence = result.to_srt(level="sentence")
    assert "1" in srt_sentence
    assert "00:00:00,000 --> 00:00:01,300" in srt_sentence
    assert "Hello world!" in srt_sentence

    # 3. SRT Word
    srt_word = result.to_srt(level="word")
    assert "00:00:00,100 --> 00:00:00,500" in srt_word
    assert "Hello" in srt_word
    assert "00:00:00,600 --> 00:00:01,200" in srt_word
    assert "world!" in srt_word

    # 4. VTT
    vtt_str = result.to_vtt(level="sentence")
    assert vtt_str.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.300" in vtt_str

    # 5. CSV
    csv_str = result.to_csv()
    assert "type,start,end,duration,text" in csv_str
    assert "sentence,0.000,1.300,1.300,Hello world!" in csv_str
    assert "word,0.100,0.500,0.400,Hello" in csv_str

    # 6. Save method
    base_file = tmp_path / "audio_out.wav"
    saved = result.save(base_file, fmt="all", level="both")
    assert len(saved) == 6
    for f in saved:
        assert Path(f).exists()


def test_align_phonemes_to_words():
    """Test grouping phoneme timings into word-level spans."""
    # "Hi there"
    # 'h', 'aɪ', ' ', 'ð', 'ɛ', 'ɹ'
    spoken = [
        MockTiming(phoneme="h", start=0.1, end=0.2),
        MockTiming(phoneme="aɪ", start=0.2, end=0.4),
        MockTiming(phoneme=" ", start=0.4, end=0.5),
        MockTiming(phoneme="ð", start=0.5, end=0.6),
        MockTiming(phoneme="ɛ", start=0.6, end=0.7),
        MockTiming(phoneme="ɹ", start=0.7, end=0.9),
    ]

    words = align_phonemes_to_words("Hi there", spoken, sentence_offset=1.0)
    assert len(words) == 2
    assert words[0].word == "Hi"
    assert pytest.approx(words[0].start, 0.01) == 1.1
    assert pytest.approx(words[0].end, 0.01) == 1.4

    assert words[1].word == "there"
    assert pytest.approx(words[1].start, 0.01) == 1.5
    assert pytest.approx(words[1].end, 0.01) == 1.9


def test_interpolate_word_timestamps():
    """Test proportional word timestamp distribution."""
    text = "Quick brown fox"
    words = interpolate_word_timestamps(text, start_time=1.0, end_time=4.0)

    assert len(words) == 3
    assert words[0].word == "Quick"
    assert pytest.approx(words[0].start) == 1.0
    assert words[1].word == "brown"
    assert words[2].word == "fox"
    assert pytest.approx(words[-1].end) == 4.0

    # Ensure strictly non-decreasing times
    for i in range(len(words) - 1):
        assert words[i].start <= words[i].end
        assert words[i].end <= words[i + 1].start
