import gc
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from kokoro_onnx import Kokoro


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Lenchos Audio Studio"

TARGET_WORDS_PER_CHUNK = 550

OUTPUT_SAMPLE_RATE = 24000

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)

VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

AMBIENT_URL = (
    "https://github.com/rafaelreis-io/rafaelreis-io/raw/main/ambient.wav"
)

MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"
AMBIENT_FILENAME = "ambient_bed.wav"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHTWEIGHT CSS
# ============================================================

st.markdown(
    "<style>"
    ".hero-container{"
    "padding:28px 10px 20px 10px;"
    "text-align:center;"
    "}"
    ".hero-title{"
    "font-size:38px;"
    "font-weight:800;"
    "letter-spacing:1px;"
    "}"
    ".hero-subtitle{"
    "font-size:15px;"
    "opacity:.75;"
    "margin-top:8px;"
    "}"
    ".lencho-highlight{"
    "font-weight:800;"
    "}"
    ".status-box{"
    "padding:14px;"
    "border-radius:12px;"
    "border:1px solid rgba(128,128,128,.25);"
    "margin-top:10px;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

hero_html = (
    '<div class="hero-container">'
    '<div class="hero-title">'
    "🎙️ LENCHOS AUDIO STUDIO"
    "</div>"
    '<div class="hero-subtitle">'
    "Kokoro-powered open-source AI narration for calm, long-form audio."
    "</div>"
    "</div>"
)

st.markdown(hero_html, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "current_job" not in st.session_state:
    st.session_state.current_job = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def get_base_work_dir():
    """
    Creates one application-level directory for generated jobs.

    Files remain available during the current Streamlit process,
    including normal Streamlit reruns.
    """
    base_dir = Path(tempfile.gettempdir()) / "lenchos_audio_studio"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def cleanup_old_jobs(keep_job_dirs=None):
    """
    Keeps disk usage under control.

    We only delete directories that are not currently needed.
    """
    if keep_job_dirs is None:
        keep_job_dirs = set()

    base_dir = get_base_work_dir()

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        if str(item) in keep_job_dirs:
            continue

        # Keep recent directories only if they are part of session history.
        # Old orphaned jobs can safely be removed.
        try:
            shutil.rmtree(item)
        except Exception:
            pass


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_file(url, destination):
    """
    Downloads a file only when it does not already exist.
    """
    destination = Path(destination)

    if destination.exists() and destination.stat().st_size > 0:
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(url, destination)

    return str(destination)


# ============================================================
# KOKORO ENGINE
# ============================================================

@st.cache_resource(show_spinner="Loading Kokoro model...")
def get_kokoro_engine():
    """
    Loads Kokoro once and keeps it cached.

    This is intentionally cached because reloading the ONNX model
    for every chunk would waste CPU and memory.
    """

    model_dir = get_base_work_dir() / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    voices_path = model_dir / VOICES_FILENAME

    download_file(MODEL_URL, model_path)
    download_file(VOICES_URL, voices_path)

    kokoro = Kokoro(
        str(model_path),
        str(voices_path),
    )

    return kokoro


# ============================================================
# AMBIENT AUDIO
# ============================================================

@st.cache_data(show_spinner=False)
def get_background_track():
    """
    Downloads and caches the ambient bed.

    Returns:
        audio: mono float32 NumPy array
        sample_rate: source sample rate
    """

    audio_dir = get_base_work_dir() / "audio_assets"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ambient_path = audio_dir / AMBIENT_FILENAME

    download_file(AMBIENT_URL, ambient_path)

    audio, sample_rate = sf.read(
        str(ambient_path),
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    return audio, int(sample_rate)


# ============================================================
# AUDIO RESAMPLING
# ============================================================

def resample_audio(audio, source_rate, target_rate):
    """
    Simple linear resampling.

    This avoids requiring an additional audio-processing package.
    """

    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    if len(audio) == 0:
        return audio.astype(np.float32)

    target_length = int(
        round(len(audio) * target_rate / source_rate)
    )

    if target_length <= 1:
        return np.zeros(1, dtype=np.float32)

    old_positions = np.linspace(
        0,
        1,
        num=len(audio),
        endpoint=False,
    )

    new_positions = np.linspace(
        0,
        1,
        num=target_length,
        endpoint=False,
    )

    resampled = np.interp(
        new_positions,
        old_positions,
        audio,
    )

    return resampled.astype(np.float32)


# ============================================================
# AMBIENT MIXING
# ============================================================

def mix_audio_bed(
    narration,
    narration_rate,
    background,
    background_rate,
    volume,
):
    """
    Mixes a looping ambient track underneath narration.
    """

    narration = np.asarray(
        narration,
        dtype=np.float32,
    )

    background = np.asarray(
        background,
        dtype=np.float32,
    )

    if len(background) == 0:
        return narration

    background = resample_audio(
        background,
        background_rate,
        narration_rate,
    )

    if len(background) == 0:
        return narration

    repeats = int(
        np.ceil(len(narration) / len(background))
    )

    tiled_background = np.tile(
        background,
        repeats,
    )

    tiled_background = tiled_background[: len(narration)]

    mixed = narration + (
        tiled_background * float(volume)
    )

    mixed = np.clip(
        mixed,
        -1.0,
        1.0,
    )

    return mixed.astype(np.float32)


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_script(text):
    """
    Cleans excessive whitespace while preserving paragraphs.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        cleaned = re.sub(
            r"\s+",
            " ",
            line.strip(),
        )

        if cleaned:
            lines.append(cleaned)

    return "\n\n".join(lines)


def split_long_sentence(sentence, target_words):
    """
    Splits an unusually long sentence into word-sized pieces.
    """

    words = sentence.split()

    if len(words) <= target_words:
        return [sentence.strip()]

    pieces = []

    for start in range(
        0,
        len(words),
        target_words,
    ):
        piece = " ".join(
            words[start:start + target_words]
        )

        if piece.strip():
            pieces.append(piece.strip())

    return pieces


def split_script_into_chunks(
    text,
    target_words=TARGET_WORDS_PER_CHUNK,
):
    """
    Creates sentence-aware chunks of roughly target_words.

    This prevents Kokoro from receiving the entire 25-minute script
    at once.
    """

    text = normalize_script(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        for sentence in paragraph_sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentences.extend(
                split_long_sentence(
                    sentence,
                    target_words,
                )
            )

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if (
            current
            and current_words + sentence_words > target_words
        ):
            chunks.append(
                " ".join(current).strip()
            )

            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(
            " ".join(current).strip()
        )

    return chunks


# ============================================================
# JOB IDENTIFICATION
# ============================================================

def make_job_id(
    script,
    voice,
    speed,
    ambient_enabled,
    ambient_volume,
):
    """
    Generates a deterministic ID.

    If generation is interrupted, launching the exact same job
    again can reuse already completed chunks.
    """

    payload = (
        f"{script}|"
        f"{voice}|"
        f"{speed:.4f}|"
        f"{ambient_enabled}|"
        f"{ambient_volume:.4f}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return digest[:16]


# ============================================================
# JOB CREATION
# ============================================================

def create_job_directory(job_id):
    base_dir = get_base_work_dir()

    job_dir = base_dir / f"job_{job_id}"
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return job_dir


# ============================================================
# CHUNK FILE HELPERS
# ============================================================

def get_chunk_path(job_dir, index):
    return (
        Path(job_dir)
        / "chunks"
        / f"chunk_{index:03d}.wav"
    )


def chunk_is_complete(path):
    """
    A chunk is considered complete if the WAV exists and has
    actual audio frames.
    """

    path = Path(path)

    if not path.exists():
        return False

    if path.stat().st_size < 1000:
        return False

    try:
        info = sf.info(str(path))

        return (
            info.frames > 0
            and info.samplerate > 0
        )

    except Exception:
        return False


# ============================================================
# GENERATE ONE CHUNK
# ============================================================

def generate_chunk(
    kokoro,
    text,
    voice,
    speed,
    output_path,
    ambient_enabled=False,
    ambient_volume=0.15,
):
    """
    Generates one independent Kokoro chunk.

    Important:
    The completed result is immediately written to disk.
    """

    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=float(speed),
        lang="en-us",
    )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    sample_rate = int(sample_rate)

    if ambient_enabled:
        background, background_rate = (
            get_background_track()
        )

        samples = mix_audio_bed(
            samples,
            sample_rate,
            background,
            background_rate,
            ambient_volume,
        )

    sf.write(
        str(output_path),
        samples,
        sample_rate,
        subtype="PCM_16",
    )

    # Explicitly release the large NumPy array.
    del samples
    gc.collect()

    return sample_rate


# ============================================================
# COMBINE WAV FILES
# ============================================================

def combine_wav_files(
    chunk_paths,
    output_path,
):
    """
    Combines WAV chunks sequentially.

    It does NOT load all chunks into RAM simultaneously.
    """

    if not chunk_paths:
        raise ValueError(
            "No chunk files were supplied."
        )

    first_path = Path(chunk_paths[0])

    first_info = sf.info(
        str(first_path)
    )

    sample_rate = first_info.samplerate
    channels = first_info.channels

    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
        format="WAV",
    ) as output_file:

        for path in chunk_paths:
            path = Path(path)

            with sf.SoundFile(
                str(path),
                mode="r",
            ) as input_file:

                if input_file.samplerate != sample_rate:
                    raise ValueError(
                        "Chunk sample rates do not match."
                    )

                if input_file.channels != channels:
                    raise ValueError(
                        "Chunk channel counts do not match."
                    )

                while True:
                    block = input_file.read(
                        65536,
                        dtype="float32",
                    )

                    if len(block) == 0:
                        break

                    output_file.write(block)

                    del block

    gc.collect()


# ============================================================
# MP3 CONVERSION
# ============================================================

def convert_wav_to_mp3(
    wav_path,
    mp3_path,
):
    """
    Converts the completed WAV to MP3 using imageio-ffmpeg.

    imageio-ffmpeg is installed from requirements.txt.
    """

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed. "
            "Add imageio-ffmpeg to requirements.txt."
        ) from exc

    ffmpeg_exe = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        "-ar",
        "24000",
        str(mp3_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg MP3 conversion failed:\n\n"
            + result.stderr[-3000:]
        )


# ============================================================
# JOB STATUS
# ============================================================

def count_completed_chunks(
    job_dir,
    total_chunks,
):
    count = 0

    for index in range(
        1,
        total_chunks + 1,
    ):
        path = get_chunk_path(
            job_dir,
            index,
        )

        if chunk_is_complete(path):
            count += 1

    return count


# ============================================================
# HISTORY MANAGEMENT
# ============================================================

def add_history_item(item):
    history = st.session_state.history

    # Remove duplicate job ID.
    history = [
        old
        for old in history
        if old.get("job_id") != item.get("job_id")
    ]

    history.insert(
        0,
        item,
    )

    # Only metadata is stored here.
    st.session_state.history = history[:2]


def cleanup_history_dirs():
    """
    Deletes job directories that are no longer represented
    in current history or current job.
    """

    keep_dirs = set()

    current_job = st.session_state.current_job

    if current_job:
        keep_dirs.add(
            str(current_job["work_dir"])
        )

    for item in st.session_state.history:
        work_dir = item.get("work_dir")

        if work_dir:
            keep_dirs.add(
                str(work_dir)
            )

    base_dir = get_base_work_dir()

    for item in base_dir.iterdir():

        if not item.is_dir():
            continue

        if item.name in {
            "model",
            "audio_assets",
        }:
            continue

        if str(item) not in keep_dirs:
            try:
                shutil.rmtree(item)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎙️ Voice Settings")

    voice_map = {
        "🇺🇸 AF Heart": "af_heart",
        "🇺🇸 AF Bella": "af_bella",
        "🇺🇸 AF Nicole": "af_nicole",
        "🇺🇸 AF Sarah": "af_sarah",
        "🇺🇸 AF Sky": "af_sky",
        "🇺🇸 AM Adam": "am_adam",
        "🇺🇸 AM Michael": "am_michael",
        "🇬🇧 BF Emma": "bf_emma",
        "🇬🇧 BF Isabella": "bf_isabella",
        "🇬🇧 BM George": "bm_george",
        "🇬🇧 BM Fable": "bm_fable",
    }

    voice_name = st.selectbox(
        "Narrator",
        list(voice_map.keys()),
        index=10,
    )

    voice_key = voice_map[voice_name]

    speed = st.slider(
        "Speech speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )

    st.caption(
        "For Slumber Tales English, "
        "around 0.90–1.00 is usually a good starting point."
    )

    st.divider()

    st.subheader("🌙 Ambient Bed")

    ambient_enabled = st.checkbox(
        "Enable ambient background",
        value=False,
    )

    ambient_volume = st.slider(
        "Ambient volume",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        disabled=not ambient_enabled,
    )

    st.divider()

    st.subheader("⚙️ Chunking")

    st.write(
        f"Target chunk size: "
        f"**{TARGET_WORDS_PER_CHUNK} words**"
    )

    st.caption(
        "This is approximately 4–5 minutes of narration "
        "at a calm speaking pace."
    )

    st.divider()

    st.subheader("🧹 Job Controls")

    if st.button(
        "Start Fresh",
        use_container_width=True,
    ):
        current_job = (
            st.session_state.current_job
        )

        if current_job:
            work_dir = current_job.get(
                "work_dir"
            )

            if work_dir:
                try:
                    shutil.rmtree(work_dir)
                except Exception:
                    pass

        st.session_state.current_job = None

        st.rerun()


# ============================================================
# MAIN SCRIPT AREA
# ============================================================

st.header("📜 Narration Script")

script = st.text_area(
    "Paste your story here",
    height=420,
    placeholder=(
        "Paste your bedtime story here...\n\n"
        "For example:\n"
        "Welcome to Slumber Tales English. "
        "Tonight, we are going to visit a little cottage "
        "at the edge of a quiet forest..."
    ),
    label_visibility="collapsed",
)

normalized_script = normalize_script(script)

word_count = (
    len(normalized_script.split())
    if normalized_script
    else 0
)

character_count = len(normalized_script)

estimated_minutes = (
    word_count / 120
    if word_count
    else 0
)

chunks_preview = (
    split_script_into_chunks(
        normalized_script
    )
    if normalized_script
    else []
)

chunk_count = len(chunks_preview)


# ============================================================
# SCRIPT STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Words",
        f"{word_count:,}",
    )

with col2:
    st.metric(
        "Characters",
        f"{character_count:,}",
    )

with col3:
    st.metric(
        "Estimated duration",
        f"{estimated_minutes:.1f} min",
    )

with col4:
    st.metric(
        "Audio chunks",
        f"{chunk_count}",
    )


if chunk_count > 0:
    st.caption(
        f"Your script will be processed as approximately "
        f"{chunk_count} independent chunks."
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🎙️ Generate Narration",
    type="primary",
    use_container_width=True,
    disabled=not bool(normalized_script),
)


# ============================================================
# GENERATION PIPELINE
# ============================================================

if generate_button:

    if word_count < 5:
        st.error(
            "Please enter a longer script."
        )
        st.stop()

    # --------------------------------------------------------
    # Create deterministic job ID
    # --------------------------------------------------------

    job_id = make_job_id(
        normalized_script,
        voice_key,
        speed,
        ambient_enabled,
        ambient_volume,
    )

    job_dir = create_job_directory(
        job_id
    )

    chunks_dir = (
        job_dir / "chunks"
    )

    final_wav_path = (
        job_dir / "final_narration.wav"
    )

    final_mp3_path = (
        job_dir / "final_narration.mp3"
    )

    # --------------------------------------------------------
    # Save the current job metadata only.
    # No audio bytes are placed in session_state.
    # --------------------------------------------------------

    st.session_state.current_job = {
        "job_id": job_id,
        "work_dir": str(job_dir),
        "voice": voice_key,
        "voice_name": voice_name,
        "speed": speed,
        "ambient_enabled": ambient_enabled,
        "ambient_volume": ambient_volume,
        "word_count": word_count,
        "total_chunks": chunk_count,
        "completed_chunks": 0,
    }

    # --------------------------------------------------------
    # Load Kokoro once
    # --------------------------------------------------------

    try:
        kokoro = get_kokoro_engine()

    except Exception as exc:
        st.error(
            "Could not load the Kokoro model."
        )
        st.exception(exc)
        st.stop()

    # --------------------------------------------------------
    # Progress UI
    # --------------------------------------------------------

    progress = st.progress(
        0,
        text="Preparing narration...",
    )

    status_box = st.empty()

    # --------------------------------------------------------
    # Generate chunks
    # --------------------------------------------------------

    try:

        for index, chunk_text in enumerate(
            chunks_preview,
            start=1,
        ):

            chunk_path = get_chunk_path(
                job_dir,
                index,
            )

            # ----------------------------------------------
            # RECOVERY CHECK
            # ----------------------------------------------

            if chunk_is_complete(
                chunk_path
            ):

                status_box.markdown(
                    f'<div class="status-box">'
                    f"♻️ Chunk <b>{index}</b> / "
                    f"<b>{chunk_count}</b> already exists. "
                    f"Skipping regeneration."
                    f"</div>",
                    unsafe_allow_html=True,
                )

                progress.progress(
                    index / chunk_count,
                    text=(
                        f"Recovered chunk "
                        f"{index}/{chunk_count}"
                    ),
                )

                continue

            # ----------------------------------------------
            # Generate this chunk
            # ----------------------------------------------

            status_box.markdown(
                f'<div class="status-box">'
                f"🎙️ Generating chunk "
                f"<b>{index}</b> / "
                f"<b>{chunk_count}</b>..."
                f"</div>",
                unsafe_allow_html=True,
            )

            progress.progress(
                (index - 1) / chunk_count,
                text=(
                    f"Generating chunk "
                    f"{index}/{chunk_count}"
                ),
            )

            generate_chunk(
                kokoro=kokoro,
                text=chunk_text,
                voice=voice_key,
                speed=speed,
                output_path=chunk_path,
                ambient_enabled=ambient_enabled,
                ambient_volume=ambient_volume,
            )

            # ----------------------------------------------
            # Force garbage collection after every chunk
            # ----------------------------------------------

            gc.collect()

            progress.progress(
                index / chunk_count,
                text=(
                    f"Completed chunk "
                    f"{index}/{chunk_count}"
                ),
            )

        # ----------------------------------------------------
        # Verify every chunk
        # ----------------------------------------------------

        chunk_paths = []

        for index in range(
            1,
            chunk_count + 1,
        ):

            path = get_chunk_path(
                job_dir,
                index,
            )

            if not chunk_is_complete(path):
                raise RuntimeError(
                    f"Chunk {index} is missing or invalid."
                )

            chunk_paths.append(path)

        # ----------------------------------------------------
        # Combine WAV files
        # ----------------------------------------------------

        status_box.markdown(
            '<div class="status-box">'
            "🔗 Combining completed chunks into one WAV..."
            "</div>",
            unsafe_allow_html=True,
        )

        progress.progress(
            0.90,
            text="Combining audio chunks...",
        )

        combine_wav_files(
            chunk_paths,
            final_wav_path,
        )

        gc.collect()

        # ----------------------------------------------------
        # Convert final WAV to MP3
        # ----------------------------------------------------

        status_box.markdown(
            '<div class="status-box">'
            "🎧 Creating MP3 version..."
            "</div>",
            unsafe_allow_html=True,
        )

        progress.progress(
            0.96,
            text="Creating MP3...",
        )

        try:

            convert_wav_to_mp3(
                final_wav_path,
                final_mp3_path,
            )

            mp3_available = (
                final_mp3_path.exists()
            )

        except Exception as exc:

            mp3_available = False

            st.warning(
                "The WAV was created successfully, "
                "but MP3 conversion failed."
            )

            st.caption(
                str(exc)
            )

        # ----------------------------------------------------
        # Final progress
        # ----------------------------------------------------

        progress.progress(
            1.0,
            text="Narration complete!",
        )

        status_box.success(
            f"✅ Finished {chunk_count} chunks "
            f"and created the final narration."
        )

        # ----------------------------------------------------
        # Store ONLY metadata in session_state
        # ----------------------------------------------------

        completed_job = {
            "job_id": job_id,
            "work_dir": str(job_dir),
            "voice": voice_key,
            "voice_name": voice_name,
            "speed": speed,
            "ambient_enabled": ambient_enabled,
            "ambient_volume": ambient_volume,
            "word_count": word_count,
            "total_chunks": chunk_count,
            "completed_chunks": chunk_count,
            "wav_path": str(final_wav_path),
            "mp3_path": (
                str(final_mp3_path)
                if mp3_available
                else None
            ),
        }

        st.session_state.current_job = (
            completed_job
        )

        add_history_item(
            completed_job
        )

        cleanup_history_dirs()

    except Exception as exc:

        st.error(
            "Generation stopped."
        )

        st.exception(exc)

        completed = count_completed_chunks(
            job_dir,
            chunk_count,
        )

        st.info(
            f"Recovery information: "
            f"{completed}/{chunk_count} chunks "
            f"are already complete."
        )

        st.warning(
            "You can press Generate Narration again "
            "with the same settings. Existing completed "
            "chunks will be skipped."
        )


# ============================================================
# CURRENT RESULT
# ============================================================

current_job = (
    st.session_state.current_job
)

if current_job:

    wav_path = current_job.get(
        "wav_path"
    )

    mp3_path = current_job.get(
        "mp3_path"
    )

    work_dir = current_job.get(
        "work_dir"
    )

    st.divider()

    st.header("🎧 Current Narration")

    if wav_path and os.path.exists(wav_path):

        st.subheader("WAV")

        st.audio(
            wav_path,
            format="audio/wav",
        )

        wav_size_mb = (
            os.path.getsize(wav_path)
            / (1024 * 1024)
        )

        st.caption(
            f"WAV size: {wav_size_mb:.1f} MB"
        )

        with open(
            wav_path,
            "rb",
        ) as wav_file:

            st.download_button(
                "⬇️ Download WAV",
                data=wav_file,
                file_name="slumber_tales_narration.wav",
                mime="audio/wav",
                use_container_width=True,
                key="download_wav_current",
            )

    if mp3_path and os.path.exists(mp3_path):

        st.subheader("MP3")

        st.audio(
            mp3_path,
            format="audio/mpeg",
        )

        mp3_size_mb = (
            os.path.getsize(mp3_path)
            / (1024 * 1024)
        )

        st.caption(
            f"MP3 size: {mp3_size_mb:.1f} MB"
        )

        with open(
            mp3_path,
            "rb",
        ) as mp3_file:

            st.download_button(
                "⬇️ Download MP3",
                data=mp3_file,
                file_name="slumber_tales_narration.mp3",
                mime="audio/mpeg",
                use_container_width=True,
                key="download_mp3_current",
            )

    if work_dir:

        st.caption(
            "Generated audio is stored on the server "
            "as files rather than inside Streamlit session state."
        )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.header("🕘 Recent Jobs")

    for number, item in enumerate(
        st.session_state.history,
        start=1,
    ):

        job_wav = item.get(
            "wav_path"
        )

        job_mp3 = item.get(
            "mp3_path"
        )

        label = (
            f"{number}. "
            f"{item.get('voice_name', 'Kokoro')} — "
            f"{item.get('word_count', 0):,} words — "
            f"{item.get('total_chunks', 0)} chunks"
        )

        with st.expander(label):

            st.write(
                f"**Voice:** "
                f"{item.get('voice_name', '-')}"
            )

            st.write(
                f"**Speed:** "
                f"{item.get('speed', '-')}"
            )

            st.write(
                f"**Words:** "
                f"{item.get('word_count', 0):,}"
            )

            st.write(
                f"**Chunks:** "
                f"{item.get('total_chunks', 0)}"
            )

            if (
                job_wav
                and os.path.exists(job_wav)
            ):

                st.audio(
                    job_wav,
                    format="audio/wav",
                )

                with open(
                    job_wav,
                    "rb",
                ) as wav_file:

                    st.download_button(
                        "⬇️ WAV",
                        data=wav_file,
                        file_name=(
                            "slumber_tales_narration.wav"
                        ),
                        mime="audio/wav",
                        key=(
                            f"history_wav_{number}_"
                            f"{item.get('job_id')}"
                        ),
                    )

            if (
                job_mp3
                and os.path.exists(job_mp3)
            ):

                with open(
                    job_mp3,
                    "rb",
                ) as mp3_file:

                    st.download_button(
                        "⬇️ MP3",
                        data=mp3_file,
                        file_name=(
                            "slumber_tales_narration.mp3"
                        ),
                        mime="audio/mpeg",
                        key=(
                            f"history_mp3_{number}_"
                            f"{item.get('job_id')}"
                        ),
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lenchos Audio Studio • Kokoro ONNX • "
    "Chunked generation with disk-based recovery"
)
