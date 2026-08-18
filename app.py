import os
import re
import tempfile
import urllib.request
import subprocess
import shutil

import numpy as np
import soundfile as sf
import streamlit as st
from kokoro_onnx import Kokoro


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lenchos Audio Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# 3. CUSTOM THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05) !important;
}

section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

.hero-container {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: #ffffff;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.025em;
}

.lencho-highlight {
    color: #0f172a;
    font-weight: 900;
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 500;
}

div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background-color: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    padding: 1.5rem !important;
}

div[data-testid="stVerticalBlock"] > div[style*="border"] h1,
div[data-testid="stVerticalBlock"] > div[style*="border"] h2,
div[data-testid="stVerticalBlock"] > div[style*="border"] h3,
div[data-testid="stVerticalBlock"] > div[style*="border"] p,
div[data-testid="stVerticalBlock"] > div[style*="border"] span,
div[data-testid="stVerticalBlock"] > div[style*="border"] label {
    color: #0f172a !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-size: 0.9rem !important;
}

.stTextArea textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #0f172a !important;
    box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.1) !important;
}

div.stButton > button[kind="primary"] {
    width: 100%;
    background-color: #0f172a !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1) !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #334155 !important;
}

section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. KOKORO ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_kokoro_engine():

    model_path = "kokoro-v1.0.onnx"
    voices_path = "voices-v1.0.bin"

    if not os.path.exists(model_path):
        with st.spinner("Downloading Kokoro ONNX model..."):
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                model_path
            )

    if not os.path.exists(voices_path):
        with st.spinner("Downloading voice configuration..."):
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                voices_path
            )

    return Kokoro(model_path, voices_path)


# ============================================================
# 5. BACKGROUND AUDIO
# ============================================================

@st.cache_resource(show_spinner=False)
def get_background_track():

    bg_path = "ambient_bed.wav"

    if not os.path.exists(bg_path):
        try:
            urllib.request.urlretrieve(
                "https://github.com/rafaelreis-io/rafaelreis-io/raw/main/ambient.wav",
                bg_path
            )
        except Exception:
            pass

    return bg_path


# ============================================================
# 6. SPLIT SCRIPT INTO ~5-MINUTE CHUNKS
# ============================================================

TARGET_WORDS_PER_CHUNK = 600


def split_into_sentences(text):
    """
    Split text at sentence boundaries while keeping punctuation.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [s.strip() for s in sentences if s.strip()]


def split_script_into_chunks(text, target_words=TARGET_WORDS_PER_CHUNK):

    sentences = split_into_sentences(text)

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(sentence.split())

        # Very long sentence:
        # keep it intact rather than cutting the sentence.
        if current and current_words + sentence_words > target_words:

            chunks.append(" ".join(current))

            current = [sentence]
            current_words = sentence_words

        else:

            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# 7. AUDIO MIXING
# ============================================================

def mix_audio_beds(
    voice_samples,
    sample_rate,
    bg_path,
    volume=0.15
):

    if not os.path.exists(bg_path):
        return voice_samples

    try:

        bg_samples, bg_rate = sf.read(bg_path)

        if len(bg_samples.shape) > 1:
            bg_samples = np.mean(bg_samples, axis=1)

        if len(voice_samples.shape) > 1:
            voice_samples = np.mean(
                voice_samples,
                axis=1
            )

        # Resample is intentionally avoided here.
        # The downloaded bed should use the same sample rate
        # as Kokoro whenever possible.

        if len(bg_samples) == 0:
            return voice_samples

        if len(bg_samples) < len(voice_samples):

            repeats = int(
                np.ceil(
                    len(voice_samples) /
                    len(bg_samples)
                )
            )

            bg_samples = np.tile(
                bg_samples,
                repeats
            )

        bg_samples = bg_samples[:len(voice_samples)]

        mixed = (
            voice_samples +
            (bg_samples * volume)
        )

        return np.clip(
            mixed,
            -1.0,
            1.0
        )

    except Exception:

        return voice_samples


# ============================================================
# 8. COMBINE WAV CHUNKS WITHOUT LOADING EVERYTHING INTO RAM
# ============================================================

def combine_wav_files(chunk_paths, output_path):

    if not chunk_paths:
        raise RuntimeError("No audio chunks were generated.")

    first_info = sf.info(chunk_paths[0])

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=first_info.samplerate,
        channels=1,
        subtype="PCM_16"
    ) as output_file:

        for chunk_path in chunk_paths:

            data, rate = sf.read(
                chunk_path,
                dtype="float32"
            )

            if len(data.shape) > 1:
                data = np.mean(
                    data,
                    axis=1
                )

            if rate != first_info.samplerate:
                raise RuntimeError(
                    f"Sample-rate mismatch in {chunk_path}"
                )

            output_file.write(data)


# ============================================================
# 9. CREATE MP3 USING IMAGEIO-FFMPEG
# ============================================================

def convert_wav_to_mp3(wav_path, mp3_path):

    try:

        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            mp3_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                result.stderr[-2000:]
            )

        return True

    except Exception as e:

        raise RuntimeError(
            f"MP3 conversion failed: {e}"
        )


# ============================================================
# 10. VOICE MAP
# ============================================================

VOICE_MAP = {

    "🇺🇸 Beza (American Female - Warm)":
        "af_heart",

    "🇺🇸 Birikti (American Female - Soft)":
        "af_bella",

    "🇺🇸 Demoze (American Female - Clear)":
        "af_nicole",

    "🇺🇸 Lalise (American Female - News)":
        "af_sarah",

    "🇺🇸 Efrata (American Female - Casual)":
        "af_sky",

    "🇺🇸 Lencho (American Male - Deep)":
        "am_adam",

    "🇺🇸 Dego (American Male - Crisp)":
        "am_michael",

    "🇬🇧 Bontu (British Female - Professional)":
        "bf_emma",

    "🇬🇧 Hawi (British Female - Warm)":
        "bf_isabella",

    "🇬🇧 Lalisa (British Male - Expressive)":
        "bm_george",

    "🇬🇧 Lemi (British Male - Narration)":
        "bm_fable"
}


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Studio Settings")

    st.markdown(
        "Customize your voice engine parameters."
    )

    st.divider()

    voice_display_name = st.selectbox(
        "🎙️ Voice Persona",
        options=list(VOICE_MAP.keys()),
        index=10
    )

    if st.button("▶️ Preview Voice"):

        voice_key = VOICE_MAP.get(
            voice_display_name,
            "bm_fable"
        )

        preview_text = (
            "Hello! This is a quick preview "
            "of this voice persona."
        )

        with st.spinner("Generating preview..."):

            try:

                kokoro_engine = get_kokoro_engine()

                samples, sample_rate = kokoro_engine.create(
                    preview_text,
                    voice=voice_key,
                    speed=1.0,
                    lang="en-us"
                )

                if samples is not None and len(samples) > 0:

                    temp_preview = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav"
                    )

                    temp_preview.close()

                    sf.write(
                        temp_preview.name,
                        samples,
                        sample_rate
                    )

                    st.audio(
                        temp_preview.name,
                        format="audio/wav",
                        autoplay=True
                    )

            except Exception as e:

                st.error(
                    f"Could not generate preview: {e}"
                )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    speed = st.slider(
        "⚡ Speed Rate",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    st.divider()

    st.markdown(
        "### 🎵 Background Music Bed"
    )

    enable_bg = st.checkbox(
        "Enable Ambient Bed",
        value=False
    )

    bg_volume = st.slider(
        "Music Volume",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.05
    )

    st.divider()

    st.caption(
        "🚀 **Studio Engine:** Active."
    )


# ============================================================
# 12. HERO
# ============================================================

st.markdown
<div class="hero-container">

    <div class="hero-title">
        🎙️
        <span class="lencho-highlight">
            LENCHOS
        </span>
        AUDIO STUDIO
    </div>

    <div class="hero-subtitle">
        Why pay for ElevenLabs when the mastermind
        <span class="lencho-highlight">
            <b><i><u>Lencho Lemessa</u></i></b>
        </span>
        is architecting the future of open-source AI audio?
    </div>

</div>
 unsafe_allow_html=True


# ============================================================
# 13. SCRIPT EDITOR
# ============================================================

with st.container(border=True):

    st.subheader("📝 Script Editor")

    text_input = st.text_area(
        "Input Script",
        height=180,
        placeholder="Type or paste your text here...",
        label_visibility="collapsed"
    )

    char_count = len(text_input)

    word_count = (
        len(text_input.split())
        if text_input
        else 0
    )

    # Approximate 120 WPM
    est_sec = (
        round(word_count / (2.0 * speed))
        if word_count > 0
        else 0
    )

    col_stat1, col_stat2, col_stat3 = st.columns(3)

    with col_stat1:
        st.caption(
            f"**Characters:** `{char_count}`"
        )

    with col_stat2:
        st.caption(
            f"**Words:** `{word_count}`"
        )

    with col_stat3:
        minutes = est_sec // 60
        seconds = est_sec % 60

        st.caption(
            f"**Est. Duration:** `~{minutes}m {seconds}s`"
        )

    if word_count > 0:

        estimated_chunks = max(
            1,
            int(
                np.ceil(
                    word_count /
                    TARGET_WORDS_PER_CHUNK
                )
            )
        )

        st.info(
            f"🧩 This script will be generated in "
            f"approximately **{estimated_chunks} chunks** "
            f"of about 5 minutes each."
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    generate_btn = st.button(
        "✨ Generate Audio",
        type="primary"
    )


# ============================================================
# 14. GENERATION
# ============================================================

if generate_btn:

    if not text_input.strip():

        st.warning(
            "Please enter some text in the script editor first."
        )

    else:

        st.markdown(
            "<h3 style='color: #0f172a;'>"
            "🔊 Studio Render Output"
            "</h3>",
            unsafe_allow_html=True
        )

        with st.container(border=True):

            progress_bar = st.progress(
                0.0,
                text="Preparing script..."
            )

            status_box = st.empty()

            try:

                # ------------------------------------------------
                # SPLIT SCRIPT
                # ------------------------------------------------

                chunks = split_script_into_chunks(
                    text_input,
                    TARGET_WORDS_PER_CHUNK
                )

                total_chunks = len(chunks)

                status_box.info(
                    f"🧩 Script divided into "
                    f"**{total_chunks} chunks**."
                )

                # ------------------------------------------------
                # LOAD KOKORO
                # ------------------------------------------------

                progress_bar.progress(
                    0.03,
                    text="Loading Kokoro model..."
                )

                kokoro = get_kokoro_engine()

                voice_key = VOICE_MAP.get(
                    voice_display_name,
                    "bm_fable"
                )

                # ------------------------------------------------
                # TEMP WORKSPACE
                # ------------------------------------------------

                work_dir = tempfile.mkdtemp(
                    prefix="lencho_kokoro_"
                )

                chunk_paths = []

                # ------------------------------------------------
                # BACKGROUND
                # ------------------------------------------------

                bg_path = None

                if enable_bg:

                    progress_bar.progress(
                        0.05,
                        text="Preparing ambient bed..."
                    )

                    bg_path = get_background_track()

                # ------------------------------------------------
                # GENERATE CHUNKS
                # ------------------------------------------------

                for index, chunk_text in enumerate(chunks):

                    chunk_number = index + 1

                    progress_start = (
                        0.05 +
                        (
                            0.75 *
                            index /
                            total_chunks
                        )
                    )

                    progress_bar.progress(
                        progress_start,
                        text=(
                            f"🎙️ Generating chunk "
                            f"{chunk_number}/{total_chunks}..."
                        )
                    )

                    status_box.info(
                        f"**Chunk {chunk_number} of "
                        f"{total_chunks}**\n\n"
                        f"Words: `{len(chunk_text.split())}`\n\n"
                        f"Kokoro is synthesizing this section..."
                    )

                    # Generate only this chunk.
                    samples, sample_rate = kokoro.create(
                        chunk_text,
                        voice=voice_key,
                        speed=speed,
                        lang="en-us"
                    )

                    if samples is None or len(samples) == 0:

                        raise RuntimeError(
                            f"Kokoro returned no audio "
                            f"for chunk {chunk_number}."
                        )

                    # Convert stereo → mono
                    if len(samples.shape) > 1:

                        samples = np.mean(
                            samples,
                            axis=1
                        )

                    # ------------------------------------------------
                    # MIX AMBIENT BED FOR THIS CHUNK
                    # ------------------------------------------------

                    if enable_bg and bg_path:

                        samples = mix_audio_beds(
                            samples,
                            sample_rate,
                            bg_path,
                            volume=bg_volume
                        )

                    # ------------------------------------------------
                    # WRITE CHUNK TO DISK
                    # ------------------------------------------------

                    chunk_path = os.path.join(
                        work_dir,
                        f"chunk_{chunk_number:02d}.wav"
                    )

                    sf.write(
                        chunk_path,
                        samples,
                        sample_rate,
                        subtype="PCM_16"
                    )

                    chunk_paths.append(
                        chunk_path
                    )

                    # Release this chunk's NumPy array
                    del samples

                    progress_done = (
                        0.05 +
                        (
                            0.75 *
                            chunk_number /
                            total_chunks
                        )
                    )

                    progress_bar.progress(
                        progress_done,
                        text=(
                            f"✅ Chunk "
                            f"{chunk_number}/{total_chunks} complete"
                        )
                    )

                # ------------------------------------------------
                # COMBINE WAV
                # ------------------------------------------------

                progress_bar.progress(
                    0.84,
                    text="🔗 Combining audio chunks..."
                )

                status_box.info(
                    "🔗 Combining all Kokoro chunks into "
                    "one continuous WAV..."
                )

                final_wav = os.path.join(
                    work_dir,
                    "lencho_latera_voice.wav"
                )

                combine_wav_files(
                    chunk_paths,
                    final_wav
                )

                # ------------------------------------------------
                # CREATE MP3
                # ------------------------------------------------

                progress_bar.progress(
                    0.91,
                    text="🎵 Creating MP3..."
                )

                final_mp3 = os.path.join(
                    work_dir,
                    "lencho_latera_voice.mp3"
                )

                mp3_success = False

                try:

                    convert_wav_to_mp3(
                        final_wav,
                        final_mp3
                    )

                    mp3_success = True

                except Exception as mp3_error:

                    st.warning(
                        f"⚠️ WAV was created successfully, "
                        f"but MP3 conversion failed: "
                        f"{mp3_error}"
                    )

                # ------------------------------------------------
                # READ FINAL FILES FOR DOWNLOAD
                # ------------------------------------------------

                progress_bar.progress(
                    0.96,
                    text="Preparing downloads..."
                )

                with open(
                    final_wav,
                    "rb"
                ) as f:

                    wav_bytes = f.read()

                mp3_bytes = None

                if mp3_success:

                    with open(
                        final_mp3,
                        "rb"
                    ) as f:

                        mp3_bytes = f.read()

                # ------------------------------------------------
                # SAVE HISTORY METADATA
                # ------------------------------------------------

                history_item = {

                    "text": text_input,

                    "voice": voice_display_name,

                    "speed": speed,

                    "wav_bytes": wav_bytes,

                    "mp3_bytes": mp3_bytes,

                    "filename": (
                        "lencho_latera_voice.wav"
                    ),

                    "mp3_filename": (
                        "lencho_latera_voice.mp3"
                    ),

                    "chunks": total_chunks
                }

                # Only keep latest 2
                st.session_state.history.insert(
                    0,
                    history_item
                )

                if len(
                    st.session_state.history
                ) > 2:

                    st.session_state.history.pop()

                # ------------------------------------------------
                # COMPLETE
                # ------------------------------------------------

                progress_bar.progress(
                    1.0,
                    text="✅ Complete!"
                )

                status_box.success(
                    f"🎉 Finished! "
                    f"Your {total_chunks}-chunk narration "
                    f"has been combined into one audio file."
                )

                # ------------------------------------------------
                # PLAYER + DOWNLOADS
                # ------------------------------------------------

                col_audio, col_download = st.columns(
                    [3, 1]
                )

                with col_audio:

                    st.audio(
                        wav_bytes,
                        format="audio/wav"
                    )

                with col_download:

                    st.download_button(
                        label="📥 Download WAV",
                        data=wav_bytes,
                        file_name="lencho_latera_voice.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

                    if mp3_bytes:

                        st.download_button(
                            label="📥 Download MP3",
                            data=mp3_bytes,
                            file_name="lencho_latera_voice.mp3",
                            mime="audio/mpeg",
                            use_container_width=True
                        )

                st.caption(
                    f"Generated using {total_chunks} Kokoro chunks."
                )

            except Exception as e:

                progress_bar.empty()

                status_box.error(
                    "❌ Generation stopped."
                )

                st.error(
                    "⚠️ An internal error occurred during synthesis:"
                )

                st.exception(e)


# ============================================================
# 15. SESSION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color: #0f172a;'>"
        "📜 Recent Session Archive (Last 2)"
        "</h3>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        for idx, item in enumerate(
            st.session_state.history
        ):

            item_num = (
                len(st.session_state.history) -
                idx
            )

            st.markdown(
                f"**#{item_num} | Persona:** "
                f"`{item['voice']}` | "
                f"**Speed:** `{item['speed']}x`"
            )

            snippet = (
                item["text"][:100] + "..."
                if len(item["text"]) > 100
                else item["text"]
            )

            st.caption(
                f"**Script:** {snippet}"
            )

            st.caption(
                f"🧩 Generated in "
                f"**{item['chunks']} chunks**"
            )

            col_ha, col_hd = st.columns(
                [3, 1]
            )

            with col_ha:

                st.audio(
                    item["wav_bytes"],
                    format="audio/wav"
                )

            with col_hd:

                st.download_button(
                    label="📥 WAV",
                    data=item["wav_bytes"],
                    file_name=item["filename"],
                    mime="audio/wav",
                    key=f"history_wav_{idx}",
                    use_container_width=True
                )

                if item.get("mp3_bytes"):

                    st.download_button(
                        label="📥 MP3",
                        data=item["mp3_bytes"],
                        file_name=item["mp3_filename"],
                        mime="audio/mpeg",
                        key=f"history_mp3_{idx}",
                        use_container_width=True
                    )

            if idx < len(
                st.session_state.history
            ) - 1:

                st.divider()
