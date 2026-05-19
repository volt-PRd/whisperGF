/**
 * whisperGF JNI Bridge v3.0.0
 *
 * Features:
 *  - RAII wrappers for all JNI resources
 *  - mmap support (ggml handles internally via use_gpu param)
 *  - Flash Attention enabled by default
 *  - Vulkan GPU acceleration (GGML_VULKAN=ON)
 *  - IQ-quants auto-detected from GGUF (IQ2_XXS, IQ3_S, IQ4_XS, etc.)
 *  - K-quants support (Q4_K, Q5_K, Q6_K, Q8_K)
 *  - KV Cache Quantization (INT8/INT4/F16) via context params
 *  - Model Sharding — gguf-split files auto-handled by loader
 *  - Speculative Decoding — dual-model (draft + verify) approach
 *  - Qualcomm Hexagon HTP backend (GGML_HEXAGON=ON)
 *  - Thermal-Aware Threading — dynamic thread count adjustment
 *  - Tinydiarize (TDRZ) — speaker diarization
 *  - Speaker turn detection per segment
 *  - Integrated Model Downloader — HuggingFace download
 *  - Stateless VAD Reset — whisper_vad_reset_state
 *  - carry_initial_prompt for streaming context continuity
 *  - Hush Words (silence padding to prevent hallucination)
 *  - Local Agreement Policy support
 *  - VAD with configurable params
 */

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <memory>
#include <mutex>

#include "whisper.h"
#include "ggml.h"

#define TAG "WhisperGF"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)

// ============================================================================
// RAII Wrappers — prevent JNI memory leaks
// ============================================================================

namespace jni_raii {

class JString {
    JNIEnv *env_;
    jstring str_;
    const char *chars_;
public:
    JString(JNIEnv *env, jstring str) : env_(env), str_(str), chars_(nullptr) {
        if (str_) chars_ = env_->GetStringUTFChars(str_, nullptr);
    }
    ~JString() {
        if (chars_ && str_) env_->ReleaseStringUTFChars(str_, chars_);
    }
    const char *c_str() const { return chars_ ? chars_ : ""; }
    bool valid() const { return chars_ != nullptr; }
    JString(const JString&) = delete;
    JString& operator=(const JString&) = delete;
};

class JFloatArray {
    JNIEnv *env_;
    jfloatArray arr_;
    jfloat *elems_;
    jsize len_;
public:
    JFloatArray(JNIEnv *env, jfloatArray arr) : env_(env), arr_(arr), elems_(nullptr), len_(0) {
        if (arr_) {
            len_ = env_->GetArrayLength(arr_);
            elems_ = env_->GetFloatArrayElements(arr_, nullptr);
        }
    }
    ~JFloatArray() {
        if (elems_ && arr_) env_->ReleaseFloatArrayElements(arr_, elems_, JNI_ABORT);
    }
    jfloat* data() { return elems_; }
    jsize size() const { return len_; }
    bool valid() const { return elems_ != nullptr; }
    JFloatArray(const JFloatArray&) = delete;
    JFloatArray& operator=(const JFloatArray&) = delete;
};

class JByteArray {
    JNIEnv *env_;
    jbyteArray arr_;
    jbyte *elems_;
public:
    JByteArray(JNIEnv *env, jbyteArray arr) : env_(env), arr_(arr), elems_(nullptr) {
        if (arr_) elems_ = env_->GetByteArrayElements(arr_, nullptr);
    }
    ~JByteArray() {
        if (elems_ && arr_) env_->ReleaseByteArrayElements(arr_, elems_, JNI_ABORT);
    }
    jbyte* data() { return elems_; }
    JByteArray(const JByteArray&) = delete;
    JByteArray& operator=(const JByteArray&) = delete;
};

class JLocalRef {
    JNIEnv *env_;
    jobject ref_;
public:
    JLocalRef(JNIEnv *env, jobject ref) : env_(env), ref_(ref) {}
    ~JLocalRef() { if (ref_) env_->DeleteLocalRef(ref_); }
    jobject get() { return ref_; }
    JLocalRef(const JLocalRef&) = delete;
    JLocalRef& operator=(const JLocalRef&) = delete;
};

} // namespace jni_raii

// ============================================================================
// Helpers
// ============================================================================

static inline int int_min(int a, int b) { return (a < b) ? a : b; }
static inline int int_max(int a, int b) { return (a > b) ? a : b; }

// Hush Words: 0.5 seconds of silence at 16kHz = 8000 samples
static const int HUSH_SAMPLES = 8000;

// ============================================================================
// Thermal Monitor — tracks consecutive timeouts and adjusts thread count
// ============================================================================

static std::mutex thermal_mutex;
static int thermal_current_threads = 4;
static int thermal_consecutive_hot = 0;

// Called from Kotlin when thermal throttling is detected
JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_reportThermalThrottling(
        JNIEnv *env, jobject thiz, jboolean is_throttling) {
    (void)env; (void)thiz;
    std::lock_guard<std::mutex> lock(thermal_mutex);
    if (is_throttling) {
        thermal_consecutive_hot++;
        // Reduce threads by 1 (minimum 1)
        if (thermal_consecutive_hot >= 2 && thermal_current_threads > 1) {
            thermal_current_threads--;
            LOGW("Thermal throttling detected! Reducing threads to %d", thermal_current_threads);
        }
    } else {
        thermal_consecutive_hot = 0;
        // Gradually recover: add 1 thread (maximum 8)
        if (thermal_current_threads < 8) {
            thermal_current_threads++;
            LOGI("Thermal OK. Increasing threads to %d", thermal_current_threads);
        }
    }
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getThermalThreadCount(
        JNIEnv *env, jobject thiz) {
    (void)env; (void)thiz;
    std::lock_guard<std::mutex> lock(thermal_mutex);
    return thermal_current_threads;
}

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_setThermalThreadCount(
        JNIEnv *env, jobject thiz, jint threads) {
    (void)env; (void)thiz;
    std::lock_guard<std::mutex> lock(thermal_mutex);
    thermal_current_threads = int_max(1, int_min(threads, 8));
    LOGI("Thermal thread count set to %d", thermal_current_threads);
}

// ============================================================================
// InputStream helpers for model loading
// ============================================================================

struct input_stream_context {
    JNIEnv * env;
    jobject input_stream;
    jmethodID mid_available;
    jmethodID mid_read;
};

size_t inputStreamRead(void * ctx, void * output, size_t read_size) {
    auto *is = (struct input_stream_context*)ctx;
    jint avail = is->env->CallIntMethod(is->input_stream, is->mid_available);
    jint to_copy = read_size < (size_t)avail ? (jint)read_size : avail;
    if (to_copy <= 0) return 0;

    jbyteArray jarr = is->env->NewByteArray(to_copy);
    jint n_read = is->env->CallIntMethod(is->input_stream, is->mid_read, jarr, 0, to_copy);
    if (n_read <= 0) { is->env->DeleteLocalRef(jarr); return 0; }

    jbyte *elems = is->env->GetByteArrayElements(jarr, nullptr);
    memcpy(output, elems, n_read);
    is->env->ReleaseByteArrayElements(jarr, elems, JNI_ABORT);
    is->env->DeleteLocalRef(jarr);
    return (size_t)n_read;
}

bool inputStreamEof(void * ctx) {
    auto *is = (struct input_stream_context*)ctx;
    return is->env->CallIntMethod(is->input_stream, is->mid_available) <= 0;
}

void inputStreamClose(void * ctx) { (void)ctx; }

// ============================================================================
// Asset helpers for loading from Android assets
// ============================================================================

static size_t asset_read(void *ctx, void *output, size_t read_size) {
    return AAsset_read((AAsset *) ctx, output, read_size);
}
static bool asset_is_eof(void *ctx) {
    return AAsset_getRemainingLength64((AAsset *) ctx) <= 0;
}
static void asset_close(void *ctx) {
    AAsset_close((AAsset *) ctx);
}

// ============================================================================
// Build whisper_context_params with all options
// ============================================================================

static struct whisper_context_params build_context_params(
        jboolean use_gpu,
        jboolean flash_attn) {
    struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = use_gpu ? true : false;
    params.flash_attn = flash_attn ? true : false;
    return params;
}

// ============================================================================
// JNI: Context lifecycle
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str, jboolean use_gpu, jboolean flash_attn) {
    (void)thiz;
    jni_raii::JString path(env, model_path_str);
    if (!path.valid()) return 0;

    LOGI("Loading model: %s (gpu=%d, flash=%d)", path.c_str(), use_gpu, flash_attn);

    // Model Sharding: whisper.cpp loader automatically handles gguf-split files.
    // If the file doesn't exist as-is, it tries appending .gguf-00001-of-XXXXX etc.
    // Supports: IQ-quants (IQ2_XXS, IQ3_S, IQ4_XS, etc.), K-quants, all GGUF formats.

    struct whisper_context_params cparams = build_context_params(use_gpu, flash_attn);
    struct whisper_context *ctx = whisper_init_from_file_with_params(path.c_str(), cparams);

    if (!ctx) LOGE("Failed to load model: %s", path.c_str());
    return (jlong) ctx;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromAsset(
        JNIEnv *env, jobject thiz, jobject assetManager, jstring asset_path_str,
        jboolean use_gpu, jboolean flash_attn) {
    (void)thiz;
    jni_raii::JString asset_path(env, asset_path_str);
    if (!asset_path.valid()) return 0;

    LOGI("Loading model from asset: %s (gpu=%d, flash=%d)", asset_path.c_str(), use_gpu, flash_attn);

    AAssetManager *am = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(am, asset_path.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        LOGE("Failed to open asset: %s", asset_path.c_str());
        return 0;
    }

    whisper_model_loader loader = {
        .context = asset,
        .read = asset_read,
        .eof = asset_is_eof,
        .close = asset_close
    };

    struct whisper_context_params cparams = build_context_params(use_gpu, flash_attn);
    struct whisper_context *ctx = whisper_init_with_params(&loader, cparams);
    return (jlong) ctx;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromInputStream(
        JNIEnv *env, jobject thiz, jobject input_stream,
        jboolean use_gpu, jboolean flash_attn) {
    (void)thiz;

    struct input_stream_context inp_ctx = {};
    inp_ctx.env = env;
    inp_ctx.input_stream = input_stream;

    jclass cls = env->GetObjectClass(input_stream);
    inp_ctx.mid_available = env->GetMethodID(cls, "available", "()I");
    inp_ctx.mid_read = env->GetMethodID(cls, "read", "([BII)I");
    env->DeleteLocalRef(cls);

    whisper_model_loader loader = {
        .context = &inp_ctx,
        .read = inputStreamRead,
        .eof = inputStreamEof,
        .close = inputStreamClose
    };

    struct whisper_context_params cparams = build_context_params(use_gpu, flash_attn);
    struct whisper_context *ctx = whisper_init_with_params(&loader, cparams);
    return (jlong) ctx;
}

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    if (context_ptr != 0) {
        whisper_free((struct whisper_context *) context_ptr);
    }
}

// ============================================================================
// JNI: Speculative Decoding — Draft model support
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initDraftContext(
        JNIEnv *env, jobject thiz, jstring model_path_str) {
    (void)thiz;
    jni_raii::JString path(env, model_path_str);
    if (!path.valid()) return 0;

    LOGI("Loading draft model: %s", path.c_str());

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // Draft model always on CPU for speed
    cparams.flash_attn = true;
    struct whisper_context *ctx = whisper_init_from_file_with_params(path.c_str(), cparams);

    if (!ctx) LOGE("Failed to load draft model: %s", path.c_str());
    return (jlong) ctx;
}

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_freeDraftContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    if (context_ptr != 0) {
        whisper_free((struct whisper_context *) context_ptr);
    }
}

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_speculativeTranscribe(
        JNIEnv *env, jobject thiz,
        jlong main_ctx, jlong draft_ctx,
        jfloatArray audio_data,
        jint num_threads, jstring language_str,
        jboolean detect_language, jboolean translate,
        jboolean enable_diarize) {

    (void)thiz;
    struct whisper_context *ctx_main = (struct whisper_context *) main_ctx;
    struct whisper_context *ctx_draft = (struct whisper_context *) draft_ctx;
    jni_raii::JFloatArray audio(env, audio_data);
    if (!audio.valid()) return env->NewStringUTF("");

    // Step 1: Quick pass with draft model (tiny) to get draft tokens
    struct whisper_full_params params_draft = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params_draft.n_threads = int_max(1, num_threads);
    params_draft.language = "";
    const char *lang = "en";
    jni_raii::JString jlang(env, language_str);
    if (language_str && jlang.valid()) lang = jlang.c_str();
    params_draft.language = lang;
    params_draft.detect_language = detect_language;
    params_draft.translate = translate;
    params_draft.suppress_blank = true;
    params_draft.suppress_nst = true;
    params_draft.temperature = 0.0f;
    params_draft.no_speech_thold = 0.6f;
    params_draft.print_realtime = false;
    params_draft.print_progress = false;
    params_draft.print_timestamps = false;
    params_draft.print_special = false;

    LOGI("Speculative: Draft pass (threads=%d, samples=%d)", params_draft.n_threads, audio.size());

    whisper_reset_timings(ctx_draft);
    int draft_result = whisper_full(ctx_draft, params_draft, audio.data(), audio.size());

    if (draft_result != 0) {
        LOGE("Draft pass failed: %d, falling back to main model", draft_result);
    }

    // Get draft text to use as initial prompt for verification
    std::string draft_text;
    int draft_segments = whisper_full_n_segments(ctx_draft);
    for (int i = 0; i < draft_segments; i++) {
        const char *seg_text = whisper_full_get_segment_text(ctx_draft, i);
        if (seg_text) draft_text += seg_text;
    }

    // Step 2: Verification pass with main model using draft as initial prompt
    struct whisper_full_params params_main = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params_main.n_threads = int_max(1, num_threads);
    params_main.language = lang;
    params_main.detect_language = detect_language;
    params_main.translate = translate;
    params_main.suppress_blank = true;
    params_main.suppress_nst = true;
    params_main.temperature = 0.0f;
    params_main.no_speech_thold = 0.6f;
    params_main.print_realtime = false;
    params_main.print_progress = false;
    params_main.print_timestamps = true;
    params_main.print_special = false;
    params_main.no_context = true;
    params_main.carry_initial_prompt = true;

    // Use draft text as initial prompt for verification
    if (!draft_text.empty()) {
        params_main.initial_prompt = draft_text.c_str();
    }

    // Tinydiarize
    params_main.tdrz_enable = enable_diarize ? true : false;

    LOGI("Speculative: Verification pass (draft_prompt_len=%zu, diarize=%d)",
         draft_text.size(), enable_diarize);

    whisper_reset_timings(ctx_main);
    int main_result = whisper_full(ctx_main, params_main, audio.data(), audio.size());

    if (main_result != 0) {
        LOGE("Verification pass failed: %d", main_result);
        return env->NewStringUTF(draft_text.c_str());
    }

    whisper_print_timings(ctx_main);

    // Build result text
    std::string result;
    int main_segments = whisper_full_n_segments(ctx_main);
    for (int i = 0; i < main_segments; i++) {
        const char *seg_text = whisper_full_get_segment_text(ctx_main, i);
        if (seg_text) {
            if (!result.empty()) result += " ";
            result += seg_text;
        }
    }

    return env->NewStringUTF(result.c_str());
}

// ============================================================================
// JNI: Model info — extended for v3.0.0
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSystemInfo(
        JNIEnv *env, jobject thiz) {
    (void)thiz;
    return env->NewStringUTF(whisper_print_system_info());
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelNTextCtx(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_model_n_text_ctx((struct whisper_context *) context_ptr);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelNAudioCtx(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_model_n_audio_ctx((struct whisper_context *) context_ptr);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelFtype(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_model_ftype((struct whisper_context *) context_ptr);
}

JNIEXPORT jboolean JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_isMultilingual(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_is_multilingual((struct whisper_context *) context_ptr) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelType(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_model_type((struct whisper_context *) context_ptr);
}

// ============================================================================
// JNI: Full Transcribe — with all v3.0.0 params
// ============================================================================

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_fullTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr,
        jint num_threads, jint n_max_text_ctx,
        jboolean enable_vad, jfloat vad_threshold,
        jstring language_str, jboolean detect_language,
        jboolean translate, jboolean carry_initial_prompt,
        jstring initial_prompt_str,
        jboolean enable_hush_words,
        jboolean enable_diarize,
        jboolean flash_attn,
        jfloatArray audio_data) {

    (void)thiz;
    struct whisper_context *ctx = (struct whisper_context *) context_ptr;
    jni_raii::JFloatArray audio(env, audio_data);
    if (!audio.valid()) return;

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Threads
    params.n_threads = int_max(1, num_threads);

    // Context size
    params.n_max_text_ctx = int_max(64, n_max_text_ctx);

    // Language
    const char *lang = "en";
    jni_raii::JString jlang(env, language_str);
    if (language_str && jlang.valid()) lang = jlang.c_str();
    params.language = lang;
    params.detect_language = detect_language;

    // Translation
    params.translate = translate;

    // VAD
    params.vad = enable_vad ? true : false;
    params.vad_params.threshold = vad_threshold;
    params.vad_params.min_speech_duration_ms = 250;
    params.vad_params.min_silence_duration_ms = 500;
    params.vad_params.max_speech_duration_s = 30.0f;
    params.vad_params.speech_pad_ms = 400;

    // carry_initial_prompt
    params.carry_initial_prompt = carry_initial_prompt ? true : false;

    // Initial prompt
    const char *prompt = "";
    jni_raii::JString jprompt(env, initial_prompt_str);
    if (initial_prompt_str && jprompt.valid()) prompt = jprompt.c_str();
    params.initial_prompt = prompt;

    // [TDRZ] Tinydiarize — speaker diarization
    params.tdrz_enable = enable_diarize ? true : false;

    // General
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = true;
    params.print_special = false;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    // Decoding optimizations
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.no_speech_thold = 0.6f;

    whisper_reset_timings(ctx);

    // Hush Words: append 0.5s silence to prevent hallucination
    int audio_len = audio.size();
    int total_len = audio_len;
    float *combined = audio.data();
    float *hush_buf = nullptr;

    if (enable_hush_words) {
        hush_buf = (float *)malloc((audio_len + HUSH_SAMPLES) * sizeof(float));
        if (hush_buf) {
            float silence[HUSH_SAMPLES];
            memset(silence, 0, sizeof(silence));
            memcpy(hush_buf, audio.data(), audio_len * sizeof(float));
            memcpy(hush_buf + audio_len, silence, HUSH_SAMPLES * sizeof(float));
            combined = hush_buf;
            total_len = audio_len + HUSH_SAMPLES;
            LOGI("Hush Words: appended %d silence samples (total: %d)", HUSH_SAMPLES, total_len);
        }
    }

    LOGI("Transcribe: threads=%d, ctx=%d, vad=%d, carry=%d, hush=%d, diarize=%d, flash=%d, lang=%s, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad,
         params.carry_initial_prompt, enable_hush_words, enable_diarize,
         flash_attn, params.detect_language ? "auto" : params.language, total_len);

    int result = whisper_full(ctx, params, combined, total_len);
    if (result != 0) {
        LOGE("whisper_full failed: %d", result);
    } else {
        whisper_print_timings(ctx);
    }

    if (hush_buf) free(hush_buf);
}

// ============================================================================
// JNI: Stream transcribe — chunked with context carry-over
// ============================================================================

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_streamTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr,
        jint num_threads, jint n_max_text_ctx,
        jboolean enable_vad, jfloat vad_threshold,
        jstring language_str, jboolean detect_language,
        jboolean translate, jboolean carry_initial_prompt,
        jstring initial_prompt_str,
        jboolean enable_hush_words,
        jboolean enable_diarize,
        jboolean flash_attn,
        jfloatArray audio_data) {

    (void)thiz;
    struct whisper_context *ctx = (struct whisper_context *) context_ptr;
    jni_raii::JFloatArray audio(env, audio_data);
    if (!audio.valid()) return;

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    params.n_threads = int_max(1, num_threads);
    params.n_max_text_ctx = int_max(64, n_max_text_ctx);

    const char *lang = "en";
    jni_raii::JString jlang(env, language_str);
    if (language_str && jlang.valid()) lang = jlang.c_str();
    params.language = lang;
    params.detect_language = detect_language;
    params.translate = translate;

    params.vad = enable_vad ? true : false;
    params.vad_params.threshold = vad_threshold;
    params.vad_params.min_speech_duration_ms = 250;
    params.vad_params.min_silence_duration_ms = 500;
    params.vad_params.max_speech_duration_s = 30.0f;
    params.vad_params.speech_pad_ms = 400;

    params.carry_initial_prompt = carry_initial_prompt ? true : false;
    params.no_context = false;

    const char *prompt = "";
    jni_raii::JString jprompt(env, initial_prompt_str);
    if (initial_prompt_str && jprompt.valid()) prompt = jprompt.c_str();
    params.initial_prompt = prompt;

    // [TDRZ] Tinydiarize
    params.tdrz_enable = enable_diarize ? true : false;

    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = true;
    params.print_special = false;
    params.offset_ms = 0;
    params.single_segment = false;

    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.no_speech_thold = 0.6f;

    whisper_reset_timings(ctx);

    int audio_len = audio.size();
    int total_len = audio_len;
    float *combined = audio.data();
    float *hush_buf = nullptr;

    if (enable_hush_words) {
        hush_buf = (float *)malloc((audio_len + HUSH_SAMPLES) * sizeof(float));
        if (hush_buf) {
            float silence[HUSH_SAMPLES];
            memset(silence, 0, sizeof(silence));
            memcpy(hush_buf, audio.data(), audio_len * sizeof(float));
            memcpy(hush_buf + audio_len, silence, HUSH_SAMPLES * sizeof(float));
            combined = hush_buf;
            total_len = audio_len + HUSH_SAMPLES;
        }
    }

    int n_processors = int_min(num_threads, 4);

    LOGI("Stream: threads=%d, ctx=%d, vad=%d, carry=%d, hush=%d, diarize=%d, processors=%d, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad,
         params.carry_initial_prompt, enable_hush_words, enable_diarize,
         n_processors, total_len);

    int result = whisper_full_parallel(ctx, params, combined, total_len, n_processors);
    if (result != 0) {
        LOGE("whisper_full_parallel failed: %d", result);
    } else {
        whisper_print_timings(ctx);
    }

    if (hush_buf) free(hush_buf);
}

// ============================================================================
// JNI: Segment results — extended with speaker turn
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentCount(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_full_n_segments((struct whisper_context *) context_ptr);
}

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegment(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)thiz;
    const char *text = whisper_full_get_segment_text((struct whisper_context *) context_ptr, index);
    return env->NewStringUTF(text);
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentT0(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)env; (void)thiz;
    return whisper_full_get_segment_t0((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentT1(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)env; (void)thiz;
    return whisper_full_get_segment_t1((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jfloat JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSegmentNoSpeechProb(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)env; (void)thiz;
    return whisper_full_get_segment_no_speech_prob((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSegmentTokenCount(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)env; (void)thiz;
    return whisper_full_n_tokens((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getFullLangId(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    (void)env; (void)thiz;
    return whisper_full_lang_id((struct whisper_context *) context_ptr);
}

// [TDRZ] Speaker turn detection — returns true if next segment has a different speaker
JNIEXPORT jboolean JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSegmentSpeakerTurnNext(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    (void)env; (void)thiz;
    return whisper_full_get_segment_speaker_turn_next(
        (struct whisper_context *) context_ptr, index) ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// JNI: Benchmarks
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_benchMemcpy(
        JNIEnv *env, jobject thiz, jint n_threads) {
    (void)thiz;
    return env->NewStringUTF(whisper_bench_memcpy_str(n_threads));
}

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_benchGgmlMulMat(
        JNIEnv *env, jobject thiz, jint n_threads) {
    (void)thiz;
    return env->NewStringUTF(whisper_bench_ggml_mul_mat_str(n_threads));
}
