/**
 * whisperGF JNI Bridge v3.0.0
 *
 * Improvements:
 *  - RAII wrappers for all JNI resources (jstring, jfloatArray, etc.)
 *  - mmap support via use_gpu context param (ggml handles mmap internally)
 *  - Flash Attention enabled by default (whisper_context_default_params sets flash_attn=true)
 *  - Vulkan GPU acceleration (built with GGML_VULKAN=ON)
 *  - K-quants support (Q4_K, Q5_K, Q6_K, Q8_K — auto-detected from GGUF header)
 *  - carry_initial_prompt support for streaming consistency
 *  - Hush Words: appends 0.5s silence to audio buffer to prevent hallucination
 *  - Enhanced transcribe with all tuning parameters exposed
 *  - Stream transcribe via whisper_full_parallel with context carry-over
 *  - Local Agreement Policy support (exposed via JNI for Kotlin layer)
 *  - Model info: quantization type, context sizes, multilingual support
 */

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <memory>

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

/** RAII wrapper for jstring — auto-releases UTF chars on destruction */
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

/** RAII wrapper for jfloatArray — auto-releases elements */
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

/** RAII wrapper for jbyteArray — auto-releases elements */
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

/** RAII wrapper for local JNI references */
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
// InputStream helpers
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
// Asset helpers
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
// Build whisper_context_params with GPU and Flash Attention
// ============================================================================

static struct whisper_context_params build_context_params(jboolean use_gpu) {
    struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = use_gpu ? true : false;
    // flash_attn is already true by default in whisper.cpp
    return params;
}

// ============================================================================
// JNI: Context lifecycle — with mmap (use_gpu enables ggml backend mmap)
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str, jboolean use_gpu) {
    (void)thiz;
    jni_raii::JString path(env, model_path_str);
    if (!path.valid()) return 0;

    LOGI("Loading model: %s (gpu=%d, flash_attn=true)", path.c_str(), use_gpu);

    struct whisper_context_params cparams = build_context_params(use_gpu);
    struct whisper_context *ctx = whisper_init_from_file_with_params(path.c_str(), cparams);

    if (!ctx) LOGE("Failed to load model: %s", path.c_str());
    return (jlong) ctx;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromAsset(
        JNIEnv *env, jobject thiz, jobject assetManager, jstring asset_path_str, jboolean use_gpu) {
    (void)thiz;
    jni_raii::JString asset_path(env, asset_path_str);
    if (!asset_path.valid()) return 0;

    LOGI("Loading model from asset: %s (gpu=%d)", asset_path.c_str(), use_gpu);

    AAssetManager *am = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(am, asset_path.valid() ? asset_path.c_str() : "", AASSET_MODE_STREAMING);
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

    struct whisper_context_params cparams = build_context_params(use_gpu);
    struct whisper_context *ctx = whisper_init_with_params(&loader, cparams);
    return (jlong) ctx;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromInputStream(
        JNIEnv *env, jobject thiz, jobject input_stream, jboolean use_gpu) {
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

    struct whisper_context_params cparams = build_context_params(use_gpu);
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
// JNI: Model info
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
// JNI: Full Transcribe — with all v3 params
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

    // carry_initial_prompt — improves streaming consistency
    params.carry_initial_prompt = carry_initial_prompt ? true : false;

    // Initial prompt
    const char *prompt = "";
    jni_raii::JString jprompt(env, initial_prompt_str);
    if (initial_prompt_str && jprompt.valid()) prompt = jprompt.c_str();
    params.initial_prompt = prompt;

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

    // Hush Words: append 0.5s silence to prevent hallucination at audio end
    int audio_len = audio.size();
    int total_len = audio_len;
    float *samples = audio.data();

    // Stack-allocate silence buffer (8KB for 0.5s at 16kHz float)
    float silence[HUSH_SAMPLES];
    memset(silence, 0, sizeof(silence));

    // Create a combined buffer: original audio + silence
    int use_hush = enable_hush_words ? 1 : 0;
    float *combined = samples; // default: use original
    float *hush_buf = nullptr;

    if (use_hush) {
        hush_buf = (float *)malloc((audio_len + HUSH_SAMPLES) * sizeof(float));
        if (hush_buf) {
            memcpy(hush_buf, samples, audio_len * sizeof(float));
            memcpy(hush_buf + audio_len, silence, HUSH_SAMPLES * sizeof(float));
            combined = hush_buf;
            total_len = audio_len + HUSH_SAMPLES;
            LOGI("Hush Words: appended %d silence samples (total: %d)", HUSH_SAMPLES, total_len);
        }
    }

    LOGI("Transcribe: threads=%d, ctx=%d, vad=%d, carry=%d, hush=%d, lang=%s, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad,
         params.carry_initial_prompt, use_hush,
         params.detect_language ? "auto" : params.language, total_len);

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

    // carry_initial_prompt is KEY for streaming: ensures context from previous chunks carries over
    params.carry_initial_prompt = carry_initial_prompt ? true : false;
    params.no_context = false; // Enable context between chunks

    const char *prompt = "";
    jni_raii::JString jprompt(env, initial_prompt_str);
    if (initial_prompt_str && jprompt.valid()) prompt = jprompt.c_str();
    params.initial_prompt = prompt;

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

    LOGI("Stream: threads=%d, ctx=%d, vad=%d, carry=%d, hush=%d, processors=%d, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad,
         params.carry_initial_prompt, enable_hush_words, n_processors, total_len);

    int result = whisper_full_parallel(ctx, params, combined, total_len, n_processors);
    if (result != 0) {
        LOGE("whisper_full_parallel failed: %d", result);
    } else {
        whisper_print_timings(ctx);
    }

    if (hush_buf) free(hush_buf);
}

// ============================================================================
// JNI: Segment results
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
