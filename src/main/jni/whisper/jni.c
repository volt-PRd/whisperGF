#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <string.h>
#include "whisper.h"
#include "ggml.h"

#define UNUSED(x) (void)(x)
#define TAG "WhisperGF"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)

static inline int int_min(int a, int b) { return (a < b) ? a : b; }
static inline int int_max(int a, int b) { return (a > b) ? a : b; }

// ============================================================================
// Input Stream helpers (for loading models from InputStream)
// ============================================================================

struct input_stream_context {
    size_t offset;
    JNIEnv * env;
    jobject thiz;
    jobject input_stream;
    jmethodID mid_available;
    jmethodID mid_read;
};

size_t inputStreamRead(void * ctx, void * output, size_t read_size) {
    struct input_stream_context* is = (struct input_stream_context*)ctx;
    jint avail_size = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_available);
    jint size_to_copy = read_size < avail_size ? (jint)read_size : avail_size;
    if (size_to_copy <= 0) return 0;

    jbyteArray byte_array = (*is->env)->NewByteArray(is->env, size_to_copy);
    jint n_read = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_read, byte_array, 0, size_to_copy);
    if (n_read <= 0) { (*is->env)->DeleteLocalRef(is->env, byte_array); return 0; }

    jbyte* byte_array_elements = (*is->env)->GetByteArrayElements(is->env, byte_array, NULL);
    memcpy(output, byte_array_elements, n_read);
    (*is->env)->ReleaseByteArrayElements(is->env, byte_array, byte_array_elements, JNI_ABORT);
    (*is->env)->DeleteLocalRef(is->env, byte_array);
    is->offset += n_read;
    return n_read;
}

bool inputStreamEof(void * ctx) {
    struct input_stream_context* is = (struct input_stream_context*)ctx;
    jint result = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_available);
    return result <= 0;
}

void inputStreamClose(void * ctx) { UNUSED(ctx); }

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

static struct whisper_context *whisper_init_from_asset(
        JNIEnv *env,
        jobject assetManager,
        const char *asset_path
) {
    LOGI("Loading model from asset '%s'", asset_path);
    AAssetManager *asset_manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(asset_manager, asset_path, AASSET_MODE_STREAMING);
    if (!asset) {
        LOGE("Failed to open asset '%s'", asset_path);
        return NULL;
    }

    whisper_model_loader loader = {
            .context = asset,
            .read = &asset_read,
            .eof = &asset_is_eof,
            .close = &asset_close
    };

    return whisper_init_with_params(&loader, whisper_context_default_params());
}

// ============================================================================
// JNI: Context lifecycle
// ============================================================================

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str) {
    UNUSED(thiz);
    const char *model_path = (*env)->GetStringUTFChars(env, model_path_str, NULL);
    LOGI("Loading model from file: %s", model_path);

    // whisper_init_from_file_with_params supports quantized models automatically:
    // q4_0, q4_1, q5_0, q5_1, q8_0 are detected from the file header
    struct whisper_context *context = whisper_init_from_file_with_params(model_path, whisper_context_default_params());
    (*env)->ReleaseStringUTFChars(env, model_path_str, model_path);

    if (!context) LOGE("Failed to load model from: %s", model_path);
    return (jlong) context;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromAsset(
        JNIEnv *env, jobject thiz, jobject assetManager, jstring asset_path_str) {
    UNUSED(thiz);
    const char *asset_path = (*env)->GetStringUTFChars(env, asset_path_str, NULL);
    struct whisper_context *context = whisper_init_from_asset(env, assetManager, asset_path);
    (*env)->ReleaseStringUTFChars(env, asset_path_str, asset_path);
    return (jlong) context;
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_initContextFromInputStream(
        JNIEnv *env, jobject thiz, jobject input_stream) {
    UNUSED(thiz);
    struct input_stream_context inp_ctx = {};
    inp_ctx.env = env;
    inp_ctx.thiz = thiz;
    inp_ctx.input_stream = input_stream;

    jclass cls = (*env)->GetObjectClass(env, input_stream);
    inp_ctx.mid_available = (*env)->GetMethodID(env, cls, "available", "()I");
    inp_ctx.mid_read = (*env)->GetMethodID(env, cls, "read", "([BII)I");

    whisper_model_loader loader = {
        .context = &inp_ctx,
        .read = inputStreamRead,
        .eof = inputStreamEof,
        .close = inputStreamClose
    };

    struct whisper_context *context = whisper_init_with_params(&loader, whisper_context_default_params());
    return (jlong) context;
}

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    whisper_free((struct whisper_context *) context_ptr);
}

// ============================================================================
// JNI: Model info
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSystemInfo(
        JNIEnv *env, jobject thiz) {
    UNUSED(thiz);
    const char *sysinfo = whisper_print_system_info();
    return (*env)->NewStringUTF(env, sysinfo);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelNTextCtx(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    return whisper_model_n_text_ctx((struct whisper_context *) context_ptr);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelNAudioCtx(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    return whisper_model_n_audio_ctx((struct whisper_context *) context_ptr);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getModelFtype(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    return whisper_model_ftype((struct whisper_context *) context_ptr);
}

JNIEXPORT jboolean JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_isMultilingual(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    return whisper_is_multilingual((struct whisper_context *) context_ptr) ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// JNI: Full Transcribe (enhanced with all params)
// ============================================================================

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_fullTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr,
        jint num_threads, jint n_max_text_ctx,
        jboolean enable_vad, jfloat vad_threshold,
        jstring language_str, jboolean detect_language,
        jboolean translate,
        jfloatArray audio_data) {

    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    jfloat *audio_data_arr = (*env)->GetFloatArrayElements(env, audio_data, NULL);
    const jsize audio_data_length = (*env)->GetArrayLength(env, audio_data);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // === Thread control ===
    params.n_threads = int_max(1, num_threads);

    // === Context size control ===
    params.n_max_text_ctx = int_max(64, n_max_text_ctx);

    // === Language ===
    const char *language = "en";
    if (language_str != NULL) {
        language = (*env)->GetStringUTFChars(env, language_str, NULL);
    }
    params.language = language;
    params.detect_language = detect_language;

    // === Translation ===
    params.translate = translate;

    // === VAD (Voice Activity Detection) ===
    params.vad = enable_vad ? true : false;
    params.vad_params.threshold = vad_threshold;
    params.vad_params.min_speech_duration_ms = 250;
    params.vad_params.min_silence_duration_ms = 500;
    params.vad_params.max_speech_duration_s = 30.0f;
    params.vad_params.speech_pad_ms = 400;

    // === General settings ===
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = true;
    params.print_special = false;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    // === Decoding optimizations ===
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.no_speech_thold = 0.6f;

    whisper_reset_timings(context);

    LOGI("Transcribing: threads=%d, ctx=%d, vad=%d, lang=%s, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad,
         params.detect_language ? "auto" : params.language, audio_data_length);

    if (whisper_full(context, params, audio_data_arr, audio_data_length) != 0) {
        LOGE("Failed to run transcription");
    } else {
        whisper_print_timings(context);
    }

    if (language_str != NULL) {
        (*env)->ReleaseStringUTFChars(env, language_str, language);
    }
    (*env)->ReleaseFloatArrayElements(env, audio_data, audio_data_arr, JNI_ABORT);
}

// ============================================================================
// JNI: Streaming transcribe — process audio in chunks via whisper_full_parallel
// ============================================================================

JNIEXPORT void JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_streamTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr,
        jint num_threads, jint n_max_text_ctx,
        jboolean enable_vad, jfloat vad_threshold,
        jstring language_str, jboolean detect_language,
        jboolean translate,
        jfloatArray audio_data) {

    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    jfloat *audio_data_arr = (*env)->GetFloatArrayElements(env, audio_data, NULL);
    const jsize audio_data_length = (*env)->GetArrayLength(env, audio_data);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // === Thread control ===
    params.n_threads = int_max(1, num_threads);

    // === Context size control ===
    params.n_max_text_ctx = int_max(64, n_max_text_ctx);

    // === Language ===
    const char *language = "en";
    if (language_str != NULL) {
        language = (*env)->GetStringUTFChars(env, language_str, NULL);
    }
    params.language = language;
    params.detect_language = detect_language;

    // === Translation ===
    params.translate = translate;

    // === VAD ===
    params.vad = enable_vad ? true : false;
    params.vad_params.threshold = vad_threshold;
    params.vad_params.min_speech_duration_ms = 250;
    params.vad_params.min_silence_duration_ms = 500;
    params.vad_params.max_speech_duration_s = 30.0f;
    params.vad_params.speech_pad_ms = 400;

    // === General settings ===
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = true;
    params.print_special = false;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    // === Decoding optimizations ===
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.no_speech_thold = 0.6f;

    whisper_reset_timings(context);

    // Use parallel processing — splits audio into 30-second chunks
    // and processes each independently. Reduces peak RAM usage significantly.
    int n_processors = int_min(num_threads, 4);
    LOGI("Stream transcribe: threads=%d, ctx=%d, vad=%d, processors=%d, samples=%d",
         params.n_threads, params.n_max_text_ctx, params.vad, n_processors, audio_data_length);

    int result = whisper_full_parallel(context, params, audio_data_arr, audio_data_length, n_processors);
    if (result != 0) {
        LOGE("whisper_full_parallel failed with code %d", result);
    } else {
        whisper_print_timings(context);
    }

    if (language_str != NULL) {
        (*env)->ReleaseStringUTFChars(env, language_str, language);
    }
    (*env)->ReleaseFloatArrayElements(env, audio_data, audio_data_arr, JNI_ABORT);
}

// ============================================================================
// JNI: Segment results
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentCount(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env); UNUSED(thiz);
    return whisper_full_n_segments((struct whisper_context *) context_ptr);
}

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegment(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    const char *text = whisper_full_get_segment_text((struct whisper_context *) context_ptr, index);
    return (*env)->NewStringUTF(env, text);
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentT0(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    return whisper_full_get_segment_t0((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jlong JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getTextSegmentT1(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    return whisper_full_get_segment_t1((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jfloat JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSegmentNoSpeechProb(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    return whisper_full_get_segment_no_speech_prob((struct whisper_context *) context_ptr, index);
}

JNIEXPORT jint JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_getSegmentTokenCount(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    return whisper_full_n_tokens((struct whisper_context *) context_ptr, index);
}

// ============================================================================
// JNI: Benchmarks
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_benchMemcpy(
        JNIEnv *env, jobject thiz, jint n_threads) {
    UNUSED(thiz);
    const char *bench = whisper_bench_memcpy_str(n_threads);
    return (*env)->NewStringUTF(env, bench);
}

JNIEXPORT jstring JNICALL
Java_com_whispercpp_whisper_WhisperLib_00024Companion_benchGgmlMulMat(
        JNIEnv *env, jobject thiz, jint n_threads) {
    UNUSED(thiz);
    const char *bench = whisper_bench_ggml_mul_mat_str(n_threads);
    return (*env)->NewStringUTF(env, bench);
}
