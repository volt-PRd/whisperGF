package com.whispercpp.whisper

import android.content.res.AssetManager
import android.os.Build
import android.util.Log
import kotlinx.coroutines.*
import java.io.File
import java.io.InputStream
import java.util.concurrent.Executors

private const val LOG_TAG = "WhisperGF"

// ============================================================================
// TranscribeConfig — all tuning parameters exposed to the user
// ============================================================================

/**
 * Configuration for whisper transcription.
 *
 * @param numThreads           Number of CPU threads for inference. Default: min(big_cores, 8).
 * @param maxTextCtx           Maximum text context tokens for the decoder. Default: 448.
 *                             Reduce to 256 to save ~30% RAM with minimal accuracy loss.
 * @param language             Language code (e.g. "en", "ar", "fr"). Null = auto-detect.
 * @param detectLanguage       Automatically detect the spoken language. Default: true.
 * @param translate            Translate to English. Default: false.
 * @param enableVad            Voice Activity Detection — skip silence segments. Default: false.
 * @param vadThreshold         VAD speech probability threshold (0.0–1.0). Default: 0.5.
 * @param carryInitialPrompt   Carry previous context into next decode window. Default: true.
 *                             Critical for streaming — improves term consistency.
 * @param initialPrompt        Optional text prompt to guide transcription. Default: null.
 * @param enableHushWords      Append 0.5s silence to audio end to prevent hallucination. Default: true.
 */
data class TranscribeConfig(
    val numThreads: Int = WhisperCpuConfig.preferredThreadCount.coerceIn(1, 8),
    val maxTextCtx: Int = 448,
    val language: String? = null,
    val detectLanguage: Boolean = true,
    val translate: Boolean = false,
    val enableVad: Boolean = false,
    val vadThreshold: Float = 0.5f,
    val carryInitialPrompt: Boolean = true,
    val initialPrompt: String? = null,
    val enableHushWords: Boolean = true
)

// ============================================================================
// TranscribeResult — structured output
// ============================================================================

data class TranscribeSegment(
    val text: String,
    val t0Ms: Long,
    val t1Ms: Long,
    val noSpeechProb: Float
)

data class TranscribeResult(
    val fullText: String,
    val segments: List<TranscribeSegment>,
    val language: String?
)

// ============================================================================
// ModelInfo — metadata about the loaded model
// ============================================================================

data class ModelInfo(
    val textCtx: Int,
    val audioCtx: Int,
    val ftype: Int,
    val modelType: Int,
    val multilingual: Boolean
) {
    val quantizationName: String get() = when (ftype) {
        1 -> "Q4_0 (4-bit uniform, ~45% less RAM)"
        2 -> "Q4_1 (4-bit uniform, ~45% less RAM)"
        3 -> "Q5_0 (5-bit uniform, ~35% less RAM)"
        7 -> "Q8_0 (8-bit, ~20% less RAM)"
        12 -> "Q4_K_M (4-bit K-quants, best quality/size ratio)"
        13 -> "Q5_K_M (5-bit K-quants, near-FP32 quality)"
        14 -> "Q6_K (6-bit K-quants, minimal quality loss)"
        15 -> "Q8_K (8-bit K-quants, high accuracy)"
        else -> "FP32 (no quantization)"
    }

    val isQuantized: Boolean get() = ftype in listOf(1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    val modelName: String get() = when (modelType) {
        0 -> "tiny"
        1 -> "base"
        2 -> "small"
        3 -> "medium"
        4 -> "large"
        5 -> "large-v1"
        6 -> "large-v2"
        7 -> "large-v3"
        8 -> "large-v3-turbo"
        else -> "unknown ($modelType)"
    }
}

// ============================================================================
// Local Agreement Policy — prevents text flashing in streaming
// ============================================================================

/**
 * Filters streaming output using Local Agreement Policy.
 * A word/segment is only "committed" when it appears consistently
 * across two consecutive inference passes, preventing visual flickering.
 */
class LocalAgreementPolicy(
    private val minConfirmations: Int = 2
) {
    private val confirmedText = StringBuilder()
    private var pendingSegments: List<TranscribeSegment> = emptyList()
    private var confirmationCount = 0

    /**
     * Process new inference result and return stable text.
     * Only returns text that has been confirmed across [minConfirmations] passes.
     */
    fun process(result: TranscribeResult): String {
        val currentFull = result.fullText

        if (currentFull == pendingSegments.joinToString("") { it.text }) {
            // Same result as last time — increase confirmation
            confirmationCount++
        } else {
            // Changed — reset counter
            confirmationCount = 1
        }

        pendingSegments = result.segments

        return if (confirmationCount >= minConfirmations) {
            // Confirmed — lock in this text
            confirmedText.append(currentFull)
            confirmationCount = 0
            confirmedText.toString()
        } else {
            // Not yet confirmed — return previous stable text
            confirmedText.toString()
        }
    }

    fun reset() {
        confirmedText.clear()
        pendingSegments = emptyList()
        confirmationCount = 0
    }

    fun getStableText(): String = confirmedText.toString()
}

// ============================================================================
// WhisperContext — main public API v3.0.0
// ============================================================================

class WhisperContext private constructor(
    private var ptr: Long,
    private val useGpu: Boolean
) {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )

    // ==========================================================================
    // Model info
    // ==========================================================================

    fun getModelInfo(): ModelInfo = runBlocking {
        withContext(scope.coroutineContext) {
            require(ptr != 0L) { "Context has been released" }
            ModelInfo(
                textCtx = WhisperLib.getModelNTextCtx(ptr),
                audioCtx = WhisperLib.getModelNAudioCtx(ptr),
                ftype = WhisperLib.getModelFtype(ptr),
                modelType = WhisperLib.getModelType(ptr),
                multilingual = WhisperLib.isMultilingual(ptr)
            )
        }
    }

    // ==========================================================================
    // Full transcribe — entire audio in memory
    // ==========================================================================

    /**
     * Transcribe audio data (full-file mode).
     * Supports FP32 and all quantized models (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)
     * Flash Attention is enabled by default.
     * Vulkan GPU acceleration if useGpu was true on creation.
     *
     * @param data    PCM float array at 16kHz, mono.
     * @param config  Transcription configuration.
     * @return TranscribeResult with full text and per-segment details.
     */
    suspend fun transcribe(
        data: FloatArray,
        config: TranscribeConfig = TranscribeConfig()
    ): TranscribeResult = withContext(scope.coroutineContext) {
        require(ptr != 0L) { "Context has been released" }

        Log.d(LOG_TAG, "transcribe: threads=${config.numThreads}, ctx=${config.maxTextCtx}, " +
                "vad=${config.enableVad}, carry=${config.carryInitialPrompt}, " +
                "hush=${config.enableHushWords}, gpu=$useGpu")

        WhisperLib.fullTranscribe(
            contextPtr = ptr,
            numThreads = config.numThreads,
            nMaxTextCtx = config.maxTextCtx,
            enableVad = config.enableVad,
            vadThreshold = config.vadThreshold,
            language = config.language,
            detectLanguage = config.detectLanguage,
            translate = config.translate,
            carryInitialPrompt = config.carryInitialPrompt,
            initialPrompt = config.initialPrompt,
            enableHushWords = config.enableHushWords,
            audioData = data
        )

        buildResult()
    }

    // ==========================================================================
    // Stream transcribe — chunked processing with context carry-over
    // ==========================================================================

    /**
     * Stream-transcribe audio data (chunked, parallel processing).
     * Splits audio into 30-second chunks processed independently.
     * Uses ~40% less peak RAM than full transcribe.
     * carry_initial_prompt ensures context continuity between chunks.
     *
     * @param data    PCM float array at 16kHz, mono.
     * @param config  Transcription configuration.
     * @return TranscribeResult with full text and per-segment details.
     */
    suspend fun transcribeStream(
        data: FloatArray,
        config: TranscribeConfig = TranscribeConfig()
    ): TranscribeResult = withContext(scope.coroutineContext) {
        require(ptr != 0L) { "Context has been released" }

        Log.d(LOG_TAG, "transcribeStream: threads=${config.numThreads}, ctx=${config.maxTextCtx}, " +
                "vad=${config.enableVad}, carry=${config.carryInitialPrompt}, gpu=$useGpu")

        WhisperLib.streamTranscribe(
            contextPtr = ptr,
            numThreads = config.numThreads,
            nMaxTextCtx = config.maxTextCtx,
            enableVad = config.enableVad,
            vadThreshold = config.vadThreshold,
            language = config.language,
            detectLanguage = config.detectLanguage,
            translate = config.translate,
            carryInitialPrompt = config.carryInitialPrompt,
            initialPrompt = config.initialPrompt,
            enableHushWords = config.enableHushWords,
            audioData = data
        )

        buildResult()
    }

    // ==========================================================================
    // Backward-compatible simple transcription
    // ==========================================================================

    suspend fun transcribeData(data: FloatArray, printTimestamp: Boolean = true): String {
        val result = transcribe(data)
        return if (printTimestamp) {
            result.segments.joinToString("\n") { seg ->
                "[${toTimestamp(seg.t0Ms)} --> ${toTimestamp(seg.t1Ms)}]: ${seg.text}"
            }
        } else {
            result.fullText
        }
    }

    // ==========================================================================
    // Benchmarks
    // ==========================================================================

    suspend fun benchMemory(nthreads: Int = WhisperCpuConfig.preferredThreadCount): String =
        withContext(scope.coroutineContext) { WhisperLib.benchMemcpy(nthreads) }

    suspend fun benchGgmlMulMat(nthreads: Int = WhisperCpuConfig.preferredThreadCount): String =
        withContext(scope.coroutineContext) { WhisperLib.benchGgmlMulMat(nthreads) }

    // ==========================================================================
    // Lifecycle
    // ==========================================================================

    suspend fun release() = withContext(scope.coroutineContext) {
        if (ptr != 0L) {
            WhisperLib.freeContext(ptr)
            ptr = 0
            Log.d(LOG_TAG, "Context released")
        }
    }

    protected fun finalize() { runBlocking { release() } }

    // -- Private --

    private fun buildResult(): TranscribeResult {
        val textCount = WhisperLib.getTextSegmentCount(ptr)
        val segments = mutableListOf<TranscribeSegment>()
        val sb = StringBuilder()
        for (i in 0 until textCount) {
            val text = WhisperLib.getTextSegment(ptr, i)
            val t0 = WhisperLib.getTextSegmentT0(ptr, i)
            val t1 = WhisperLib.getTextSegmentT1(ptr, i)
            val noSpeechProb = WhisperLib.getSegmentNoSpeechProb(ptr, i)
            segments.add(TranscribeSegment(text, t0, t1, noSpeechProb))
            if (i > 0) sb.append(" ")
            sb.append(text)
        }
        return TranscribeResult(
            fullText = sb.toString().trim(),
            segments = segments,
            language = null // Detected by whisper internally
        )
    }

    companion object {
        /**
         * Create context from model file.
         * Supports all model types: FP32, Q4_0, Q4_K_M, Q5_K_M, Q8_0, large-v3-turbo, etc.
         * Quantization is auto-detected from GGUF header.
         *
         * @param filePath  Path to .bin or .gguf model file.
         * @param useGpu    Enable Vulkan GPU acceleration and mmap (recommended). Default: true.
         */
        fun createContextFromFile(filePath: String, useGpu: Boolean = true): WhisperContext {
            val ptr = WhisperLib.initContext(filePath, useGpu)
            if (ptr == 0L) throw RuntimeException("Failed to load model from: $filePath")
            return WhisperContext(ptr, useGpu)
        }

        /**
         * Create context from Android Asset.
         *
         * @param assetManager  Android AssetManager.
         * @param assetPath     Path within assets/ directory.
         * @param useGpu        Enable GPU/mmap. Default: true.
         */
        fun createContextFromAsset(
            assetManager: AssetManager, assetPath: String, useGpu: Boolean = true
        ): WhisperContext {
            val ptr = WhisperLib.initContextFromAsset(assetManager, assetPath, useGpu)
            if (ptr == 0L) throw RuntimeException("Failed to load model from asset: $assetPath")
            return WhisperContext(ptr, useGpu)
        }

        /**
         * Create context from InputStream.
         *
         * @param stream   InputStream to model data.
         * @param useGpu   Enable GPU/mmap. Default: true.
         */
        fun createContextFromInputStream(stream: InputStream, useGpu: Boolean = true): WhisperContext {
            val ptr = WhisperLib.initContextFromInputStream(stream, useGpu)
            if (ptr == 0L) throw RuntimeException("Failed to load model from InputStream")
            return WhisperContext(ptr, useGpu)
        }

        /** Get GGML system info (backend, Vulkan status, CPU features, etc.) */
        fun getSystemInfo(): String = WhisperLib.getSystemInfo()
    }
}

// ============================================================================
// WhisperLib — private JNI bridge
// ============================================================================

private class WhisperLib {
    companion object {
        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS[0]}")
            val loadV8fp16 = Build.SUPPORTED_ABIS[0] == "arm64-v8a" &&
                cpuInfo()?.contains("fphp") == true

            if (loadV8fp16) {
                Log.d(LOG_TAG, "Loading libwhisper_v8fp16_va.so (ARMv8.2 FP16 optimized)")
                System.loadLibrary("whisper_v8fp16_va")
            } else {
                System.loadLibrary("whisper")
            }
            Log.d(LOG_TAG, "Native library loaded successfully")
        }

        // Context lifecycle
        external fun initContext(modelPath: String, useGpu: Boolean): Long
        external fun initContextFromAsset(assetManager: AssetManager, assetPath: String, useGpu: Boolean): Long
        external fun initContextFromInputStream(inputStream: InputStream, useGpu: Boolean): Long
        external fun freeContext(contextPtr: Long)

        // Model info
        external fun getSystemInfo(): String
        external fun getModelNTextCtx(contextPtr: Long): Int
        external fun getModelNAudioCtx(contextPtr: Long): Int
        external fun getModelFtype(contextPtr: Long): Int
        external fun getModelType(contextPtr: Long): Int
        external fun isMultilingual(contextPtr: Long): Boolean

        // Full transcribe (with hush words + carry prompt + all params)
        external fun fullTranscribe(
            contextPtr: Long,
            numThreads: Int,
            nMaxTextCtx: Int,
            enableVad: Boolean,
            vadThreshold: Float,
            language: String?,
            detectLanguage: Boolean,
            translate: Boolean,
            carryInitialPrompt: Boolean,
            initialPrompt: String?,
            enableHushWords: Boolean,
            audioData: FloatArray
        )

        // Stream transcribe (chunked, parallel, with carry-over)
        external fun streamTranscribe(
            contextPtr: Long,
            numThreads: Int,
            nMaxTextCtx: Int,
            enableVad: Boolean,
            vadThreshold: Float,
            language: String?,
            detectLanguage: Boolean,
            translate: Boolean,
            carryInitialPrompt: Boolean,
            initialPrompt: String?,
            enableHushWords: Boolean,
            audioData: FloatArray
        )

        // Segment results
        external fun getTextSegmentCount(contextPtr: Long): Int
        external fun getTextSegment(contextPtr: Long, index: Int): String
        external fun getTextSegmentT0(contextPtr: Long, index: Int): Long
        external fun getTextSegmentT1(contextPtr: Long, index: Int): Long
        external fun getSegmentNoSpeechProb(contextPtr: Long, index: Int): Float
        external fun getSegmentTokenCount(contextPtr: Long, index: Int): Int
        external fun getFullLangId(contextPtr: Long): Int

        // Benchmarks
        external fun benchMemcpy(nthread: Int): String
        external fun benchGgmlMulMat(nthread: Int): String
    }
}

// ============================================================================
// Utilities
// ============================================================================

private fun toTimestamp(t: Long, comma: Boolean = false): String {
    var msec = t * 10
    val hr = msec / (1000 * 60 * 60); msec -= hr * (1000 * 60 * 60)
    val min = msec / (1000 * 60); msec -= min * (1000 * 60)
    val sec = msec / 1000; msec -= sec * 1000
    val delim = if (comma) "," else "."
    return String.format("%02d:%02d:%02d%s%03d", hr, min, sec, delim, msec)
}

private fun cpuInfo(): String? = try {
    File("/proc/cpuinfo").inputStream().bufferedReader().use { it.readText() }
} catch (e: Exception) {
    Log.w(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
    null
}
