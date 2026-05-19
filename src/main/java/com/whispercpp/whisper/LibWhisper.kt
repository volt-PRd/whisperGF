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
 * @param numThreads       Number of CPU threads for inference. Default: min(4, availableCores).
 *                         Higher values speed up processing but increase CPU usage.
 * @param maxTextCtx       Maximum text context tokens for the decoder. Default: 448.
 *                         Reduce to 256 to save ~30% RAM with minimal accuracy loss.
 * @param language         Language code (e.g. "en", "ar", "fr"). Null = auto-detect.
 * @param detectLanguage   Automatically detect the spoken language. Default: true.
 * @param translate        Translate to English. Default: false.
 * @param enableVad        Enable Voice Activity Detection to skip silence. Default: false.
 * @param vadThreshold     VAD speech probability threshold (0.0–1.0). Default: 0.5.
 *                         Lower = more sensitive; higher = less false positives.
 */
data class TranscribeConfig(
    val numThreads: Int = WhisperCpuConfig.preferredThreadCount.coerceIn(1, 8),
    val maxTextCtx: Int = 448,
    val language: String? = null,
    val detectLanguage: Boolean = true,
    val translate: Boolean = false,
    val enableVad: Boolean = false,
    val vadThreshold: Float = 0.5f
)

// ============================================================================
// TranscribeResult — structured output from transcription
// ============================================================================

data class TranscribeSegment(
    val text: String,
    val t0Ms: Long,      // start timestamp in milliseconds
    val t1Ms: Long,      // end timestamp in milliseconds
    val noSpeechProb: Float  // probability this segment is not speech (0.0–1.0)
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
    val ftype: Int,      // quantization type: 0=FP32, 1=Q4_0, 2=Q4_1, 3=Q5_0, 7=Q8_0, etc.
    val multilingual: Boolean
) {
    /**
     * Returns a human-readable quantization name.
     * Supported quantized formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
     */
    val quantizationName: String get() = when (ftype) {
        1 -> "Q4_0 (4-bit, ~45% less RAM)"
        2 -> "Q4_1 (4-bit, ~45% less RAM)"
        3 -> "Q5_0 (5-bit, ~35% less RAM)"
        7 -> "Q8_0 (8-bit, ~20% less RAM)"
        else -> "FP32 (no quantization)"
    }

    val isQuantized: Boolean get() = ftype in listOf(1, 2, 3, 7, 8, 9, 10)
}

// ============================================================================
// WhisperContext — main public API
// ============================================================================

class WhisperContext private constructor(private var ptr: Long) {

    // Single-threaded scope: whisper.cpp is NOT thread-safe for the same context
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )

    /**
     * Get information about the loaded model (quantization type, context sizes, etc.)
     */
    fun getModelInfo(): ModelInfo = runBlocking {
        withContext(scope.coroutineContext) {
            require(ptr != 0L) { "Context has been released" }
            ModelInfo(
                textCtx = WhisperLib.getModelNTextCtx(ptr),
                audioCtx = WhisperLib.getModelNAudioCtx(ptr),
                ftype = WhisperLib.getModelFtype(ptr),
                multilingual = WhisperLib.isMultilingual(ptr)
            )
        }
    }

    /**
     * Transcribe audio data (full-file mode — loads entire audio in memory).
     *
     * @param data    PCM float array at 16kHz, mono.
     * @param config  Transcription configuration (threads, VAD, language, etc.)
     * @return TranscribeResult with full text and per-segment details
     */
    suspend fun transcribe(
        data: FloatArray,
        config: TranscribeConfig = TranscribeConfig()
    ): TranscribeResult = withContext(scope.coroutineContext) {
        require(ptr != 0L) { "Context has been released" }

        Log.d(LOG_TAG, "Transcribing: threads=${config.numThreads}, ctx=${config.maxTextCtx}, vad=${config.enableVad}, lang=${config.language ?: "auto"}")

        WhisperLib.fullTranscribe(
            contextPtr = ptr,
            numThreads = config.numThreads,
            nMaxTextCtx = config.maxTextCtx,
            enableVad = config.enableVad,
            vadThreshold = config.vadThreshold,
            language = config.language,
            detectLanguage = config.detectLanguage,
            translate = config.translate,
            audioData = data
        )

        return@withContext buildResult(config.language)
    }

    /**
     * Stream-transcribe audio data (chunked processing).
     * Splits audio into 30-second chunks processed independently.
     * Uses ~40% less peak RAM than full transcribe.
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

        Log.d(LOG_TAG, "Stream transcribe: threads=${config.numThreads}, ctx=${config.maxTextCtx}, vad=${config.enableVad}")

        WhisperLib.streamTranscribe(
            contextPtr = ptr,
            numThreads = config.numThreads,
            nMaxTextCtx = config.maxTextCtx,
            enableVad = config.enableVad,
            vadThreshold = config.vadThreshold,
            language = config.language,
            detectLanguage = config.detectLanguage,
            translate = config.translate,
            audioData = data
        )

        return@withContext buildResult(config.language)
    }

    /**
     * Simple transcription — returns plain text with optional timestamps.
     * Backward-compatible convenience method.
     *
     * @param data            PCM float array at 16kHz, mono.
     * @param printTimestamp  Include timestamps in the output string.
     * @return Formatted transcription text.
     */
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

    /**
     * Run memory copy benchmark.
     */
    suspend fun benchMemory(nthreads: Int = WhisperCpuConfig.preferredThreadCount): String =
        withContext(scope.coroutineContext) {
            WhisperLib.benchMemcpy(nthreads)
        }

    /**
     * Run GGML matrix multiplication benchmark.
     */
    suspend fun benchGgmlMulMat(nthreads: Int = WhisperCpuConfig.preferredThreadCount): String =
        withContext(scope.coroutineContext) {
            WhisperLib.benchGgmlMulMat(nthreads)
        }

    /**
     * Release the native context and free all associated memory.
     * Must be called when done with this context.
     */
    suspend fun release() = withContext(scope.coroutineContext) {
        if (ptr != 0L) {
            WhisperLib.freeContext(ptr)
            ptr = 0
            Log.d(LOG_TAG, "Context released")
        }
    }

    protected fun finalize() {
        runBlocking { release() }
    }

    // -- Private helpers --

    private fun buildResult(configLang: String?): TranscribeResult {
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
            language = configLang
        )
    }

    companion object {
        /**
         * Create context from a model file path.
         * Supports both FP32 and quantized models (q4_0, q4_1, q5_0, q5_1, q8_0).
         * The quantization type is auto-detected from the model file header.
         */
        fun createContextFromFile(filePath: String): WhisperContext {
            val ptr = WhisperLib.initContext(filePath)
            if (ptr == 0L) {
                throw RuntimeException("Failed to load model from: $filePath")
            }
            return WhisperContext(ptr)
        }

        /**
         * Create context from an Android Asset.
         * Supports quantized models (auto-detected).
         */
        fun createContextFromAsset(assetManager: AssetManager, assetPath: String): WhisperContext {
            val ptr = WhisperLib.initContextFromAsset(assetManager, assetPath)
            if (ptr == 0L) {
                throw RuntimeException("Failed to load model from asset: $assetPath")
            }
            return WhisperContext(ptr)
        }

        /**
         * Create context from an InputStream.
         * Supports quantized models (auto-detected).
         */
        fun createContextFromInputStream(stream: InputStream): WhisperContext {
            val ptr = WhisperLib.initContextFromInputStream(stream)
            if (ptr == 0L) {
                throw RuntimeException("Failed to load model from InputStream")
            }
            return WhisperContext(ptr)
        }

        /**
         * Get system information (GGML backend, CPU features, etc.)
         */
        fun getSystemInfo(): String {
            return WhisperLib.getSystemInfo()
        }
    }
}

// ============================================================================
// WhisperLib — private JNI bridge
// ============================================================================

private class WhisperLib {
    companion object {
        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS[0]}")
            var loadV8fp16 = false
            if (isArmEabiV8a()) {
                val cpuInfo = cpuInfo()
                cpuInfo?.let {
                    Log.d(LOG_TAG, "CPU info: $cpuInfo")
                    if (cpuInfo.contains("fphp")) {
                        Log.d(LOG_TAG, "CPU supports FP16 arithmetic — loading optimized variant")
                        loadV8fp16 = true
                    }
                }
            }

            if (loadV8fp16) {
                System.loadLibrary("whisper_v8fp16_va")
            } else {
                System.loadLibrary("whisper")
            }
        }

        // Context lifecycle
        external fun initContext(modelPath: String): Long
        external fun initContextFromAsset(assetManager: AssetManager, assetPath: String): Long
        external fun initContextFromInputStream(inputStream: InputStream): Long
        external fun freeContext(contextPtr: Long)

        // Model info
        external fun getSystemInfo(): String
        external fun getModelNTextCtx(contextPtr: Long): Int
        external fun getModelNAudioCtx(contextPtr: Long): Int
        external fun getModelFtype(contextPtr: Long): Int
        external fun isMultilingual(contextPtr: Long): Boolean

        // Transcription (full)
        external fun fullTranscribe(
            contextPtr: Long,
            numThreads: Int,
            nMaxTextCtx: Int,
            enableVad: Boolean,
            vadThreshold: Float,
            language: String?,
            detectLanguage: Boolean,
            translate: Boolean,
            audioData: FloatArray
        )

        // Transcription (streaming/chunked)
        external fun streamTranscribe(
            contextPtr: Long,
            numThreads: Int,
            nMaxTextCtx: Int,
            enableVad: Boolean,
            vadThreshold: Float,
            language: String?,
            detectLanguage: Boolean,
            translate: Boolean,
            audioData: FloatArray
        )

        // Segment results
        external fun getTextSegmentCount(contextPtr: Long): Int
        external fun getTextSegment(contextPtr: Long, index: Int): String
        external fun getTextSegmentT0(contextPtr: Long, index: Int): Long
        external fun getTextSegmentT1(contextPtr: Long, index: Int): Long
        external fun getSegmentNoSpeechProb(contextPtr: Long, index: Int): Float
        external fun getSegmentTokenCount(contextPtr: Long, index: Int): Int

        // Benchmarks
        external fun benchMemcpy(nthread: Int): String
        external fun benchGgmlMulMat(nthread: Int): String
    }
}

// ============================================================================
// Utility functions
// ============================================================================

private fun toTimestamp(t: Long, comma: Boolean = false): String {
    var msec = t * 10
    val hr = msec / (1000 * 60 * 60)
    msec -= hr * (1000 * 60 * 60)
    val min = msec / (1000 * 60)
    msec -= min * (1000 * 60)
    val sec = msec / 1000
    msec -= sec * 1000
    val delimiter = if (comma) "," else "."
    return String.format("%02d:%02d:%02d%s%03d", hr, min, sec, delimiter, msec)
}

private fun isArmEabiV8a(): Boolean = Build.SUPPORTED_ABIS[0].equals("arm64-v8a")

private fun cpuInfo(): String? = try {
    File("/proc/cpuinfo").inputStream().bufferedReader().use { it.readText() }
} catch (e: Exception) {
    Log.w(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
    null
}
