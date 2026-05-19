package com.whispergf.test

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.method.ScrollingMovementMethod
import android.view.MotionEvent
import android.view.View
import android.widget.*
import com.whispercpp.whisper.*
import kotlinx.coroutines.*
import java.io.File
import kotlin.system.measureTimeMillis

class MainActivity : Activity() {

    private lateinit var tvModelStatus: TextView
    private lateinit var tvSystemInfo: TextView
    private lateinit var tvModelInfo: TextView
    private lateinit var tvResult: TextView
    private lateinit var tvLog: TextView
    private lateinit var tvTimer: TextView
    private lateinit var tvInferenceTime: TextView
    private lateinit var tvDownloadStatus: TextView
    private lateinit var tvThreadCount: TextView
    private lateinit var btnRecord: Button
    private lateinit var btnDownloadModel: Button
    private lateinit var btnPickModel: Button
    private lateinit var btnCopy: Button
    private lateinit var btnClearLog: Button
    private lateinit var progressDownload: ProgressBar
    private lateinit var cbVad: CheckBox
    private lateinit var cbTranslate: CheckBox
    private lateinit var cbDiarize: CheckBox
    private lateinit var cbHush: CheckBox
    private lateinit var spinnerModel: Spinner
    private lateinit var spinnerLang: Spinner
    private lateinit var seekBarThreads: SeekBar

    private var whisperContext: WhisperContext? = null
    private var isRecording = false
    private var audioRecord: AudioRecord? = null
    private var recordThread: Thread? = null
    private var audioBuffer = mutableListOf<Float>()
    private val SAMPLE_RATE = 16000
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var recordStartTime = 0L
    private val timerHandler = Handler(Looper.getMainLooper())
    private var selectedModelIndex = 4 // Default: large-v3-turbo

    private val modelMap = mapOf(
        0 to WhisperModel.TINY_Q5_K_M,
        1 to WhisperModel.BASE_Q5_K_M,
        2 to WhisperModel.SMALL_Q5_K_M,
        3 to WhisperModel.MEDIUM_IQ4_XS,
        4 to WhisperModel.LARGE_V3_TURBO_Q5_K_M,
        5 to WhisperModel.TINY_TDRZ,
        6 to WhisperModel.BASE_TDRZ
    )

    private val langCodes = arrayOf("", "ar", "en", "fr", "es", "de", "tr", "zh", "ja", "ko", "ru", "hi", "ur")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestPermissions()
        setupRecordButton()
        setupSpinner()
        setupSeekBar()
        loadExistingModel()
        appendLog("App started. ABI: ${Build.SUPPORTED_ABIS[0]}")
        appendLog("GGML: ${tryGetSystemInfo()}")
    }

    private fun initViews() {
        tvModelStatus = findViewById(R.id.tvModelStatus)
        tvSystemInfo = findViewById(R.id.tvSystemInfo)
        tvModelInfo = findViewById(R.id.tvModelInfo)
        tvResult = findViewById(R.id.tvResult)
        tvResult.movementMethod = ScrollingMovementMethod()
        tvLog = findViewById(R.id.tvLog)
        tvLog.movementMethod = ScrollingMovementMethod()
        tvTimer = findViewById(R.id.tvTimer)
        tvInferenceTime = findViewById(R.id.tvInferenceTime)
        tvDownloadStatus = findViewById(R.id.tvDownloadStatus)
        tvThreadCount = findViewById(R.id.tvThreadCount)
        btnRecord = findViewById(R.id.btnRecord)
        btnDownloadModel = findViewById(R.id.btnDownloadModel)
        btnPickModel = findViewById(R.id.btnPickModel)
        btnCopy = findViewById(R.id.btnCopy)
        btnClearLog = findViewById(R.id.btnClearLog)
        progressDownload = findViewById(R.id.progressDownload)
        cbVad = findViewById(R.id.cbVad)
        cbTranslate = findViewById(R.id.cbTranslate)
        cbDiarize = findViewById(R.id.cbDiarize)
        cbHush = findViewById(R.id.cbHush)
        spinnerModel = findViewById(R.id.spinnerModel)
        spinnerLang = findViewById(R.id.spinnerLang)
        seekBarThreads = findViewById(R.id.seekBarThreads)

        tvSystemInfo.text = Build.SUPPORTED_ABIS[0]

        btnDownloadModel.setOnClickListener { downloadModel() }
        btnPickModel.setOnClickListener { pickModelFile() }
        btnCopy.setOnClickListener { copyResult() }
        btnClearLog.setOnClickListener { tvLog.text = "" }
    }

    private fun setupSpinner() {
        val modelNames = resources.getStringArray(R.array.models)
        val langNames = resources.getStringArray(R.array.languages)

        spinnerModel.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modelNames)
        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedModelIndex = position
                cbDiarize.isChecked = position == 5 || position == 6
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        spinnerModel.setSelection(4)

        spinnerLang.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, langNames)
    }

    private fun setupSeekBar() {
        seekBarThreads.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                tvThreadCount.text = maxOf(1, progress).toString()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupRecordButton() {
        btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED) {
                        startRecording()
                    } else {
                        requestPermissions()
                        appendLog("ERROR: Microphone permission required")
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    stopRecording()
                    true
                }
                else -> false
            }
        }
    }

    private fun requestPermissions() {
        val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) perms.add(Manifest.permission.POST_NOTIFICATIONS)
        requestPermissions(perms.toTypedArray(), 100)
    }

    private fun loadExistingModel() {
        scope.launch {
            try {
                val model = modelMap[selectedModelIndex] ?: return@launch
                val file = File(filesDir, model.fileName)
                if (file.exists()) {
                    appendLog("Found cached model: ${model.fileName}")
                    loadModel(file.absolutePath)
                } else {
                    appendLog("No cached model. Download or pick a file.")
                }
            } catch (e: Exception) {
                appendLog("Load check error: ${e.message}")
            }
        }
    }

    private fun downloadModel() {
        scope.launch {
            try {
                val model = modelMap[selectedModelIndex] ?: return@launch
                appendLog("Downloading: ${model.displayName}...")

                btnDownloadModel.isEnabled = false
                btnDownloadModel.text = "Downloading..."
                progressDownload.visibility = View.VISIBLE
                tvDownloadStatus.visibility = View.VISIBLE

                val elapsed = measureTimeMillis {
                    WhisperModelDownloader.download(model, filesDir) { progress ->
                        runOnUiThread {
                            progressDownload.progress = progress.percent
                            val mb = String.format("%.1f", progress.bytesRead / 1e6)
                            val totalMb = String.format("%.1f", progress.totalBytes / 1e6)
                            tvDownloadStatus.text = "$mb / $totalMb MB (${progress.percent}%)"
                        }
                    }
                }

                tvDownloadStatus.text = "Done in ${elapsed / 1000}s"
                appendLog("Downloaded ${model.displayName} in ${elapsed / 1000}s")

                val file = File(filesDir, model.fileName)
                loadModel(file.absolutePath)
            } catch (e: Exception) {
                appendLog("Download error: ${e.message}")
                tvDownloadStatus.text = "Error: ${e.message}"
            } finally {
                runOnUiThread {
                    btnDownloadModel.isEnabled = true
                    btnDownloadModel.text = "Download Model"
                }
            }
        }
    }

    private fun pickModelFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "*/*"
            addCategory(Intent.CATEGORY_OPENABLE)
            putExtra(Intent.EXTRA_TITLE, "Select GGUF model file")
        }
        startActivityForResult(intent, 200)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 200 && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                try {
                    val inputStream = contentResolver.openInputStream(uri)
                    val file = File(cacheDir, "picked_model.gguf")
                    inputStream?.use { input ->
                        file.outputStream().use { output -> input.copyTo(output) }
                    }
                    appendLog("Picked model: ${file.absolutePath} (${file.length() / 1e6} MB)")
                    scope.launch { loadModel(file.absolutePath) }
                } catch (e: Exception) {
                    appendLog("Pick error: ${e.message}")
                }
            }
        }
    }

    private suspend fun loadModel(path: String) {
        try {
            withContext(Dispatchers.IO) {
                whisperContext?.release()
                val elapsed = measureTimeMillis {
                    whisperContext = WhisperContext.createContextFromFile(path, useGpu = false, flashAttn = true)
                }
                val info = whisperContext!!.getModelInfo()
                withContext(Dispatchers.Main) {
                    tvModelStatus.text = "Loaded: ${info.modelName}"
                    tvModelStatus.setTextColor(Color.parseColor("#3FB950"))
                    tvModelInfo.text = "${info.quantizationName} | ctx=${info.textCtx} | audio=${info.audioCtx} | multi=${info.multilingual}"
                    btnRecord.isEnabled = true
                    btnRecord.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#DA3633"))
                    appendLog("Model loaded in ${elapsed}ms: ${info.modelName} (${info.quantizationName})")
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                tvModelStatus.text = "Load failed: ${e.message}"
                tvModelStatus.setTextColor(Color.parseColor("#F85149"))
                appendLog("ERROR loading model: ${e.message}")
            }
        }
    }

    private fun startRecording() {
        if (whisperContext == null) {
            Toast.makeText(this, "Load a model first!", Toast.LENGTH_SHORT).show()
            return
        }

        isRecording = true
        audioBuffer.clear()
        recordStartTime = System.currentTimeMillis()
        tvTimer.visibility = View.VISIBLE
        tvResult.text = "Recording..."
        tvResult.setTextColor(Color.parseColor("#8B949E"))
        btnRecord.text = "Recording..."
        btnRecord.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#B62324"))
        btnCopy.visibility = View.GONE

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize * 2
        )

        audioRecord?.startRecording()
        appendLog("Recording started (16kHz mono)...")

        recordThread = Thread {
            val readBuffer = ShortArray(bufferSize)
            while (isRecording) {
                val read = audioRecord?.read(readBuffer, 0, bufferSize) ?: -1
                if (read > 0) {
                    for (i in 0 until read) {
                        audioBuffer.add(readBuffer[i].toFloat() / 32768.0f)
                    }
                }
            }
        }.apply { start() }

        // Timer
        timerHandler.postDelayed(object : Runnable {
            override fun run() {
                if (isRecording) {
                    val elapsed = ((System.currentTimeMillis() - recordStartTime) / 1000).toInt()
                    val min = elapsed / 60
                    val sec = elapsed % 60
                    tvTimer.text = String.format("%02d:%02d", min, sec)
                    timerHandler.postDelayed(this, 500)
                }
            }
        }, 500)
    }

    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false

        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            recordThread?.join(1000)
        } catch (e: Exception) {
            appendLog("Stop recording error: ${e.message}")
        }

        timerHandler.removeCallbacksAndMessages(null)
        tvTimer.visibility = View.GONE
        btnRecord.text = "Hold to Record"
        btnRecord.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#DA3633"))

        val samples = audioBuffer.toFloatArray()
        val durationSec = samples.size.toFloat() / SAMPLE_RATE
        appendLog("Recorded ${String.format("%.1f", durationSec)}s (${samples.size} samples)")

        if (samples.size < SAMPLE_RATE) {
            tvResult.text = "Too short. Hold longer to record at least 1 second."
            tvResult.setTextColor(Color.parseColor("#F85149"))
            return
        }

        tvResult.text = "Transcribing..."
        tvResult.setTextColor(Color.parseColor("#58A6FF"))
        transcribeAudio(samples)
    }

    private fun transcribeAudio(samples: FloatArray) {
        scope.launch {
            try {
                val langIdx = spinnerLang.selectedItemPosition
                val lang = if (langIdx == 0) null else langCodes.getOrNull(langIdx)
                val threads = maxOf(1, seekBarThreads.progress)

                val config = TranscribeConfig(
                    numThreads = threads,
                    maxTextCtx = 448,
                    language = lang,
                    detectLanguage = lang == null,
                    translate = cbTranslate.isChecked,
                    enableVad = cbVad.isChecked,
                    vadThreshold = 0.5f,
                    carryInitialPrompt = true,
                    enableHushWords = cbHush.isChecked,
                    enableDiarize = cbDiarize.isChecked,
                    flashAttn = true
                )

                appendLog("Transcribing: threads=$threads, vad=${config.enableVad}, " +
                        "diarize=${config.enableDiarize}, hush=${config.enableHushWords}, lang=${lang ?: "auto"}")

                var timeMs = 0L
                val transcription = withContext(Dispatchers.IO) {
                    var result: TranscribeResult? = null
                    timeMs = measureTimeMillis {
                        result = whisperContext!!.transcribe(samples, config)
                    }
                    result!!
                }
                val speedup = if (timeMs > 0) String.format("%.1f", samples.size.toFloat() / SAMPLE_RATE / (timeMs / 1000f)) else "N/A"

                withContext(Dispatchers.Main) {
                    val builder = StringBuilder()
                    for (seg in transcription.segments) {
                        val ts0 = formatTimestamp(seg.t0Ms)
                        val ts1 = formatTimestamp(seg.t1Ms)
                        val speaker = if (seg.speakerTurnNext) " [>> TURN]" else ""
                        builder.append("[$ts0 --> $ts1]$speaker ${seg.text}\n")
                    }
                    tvResult.text = builder.toString().trim()
                    tvResult.setTextColor(Color.parseColor("#C9D1D9"))
                    tvInferenceTime.text = "${timeMs}ms (${speedup}x realtime)"
                    btnCopy.visibility = View.VISIBLE
                    appendLog("Transcribed in ${timeMs}ms (${speedup}x realtime)")
                    appendLog("Result: ${transcription.fullText.take(100)}...")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvResult.text = "Transcription error: ${e.message}"
                    tvResult.setTextColor(Color.parseColor("#F85149"))
                    appendLog("ERROR: ${e.message}")
                    e.printStackTrace()
                }
            }
        }
    }

    private fun formatTimestamp(t: Long): String {
        var ms = t * 10
        val hr = ms / 3600000; ms -= hr * 3600000
        val min = ms / 60000; ms -= min * 60000
        val sec = ms / 1000; ms -= sec * 1000
        return String.format("%02d:%02d:%02d.%03d", hr, min, sec, ms)
    }

    private fun copyResult() {
        val text = tvResult.text.toString()
        if (text.isNotEmpty()) {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as android.text.ClipboardManager
            clipboard.text = text
            Toast.makeText(this, "Copied to clipboard", Toast.LENGTH_SHORT).show()
        }
    }

    private fun appendLog(msg: String) {
        val time = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
        runOnUiThread {
            tvLog.append("[$time] $msg\n")
            val scroll = tvLog.parent as? ScrollView
            scroll?.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun tryGetSystemInfo(): String {
        return try { WhisperContext.getSystemInfo().take(60) } catch (_: Exception) { "N/A" }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        runBlocking { whisperContext?.release() }
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        timerHandler.removeCallbacksAndMessages(null)
    }
}
