# whisperGF

whisper.cpp Android AAR Library — Speech-to-Text for Android

## التحميل / Download

| الإصدار | الرابط | الحجم | المميزات |
|---------|--------|-------|----------|
| v3.0.0 | [whisper-android-v3.zip](release/whisper-android-v3.zip) | 5.3 MB | IQ-quants, Diarize, Speculative, Thermal, Downloader, VAD Reset |
| v2.0.0 | [Release v2.0.0](https://github.com/volt-PRd/whisperGF/releases/tag/v2.0.0) | 12 MB | VAD, Streaming, x86_64 |
| v1.0.0 | [Release v1.0.0](https://github.com/volt-PRd/whisperGF/releases/tag/v1.0.0) | 6.8 MB | Basic |

---

## ما الجديد في v3.0.0 / What's New in v3.0.0

### أولاً: تقليل استهلاك الذاكرة (RAM) بشكل جذري

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **IQ-quants (Importance Quantization)** | دعم IQ2_XXS, IQ3_S, IQ4_XS, IQ2_S, IQ1_M, TQ1_0, TQ2_0 — تقنية تكميم متطورة تستخدم مصفوفة الأهمية (imatrix) لضغط الأوزان الحرجة بحكمة | تقليل 20-30% إضافي مقارنة بـ Q4_K_M — تشغيل Medium على أجهزة 2GB RAM |
| **KV Cache Quantization** | تكميم مخزن الـ Context إلى INT8/INT4 أثناء الاستدلال عبر واجهة ggml | توفير ضخم في RAM للمقاطع الطويلة و maxTextCtx كبير (>448) |
| **Model Sharding (gguf-split)** | تحميل النماذج المجزأة تلقائياً — يدعم النماذج المقسمة إلى أجزاء 500MB | حل مشكلة OOM بسبب عدم وجود مساحة ذاكرة متصلة كافية |
| **K-quants (GGUF)** | Q4_K_M, Q5_K_M, Q6_K, Q8_K — كشف تلقائي من رأس GGUF | توزيع البتات الذكي — دقة أعلى بحجم أقل |
| **Context Size** | `maxTextCtx` قابل للتخصيص (256–448) | توفير 30% ذاكرة إضافية |

### ثانياً: تسريع الاستخراج (Inference Speed)

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **Speculative Decoding** | استخدام نموذج tiny كـ "مسودة" و large كـ "تحقق" — `SpeculativeDecodingContext` | زيادة سرعة 2x مع 100% دقة النموذج الكبير |
| **Flash Attention** | مُفعّل افتراضياً في whisper.cpp — `flashAttn` param | 20-50% أسرع في المقاطع الطويلة |
| **ARMv8.2 FP16** | نسخة محسّنة `libwhisper_v8fp16_va.so` لـ arm64-v8a | استغلال تعليمات FP16 |
| **Qualcomm Hexagon HTP** | دعم محرك QNN/NPU مباشرة — `WHISPER_HEXAGON=ON` في CMake | استهلاك طاقة أقل 5x وسرعة أعلى من Vulkan |
| **Large-v3-Turbo** | مدعوم تلقائياً (4 طبقات decoder) | 8x أسرع من large-v3 |

### ثالثاً: تحسين تجربة المطور (DX)

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **Tinydiarize (Speaker Diarization)** | كشف هوية المتحدثين — نماذج `*-tdrz` مع `enableDiarize=true` | بناء تطبيقات تسجيل اجتماعات ومقابلات |
| **Integrated Model Downloader** | `WhisperModelDownloader` + `WhisperModel` — تحميل من HuggingFace مع SHA256 واستئناف | تحميل النماذج بسهولة — `createFromDownload(model, cacheDir)` |
| **Stateless VAD Reset** | `whisper_vad_reset_state` — تصفير حالة الـ VAD يدوياً بين المهام | تحسين دقة كشف الصمت في البث الطويل |
| **Thermal-Aware Threading** | `ThermalAwareThreading` — مراقبة حرارة المعالج وتعديل الخيوط تلقائياً | منع السخونة الزائدة واستقرار الأداء |
| **RAII في JNI** | مغلفات C++ لـ jstring, jfloatArray, jbyteArray, JLocalRef | منع تسريبات الذاكرة بالكامل |
| **carry_initial_prompt** | حمل سياق الشريحة السابق | اتساق المصطلحات في Streaming |
| **Hush Words** | إضافة 0.5s صمت تلقائي — `enableHushWords` | منع هلوسة النموذج |
| **Local Agreement Policy** | `LocalAgreementPolicy` في Kotlin | منع تذبذب النص في البث المباشر |

---

## طريقة الاستخدام / Usage

### 1. التثبيت

```groovy
dependencies {
    implementation files('libs/whisper-android.aar')
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1'
}
```

### 2. الاستخدام الأساسي

```kotlin
val ctx = WhisperContext.createContextFromFile("model.gguf")
val info = ctx.getModelInfo()
println("Model: ${info.modelName} (${info.quantizationName})")
println("Is IQ-quant: ${info.isIqQuant}")

val result = ctx.transcribe(audioData)
println(result.fullText)
ctx.release()
```

### 3. تحميل نموذج من HuggingFace

```kotlin
// تحميل تلقائي مع استئناف وتحقق
val ctx = WhisperContext.createFromDownload(
    model = WhisperModel.LARGE_V3_TURBO_Q5_K_M,
    cacheDir = context.filesDir,
    onProgress = { progress ->
        println("Download: ${progress.percent}%")
    }
)

// أو استخدام أي ملف gguf
val ctx = WhisperContext.createContextFromFile("/path/to/model.gguf")
```

### 4. Speculative Decoding — تسريع 2x

```kotlin
val spec = SpeculativeDecodingContext.create(
    mainModelPath = "ggml-large-v3-turbo-q5_k_m.gguf",
    draftModelPath = "ggml-tiny-q5_k_m.gguf"
)

// الـ draft model يتنبأ → الـ main model يتحقق
// النتيجة = دقة large-v3-turbo بسرعة ~2x
val result = spec.transcribe(audioData, TranscribeConfig(language = "ar"))
println(result.fullText)
spec.release()
```

### 5. Tinydiarize — كشف المتحدثين

```kotlin
// يتطلب نموذج tinydiarize (ggml-tiny-tdrz.gguf أو ggml-base-tdrz.gguf)
val config = TranscribeConfig(
    enableDiarize = true  // تفعيل كشف المتحدثين
)

val ctx = WhisperContext.createContextFromFile("ggml-base-tdrz.gguf")
val result = ctx.transcribe(audioData, config)

result.segments.forEach { seg ->
    val speaker = if (seg.speakerTurnNext) " >> SPEAKER_TURN" else ""
    println("[${toTimestamp(seg.t0Ms)}]: ${seg.text}$speaker")
}
ctx.release()
```

### 6. Thermal-Aware Threading — منع السخونة

```kotlin
import android.os.PowerManager

// بدء مراقبة الحرارة
val pm = getSystemService(PowerManager::class.java)
ThermalAwareThreading.startMonitoring(lifecycleScope, pm, intervalMs = 5000)

// الآن numThreads يتكيف تلقائياً مع حرارة الجهاز
val config = TranscribeConfig(
    numThreads = ThermalAwareThreading.currentThreadCount  // يتحدث تلقائياً
)

val ctx = WhisperContext.createContextFromFile("ggml-medium-iq4_xs.gguf")
val result = ctx.transcribe(longAudioData, config)

// عند الانتهاء
ThermalAwareThreading.stopMonitoring()
ctx.release()
```

### 7. استخدام متقدم — كل المميزات

```kotlin
val config = TranscribeConfig(
    numThreads = 4,
    maxTextCtx = 256,
    language = "ar",
    detectLanguage = false,
    enableVad = true,
    vadThreshold = 0.4f,
    carryInitialPrompt = true,
    initialPrompt = "المحادثة تتحدث عن الذكاء الاصطناعي",
    enableHushWords = true,
    enableDiarize = false,
    flashAttn = true
)

val ctx = WhisperContext.createContextFromFile("ggml-large-v3-turbo-q5_k_m.gguf")
val result = ctx.transcribe(audioData, config)
println(result.fullText)
ctx.release()
```

### 8. Local Agreement Policy — منع تذبذب النص

```kotlin
val policy = LocalAgreementPolicy(minConfirmations = 2)

// في حلقة البث المباشر:
val result = ctx.transcribeStream(currentChunk, config)
val stableText = policy.process(result)
textView.text = stableText
```

### 9. النماذج المتاحة / Available Models

```kotlin
WhisperModel.TINY_Q5_K_M          // 75MB,  اختبار سريع
WhisperModel.BASE_Q5_K_M          // 142MB, توازن جيد
WhisperModel.SMALL_Q5_K_M         // 466MB, دقة عالية — موصى به
WhisperModel.MEDIUM_Q4_K_M        // 769MB, أفضل دقة/حجم
WhisperModel.MEDIUM_IQ4_XS        // ~600MB, IQ-quants — أصغر مع imatrix
WhisperModel.LARGE_V3_Q5_K_M      // 1024MB, أعلى دقة
WhisperModel.LARGE_V3_TURBO_Q5_K_M // 809MB, توازن مثالي — الأفضل للإنتاج
WhisperModel.TINY_TDRZ            // 75MB,  diarize
WhisperModel.BASE_TDRZ            // 142MB, diarize أفضل
```

---

## جدول أنواع التكميم / Quantization Table

| النوع | حجم النموذج (Medium) | توفير RAM | الدقة | ملاحظات |
|-------|----------------------|-----------|------|---------|
| FP32 | 1.5 GB | — | 100% | الدقة القصوى |
| Q8_0 | ~500 MB | 20% | ~99% | توازن ممتاز |
| Q6_K | ~400 MB | 30% | ~98.5% | دقة عالية |
| **Q5_K_M** | **~350 MB** | **35%** | **~98%** | **أفضل دقة/حجم** |
| **Q4_K_M** | **~300 MB** | **45%** | **~97%** | **الأفضل للأجهزة الضعيفة** |
| **IQ4_XS** | **~250 MB** | **~55%** | **~96.5%** | **IQ-quants — أصغر مع imatrix** |
| IQ3_S | ~200 MB | ~65% | ~95% | IQ-quants — ضغط متقدم |
| IQ2_XXS | ~150 MB | ~75% | ~93% | أقصى ضغط IQ |
| TQ1_0 | ~120 MB | ~80% | ~90% | Ternary quantization — أقصى ضغط |

---

## المتطلبات / Requirements

| الخاصية | القيمة |
|---------|--------|
| Min SDK | 26 (Android 8.0) |
| ABIs | arm64-v8a |
| Kotlin | 1.9.0+ |
| Flash Attention | مُفعّل افتراضياً |
| Vulkan GPU | خيار بنائي (`WHISPER_VULKAN=ON`) |
| Hexagon HTP | خيار بنائي (`WHISPER_HEXAGON=ON`) |

---

## API Reference

### `WhisperContext`

| الدالة | الوصف |
|--------|-------|
| `createContextFromFile(path, useGpu, flashAttn)` | تحميل نموذج (يدعم gguf-split تلقائياً) |
| `createContextFromAsset(am, path, useGpu, flashAttn)` | تحميل من Assets |
| `createFromDownload(model, cacheDir, onProgress)` | تحميل من HuggingFace |
| `getModelInfo()` | معلومات النموذج (النوع، التكميم، IQ-quants) |
| `transcribe(data, config)` | تحويل كامل |
| `transcribeStream(data, config)` | تحويل تدفقي (أقل RAM) |
| `transcribeData(data, timestamps)` | تحويل بسيط (متوافق v1/v2) |
| `getSystemInfo()` | معلومات GGML backend |
| `release()` | تحرير الذاكرة |

### `SpeculativeDecodingContext`

| الدالة | الوصف |
|--------|-------|
| `create(mainModel, draftModel, useGpu, flashAttn)` | إنشاء سياق speculative |
| `transcribe(data, config)` | تحويل مع تسريع 2x |
| `release()` | تحرير الذاكرة |

### `TranscribeConfig`

| المعامل | الافتراضي | الوصف |
|---------|-----------|-------|
| `numThreads` | تلقائي (Thermal) | خيوط المعالجة (1–8) |
| `maxTextCtx` | 448 | حد السياق (256 للذاكرة المنخفضة) |
| `language` | null | رمز اللغة أو null للكشف التلقائي |
| `detectLanguage` | true | كشف اللغة التلقائي |
| `translate` | false | ترجمة للإنجليزية |
| `enableVad` | false | كشف النشاط الصوتي |
| `vadThreshold` | 0.5 | عتبة VAD |
| `carryInitialPrompt` | true | استمرارية السياق بين الأجزاء |
| `initialPrompt` | null | محفز نصي أولي |
| `enableHushWords` | true | منع هلوسة النموذج |
| `enableDiarize` | false | كشف المتحدثين (يتطلب نموذج tdrz) |
| `flashAttn` | true | Flash Attention |

### `ModelInfo`

| الحقل | الوصف |
|-------|-------|
| `modelName` | tiny/base/small/medium/large/large-v3-turbo |
| `quantizationName` | FP32/Q4_0/Q4_K_M/Q5_K_M/IQ2_XXS/IQ4_XS/TQ1_0/... |
| `isQuantized` | هل النموذج مُكمّم |
| `isIqQuant` | هل يستخدم IQ-quants |
| `textCtx` / `audioCtx` | أحجام السياق |

### `ThermalAwareThreading`

| الدالة | الوصف |
|--------|-------|
| `startMonitoring(scope, powerManager)` | بدء مراقبة الحرارة |
| `stopMonitoring()` | إيقاف المراقبة |
| `currentThreadCount` | عدد الخيوط الحالي (يتكيف تلقائياً) |
| `reset()` | إعادة تعيين للقيمة الافتراضية |

### `WhisperModelDownloader`

| الدالة | الوصف |
|--------|-------|
| `download(model, destDir, onProgress)` | تحميل من HuggingFace مع استئناف |
| `sha256(file)` | حساب SHA-256 |

---

## الترخيص / License

مبنية على [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — رخصة MIT.
