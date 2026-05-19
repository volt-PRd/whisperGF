# whisperGF

whisper.cpp Android AAR Library - Speech-to-Text for Android

## التحميل / Download

| الإصدار | الرابط | الحجم | المميزات |
|---------|--------|-------|----------|
| v3.0.0 | [whisper-android.zip](release/whisper-android.zip) | 12 MB | K-quants, Flash Attention, RAII, Hush Words, Local Agreement |
| v2.0.0 | [Release v2.0.0](https://github.com/volt-PRd/whisperGF/releases/tag/v2.0.0) | 12 MB | VAD, Streaming, x86_64 |
| v1.0.0 | [Release v1.0.0](https://github.com/volt-PRd/whisperGF/releases/tag/v1.0.0) | 6.8 MB | Basic |

---

## ما الجديد في v3.0.0 / What's New in v3.0.0

### أولاً: تقليل استهلاك الذاكرة (RAM)

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **GPU/mmap** | تفعيل `use_gpu=true` يُفعّل backend ذكي يستخدم mmap لتحميل النماذج | تحميل فوري + تقليل RSS |
| **K-quants (GGUF)** | دعم Q4_K_M, Q5_K_M, Q6_K, Q8_K | توزيع البتات الذكي — دقة أعلى بحجم أقل |
| **Context Size** | `maxTextCtx` قابل للتخصيص (256–448) | توفير 30% ذاكرة إضافية |

### ثانياً: تسريع الاستخراج (Inference Speed)

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **Flash Attention** | مُفعّل افتراضياً في whisper.cpp | 20-50% أسرع في المقاطع الطويلة |
| **ARMv8.2 FP16** | نسخة محسّنة لـ arm64-v8a | استغلال تعليمات FP16 |
| **Vulkan GPU** | خيار بنائي متاح (`WHISPER_VULKAN=ON`) | 10-15% تسريع إضافي |
| **Large-v3-Turbo** | مدعوم تلقائياً (4 طبقات decoder) | 8x أسرع من large-v3 |

### ثالثاً: تحسين تجربة المطور

| الميزة | الوصف | التأثير |
|--------|-------|---------|
| **RAII في JNI** | مغلفات C++ لـ jstring, jfloatArray | منع تسريبات الذاكرة |
| **carry_initial_prompt** | حمل سياق الشريح السابق | اتساق المصطلحات في Streaming |
| **Hush Words** | إضافة 0.5s صمت تلقائي | منع هلوسة النموذج |
| **Local Agreement Policy** | `LocalAgreementPolicy` في Kotlin | منع تذبذب النص في البث |
| **K-quants auto-detect** | كشف تلقائي من رأس GGUF | دعم شامل لكل أنواع التكميم |

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
val ctx = WhisperContext.createContextFromFile("model.bin")
val info = ctx.getModelInfo()
println("Model: ${info.modelName} (${info.quantizationName})")
println("Is quantized: ${info.isQuantized}")

val result = ctx.transcribe(audioData)
println(result.fullText)
ctx.release()
```

### 3. استخدام متقدم — كل المميزات الجديدة

```kotlin
val config = TranscribeConfig(
    numThreads = 4,
    maxTextCtx = 256,           // توفير ذاكرة
    language = "ar",
    detectLanguage = false,
    translate = false,
    enableVad = true,           // تخطي الصمت
    vadThreshold = 0.4f,
    carryInitialPrompt = true,  // استمرارية السياق
    initialPrompt = "المحادثة تتحدث عن الذكاء الاصطناعي",
    enableHushWords = true      // منع الهلوسة
)

val ctx = WhisperContext.createContextFromFile("ggml-large-v3-turbo-q4_k_m.bin")

// تحويل كامل
val result = ctx.transcribe(audioData, config)

// أو تحويل تدفقي مع Local Agreement Policy
val policy = LocalAgreementPolicy(minConfirmations = 2)
val streamResult = ctx.transcribeStream(audioChunks, config)
val stableText = policy.process(streamResult)
println(stableText)

ctx.release()
```

### 4. Local Agreement Policy — منع تذبذب النص

```kotlin
val policy = LocalAgreementPolicy(minConfirmations = 2)

// في حلقة البث المباشر:
// في كل مرة تصل نتيجة جديدة من الـ stream
val result = ctx.transcribeStream(currentChunk, config)
val stableText = policy.process(result)

// عرض stableText فقط — النص لن يتغير أمام المستخدم
textView.text = stableText
```

### 5. Hush Words — منع الهلوسة

```kotlin
// مفعّل افتراضياً. يضيف 0.5 ثانية صمت لنهاية الصوت
val config = TranscribeConfig(enableHushWords = true)
val result = ctx.transcribe(audioData, config)
// لن تحدث هلوسة أو تكرار في نهاية المقاطع الصامتة
```

---

## نماذج Quantized / Quantized Models

| النوع | حجم النموذج (تقريبي) | توفير RAM | الدقة | الأفضل لـ |
|-------|----------------------|-----------|------|----------|
| FP32 | 1.5 GB | — | 100% | الدقة القصوى |
| Q8_0 | ~500 MB | 20% | ~99% | توازن ممتاز |
| Q6_K | ~400 MB | 30% | ~98.5% | دقة عالية |
| **Q5_K_M** | **~350 MB** | **35%** | **~98%** | **أفضل دقة/حجم** |
| **Q4_K_M** | **~300 MB** | **45%** | **~97%** | **الأفضل للأجهزة الضعيفة** |
| Q4_0 | ~280 MB | 45% | ~97% | الأجهزة ذات RAM محدود جداً |

> Q4_K_M و Q5_K_M يستخدمان تقنية توزيع البتات غير المتساوي — بتات أكثر للطبقات الحرجة وبتات أقل للطبقات الأقل أهمية.

---

## المتطلبات / Requirements

| الخاصية | القيمة |
|---------|--------|
| Min SDK | 26 (Android 8.0) |
| ABIs | arm64-v8a, x86_64 |
| Kotlin | 1.9.0+ |
| Flash Attention | مُفعّل افتراضياً |
| Vulkan GPU | متاح كخيار بنائي |

---

## API Reference

### `WhisperContext`

| الدالة | الوصف |
|--------|-------|
| `createContextFromFile(path, useGpu=true)` | تحميل نموذج مع GPU/mmap |
| `createContextFromAsset(am, path, useGpu=true)` | تحميل من Assets |
| `getModelInfo()` | معلومات النموذج (النوع، التكميم، السياق) |
| `transcribe(data, config)` | تحويل كامل |
| `transcribeStream(data, config)` | تحويل تدفقي (أقل RAM) |
| `transcribeData(data, timestamps)` | تحويل بسيط (متوافق v1/v2) |
| `getSystemInfo()` | معلومات GGML backend |
| `release()` | تحرير الذاكرة |

### `TranscribeConfig`

| المعامل | الافتراضي | الوصف |
|---------|-----------|-------|
| `numThreads` | تلقائي | خيوط المعالجة (1–8) |
| `maxTextCtx` | 448 | حد السياق (256 للذاكرة المنخفضة) |
| `language` | null | رمز اللغة أو null للكشف التلقائي |
| `detectLanguage` | true | كشف اللغة التلقائي |
| `translate` | false | ترجمة للإنجليزية |
| `enableVad` | false | كشف النشاط الصوتي |
| `vadThreshold` | 0.5 | عتبة VAD |
| `carryInitialPrompt` | true | استمرارية السياق بين الأجزاء |
| `initialPrompt` | null | محفز نصي أولي |
| `enableHushWords` | true | منع هلوسة النموذج |

### `LocalAgreementPolicy`

| الدالة | الوصف |
|--------|-------|
| `process(result)` | معالجة النتيجة وإرجاع نص مستقر |
| `getStableText()` | النص المؤكد |
| `reset()` | إعادة تعيين |

### `ModelInfo`

| الحقل | الوصف |
|-------|-------|
| `modelName` | tiny/base/small/medium/large/large-v3-turbo |
| `quantizationName` | FP32/Q4_0/Q4_K_M/Q5_K_M/Q8_0 |
| `isQuantized` | هل النموذج مُكمّم |
| `textCtx` / `audioCtx` | أحجام السياق |

---

## الترخيص / License

مبنية على [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — رخصة MIT.
