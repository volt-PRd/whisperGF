# whisperGF

whisper.cpp Android AAR Library - Speech-to-Text for Android

## التحميل / Download

| الإصدار | الرابط | الحجم | ABI |
|---------|--------|-------|-----|
| v2.0.0 | [whisper-android.zip](release/whisper-android.zip) | 12 MB | arm64-v8a + x86_64 |
| v1.0.0 | [Release v1.0.0](https://github.com/volt-PRd/whisperGF/releases/tag/v1.0.0) | 6.8 MB | arm64-v8a |

---

## ما الجديد في v2.0.0 / What's New in v2.0.0

| الميزة | الوصف |
|--------|-------|
| Quantized Models | دعم نماذج q4_0, q5_0, q8_0 تلقائياً — تقليل استهلاك RAM بنسبة تصل إلى 45% |
| VAD | Voice Activity Detection — تخطي الأجزاء الصامتة وتسريع المعالجة |
| Context Size | التحكم في `n_max_text_ctx` — خفضه إلى 256 يوفر ~30% ذاكرة إضافية |
| Thread Control | `numThreads` كمعامل — افتراضي ذكي: min(big_cores, 8) |
| Streaming | `transcribeStream()` معالجة chunks — تقليل ذروة RAM بنسبة ~40% |
| x86_64 ABI | دعم المحاكيات والأجهزة x86_64 بجانب arm64-v8a |

---

## طريقة الاستخدام / Usage

### 1. التثبيت / Installation

انسخ ملف `whisper-android.aar` إلى مجلد `app/libs/` في مشروعك، وأضف التبعية في `build.gradle`:

```groovy
dependencies {
    implementation files('libs/whisper-android.aar')
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1'
}
```

### 2. الاستخدام الأساسي / Basic Usage

```kotlin
import com.whispercpp.whisper.WhisperContext

// تحميل النموذج (يدعم FP32 و quantized تلقائياً)
val ctx = WhisperContext.createContextFromFile("/path/to/model.bin")

// معلومات النموذج
val info = ctx.getModelInfo()
println("Quantization: ${info.quantizationName}")  // مثلاً: "Q4_0 (4-bit, ~45% less RAM)"

// تحويل صوت إلى نص
val result = ctx.transcribe(audioFloatArray)
println(result.fullText)

// تحرير الذاكرة
ctx.release()
```

### 3. استخدام متقدم / Advanced Usage

```kotlin
import com.whispercpp.whisper.*

// إعدادات مخصصة
val config = TranscribeConfig(
    numThreads = 4,              // عدد خيوط المعالجة (افتراضي: تلقائي)
    maxTextCtx = 256,            // تقليل الذاكرة (افتراضي: 448)
    language = "ar",             // لغة محددة، أو null للكشف التلقائي
    detectLanguage = false,      // تفعيل/تعطيل كشف اللغة
    translate = false,           // ترجمة للإنجليزية
    enableVad = true,            // تفعيل VAD لتخطي الصمت
    vadThreshold = 0.4f          // حساسية VAD (0.0–1.0)
)

val ctx = WhisperContext.createContextFromFile("/path/to/model.bin")

// تحويل عادي (كامل الملف في الذاكرة)
val result = ctx.transcribe(audioData, config)

// أو معالجة تدفقية (أقل استهلاك RAM)
val streamResult = ctx.transcribeStream(audioData, config)

// الوصول للنتائج التفصيلية
for (seg in result.segments) {
    println("[${seg.t0Ms}ms → ${seg.t1Ms}ms] ${seg.text}")
    println("No-speech probability: ${seg.noSpeechProb}")
}

ctx.release()
```

### 4. تحميل من Assets

```kotlin
val ctx = WhisperContext.createContextFromAsset(assetManager, "models/ggml-base-q4_0.bin")
```

### 5. نماذج Quantized / Quantized Models

المكتبة تكتشف نوع النموذج تلقائياً من رأس الملف. ما عليك سوى تحميل ملف النموذج:

| النوع | حجم النموذج ( تقريباً) | توفير RAM | الدقة |
|-------|----------------------|-----------|------|
| FP32 (الافتراضي) | 1.5 GB | — | 100% |
| Q8_0 | ~500 MB | ~20% أقل | ~99% |
| Q5_0 | ~400 MB | ~35% أقل | ~98% |
| Q4_0 | ~300 MB | ~45% أقل | ~97% |

لتحويل نموذج إلى quantized، استخدم أداة `quantize` من whisper.cpp:
```bash
./quantize models/ggml-base.bin models/ggml-base-q4_0.bin q4_0
```

---

## المتطلبات / Requirements

| الخاصية | القيمة |
|---------|--------|
| Min SDK | 26 (Android 8.0) |
| Target SDK | 34 |
| ABIs | arm64-v8a, x86_64 |
| Kotlin | 1.9.0+ |
| NDK | r25c (25.2.9519653) |
| C++ Standard | C++17 |

---

## واجهة برمجة التطبيق / API Reference

### `WhisperContext`

| الدالة | الوصف |
|--------|-------|
| `createContextFromFile(path)` | تحميل نموذج من ملف |
| `createContextFromAsset(am, path)` | تحميل نموذج من Assets |
| `createContextFromInputStream(stream)` | تحميل نموذج من InputStream |
| `getModelInfo()` | معلومات النموذج (نوع التكميم، حجم السياق) |
| `transcribe(data, config)` | تحويل صوت لنص (وضع كامل) |
| `transcribeStream(data, config)` | تحويل صوت لنص (وضع تدفقي — أقل RAM) |
| `transcribeData(data, timestamps)` | تحويل بسيط (متوافق مع v1) |
| `getSystemInfo()` | معلومات النظام |
| `benchMemory(n)` | اختبار الذاكرة |
| `benchGgmlMulMat(n)` | اختبار المصفوفات |
| `release()` | تحرير الذاكرة |

### `TranscribeConfig`

| المعامل | الافتراضي | الوصف |
|---------|-----------|-------|
| `numThreads` | تلقائي (big cores) | عدد خيوط المعالجة (1–8) |
| `maxTextCtx` | 448 | الحد الأقصى لـ tokens السياق (256 للذاكرة المنخفضة) |
| `language` | null (كشف تلقائي) | رمز اللغة ("en", "ar", "fr") |
| `detectLanguage` | true | كشف اللغة تلقائياً |
| `translate` | false | ترجمة للإنجليزية |
| `enableVad` | false | تفعيل كشف النشاط الصوتي |
| `vadThreshold` | 0.5 | عتبة حساسية VAD |

### `TranscribeResult`

| الحقل | الوصف |
|-------|-------|
| `fullText` | النص الكامل |
| `segments` | قائمة الشرائح مع الطوابع الزمنية |
| `language` | اللغة المكتشفة/المحددة |

### `TranscribeSegment`

| الحقل | الوصف |
|-------|-------|
| `text` | نص الشريحة |
| `t0Ms` | بداية الشريحة (ميلي ثانية) |
| `t1Ms` | نهاية الشريحة (ميلي ثانية) |
| `noSpeechProb` | احتمال عدم وجود كلام (0.0–1.0) |

---

## الترخيص / License

هذه المكتبة مبنية على [whisper.cpp](https://github.com/ggerganov/whisper.cpp) المرخص تحت رخصة MIT.
