# whisperGF

whisper.cpp Android AAR Library - Speech-to-Text for Android

## 📦 التحميل / Download

| الإصدار | الرابط | الحجم |
|---------|--------|-------|
| v1.0.0 | [whisper-android.zip](release/whisper-android.zip) | 6.8 MB |

## 🚀 طريقة الاستخدام / Usage

1. انسخ ملف `whisper-android.aar` إلى مجلد `app/libs/` في مشروعك
2. أضف التبعية في `build.gradle`:

```groovy
dependencies {
    implementation files('libs/whisper-android.aar')
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1'
}
```

3. استخدم المكتبة في كودك:

```kotlin
import com.whispercpp.whisper.WhisperContext

// تحميل النموذج
val context = WhisperContext.createContextFromFile("/path/to/model.bin")

// تحويل الصوت إلى نص
val audioData: FloatArray = // بيانات الصوت (16kHz, mono, f32)
val transcription = context.transcribeData(audioData)
println(transcription)

// أو تحميل من Assets
val context = WhisperContext.createContextFromAsset(assetManager, "model.bin")

// تحرير الذاكرة
context.release()
```

## 📋 المتطلبات / Requirements

| الخاصية | القيمة |
|---------|--------|
| Min SDK | 26 (Android 8.0) |
| Target SDK | 34 |
| ABI | arm64-v8a |
| Kotlin | 1.9.0 |
| NDK | r25c (25.2.9519653) |

## 🏗️ هيكل المكتبة / Library Structure

```
whisperGF/
├── release/
│   ├── aar/
│   │   └── whisper-android.aar    # ملف المكتبة
│   └── whisper-android.zip         # نسخة مضغوطة
├── src/
│   ├── main/
│   │   ├── java/com/whispercpp/whisper/
│   │   │   ├── LibWhisper.kt       # واجهة Kotlin API
│   │   │   └── WhisperCpuConfig.kt # إعدادات المعالج
│   │   ├── jni/whisper/
│   │   │   ├── jni.c               # JNI bridge
│   │   │   └── CMakeLists.txt      # إعدادات البناء
│   │   └── AndroidManifest.xml
├── lib.build.gradle                # إعدادات بناء المكتبة
└── README.md
```

## 📝 أمر التحديث / Update Command

لتحديث المكتبة وإصدار نسخة جديدة:

```bash
# بناء الإصدار الجديد
./build-release.sh <new_version>

# أو يدوياً:
# 1. حدث الملفات في مجلد release/
# 2. أنشئ tag جديد: git tag v1.x.x
# 3. ارفع: git push && git push --tags
```

## 📄 الترخيص / License

هذه المكتبة مبنية على [whisper.cpp](https://github.com/ggerganov/whisper.cpp) المرخص تحت رخصة MIT.
