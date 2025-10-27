package com.example.client

import android.Manifest
import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Process
import android.speech.RecognitionListener
import android.speech.RecognitionService
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.client.ui.theme.HandsFreeTheme
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.max
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor

enum class AssistantState { IDLE, LISTENING, THINKING, SPEAKING }

private const val TAG = "HF"
private const val TAG_VAD = "HF-VAD"
private const val TAG_STT = "HF-STT"

/* ---------- 유틸 ---------- */
private fun AudioRecord?.safeStopRelease() {
    this?.let {
        try { it.stop() } catch (_: Exception) {}
        try { it.release() } catch (_: Exception) {}
    }
}
private fun VadSilero?.safeClose() {
    try { this?.close() } catch (_: Exception) {}
}

private data class AudioEffects(
    val aec: AcousticEchoCanceler? = null,
    val ns: NoiseSuppressor? = null,
    val agc: AutomaticGainControl? = null
)
private fun attachEffects(sessionId: Int): AudioEffects {
    var aec: AcousticEchoCanceler? = null
    var ns: NoiseSuppressor? = null
    var agc: AutomaticGainControl? = null
    try {
        if (AcousticEchoCanceler.isAvailable()) {
            aec = AcousticEchoCanceler.create(sessionId)
            aec?.enabled = true
            Log.i(TAG_VAD, "AEC enabled")
        } else Log.i(TAG_VAD, "AEC not available")
    } catch (e: Exception) { Log.w(TAG_VAD, "AEC attach failed", e) }

    try {
        if (NoiseSuppressor.isAvailable()) {
            ns = NoiseSuppressor.create(sessionId)
            ns?.enabled = true
            Log.i(TAG_VAD, "NS enabled")
        } else Log.i(TAG_VAD, "NS not available")
    } catch (e: Exception) { Log.w(TAG_VAD, "NS attach failed", e) }

    try {
        if (AutomaticGainControl.isAvailable()) {
            agc = AutomaticGainControl.create(sessionId)
            agc?.enabled = true
            Log.i(TAG_VAD, "AGC enabled")
        } else Log.i(TAG_VAD, "AGC not available")
    } catch (e: Exception) { Log.w(TAG_VAD, "AGC attach failed", e) }

    return AudioEffects(aec, ns, agc)
}
private fun AudioEffects.release() {
    try { aec?.release() } catch (_: Exception) {}
    try { ns?.release() } catch (_: Exception) {}
    try { agc?.release() } catch (_: Exception) {}
}

/* ---------- 인식 서비스 선택(있으면 Google 우선) ---------- */
private fun pickRecognitionService(pm: PackageManager): ComponentName? {
    val googlePkgs = listOf("com.google.android.googlequicksearchbox")
    val services = pm.queryIntentServices(
        Intent(RecognitionService.SERVICE_INTERFACE), PackageManager.MATCH_ALL
    ).orEmpty()

    val google = services.firstOrNull { ri -> googlePkgs.any { ri.serviceInfo?.packageName == it } }?.serviceInfo
    return if (google != null) {
        Log.i(TAG_STT, "Using Google service: ${google.packageName}/${google.name}")
        ComponentName(google.packageName, google.name)
    } else {
        services.firstOrNull()?.serviceInfo?.let {
            Log.w(TAG_STT, "Using default service: ${it.packageName}/${it.name}")
            ComponentName(it.packageName, it.name)
        }.also { if (it == null) Log.e(TAG_STT, "No recognition service found") }
    }
}

/* ---------- Activity ---------- */
class MainActivity : ComponentActivity() {
    private val reqPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* 화면에서 처리 */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            reqPerm.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            HandsFreeTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { inner ->
                    AssistantScreen(modifier = Modifier.padding(inner))
                }
            }
        }
    }
}

/* ---------- UI / 상태 머신 ---------- */
@SuppressLint("MissingPermission")
@Composable
fun AssistantScreen(modifier: Modifier = Modifier) {
    var assistantState by remember { mutableStateOf(AssistantState.IDLE) }
    var spokenText by remember { mutableStateOf("") }     // 화면에는 결과만
    var vadCycleKey by remember { mutableStateOf(0) }
    var vadMode by remember { mutableStateOf(Mode.AGGRESSIVE) }
    var audioLevel by remember { mutableStateOf(0f) }

    val context = LocalContext.current
    val pm = context.packageManager
    val serviceComponent = remember { pickRecognitionService(pm) }

    DisposableEffect(Unit) {
        Log.i(TAG, "AssistantScreen mounted")
        onDispose { Log.i(TAG, "AssistantScreen disposed") }
    }

    /* ---------- VAD: 오직 IDLE에서만 마이크 점유 ---------- */
    LaunchedEffect(vadCycleKey, vadMode, assistantState) {
        val hasPerm = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        if (assistantState != AssistantState.IDLE || !hasPerm) return@LaunchedEffect

        withContext(Dispatchers.IO) {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)

            var rec: AudioRecord? = null
            var vad: VadSilero? = null
            var fx: AudioEffects? = null
            try {
                vad = VadSilero(context, SampleRate.SAMPLE_RATE_16K, FrameSize.FRAME_SIZE_512, vadMode)
                val sr = vad?.sampleRate?.value ?: 16000
                val fs = vad?.frameSize?.value ?: 512

                fun build(source: Int) = AudioRecord(
                    source, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    max(AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT), fs * 2 * 4)
                )

                // ✅ 소스 일관화: VOICE_RECOGNITION 고정, 실패 시 단 1회 MIC 폴백만
                rec = build(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                var usedSrc = if (rec?.state == AudioRecord.STATE_INITIALIZED) {
                    "VOICE_RECOGNITION"
                } else {
                    rec.safeStopRelease()
                    rec = build(MediaRecorder.AudioSource.MIC)
                    if (rec?.state == AudioRecord.STATE_INITIALIZED) "MIC(FALLBACK-ONCE)" else "INIT_FAILED"
                }
                if (rec?.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG_VAD, "AudioRecord init failed")
                    return@withContext
                }
                rec!!.startRecording()

                // ✅ 하드웨어 오디오 이펙트 부착 (가능한 경우만)
                fx = attachEffects(rec!!.audioSessionId)

                Log.i(TAG_VAD, "AudioRecord started. source=$usedSrc sr=$sr frame=$fs")

                val frame = ShortArray(fs)
                var hit = 0
                val need = 2
                var lastLog = 0L

                // ✅ 이펙트/AGC 안정화용 프리롤(약 200ms): 판정에는 영향 없음
                var prerollSamples = (sr / fs) / 5  // ≈ 0.2s 분량
                while (prerollSamples-- > 0 && isActive) {
                    val r = rec?.read(frame, 0, frame.size) ?: -1
                    if (r <= 0) break
                }

                // ❗ 판정 로직(peak, ampGate, need=2) 그대로 유지
                while (isActive) {
                    val read: Int = rec?.read(frame, 0, frame.size) ?: -1
                    if (read <= 0) continue

                    val amp = frame.maxOrNull()?.toDouble() ?: 0.0
                    val lvl = (amp / 32767.0).toFloat().coerceIn(0f, 1f)
                    withContext(Dispatchers.Main) { audioLevel = lvl }

                    val now = System.currentTimeMillis()
                    if (now - lastLog > 500) {
                        Log.d(TAG_VAD, "level=${"%.2f".format(lvl)} src=$usedSrc")
                        lastLog = now
                    }

                    // ❌ 저레벨 지속 재초기화 제거 (잡음으로 인한 잦은 재시작 방지)
                    //   → 오로지 최초 초기화 실패 시에만 MIC 폴백됨.

                    val speech = vad?.isSpeech(frame) == true
                    val ampGate = lvl > 0.08f
                    if (speech || ampGate) {
                        if (++hit >= need) {
                            Log.i(TAG_VAD, "Speech detected → LISTENING (src=$usedSrc)")
                            // STT 시작 전에 완전 반납
                            rec.safeStopRelease(); rec = null
                            fx?.release(); fx = null
                            vad.safeClose(); vad = null
                            withContext(Dispatchers.Main) { assistantState = AssistantState.LISTENING }
                            break
                        }
                    } else hit = 0
                }
            } catch (e: Exception) {
                Log.e(TAG_VAD, "VAD error: ${e.message}", e)
            } finally {
                rec.safeStopRelease()
                fx?.release()
                vad.safeClose()
                withContext(Dispatchers.Main) { audioLevel = 0f }
            }
        }
    }

    /* ---------- 상태 머신 ---------- */
    LaunchedEffect(assistantState) {
        when (assistantState) {
            AssistantState.LISTENING -> {
                Log.i(TAG, "STATE → LISTENING")
                delay(600) // 자원 전환 안정화

                if (!SpeechRecognizer.isRecognitionAvailable(context) || serviceComponent == null) {
                    Log.e(TAG_STT, "Recognition not available; skip")
                    assistantState = AssistantState.THINKING
                    return@LaunchedEffect
                }

                val sr = SpeechRecognizer.createSpeechRecognizer(context, serviceComponent)
                Log.d(TAG_STT, "Recognizer created for session")

                val sttIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                    // 요청대로 기존 값 유지
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 2300L)
                }

                var finished = false
                try {
                    sr.setRecognitionListener(object : RecognitionListener {
                        override fun onReadyForSpeech(params: Bundle?) { Log.d(TAG_STT, "onReadyForSpeech") }
                        override fun onBeginningOfSpeech() { Log.d(TAG_STT, "onBeginningOfSpeech") }
                        override fun onRmsChanged(rmsdB: Float) {}
                        override fun onBufferReceived(buffer: ByteArray?) {}
                        override fun onEndOfSpeech() { Log.d(TAG_STT, "onEndOfSpeech (no stop/cancel)") }
                        override fun onError(error: Int) {
                            Log.w(TAG_STT, "onError: $error")
                            finished = true
                        }
                        override fun onResults(results: Bundle?) {
                            val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                            if (text.isNotBlank()) spokenText = text
                            Log.i(TAG_STT, "onResults: $text")
                            finished = true
                        }
                        override fun onPartialResults(partialResults: Bundle?) {}
                        override fun onEvent(eventType: Int, params: Bundle?) {}
                    })

                    sr.startListening(sttIntent)
                    Log.d(TAG_STT, "startListening")

                    val start = System.currentTimeMillis()
                    while (!finished && System.currentTimeMillis() - start < 12_000) delay(50)
                } finally {
                    try { sr.setRecognitionListener(null) } catch (_: Exception) {}
                    try { sr.destroy(); Log.d(TAG_STT, "Recognizer destroyed (finally)") } catch (_: Exception) {}
                    assistantState = AssistantState.THINKING
                }
            }

            AssistantState.THINKING -> {
                Log.i(TAG, "STATE → THINKING")
                delay(2000)
                delay(1000) // 오디오 포커스 안정화
                assistantState = AssistantState.IDLE
                vadCycleKey++
                Log.i(TAG, "STATE → IDLE (VAD restart)")
            }

            else -> Unit
        }
    }

    /* ---------- UI ---------- */
    Column(
        modifier = modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = spokenText,
            modifier = Modifier.weight(1f).padding(horizontal = 24.dp, vertical = 8.dp),
            fontSize = 24.sp,
            textAlign = TextAlign.Center
        )
        Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
            AssistantAvatar(state = assistantState)
        }
        AudioLevelIndicator(level = audioLevel, modifier = Modifier.padding(vertical = 8.dp))
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { assistantState = AssistantState.IDLE; vadCycleKey++ }) { Text("Idle") }
            Button(onClick = { if (assistantState == AssistantState.IDLE) assistantState = AssistantState.LISTENING }) { Text("Listen") }
            Button(onClick = { assistantState = AssistantState.THINKING }) { Text("Think") }
            Button(onClick = { assistantState = AssistantState.SPEAKING }) { Text("Speak") }
        }
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { vadMode = Mode.NORMAL }) { Text("Normal") }
            Button(onClick = { vadMode = Mode.AGGRESSIVE }) { Text("Aggressive") }
            Button(onClick = { vadMode = Mode.VERY_AGGRESSIVE }) { Text("V. Aggressive") }
        }
    }
}

/* ---------- 보조 UI ---------- */
@Composable
fun AudioLevelIndicator(level: Float, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.height(50.dp).fillMaxWidth().padding(horizontal = 32.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        val barCount = 24
        for (i in 0 until barCount) {
            val active = i < (level * barCount)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(if (active) 1f else 0.2f)
                    .background(if (active) Color(0xFF1E88E5) else Color.LightGray)
            )
        }
    }
}

@Composable
fun AssistantAvatar(state: AssistantState, modifier: Modifier = Modifier) {
    val baseColor = Color(0xFF1E88E5)
    val avatarSize = 250.dp
    val tr = rememberInfiniteTransition(label = "avatar")
    val pulse by tr.animateFloat(1f, 1.6f, infiniteRepeatable(tween(1200), RepeatMode.Restart), label = "pulse")
    val alpha by tr.animateFloat(1f, 0f, infiniteRepeatable(tween(1200), RepeatMode.Restart), label = "alpha")
    val rot by tr.animateFloat(0f, 360f, infiniteRepeatable(tween(2000)), label = "rot")

    Box(modifier = modifier.size(avatarSize), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            when (state) {
                AssistantState.LISTENING -> drawCircle(
                    color = baseColor.copy(alpha = alpha),
                    radius = size.minDimension / 2 * pulse,
                    style = Stroke(width = (size.minDimension * 0.05f))
                )
                AssistantState.THINKING -> drawArc(
                    color = baseColor,
                    startAngle = rot,
                    sweepAngle = 90f,
                    useCenter = false,
                    style = Stroke(width = (size.minDimension * 0.1f))
                )
                else -> {}
            }
        }
        Box(
            modifier = Modifier.fillMaxSize().clip(CircleShape).background(baseColor),
            contentAlignment = Alignment.Center
        ) { Text(text = state.name, color = Color.White, fontSize = 32.sp, textAlign = TextAlign.Center) }
    }
}
