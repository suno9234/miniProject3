package com.example.client.ui

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.client.audio.*
import com.example.client.state.AssistantState
import com.example.client.ui.components.AssistantAvatar
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.max
import kotlin.math.min

/* =========================== Config =========================== */
object AssistantConfig {
    // VAD
    const val DEFAULT_SAMPLE_RATE_NORMAL = 16000
    const val DEFAULT_SAMPLE_RATE_BT = 8000
    const val DEFAULT_FRAME_SIZE_NORMAL = 512
    const val DEFAULT_FRAME_SIZE_BT = 256
    const val AMP_GATE_NORMAL = 0.08f
    const val AMP_GATE_BT = 0.12f
    val DEFAULT_VAD_MODE: Mode = Mode.AGGRESSIVE

    // SCO
    const val SCO_CONNECT_TIMEOUT_MS = 2000L

    // 상태/세션
    const val LISTENING_DELAY_MS = 0L
    const val THINKING_DELAY_MS = 400L
    const val SILENCE_FINALIZE_MS = 5000L      // 앱 자체 종료(마지막 partial부터)
    const val WATCHDOG_INTERVAL_MS = 200L

    // STT Intent
    const val STT_MAX_RESULTS = 1
    const val STT_LANGUAGE = "ko-KR"

    // 재기동 히스테리시스/백오프
    const val MIN_RESTART_GAP_MS = 150L        // 두 start 사이 최소 간격
    const val BACKOFF_BASE_MS = 200L           // 1회 idle 종료 후 지연 시작점
    const val BACKOFF_MAX_MS = 1500L           // 백오프 상한
    const val RESTART_DELAY_MS = 0L            // 기본 지연(백오프에 더해짐; 보통 0)

    // Google STT
    const val GOOGLE_APP_PACKAGE = "com.google.android.googlequicksearchbox"
    const val GOOGLE_SERVICE_CLASS =
        "com.google.android.voicesearch.serviceapi.GoogleRecognitionService"

    // RMS → UI level 정규화(표시용)
    const val RMS_MIN_DB = -50f
    const val RMS_MAX_DB = -10f
}

private const val TAG_VAD = "HF-VAD"
private const val TAG_STT = "HF-STT"

/* ================== Google STT Component ================== */
private fun googleRecognizerComponent(context: Context): ComponentName? {
    val pm = context.packageManager
    val cn = ComponentName(
        AssistantConfig.GOOGLE_APP_PACKAGE,
        AssistantConfig.GOOGLE_SERVICE_CLASS
    )
    return try { pm.getServiceInfo(cn, 0); cn } catch (_: Exception) { null }
}

/* ============================ UI ============================ */
@Composable
fun AssistantScreen(modifier: Modifier = Modifier) {
    var assistantState by remember { mutableStateOf(AssistantState.IDLE) }

    // 텍스트 버퍼
    var transcriptConfirmed by remember { mutableStateOf("") }
    var transcriptLive by remember { mutableStateOf("") }
    var displayText by remember { mutableStateOf("") }

    // 상태 표시
    var isSpeaking by remember { mutableStateOf(false) }
    var audioLevel by remember { mutableStateOf(0f) }  // 0~1, 표시용

    // 디버그
    var vadCycleKey by remember { mutableStateOf(0) }
    var vadMode by remember { mutableStateOf(AssistantConfig.DEFAULT_VAD_MODE) }
    var isBluetoothInput by remember { mutableStateOf(false) }
    var currentSampleRate by remember { mutableStateOf(AssistantConfig.DEFAULT_SAMPLE_RATE_NORMAL) }
    var currentFrameSize by remember { mutableStateOf(AssistantConfig.DEFAULT_FRAME_SIZE_NORMAL) }
    var scoConnected by remember { mutableStateOf(false) }
    var audioMode by remember { mutableStateOf("NORMAL") }
    var deviceName by remember { mutableStateOf("N/A") }

    val context = LocalContext.current

    /* -------------------- VAD: IDLE에서만 점유 -------------------- */
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
            val am = context.getSystemService(AudioManager::class.java)

            try {
                val usingBt = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
                        (findPreferredInputDevice(context, true) != null)
                withContext(Dispatchers.Main) { isBluetoothInput = usingBt }

                if (usingBt) {
                    val ok = ensureScoConnected(context, AssistantConfig.SCO_CONNECT_TIMEOUT_MS)
                    withContext(Dispatchers.Main) { scoConnected = ok }
                } else withContext(Dispatchers.Main) { scoConnected = false }

                val tgtSr = if (usingBt) AssistantConfig.DEFAULT_SAMPLE_RATE_BT else AssistantConfig.DEFAULT_SAMPLE_RATE_NORMAL
                val tgtFs = if (usingBt) AssistantConfig.DEFAULT_FRAME_SIZE_BT else AssistantConfig.DEFAULT_FRAME_SIZE_NORMAL
                val vadSampleRate = if (tgtSr == 8000) SampleRate.SAMPLE_RATE_8K else SampleRate.SAMPLE_RATE_16K
                val vadFrameSize = if (tgtFs == 256) FrameSize.FRAME_SIZE_256 else FrameSize.FRAME_SIZE_512

                vad = VadSilero(context, vadSampleRate, vadFrameSize, vadMode)

                val sr = vad.sampleRate.value
                val fs = vad.frameSize.value
                withContext(Dispatchers.Main) {
                    currentSampleRate = sr
                    currentFrameSize = fs
                    audioMode = when (am?.mode) {
                        AudioManager.MODE_IN_COMMUNICATION -> "IN_COMMUNICATION"
                        AudioManager.MODE_IN_CALL -> "IN_CALL"
                        else -> "NORMAL"
                    }
                    deviceName = getCurrentInputDeviceName(context) ?: "N/A"
                }

                fun build(source: Int) = AudioRecord(
                    source, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    max(AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT), fs * 4)
                )

                rec = build(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                val ok = if (rec?.state == AudioRecord.STATE_INITIALIZED) true else run {
                    rec.safeStopRelease(); rec = build(MediaRecorder.AudioSource.MIC)
                    rec?.state == AudioRecord.STATE_INITIALIZED
                }
                if (!ok) {
                    Log.e(TAG_VAD, "AudioRecord init failed")
                    return@withContext
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    rec.applyPreferredInput(context, preferBluetooth = usingBt)
                }

                fx = attachAudioEffects(rec?.audioSessionId ?: -1)
                rec?.startRecording()

                val buf = ShortArray(fs)
                while (assistantState == AssistantState.IDLE) {
                    val n = rec?.read(buf, 0, buf.size) ?: 0
                    if (n <= 0) { withContext(Dispatchers.Main) { audioLevel = 0f }; delay(10); continue }

                    var peak = 0
                    for (i in 0 until n) {
                        val v = kotlin.math.abs(buf[i].toInt())
                        if (v > peak) peak = v
                    }
                    val lvl = (peak / 32767.0f).coerceIn(0f, 1f)
                    withContext(Dispatchers.Main) { audioLevel = lvl }

                    val ampGate = lvl > if (usingBt) AssistantConfig.AMP_GATE_BT else AssistantConfig.AMP_GATE_NORMAL
                    val speech = ampGate && (vad?.isSpeech(buf) == true)
                    if (speech) {
                        withContext(Dispatchers.Main) {
                            isSpeaking = true
                            transcriptConfirmed = ""
                            transcriptLive = ""
                            displayText = ""
                            assistantState = AssistantState.LISTENING
                        }
                        break
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG_VAD, "VAD loop error", t)
            } finally {
                rec.safeStopRelease()
                fx?.releaseAll()
                if (scoConnected) {
                    tearDownSco(context)
                    withContext(Dispatchers.Main) { scoConnected = false }
                }
                withContext(Dispatchers.Main) { audioLevel = 0f }
            }
        }
    }

    /* --------------- STT 세션 (LISTENING에서만 열고 닫음) --------------- */
    if (assistantState == AssistantState.LISTENING) {
        DisposableEffect(Unit) {
            val main = CoroutineScope(Dispatchers.Main.immediate)
            val bg = CoroutineScope(SupervisorJob() + Dispatchers.Default)

            val google = googleRecognizerComponent(context)
            val sr: SpeechRecognizer = try {
                if (!SpeechRecognizer.isRecognitionAvailable(context)) throw IllegalStateException("STT not available")
                if (google != null) SpeechRecognizer.createSpeechRecognizer(context, google)
                else SpeechRecognizer.createSpeechRecognizer(context)
            } catch (e: Exception) {
                Log.e(TAG_STT, "createSpeechRecognizer failed", e)
                main.launch {
                    isSpeaking = false
                    displayText = (transcriptConfirmed + " " + transcriptLive).trim()
                    transcriptLive = ""
                    assistantState = AssistantState.THINKING
                }
                return@DisposableEffect onDispose {}
            }

            val sttIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, AssistantConfig.STT_LANGUAGE)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, AssistantConfig.STT_MAX_RESULTS)
                // 엔진 종료를 늦춰 앱 종료 규칙이 우선되게
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, AssistantConfig.SILENCE_FINALIZE_MS)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, AssistantConfig.SILENCE_FINALIZE_MS)
            }

            val mutex = Mutex()
            var lastHeardAt = SystemClock.elapsedRealtime()
            var sessionActive = true
            var finalized = false

            // 재기동 히스테리시스
            var isRestarting = false
            var lastStartAt = 0L
            var consecutiveIdleEnds = 0          // partial 없이 끝난 횟수
            var hadPartialSinceStart = false     // 이번 start 이후 partial 수신 여부

            fun mapRmsToLevel(rms: Float): Float {
                val clamped = min(AssistantConfig.RMS_MAX_DB, maxOf(rms, AssistantConfig.RMS_MIN_DB))
                val norm = (clamped - AssistantConfig.RMS_MIN_DB) / (AssistantConfig.RMS_MAX_DB - AssistantConfig.RMS_MIN_DB)
                return norm.coerceIn(0f, 1f)
            }

            val finalizeSession: (String) -> Unit = { reason ->
                if (finalized) {
                    // 이미 종료된 세션이면 아무것도 하지 않음
                } else {
                    finalized = true
                    sessionActive = false
                    Log.i(TAG_STT, "finalize by: $reason")
                    main.launch {
                        try { sr.cancel() } catch (_: Exception) {}
                        try { sr.stopListening() } catch (_: Exception) {}
                        try { sr.destroy() } catch (_: Exception) {}
                        displayText = buildString {
                            append(transcriptConfirmed)
                            if (transcriptLive.isNotBlank()) {
                                if (isNotBlank()) append(' ')
                                append(transcriptLive)
                            }
                        }.trim()
                        transcriptLive = ""
                        isSpeaking = false
                        assistantState = AssistantState.THINKING
                    }
                }
            }

            fun calcBackoffMs(): Long {
                if (consecutiveIdleEnds <= 0) return AssistantConfig.RESTART_DELAY_MS
                val exp = 1L shl (consecutiveIdleEnds - 1).coerceAtMost(10)
                val ms = AssistantConfig.BACKOFF_BASE_MS * exp
                return min(ms, AssistantConfig.BACKOFF_MAX_MS) + AssistantConfig.RESTART_DELAY_MS
            }

            fun restartIfAllowed(reason: String, idleEnd: Boolean) {
                if (!sessionActive || isRestarting || finalized) return
                val now = SystemClock.elapsedRealtime()
                if (now - lastStartAt < AssistantConfig.MIN_RESTART_GAP_MS) return

                if (idleEnd) {
                    consecutiveIdleEnds += 1
                }
                val delayMs = calcBackoffMs()
                isRestarting = true
                Log.d(TAG_STT, "restart listening: $reason | backoff=${delayMs}ms, idleEnds=$consecutiveIdleEnds")
                main.launch {
                    try { sr.cancel() } catch (_: Exception) {}
                    delay(delayMs)
                    try {
                        hadPartialSinceStart = false
                        lastStartAt = SystemClock.elapsedRealtime()
                        sr.startListening(sttIntent)
                    } catch (e: Exception) {
                        Log.e(TAG_STT, "startListening failed ($reason)", e)
                        finalizeSession("start_fail")
                    } finally {
                        isRestarting = false
                    }
                }
            }

            // 워치독: partial 기준 5초 무음 → finalize
            val watchdog = bg.launch {
                while (isActive && sessionActive && !finalized) {
                    val gap = SystemClock.elapsedRealtime() - lastHeardAt
                    if (gap >= AssistantConfig.SILENCE_FINALIZE_MS) {
                        finalizeSession("app_silence_${gap}ms")
                        break
                    }
                    delay(AssistantConfig.WATCHDOG_INTERVAL_MS)
                }
            }

            sr.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}

                override fun onPartialResults(partialResults: Bundle?) {
                    val list = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!list.isNullOrEmpty() && !finalized) {
                        lastHeardAt = SystemClock.elapsedRealtime()
                        val partial = list.joinToString(" ")
                        main.launch {
                            mutex.withLock {
                                if (!isSpeaking) isSpeaking = true
                                hadPartialSinceStart = true
                                consecutiveIdleEnds = 0 // 입력 확인 → 백오프 리셋
                                transcriptLive = partial
                                displayText = buildString {
                                    append(transcriptConfirmed)
                                    if (transcriptLive.isNotBlank()) {
                                        if (isNotBlank()) append(' ')
                                        append(transcriptLive)
                                    }
                                }.trim()
                            }
                        }
                    }
                }

                override fun onResults(results: Bundle?) {
                    if (finalized) return
                    val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                    main.launch {
                        mutex.withLock {
                            if (text.isNotBlank()) {
                                transcriptConfirmed = buildString {
                                    append(transcriptConfirmed)
                                    if (transcriptConfirmed.isNotBlank()) append(' ')
                                    append(text)
                                }.trim()
                                transcriptLive = ""
                                displayText = transcriptConfirmed
                            }
                        }
                    }
                    // 입력 없이 끝났다면 idle 종료로 간주(백오프 가산)
                    val idleEnd = !hadPartialSinceStart && text.isBlank()
                    restartIfAllowed("onResults", idleEnd)
                }

                override fun onError(error: Int) {
                    if (finalized) return
                    Log.w(TAG_STT, "onError=$error")
                    // NO_MATCH(7), SPEECH_TIMEOUT(6)은 idle 종료로 간주 → 백오프 가산
                    val idleEnd = (error == SpeechRecognizer.ERROR_NO_MATCH ||
                            error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT)
                    restartIfAllowed("onError=$error", idleEnd)
                }

                override fun onEndOfSpeech() {
                    if (finalized) return
                    // 엔진 종료 신호 → 입력 없었던 세션이면 idle 종료로 간주
                    restartIfAllowed("onEndOfSpeech", idleEnd = !hadPartialSinceStart)
                }

                override fun onRmsChanged(rmsdB: Float) {
                    val lvl = mapRmsToLevel(rmsdB)
                    main.launch { audioLevel = lvl } // 표시용
                }

                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })

            // 최초 시작(메인)
            main.launch {
                try {
                    delay(AssistantConfig.LISTENING_DELAY_MS)
                    hadPartialSinceStart = false
                    lastStartAt = SystemClock.elapsedRealtime()
                    sr.startListening(sttIntent)
                } catch (e: Exception) {
                    Log.e(TAG_STT, "startListening failed (initial)", e)
                    finalizeSession("start_fail")
                }
            }

            onDispose {
                try { watchdog.cancel() } catch (_: Exception) {}
                finalizeSession("dispose")
                audioLevel = 0f
            }
        }
    }

    /* ---------------- THINKING → IDLE 전환 ---------------- */
    LaunchedEffect(assistantState) {
        if (assistantState == AssistantState.THINKING) {
            delay(AssistantConfig.THINKING_DELAY_MS)
            assistantState = AssistantState.IDLE
            vadCycleKey++
        }
    }

    /* ------------------------------ UI ------------------------------ */
    Box(
        modifier = modifier.fillMaxSize().background(Color(0xFF0B0F14)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            AssistantAvatar(
                state = assistantState,
                level = audioLevel,
                modifier = Modifier.fillMaxWidth(0.65f).aspectRatio(1f)
            )
            Spacer(Modifier.height(24.dp))
            Text(
                text = displayText,
                color = Color(0xFFE6EDF3),
                fontSize = 22.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp)
            )
            Spacer(Modifier.height(12.dp))
            AudioLevelIndicator(level = audioLevel, modifier = Modifier.padding(vertical = 8.dp))
            Spacer(Modifier.height(12.dp))
            DebugPanel(
                isBt = isBluetoothInput,
                device = deviceName,
                sco = if (scoConnected) "CONNECTED" else "DISCONNECTED",
                sampleRate = currentSampleRate,
                frameSize = currentFrameSize,
                vadMode = vadMode.name,
                level = audioLevel,
                audioMode = audioMode
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = if (isSpeaking) "말하는 중(앱 자체 판단)" else "대기 중",
                color = Color(0xFF9EB5D1),
                fontSize = 14.sp
            )
        }
    }
}

/* ============================= SCO helpers ============================= */
suspend fun ensureScoConnected(context: Context, timeoutMs: Long = AssistantConfig.SCO_CONNECT_TIMEOUT_MS): Boolean {
    val am = context.getSystemService(AudioManager::class.java) ?: return false
    var connected = false
    val latch = CompletableDeferred<Boolean>()
    val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            if (i?.action == AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED) {
                val state = i.getIntExtra(AudioManager.EXTRA_SCO_AUDIO_STATE, -1)
                if (state == AudioManager.SCO_AUDIO_STATE_CONNECTED) {
                    connected = true
                    if (!latch.isCompleted) latch.complete(true)
                }
            }
        }
    }
    context.registerReceiver(receiver, IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED))
    try {
        am.mode = AudioManager.MODE_IN_COMMUNICATION
        am.startBluetoothSco()
        am.isBluetoothScoOn = true
        withTimeout(timeoutMs) { latch.await() }
    } catch (_: Exception) {
    } finally {
        try { context.unregisterReceiver(receiver) } catch (_: Exception) {}
    }
    return connected
}

fun tearDownSco(context: Context) {
    val am = context.getSystemService(AudioManager::class.java) ?: return
    try { am.isBluetoothScoOn = false; am.stopBluetoothSco() } catch (_: Exception) {}
    am.mode = AudioManager.MODE_NORMAL
}

/* =============================== UI bits ============================== */

@Composable
fun DebugPanel(
    isBt: Boolean,
    device: String,
    sco: String,
    sampleRate: Int,
    frameSize: Int,
    vadMode: String,
    level: Float,
    audioMode: String
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(20.dp).background(Color(0xFF121A22)).padding(16.dp)
    ) {
        RowItem("Input Route", if (isBt) "Bluetooth" else "Normal")
        RowItem("Device", device)
        RowItem("SCO", sco)
        RowItem("SampleRate", "$sampleRate Hz")
        RowItem("FrameSize", "$frameSize")
        RowItem("VAD Mode", vadMode)
        RowItem("Level (0~1)", String.format("%.2f", level))
        RowItem("AudioMode", audioMode)
    }
}

@Composable
private fun RowItem(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = Color(0xFF9EB5D1), fontSize = 16.sp)
        Text(value, color = Color(0xFFE6EDF3), fontSize = 16.sp)
    }
}

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
                modifier = Modifier.weight(1f).fillMaxHeight(if (active) 1f else 0.2f)
                    .background(if (active) Color(0xFF1E88E5) else Color(0xFF33424F))
            )
        }
    }
}
