package com.example.client.ui

import android.Manifest
import android.content.BroadcastReceiver
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
import android.speech.RecognitionListener
import android.speech.RecognitionService
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
import com.example.client.stt.pickRecognitionService
import com.example.client.ui.components.AssistantAvatar
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.max

private const val TAG_VAD = "HF-VAD"
private const val TAG_STT = "HF-STT"

@Composable
fun AssistantScreen(modifier: Modifier = Modifier) {
    var assistantState by remember { mutableStateOf(AssistantState.IDLE) }
    var spokenText by remember { mutableStateOf("") }
    var vadCycleKey by remember { mutableStateOf(0) }
    var vadMode by remember { mutableStateOf(Mode.AGGRESSIVE) }
    var audioLevel by remember { mutableStateOf(0f) }

    val context = LocalContext.current
    val pm = context.packageManager
    val serviceComponent = remember { pickRecognitionService(pm) }

    // 디버깅 UI 상태들
    var isBluetoothInput by remember { mutableStateOf(false) }
    var currentSampleRate by remember { mutableStateOf(16000) }
    var currentFrameSize by remember { mutableStateOf(512) }
    var scoConnected by remember { mutableStateOf(false) }
    var audioMode by remember { mutableStateOf("NORMAL") }
    var deviceName by remember { mutableStateOf("N/A") }

    // ---------- VAD: IDLE에서만 마이크 점유 ----------
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
                // BT 사용 여부 판정
                val usingBt = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
                        (findPreferredInputDevice(context, true) != null)
                withContext(Dispatchers.Main) { isBluetoothInput = usingBt }

                // ⚠️ IDLE 단계에서도 BT라면 SCO를 먼저 연결한다.
                var scoOn = false
                if (usingBt) {
                    scoOn = ensureScoConnected(context, timeoutMs = 2000)
                    withContext(Dispatchers.Main) { scoConnected = scoOn }
                } else {
                    withContext(Dispatchers.Main) { scoConnected = false }
                }

                // 샘플레이트/프레임 설정
                val vadSampleRate = if (usingBt) SampleRate.SAMPLE_RATE_8K else SampleRate.SAMPLE_RATE_16K
                val vadFrameSize = if (usingBt) FrameSize.FRAME_SIZE_256 else FrameSize.FRAME_SIZE_512
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
                    source, sr,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    max(
                        AudioRecord.getMinBufferSize(
                            sr,
                            AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT
                        ),
                        fs * 4
                    )
                )

                // VOICE_RECOGNITION 우선, 실패 시 MIC 폴백
                rec = build(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                val ok = if (rec?.state == AudioRecord.STATE_INITIALIZED) true else run {
                    rec.safeStopRelease(); rec = build(MediaRecorder.AudioSource.MIC)
                    rec?.state == AudioRecord.STATE_INITIALIZED
                }
                if (!ok) {
                    Log.e(TAG_VAD, "AudioRecord init failed")
                    return@withContext
                }

                // 선호 입력 장치 적용
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    rec.applyPreferredInput(context, preferBluetooth = usingBt)
                }

                // 캔슬러/노이즈 억제/AGC 적용
                fx = attachAudioEffects(rec?.audioSessionId ?: -1)

                // 녹음 시작
                rec?.startRecording()

                // 레벨 기반 VAD 게이트 (8k는 약하게)
                val buf = ShortArray(fs)
                while (assistantState == AssistantState.IDLE) {
                    val n = rec?.read(buf, 0, buf.size) ?: 0
                    if (n <= 0) {
                        withContext(Dispatchers.Main) { audioLevel = 0f }
                        delay(10)
                        continue
                    }

                    // amplitude 계산
                    var peak = 0
                    for (i in 0 until n) {
                        val v = kotlin.math.abs(buf[i].toInt())
                        if (v > peak) peak = v
                    }
                    val lvl = (peak / 32767.0).toFloat().coerceIn(0f, 1f)
                    withContext(Dispatchers.Main) { audioLevel = lvl }

                    val ampGate = lvl > if (usingBt) 0.12f else 0.08f
                    val speech = ampGate && (vad?.isSpeech(buf) == true)
                    if (speech) {
                        withContext(Dispatchers.Main) { assistantState = AssistantState.LISTENING }
                        break
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG_VAD, "VAD loop error", t)
            } finally {
                rec.safeStopRelease()
                fx?.releaseAll()
                // IDLE 루프 종료 시 SCO 정리
                if (scoConnected) {
                    tearDownSco(context)
                    withContext(Dispatchers.Main) { scoConnected = false }
                }
                withContext(Dispatchers.Main) { audioLevel = 0f }
            }
        }
    }

    // ---------- STT 상태 머신 ----------
    LaunchedEffect(assistantState) {
        when (assistantState) {
            AssistantState.LISTENING -> {
                // BT인 경우 SCO는 이미 켜져 있음(위에서). 안정화만 소폭 대기.
                delay(300)

                val usingBt = isBluetoothInput
                if (!SpeechRecognizer.isRecognitionAvailable(context) || serviceComponent == null) {
                    assistantState = AssistantState.THINKING
                    if (usingBt && scoConnected) tearDownSco(context)
                    return@LaunchedEffect
                }

                val sr = SpeechRecognizer.createSpeechRecognizer(context, serviceComponent)
                val sttIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                }

                val mutex = Mutex()
                sr.setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        Log.i(TAG_STT, "onReadyForSpeech")
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        val list = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!list.isNullOrEmpty()) {
                            val text = list.joinToString(" / ")
                            CoroutineScope(Dispatchers.Main).launch {
                                mutex.withLock { spokenText = text }
                            }
                        }
                    }

                    override fun onResults(results: Bundle?) {
                        val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        CoroutineScope(Dispatchers.Main).launch {
                            mutex.withLock { spokenText = list?.firstOrNull().orEmpty() }
                            assistantState = AssistantState.THINKING
                        }
                    }

                    override fun onError(error: Int) {
                        Log.w(TAG_STT, "onError: $error")
                        CoroutineScope(Dispatchers.Main).launch {
                            mutex.withLock { assistantState = AssistantState.THINKING }
                        }
                    }

                    // no-op
                    override fun onBeginningOfSpeech() {}
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {}
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                    override fun onRmsChanged(rmsdB: Float) {}
                })

                sr.startListening(sttIntent)
            }

            AssistantState.THINKING -> {
                // 실제 응답 생성 등 비즈니스 로직...
                delay(400)
                assistantState = AssistantState.IDLE
                // 다음 VAD 사이클
                vadCycleKey++
            }

            else -> Unit
        }
    }

    // ---------- UI ----------
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0B0F14)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            AssistantAvatar(
                state = assistantState,
                level = audioLevel,
                modifier = Modifier
                    .fillMaxWidth(0.65f)
                    .aspectRatio(1f)
            )
            Spacer(Modifier.height(24.dp))
            Text(
                text = spokenText,
                color = Color(0xFFE6EDF3),
                fontSize = 22.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp)
            )
            Spacer(Modifier.height(12.dp))
            AudioLevelIndicator(level = audioLevel, modifier = Modifier.padding(vertical = 8.dp))
            Spacer(Modifier.height(12.dp))
            // 디버그 패널
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
        }
    }
}

/* ---------- SCO helpers ---------- */
suspend fun ensureScoConnected(context: Context, timeoutMs: Long = 2000): Boolean {
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
        // timeout or failure
    } finally {
        try { context.unregisterReceiver(receiver) } catch (_: Exception) {}
    }
    return connected
}

fun tearDownSco(context: Context) {
    val am = context.getSystemService(AudioManager::class.java) ?: return
    try {
        am.isBluetoothScoOn = false
        am.stopBluetoothSco()
    } catch (_: Exception) { }
    am.mode = AudioManager.MODE_NORMAL
}

/* ---------- UI bits ---------- */

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
        modifier = Modifier
            .fillMaxWidth()
            .padding(20.dp)
            .background(Color(0xFF121A22))
            .padding(16.dp)
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
        modifier = modifier
            .height(50.dp)
            .fillMaxWidth()
            .padding(horizontal = 32.dp),
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
                    .background(if (active) Color(0xFF1E88E5) else Color(0xFF33424F))
            )
        }
    }
}
