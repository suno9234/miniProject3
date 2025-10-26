package com.example.client

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
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
import com.konovalov.vad.silero.config.SampleRate
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.max

enum class AssistantState { IDLE, LISTENING, THINKING, SPEAKING }

class MainActivity : ComponentActivity() {
    private val reqPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* granted/denied는 화면에서 처리 */ }

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

@SuppressLint("MissingPermission")
@Composable
fun AssistantScreen(modifier: Modifier = Modifier) {
    var assistantState by remember { mutableStateOf(AssistantState.IDLE) }
    var spokenText by remember { mutableStateOf("자유롭게 말씀하세요...") }
    var vadCycleKey by remember { mutableStateOf(0) } // VAD 재시작 트리거
    val context = LocalContext.current

    // SpeechRecognizer는 한 번 만들고 수명에 맞춰 파괴
    val speechRecognizer = remember { SpeechRecognizer.createSpeechRecognizer(context) }
    val currentState by rememberUpdatedState(assistantState)

    val speechIntent = remember {
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
    }

    val recognitionListener = remember {
        object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                assistantState = AssistantState.LISTENING
                spokenText = "듣는 중…"
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                assistantState = AssistantState.THINKING
            }
            override fun onError(error: Int) {
                // 바쁨 오류가 아니면 VAD 재시작
                if (error != SpeechRecognizer.ERROR_RECOGNIZER_BUSY) {
                    assistantState = AssistantState.IDLE
                    vadCycleKey++ // VAD 다시 시작
                }
            }
            override fun onResults(results: Bundle?) {
                val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!list.isNullOrEmpty()) {
                    spokenText = list[0]
                }
                assistantState = AssistantState.IDLE
                vadCycleKey++ // VAD 다시 시작
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
    }

    DisposableEffect(Unit) {
        speechRecognizer.setRecognitionListener(recognitionListener)
        onDispose { speechRecognizer.destroy() }
    }

    // ===== VAD 루프 =====
    LaunchedEffect(vadCycleKey) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) return@LaunchedEffect

        withContext(Dispatchers.IO) {
            var audioRecord: AudioRecord? = null
            var vad: VadSilero? = null
            try {
                // ▼ 생성자 사용 (create() 없음)
                vad = VadSilero(
                    context = context,
                    sampleRate = SampleRate.SAMPLE_RATE_16K,   // 지원 조합만 사용
                    frameSize = FrameSize.FRAME_SIZE_512,      // 위 sampleRate와 호환되는 값
                    mode = Mode.NORMAL,                        // NORMAL/AGGRESSIVE/VERY_AGGRESSIVE
                    speechDurationMs = 150,                    // 연속 발화 최소 ms(선택)
                    silenceDurationMs = 300                    // 침묵 최소 ms(선택)
                )

                val sampleRateHz = vad.sampleRate.value       // Int로 변환해 AudioRecord에 사용
                val frameSamples = vad.frameSize.value        // 프레임 샘플 개수

                val minBuf = AudioRecord.getMinBufferSize(
                    sampleRateHz,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )
                val bufferSizeInBytes = max(minBuf, frameSamples * 2 * 4) // 여유 있게

                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRateHz,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSizeInBytes
                )
                audioRecord.startRecording()

                val frame = ShortArray(frameSamples)
                while (isActive && assistantState == AssistantState.IDLE) {
                    val read = audioRecord.read(frame, 0, frame.size)
                    if (read > 0 && vad!!.isSpeech(frame)) {
                        withContext(Dispatchers.Main) {
                            if (assistantState == AssistantState.IDLE) {
                                assistantState = AssistantState.LISTENING
                                speechRecognizer.startListening(speechIntent)
                            }
                        }
                        break
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { /* 에러 표시 */ }
            } finally {
                try { audioRecord?.stop() } catch (_: Throwable) {}
                audioRecord?.release()
                try { vad?.close() } catch (_: Throwable) {}
            }
        }
    }

    // ===== UI =====
    Column(
        modifier = modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = spokenText,
            modifier = Modifier
                .weight(0.5f)
                .padding(24.dp),
            fontSize = 24.sp,
            textAlign = TextAlign.Center
        )
        Box(
            modifier = Modifier.weight(1f),
            contentAlignment = Alignment.Center
        ) {
            AssistantAvatar(state = assistantState)
        }
    }
}

@Composable
fun AssistantAvatar(state: AssistantState, modifier: Modifier = Modifier) {
    val baseColor = Color(0xFF1E88E5)
    val avatarSize = 250.dp
    val tr = rememberInfiniteTransition(label = "avatar")

    val listeningPulse by tr.animateFloat(
        initialValue = 1f, targetValue = 1.6f,
        animationSpec = infiniteRepeatable(tween(1200), RepeatMode.Restart), label = "pulse"
    )
    val listeningAlpha by tr.animateFloat(
        initialValue = 1f, targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(1200), RepeatMode.Restart), label = "alpha"
    )
    val thinkingRotation by tr.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(2000)), label = "rot"
    )

    Box(modifier = modifier.size(avatarSize), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            when (state) {
                AssistantState.LISTENING -> drawCircle(
                    color = baseColor.copy(alpha = listeningAlpha),
                    radius = size.minDimension / 2 * listeningPulse,
                    style = Stroke(width = (size.minDimension * 0.05f))
                )
                AssistantState.THINKING -> drawArc(
                    color = baseColor, startAngle = thinkingRotation, sweepAngle = 90f,
                    useCenter = false, style = Stroke(width = (size.minDimension * 0.1f))
                )
                else -> {}
            }
        }
        Box(
            modifier = Modifier.fillMaxSize().clip(CircleShape).background(baseColor),
            contentAlignment = Alignment.Center
        ) {}
    }
}
