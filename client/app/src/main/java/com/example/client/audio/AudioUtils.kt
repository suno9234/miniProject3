package com.example.client.audio

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.AudioRecord
import android.os.Build
import androidx.annotation.RequiresApi
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor

data class AudioEffects(
    val aec: AcousticEchoCanceler? = null,
    val ns: NoiseSuppressor? = null,
    val agc: AutomaticGainControl? = null
)

fun AudioEffects.releaseAll() {
    try { aec?.release() } catch (_: Throwable) {}
    try { ns?.release() } catch (_: Throwable) {}
    try { agc?.release() } catch (_: Throwable) {}
}

fun attachAudioEffects(sessionId: Int): AudioEffects? {
    if (sessionId <= 0) return null
    return try {
        AudioEffects(
            aec = if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(sessionId) else null,
            ns  = if (NoiseSuppressor.isAvailable()) NoiseSuppressor.create(sessionId) else null,
            agc = if (AutomaticGainControl.isAvailable()) AutomaticGainControl.create(sessionId) else null
        )
    } catch (_: Exception) { null }
}

fun AudioRecord?.safeStopRelease() {
    try { this?.stop() } catch (_: Exception) {}
    try { this?.release() } catch (_: Exception) {}
}

/** 선호 입력 장치 탐색 (BT 우선) */
fun findPreferredInputDevice(context: Context, preferBluetooth: Boolean = true): AudioDeviceInfo? {
    val am = context.getSystemService(AudioManager::class.java) ?: return null
    val inputs = am.getDevices(AudioManager.GET_DEVICES_INPUTS).orEmpty()

    val bt = inputs.firstOrNull {
        it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO || it.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP
    }
    val mic = inputs.firstOrNull {
        it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC || it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET
    }
    return if (preferBluetooth) bt ?: mic else mic ?: bt
}

fun getCurrentInputDeviceName(context: Context): String? {
    val am = context.getSystemService(AudioManager::class.java) ?: return null
    val dev = findPreferredInputDevice(context, true) ?: return null
    return dev.productName?.toString()
}

/** AudioRecord에 선호 입력 장치 적용 (성공 시 true) */
@RequiresApi(Build.VERSION_CODES.M)
fun AudioRecord?.applyPreferredInput(context: Context, preferBluetooth: Boolean = true): Boolean {
    val dev = findPreferredInputDevice(context, preferBluetooth) ?: return false
    return try { this?.preferredDevice = dev; true } catch (_: Exception) { false }
}
