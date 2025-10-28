package com.example.client.stt

import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.speech.RecognitionService

fun pickRecognitionService(pm: PackageManager): ComponentName? {
    val googlePkgs = listOf("com.google.android.googlequicksearchbox")
    val services = pm.queryIntentServices(
        Intent(RecognitionService.SERVICE_INTERFACE), PackageManager.MATCH_ALL
    ).orEmpty()
    val google = services.firstOrNull { ri -> googlePkgs.any { ri.serviceInfo?.packageName == it } }?.serviceInfo
    return if (google != null) ComponentName(google.packageName, google.name)
    else services.firstOrNull()?.serviceInfo?.let { ComponentName(it.packageName, it.name) }
}
