package com.example.client

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import androidx.compose.foundation.layout.padding
import com.example.client.ui.AssistantScreen
import com.example.client.ui.theme.HandsFreeTheme

class MainActivity : ComponentActivity() {
    private val reqPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* 필요 시 처리 */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) reqPerm.launch(Manifest.permission.RECORD_AUDIO)

        setContent {
            HandsFreeTheme {
                Scaffold(
                    containerColor = Color(0xFF0B0F14),
                    contentColor = Color(0xFFE6EDF3)
                ) { inner ->
                    AssistantScreen(modifier = Modifier.padding(inner))
                }
            }
        }
    }
}
