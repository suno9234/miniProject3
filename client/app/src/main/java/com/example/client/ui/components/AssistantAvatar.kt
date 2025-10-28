package com.example.client.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.material3.Text
import com.example.client.state.AssistantState

@Composable
fun AssistantAvatar(
    state: AssistantState,
    level: Float,
    modifier: Modifier = Modifier
) {
    val tr = rememberInfiniteTransition(label = "avatar")
    val rot by tr.animateFloat(0f, 360f, infiniteRepeatable(tween(5000)), label = "rot")
    val pulse by tr.animateFloat(0.92f, 1.08f, infiniteRepeatable(tween(1200), RepeatMode.Reverse), label = "pulse")

    val rainbow = listOf(
        Color(0xFFFF5F6D), Color(0xFFFFC371), Color(0xFFF9F871),
        Color(0xFF5FFBF1), Color(0xFF64B5F6), Color(0xFFA78BFA), Color(0xFFFF5F6D)
    )
    val coreColor = when (state) {
        AssistantState.LISTENING -> Color(0xFF80D8FF)
        AssistantState.THINKING  -> Color(0xFFB39DDB)
        AssistantState.SPEAKING  -> Color(0xFFFFB74D)
        else                     -> Color(0xFF1B232B)
    }

    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val r = size.minDimension / 2f
            val center = this.center

            drawCircle(
                brush = Brush.radialGradient(rainbow.map { it.copy(alpha = 0.18f) }, center, r * 1.35f),
                radius = r * 1.35f
            )
            rotate(rot) {
                drawCircle(brush = Brush.sweepGradient(rainbow, center), radius = r * pulse)
            }
            drawCircle(color = Color.White.copy(alpha = 0.12f), radius = r * pulse, style = Stroke(width = r * 0.06f))
            drawCircle(
                brush = Brush.radialGradient(listOf(coreColor.copy(alpha = 0.95f), Color.Black.copy(alpha = 0.55f)), center, r * 0.9f),
                radius = r * 0.9f
            )
            val halo = 0.2f + 0.8f * level
            drawCircle(
                color = Color.White.copy(alpha = 0.10f + 0.25f * level),
                radius = r * (1.05f + 0.18f * halo),
                style = Stroke(width = r * (0.03f + 0.05f * halo))
            )
        }

        Text(
            text = state.name,
            color = Color(0xFFEFF6FB),
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.12f), CircleShape)
                .padding(horizontal = 12.dp, vertical = 4.dp),
            textAlign = TextAlign.Center
        )
    }
}
