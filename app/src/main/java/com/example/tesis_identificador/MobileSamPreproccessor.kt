package com.example.tesis_identificador

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.roundToInt

data class MobileSamImage(
    val data: FloatArray,
    val width: Int,
    val height: Int,
    val resizedWidth: Int,
    val resizedHeight: Int
)

object MobileSamPreprocessor {

    private const val TARGET_SIZE = 1024

    fun prepare(bitmap: Bitmap): MobileSamImage {

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        val scale = TARGET_SIZE.toFloat() / maxOf(originalWidth, originalHeight)

        val resizedWidth = (originalWidth * scale).roundToInt()
        val resizedHeight = (originalHeight * scale).roundToInt()

        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            resizedWidth,
            resizedHeight,
            true
        )

        val paddedBitmap = Bitmap.createBitmap(
            TARGET_SIZE,
            TARGET_SIZE,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK)

        canvas.drawBitmap(
            resizedBitmap,
            0f,
            0f,
            Paint(Paint.FILTER_BITMAP_FLAG)
        )

        val pixels = IntArray(TARGET_SIZE * TARGET_SIZE)

        paddedBitmap.getPixels(
            pixels,
            0,
            TARGET_SIZE,
            0,
            0,
            TARGET_SIZE,
            TARGET_SIZE
        )

        // HWC estricto que exige tu modelo ONNX: [1024, 1024, 3]
        // El encoder ONNX ya incluye la normalización ImageNet internamente,
        // por lo que se pasan valores de píxel raw [0-255] como Float.
        val input = FloatArray(TARGET_SIZE * TARGET_SIZE * 3)

        var index = 0
        for (y in 0 until TARGET_SIZE) {
            for (x in 0 until TARGET_SIZE) {

                val pixel = pixels[y * TARGET_SIZE + x]

                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                input[index++] = r.toFloat()
                input[index++] = g.toFloat()
                input[index++] = b.toFloat()
            }
        }

        resizedBitmap.recycle()
        paddedBitmap.recycle()

        return MobileSamImage(
            data = input,
            width = originalWidth,
            height = originalHeight,
            resizedWidth = resizedWidth,
            resizedHeight = resizedHeight
        )
    }
}