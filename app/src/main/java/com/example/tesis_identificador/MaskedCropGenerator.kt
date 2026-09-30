package com.example.tesis_identificador

import android.graphics.Bitmap
import android.graphics.Color

object MaskedCropGenerator {

    fun createTransparentCrop(
        image: Bitmap,
        mask: Array<FloatArray>,
        threshold: Float = 0f,
        padding: Int = 30
    ): Bitmap? {

        if (mask.isEmpty()) {
            return null
        }

        val height = mask.size
        val width = mask[0].size

        if (width != image.width || height != image.height) {
            throw IllegalArgumentException(
                "Dimensiones incompatibles. " +
                        "Image=${image.width}x${image.height}, " +
                        "Mask=${width}x${height}"
            )
        }

        // ----------------------------------------
        // Buscar bounding box de la máscara
        // ----------------------------------------

        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1

        for (y in 0 until height) {
            val row = mask[y]
            for (x in 0 until width) {
                if (row[x] > threshold) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        // No hay píxeles de máscara válidos
        if (maxX < 0) {
            return null
        }

        // ----------------------------------------
        // Padding
        // ----------------------------------------

        val cropLeft = maxOf(0, minX - padding)
        val cropTop = maxOf(0, minY - padding)
        val cropRight = minOf(width - 1, maxX + padding)
        val cropBottom = minOf(height - 1, maxY + padding)

        val cropWidth = cropRight - cropLeft + 1
        val cropHeight = cropBottom - cropTop + 1

        // ----------------------------------------
        // Crear Bitmap ARGB
        // ----------------------------------------

        val result = Bitmap.createBitmap(
            cropWidth,
            cropHeight,
            Bitmap.Config.ARGB_8888
        )

        val pixels = IntArray(cropWidth * cropHeight)

        image.getPixels(
            pixels,
            0,
            cropWidth,
            cropLeft,
            cropTop,
            cropWidth,
            cropHeight
        )

        // ----------------------------------------
        // Aplicar máscara (Conservar color si > threshold, else transparente)
        // ----------------------------------------

        for (y in 0 until cropHeight) {
            for (x in 0 until cropWidth) {
                val maskValue = mask[cropTop + y][cropLeft + x]
                val index = y * cropWidth + x

                if (maskValue <= threshold) {
                    // Fondo transparente
                    pixels[index] = Color.TRANSPARENT
                }
            }
        }

        result.setPixels(
            pixels,
            0,
            cropWidth,
            0,
            0,
            cropWidth,
            cropHeight
        )

        return result
    }
}