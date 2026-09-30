package com.example.tesis_identificador

import android.graphics.Bitmap
import kotlin.math.sqrt
import kotlin.math.pow

data class ColorFeature(
    val lab: FloatArray
)

object ColorFeatureExtractor {

    fun extract(bitmap: Bitmap): ColorFeature {
        val width = bitmap.width
        val height = bitmap.height

        var sumL = 0.0
        var sumA = 0.0
        var sumB = 0.0
        var count = 0

        for (y in 0 until height) {
            for (x in 0 until width) {

                val pixel = bitmap.getPixel(x, y)

                val r = android.graphics.Color.red(pixel) / 255.0
                val g = android.graphics.Color.green(pixel) / 255.0
                val b = android.graphics.Color.blue(pixel) / 255.0

                // RGB -> sRGB lineal
                val rLin = if (r <= 0.04045) r / 12.92
                else ((r + 0.055) / 1.055).pow(2.4)

                val gLin = if (g <= 0.04045) g / 12.92
                else ((g + 0.055) / 1.055).pow(2.4)

                val bLin = if (b <= 0.04045) b / 12.92
                else ((b + 0.055) / 1.055).pow(2.4)

                val xLab =
                    (rLin * 0.4124 +
                            gLin * 0.3576 +
                            bLin * 0.1805) / 0.95047

                val yLab =
                    (rLin * 0.2126 +
                            gLin * 0.7152 +
                            bLin * 0.0722)

                val zLab =
                    (rLin * 0.0193 +
                            gLin * 0.1192 +
                            bLin * 0.9505) / 1.08883

                fun f(t: Double): Double {
                    return if (t > 0.008856) {
                        t.pow(1.0 / 3.0)
                    } else {
                        7.787 * t + 16.0 / 116.0
                    }
                }

                val fx = f(xLab)
                val fy = f(yLab)
                val fz = f(zLab)

                val l = 116.0 * fy - 16.0
                val a = 500.0 * (fx - fy)
                val bValue = 200.0 * (fy - fz)

                sumL += l
                sumA += a
                sumB += bValue
                count++
            }
        }

        return ColorFeature(
            floatArrayOf(
                (sumL / count).toFloat(),
                (sumA / count).toFloat(),
                (sumB / count).toFloat()
            )
        )
    }

    fun similarity(a: ColorFeature, b: ColorFeature): Double {
        val dl = a.lab[0] - b.lab[0]
        val da = a.lab[1] - b.lab[1]
        val db = a.lab[2] - b.lab[2]

        val distance = sqrt(dl * dl + da * da + db * db)

        return 1.0 / (1.0 + distance / 50.0)
    }

}