package com.example.tesis_identificador

import android.graphics.PointF

object MobileSamPointGrid {

    fun generate(
        width: Int,
        height: Int,
        pointsPerSide: Int = 8,
        marginPercentage: Float = 0.1f // 10% de margen respecto a los bordes
    ): List<PointF> {
        val points = mutableListOf<PointF>()

        val minX = width * marginPercentage
        val maxX = width * (1f - marginPercentage)
        val minY = height * marginPercentage
        val maxY = height * (1f - marginPercentage)

        val stepX = (maxX - minX) / (pointsPerSide - 1)
        val stepY = (maxY - minY) / (pointsPerSide - 1)

        for (i in 0 until pointsPerSide) {
            for (j in 0 until pointsPerSide) {
                val x = minX + j * stepX
                val y = minY + i * stepY
                points.add(PointF(x, y))
            }
        }

        return points
    }
}