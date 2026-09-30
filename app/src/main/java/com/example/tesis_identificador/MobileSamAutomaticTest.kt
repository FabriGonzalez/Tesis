package com.example.tesis_identificador

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import java.io.File

class MobileSamAutomaticTest(
    private val encoder: MobileSamEncoder,
    private val decoder: MobileSamDecoder,
    private val context: android.content.Context
) {

    companion object {
        const val MASK_PREFIX = "mobile_sam_mask_"
    }

    fun generate(
        bitmap: Bitmap,
        pointsPerSide: Int = 8
    ) {
        Log.d("MobileSAM", "========== AUTOMATIC MASK TEST ==========")

        // Borrar crops de corridas anteriores para no mostrar resultados viejos
        context.cacheDir.listFiles { f ->
            f.name.startsWith(MASK_PREFIX) && f.name.endsWith(".png")
        }?.forEach { it.delete() }

        val prepared = MobileSamPreprocessor.prepare(bitmap)

        val embedding = encoder.encode(
            prepared.data,
            1024,
            1024
        )

        val points = MobileSamPointGrid.generate(
            width = prepared.width,
            height = prepared.height,
            pointsPerSide = pointsPerSide
        )

        Log.d("MobileSAM", "Total puntos grilla: ${points.size}")

        val lowResMasks = mutableListOf<Array<FloatArray>>()
        val pointList = mutableListOf<PointF>()

        // FASE 1: RECOLECTAR MÁSCARAS LIVIANAS (256x256)
        for (i in points.indices) {
            val point = points[i]
            try {
                val result = decoder.predict(
                    imageEmbedding = embedding.data,
                    pointX = point.x,
                    pointY = point.y,
                    imageWidth = prepared.width,
                    imageHeight = prepared.height
                )

                @Suppress("UNCHECKED_CAST")
                val lowResMask = result.lowResMasks as Array<FloatArray>

                lowResMasks.add(lowResMask)
                pointList.add(point)

            } catch (e: Exception) {
                Log.e("MobileSAM", "Error en punto $i (${point.x}, ${point.y})", e)
            }
        }

        Log.d("MobileSAM", "Candidatos recolectados: ${lowResMasks.size}")

        // FASE 2: FILTRADO UNIFICADO
        val winningIndices = SamFilterPipeline.processPipeline(
            rawMasks = lowResMasks,
            topK = 30
        )

        Log.d("MobileSAM", "Máscaras finales seleccionadas tras pipeline: ${winningIndices.size}")

        // FASE 3: GENERAR CROPS DE ALTA RESOLUCIÓN SOLO PARA WINNERS
        for ((cropIndex, winIdx) in winningIndices.withIndex()) {
            val winPoint = pointList[winIdx]

            val result = decoder.predict(
                imageEmbedding = embedding.data,
                pointX = winPoint.x,
                pointY = winPoint.y,
                imageWidth = prepared.width,
                imageHeight = prepared.height
            )

            @Suppress("UNCHECKED_CAST")
            val fullMask = result.masks as Array<FloatArray>

            val crop = MaskedCropGenerator.createTransparentCrop(
                image = bitmap,
                mask = fullMask,
                threshold = 0f,
                padding = 30
            )

            if (crop != null) {
                saveMaskSample(crop, cropIndex + 1)
                crop.recycle()
            }
        }

        System.gc()

        Log.d("MobileSAM", "========== AUTOMATIC MASK TEST END ==========")
    }

    private fun saveMaskSample(bitmap: Bitmap, index: Int) {
        val file = File(
            context.cacheDir,
            "$MASK_PREFIX%02d.png".format(index)
        )

        file.outputStream().use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }

        Log.d("MobileSAM", "Guardado crop único $index (${bitmap.width}x${bitmap.height}) en: ${file.absolutePath}")
    }
}