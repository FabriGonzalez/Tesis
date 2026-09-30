package com.example.tesis_identificador

import android.graphics.Bitmap
import android.util.Log
import java.io.File

class MobileSamBatchTest(
    private val encoder: MobileSamEncoder,
    private val decoder: MobileSamDecoder,
    private val context: android.content.Context
) {

    fun test(
        bitmap: Bitmap,
        pointsCount: Int = 8
    ) {
        Log.d("MobileSAM", "========== BATCH TEST ==========")

        val prepared = MobileSamPreprocessor.prepare(bitmap)

        val embedding = encoder.encode(
            prepared.data,
            1024,
            1024
        )

        val allPoints = MobileSamPointGrid.generate(
            width = prepared.width,
            height = prepared.height,
            pointsPerSide = pointsCount
        )

        val points = allPoints.take(pointsCount)

        Log.d("MobileSAM", "Batch points: ${points.size}")

        val start = System.currentTimeMillis()

        try {
            val results = decoder.predictBatch(
                imageEmbedding = embedding.data,
                points = points,
                imageWidth = prepared.width,
                imageHeight = prepared.height
            )

            val elapsed = System.currentTimeMillis() - start

            Log.d("MobileSAM", "Batch decoder total time: $elapsed ms")
            Log.d("MobileSAM", "Average time per point: ${elapsed.toFloat() / points.size} ms")
            Log.d("MobileSAM", "Resulting masks count: ${results.size}")

            for ((index, result) in results.withIndex()) {
                val mask = extractMask(result.masks)

                // Generamos el crop de prueba
                val crop = MaskedCropGenerator.createTransparentCrop(
                    image = bitmap,
                    mask = mask,
                    threshold = 0f,
                    padding = 30
                )

                if (crop != null) {
                    Log.d("MobileSAM", "Crop $index generado con éxito (${crop.width}x${crop.height})")
                    saveCrop(crop, index)

                    // LIBERAR MEMORIA RAM
                    crop.recycle()
                } else {
                    Log.d("MobileSAM", "Crop $index resultó vacío o fuera de rango")
                }
            }

        } catch (e: Exception) {
            Log.e("MobileSAM", "Error en batch decoder", e)
        }

        // Forzar recolección de basura tras procesar lote
        System.gc()

        Log.d("MobileSAM", "========== BATCH TEST END ==========")
    }

    private fun extractMask(
        value: Any?
    ): Array<FloatArray> {
        @Suppress("UNCHECKED_CAST")
        val batch = value as Array<Array<Array<FloatArray>>>
        return batch[0][0]
    }

    private fun saveCrop(bitmap: Bitmap, index: Int) {
        val file = File(
            context.cacheDir,
            "mobile_sam_batch_crop_$index.png"
        )

        file.outputStream().use { output ->
            bitmap.compress(
                Bitmap.CompressFormat.PNG,
                100,
                output
            )
        }

        Log.d("MobileSAM", "Crop $index guardado en: ${file.absolutePath}")
    }
}