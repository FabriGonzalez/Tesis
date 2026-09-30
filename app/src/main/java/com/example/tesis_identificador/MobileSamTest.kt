package com.example.tesis_identificador

import android.graphics.Bitmap
import android.util.Log
import java.io.File

class MobileSamTest(
    private val encoder: MobileSamEncoder,
    private val decoder: MobileSamDecoder,
    private val context: android.content.Context
) {

    fun testSinglePoint(bitmap: Bitmap) {

        Log.d("MobileSAM", "========== SINGLE POINT TEST ==========")

        // ----------------------------------------
        // PREPROCESSING & ENCODER
        // ----------------------------------------

        val prepared = MobileSamPreprocessor.prepare(bitmap)

        val embedding = encoder.encode(
            prepared.data,
            1024,
            1024
        )

        Log.d("MobileSAM", "Image: ${prepared.width} x ${prepared.height}")
        Log.d("MobileSAM", "Embedding floats: ${embedding.data.size}")

        // ----------------------------------------
        // PUNTO CENTRAL (El escalado lo maneja el Decoder internamente)
        // ----------------------------------------

        val pointX = prepared.width / 2f
        val pointY = prepared.height / 2f

        Log.d("MobileSAM", "Point Original: x=$pointX y=$pointY")

        // ----------------------------------------
        // DECODER
        // ----------------------------------------

        val result = decoder.predict(
            imageEmbedding = embedding.data,
            pointX = pointX,
            pointY = pointY,
            imageWidth = prepared.width,
            imageHeight = prepared.height
        )

        Log.d("MobileSAM", "Decoder executed successfully")

        @Suppress("UNCHECKED_CAST")
        val mask = result.masks as Array<FloatArray>

        Log.d("MobileSAM", "Mask dimensions: ${mask[0].size} x ${mask.size}")

        // ----------------------------------------
        // CROP TRANSPARENTE
        // ----------------------------------------

        val crop = MaskedCropGenerator.createTransparentCrop(
            image = bitmap,
            mask = mask,
            threshold = 0f,
            padding = 30
        )

        if (crop == null) {
            Log.d("MobileSAM", "No se pudo generar crop")
        } else {
            Log.d("MobileSAM", "Transparent crop: ${crop.width} x ${crop.height}")
            saveCrop(crop)
            crop.recycle() // Liberar memoria nativa de Bitmap
        }

        logRecursive("MASKS", result.masks)
        logRecursive("IOU", result.iouPredictions)
        logRecursive("LOW_RES", result.lowResMasks)
    }

    private fun logRecursive(
        name: String,
        value: Any?,
        depth: Int = 0
    ) {
        if (value == null) {
            Log.d("MobileSAM", "$name = null")
            return
        }

        if (depth > 3) {
            Log.d("MobileSAM", "$name = " + value.javaClass.name)
            return
        }

        when (value) {
            is FloatArray -> {
                val preview = value.take(10).joinToString()
                Log.d("MobileSAM", "$name FloatArray size=${value.size} preview=[$preview]")
            }

            is Array<*> -> {
                Log.d("MobileSAM", "$name Array size=${value.size}")
                for (i in value.indices.take(3)) {
                    logRecursive("$name[$i]", value[i], depth + 1)
                }
            }

            else -> {
                Log.d("MobileSAM", "$name type=" + value.javaClass.name + " value=$value")
            }
        }
    }

    private fun saveCrop(bitmap: Bitmap) {

        val file = File(
            context.cacheDir,
            "mobile_sam_test_crop.png"
        )

        file.outputStream().use { output ->
            bitmap.compress(
                Bitmap.CompressFormat.PNG,
                100,
                output
            )
        }

        Log.d("MobileSAM", "Crop guardado en: ${file.absolutePath}")
    }
}