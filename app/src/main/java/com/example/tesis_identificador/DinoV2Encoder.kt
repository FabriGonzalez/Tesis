package com.example.tesis_identificador

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.sqrt

enum class ExecutionProvider {
    CPU,
    XNNPACK,
    NNAPI
}

class DinoV2Encoder(
    context: Context,
    private val executionProvider: ExecutionProvider = ExecutionProvider.XNNPACK
) {

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {

        val modelBytes = context.assets
            .open("dinoV2.onnx")
            .use { it.readBytes() }

        val sessionOptions = OrtSession.SessionOptions()

        // Optimizaciones generales del grafo.
        sessionOptions.setOptimizationLevel(
            OrtSession.SessionOptions.OptLevel.ALL_OPT
        )

        when (executionProvider) {

            ExecutionProvider.XNNPACK -> {
                sessionOptions.addXnnpack(emptyMap())
            }

            ExecutionProvider.NNAPI -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    sessionOptions.addNnapi()
                } else {
                    // NNAPI requiere Android 8.1 o superior.
                    sessionOptions.addXnnpack(emptyMap())
                }
            }

            ExecutionProvider.CPU -> {
                sessionOptions.addCPU(true)
            }
        }

        session = environment.createSession(
            modelBytes,
            sessionOptions
        )

        sessionOptions.close()
    }

    fun encode(bitmap: Bitmap): DinoOutput {

        val startTotal = System.nanoTime()

        // --------------------------------------------------
        // 1. PREPROCESAMIENTO
        // --------------------------------------------------

        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            224,
            224,
            true
        )

        val inputData = FloatArray(1 * 3 * 224 * 224)

        val mean = floatArrayOf(
            0.485f,
            0.456f,
            0.406f
        )

        val std = floatArrayOf(
            0.229f,
            0.224f,
            0.225f
        )

        val pixels = IntArray(224 * 224)

        resizedBitmap.getPixels(
            pixels,
            0,
            224,
            0,
            0,
            224,
            224
        )

        var index = 0

        // NCHW
        for (channel in 0 until 3) {

            for (pixel in pixels) {

                val value = when (channel) {
                    0 -> android.graphics.Color.red(pixel)
                    1 -> android.graphics.Color.green(pixel)
                    else -> android.graphics.Color.blue(pixel)
                }

                val normalized =
                    (value / 255.0f - mean[channel]) / std[channel]

                inputData[index++] = normalized
            }
        }

        // --------------------------------------------------
        // 2. CREAR TENSOR
        // --------------------------------------------------

        val inputBuffer = FloatBuffer.wrap(inputData)

        val inputTensor = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            longArrayOf(1, 3, 224, 224)
        )

        // --------------------------------------------------
        // 3. INFERENCIA
        // --------------------------------------------------

        val inputName = session.inputNames.first()

        val inputs = mapOf(
            inputName to inputTensor
        )

        val startInference = System.nanoTime()

        val results = session.run(inputs)

        val endInference = System.nanoTime()

        // --------------------------------------------------
        // 4. EXTRAER OUTPUT
        // --------------------------------------------------

        @Suppress("UNCHECKED_CAST")
        val output =
            results[0].value as Array<Array<FloatArray>>

        // [1, 257, 384]

        val cls = output[0][0].clone()

        val patches = Array(256) { patchIndex ->
            output[0][patchIndex + 1].clone()
        }

        // --------------------------------------------------
        // 5. NORMALIZAR CLS
        // --------------------------------------------------

        normalizeInPlace(cls)

        // --------------------------------------------------
        // 6. NORMALIZAR PATCHES
        // --------------------------------------------------

        for (patch in patches) {
            normalizeInPlace(patch)
        }

        // --------------------------------------------------
        // 7. TIEMPOS
        // --------------------------------------------------

        val endTotal = System.nanoTime()

        val inferenceTimeMs =
            (endInference - startInference) / 1_000_000.0

        val totalTimeMs =
            (endTotal - startTotal) / 1_000_000.0

        // Liberar recursos
        inputTensor.close()
        results.close()

        return DinoOutput(
            cls = cls,
            patches = patches,
            inferenceTimeMs = inferenceTimeMs,
            totalTimeMs = totalTimeMs,
            executionProvider = executionProvider
        )
    }

    private fun normalizeInPlace(vector: FloatArray) {

        var sum = 0.0

        for (value in vector) {
            sum += value * value
        }

        val norm = sqrt(sum)

        if (norm == 0.0) return

        for (i in vector.indices) {
            vector[i] = (vector[i] / norm).toFloat()
        }
    }

    fun getInputNames(): Set<String> {
        return session.inputNames
    }

    fun getOutputNames(): Set<String> {
        return session.outputNames
    }
}

data class DinoOutput(
    val cls: FloatArray,
    val patches: Array<FloatArray>,
    val inferenceTimeMs: Double,
    val totalTimeMs: Double,
    val executionProvider: ExecutionProvider
)