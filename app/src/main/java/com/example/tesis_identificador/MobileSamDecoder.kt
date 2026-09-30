package com.example.tesis_identificador

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.PointF
import java.nio.FloatBuffer

data class MobileSamDecoderResult(
    val masks: Any?,
    val iouPredictions: Any?,
    val lowResMasks: Any?
)

class MobileSamDecoder(
    private val env: OrtEnvironment,
    private val session: OrtSession
) {
    fun predict(
        imageEmbedding: FloatArray,
        pointX: Float,
        pointY: Float,
        imageWidth: Int,
        imageHeight: Int
    ): MobileSamDecoderResult {

        val longestSide = maxOf(imageWidth, imageHeight).toFloat()
        val scale = 1024f / longestSide

        val transformedX = pointX * scale
        val transformedY = pointY * scale

        val embeddingTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(imageEmbedding),
            longArrayOf(1L, 256L, 64L, 64L)
        )

        val pointCoordsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArrayOf(transformedX, transformedY, 0f, 0f)),
            longArrayOf(1L, 2L, 2L)
        )

        val pointLabelsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArrayOf(1f, -1f)),
            longArrayOf(1L, 2L)
        )

        val maskInputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(FloatArray(256 * 256)),
            longArrayOf(1L, 1L, 256L, 256L)
        )

        val hasMaskInputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArrayOf(0f)),
            longArrayOf(1L)
        )

        val origImSizeTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArrayOf(imageHeight.toFloat(), imageWidth.toFloat())),
            longArrayOf(2L)
        )

        var resultHolder: MobileSamDecoderResult? = null

        embeddingTensor.use {
            pointCoordsTensor.use {
                pointLabelsTensor.use {
                    maskInputTensor.use {
                        hasMaskInputTensor.use {
                            origImSizeTensor.use {
                                val inputs = mapOf(
                                    "image_embeddings" to embeddingTensor,
                                    "point_coords" to pointCoordsTensor,
                                    "point_labels" to pointLabelsTensor,
                                    "mask_input" to maskInputTensor,
                                    "has_mask_input" to hasMaskInputTensor,
                                    "orig_im_size" to origImSizeTensor
                                )

                                session.run(inputs).use { result ->
                                    @Suppress("UNCHECKED_CAST")
                                    val rawMasks = result[0].value as Array<Array<Array<FloatArray>>>
                                    @Suppress("UNCHECKED_CAST")
                                    val rawIou = result[1].value as Array<FloatArray>
                                    @Suppress("UNCHECKED_CAST")
                                    val rawLowRes = result[2].value as Array<Array<Array<FloatArray>>>

                                    // Copia liviana de la máscara reducida (256x256) para NMS
                                    val clonedLowRes = rawLowRes[0][0].map { it.clone() }.toTypedArray()
                                    val clonedIou = rawIou[0].clone()

                                    // Solo si la máscara completa es pequeña o para usar directamente,
                                    // clonamos la de alta resolución
                                    val clonedFullMask = rawMasks[0][0].map { it.clone() }.toTypedArray()

                                    resultHolder = MobileSamDecoderResult(
                                        masks = clonedFullMask,
                                        lowResMasks = clonedLowRes,
                                        iouPredictions = clonedIou
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        return resultHolder ?: throw IllegalStateException("Error al ejecutar predict en MobileSamDecoder")
    }


    fun predictBatch(
        imageEmbedding: FloatArray,
        points: List<PointF>,
        imageWidth: Int,
        imageHeight: Int
    ): List<MobileSamDecoderResult> {

        require(points.isNotEmpty()) {
            "La lista de puntos no puede estar vacía"
        }

        val results = ArrayList<MobileSamDecoderResult>(points.size)

        val longestSide = maxOf(imageWidth, imageHeight).toFloat()
        val scale = 1024f / longestSide

        for (point in points) {
            val transformedX = point.x * scale
            val transformedY = point.y * scale

            val singleResult = predict(
                imageEmbedding = imageEmbedding,
                pointX = transformedX,
                pointY = transformedY,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )

            results.add(singleResult)
        }

        return results
    }
}