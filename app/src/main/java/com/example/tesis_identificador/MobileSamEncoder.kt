package com.example.tesis_identificador

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

data class MobileSamImageEmbedding(
    val data: FloatArray,
    val shape: LongArray
)

class MobileSamEncoder(
    private val env: OrtEnvironment,
    private val session: OrtSession
) {

    fun encode(input: FloatArray, height: Int, width: Int): MobileSamImageEmbedding {

        require(input.size == height * width * 3) {
            "Input size incorrecto: ${input.size}, esperado=${height * width * 3}"
        }

        val shape = longArrayOf(
            height.toLong(),
            width.toLong(),
            3L
        )

        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            shape
        )

        tensor.use {

            val inputs = mapOf(
                "input_image" to tensor
            )

            session.run(inputs).use { result ->

                val outputTensor = result[0]

                val outputValue = outputTensor.value

                val outputArray = outputValue as Array<*>

                val flattened = mutableListOf<Float>()

                fun flatten(value: Any?) {
                    when (value) {

                        is FloatArray -> {
                            flattened.addAll(value.toList())
                        }

                        is Array<*> -> {
                            value.forEach {
                                flatten(it)
                            }
                        }
                    }
                }

                flatten(outputArray)

                return MobileSamImageEmbedding(
                    data = flattened.toFloatArray(),
                    shape = longArrayOf(1L, 256L, 64L, 64L)
                )
            }
        }
    }
}