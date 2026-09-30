package com.example.tesis_identificador

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

class CatalogEmbeddingStore(
    private val context: Context
) {

    private val fileName = "catalog_embeddings.bin"

    private val embeddingSize = 384
    private val patchCount = 256

    fun exists(): Boolean {
        return File(
            context.filesDir,
            fileName
        ).exists()
    }

    fun save(
        embeddings: List<CatalogEmbedding>
    ) {

        val file = File(
            context.filesDir,
            fileName
        )

        DataOutputStream(
            FileOutputStream(file)
        ).use { output ->

            // --------------------------------------------------
            // CANTIDAD DE PRODUCTOS
            // --------------------------------------------------

            output.writeInt(
                embeddings.size
            )

            for (embedding in embeddings) {

                // --------------------------------------------------
                // NOMBRE DEL PRODUCTO
                // --------------------------------------------------

                output.writeUTF(
                    embedding.productName
                )

                // --------------------------------------------------
                // CLS
                // --------------------------------------------------

                require(
                    embedding.cls.size == embeddingSize
                )

                for (value in embedding.cls) {
                    output.writeFloat(value)
                }

                // --------------------------------------------------
                // PATCHES
                // 256 patches × 384 dimensiones
                // --------------------------------------------------

                require(
                    embedding.patches.size == patchCount
                )

                for (patch in embedding.patches) {

                    require(
                        patch.size == embeddingSize
                    )

                    for (value in patch) {
                        output.writeFloat(value)
                    }
                }

                // --------------------------------------------------
                // COLOR LAB
                // --------------------------------------------------

                require(
                    embedding.color.lab.size == 3
                )

                for (value in embedding.color.lab) {
                    output.writeFloat(value)
                }
            }
        }
    }

    fun load(): List<CatalogEmbedding> {

        val file = File(
            context.filesDir,
            fileName
        )

        if (!file.exists()) {
            return emptyList()
        }

        return DataInputStream(
            FileInputStream(file)
        ).use { input ->

            val count =
                input.readInt()

            val embeddings =
                mutableListOf<CatalogEmbedding>()

            repeat(count) {

                // --------------------------------------------------
                // NOMBRE
                // --------------------------------------------------

                val productName =
                    input.readUTF()

                // --------------------------------------------------
                // CLS
                // --------------------------------------------------

                val cls =
                    FloatArray(embeddingSize)

                for (i in cls.indices) {

                    cls[i] =
                        input.readFloat()
                }

                // --------------------------------------------------
                // PATCHES
                // --------------------------------------------------

                val patches =
                    Array(patchCount) {

                        FloatArray(embeddingSize)
                    }

                for (patchIndex in 0 until patchCount) {

                    for (dimension in 0 until embeddingSize) {

                        patches[patchIndex][dimension] =
                            input.readFloat()
                    }
                }

                // --------------------------------------------------
                // COLOR LAB
                // --------------------------------------------------

                val lab =
                    FloatArray(3)

                for (i in lab.indices) {

                    lab[i] =
                        input.readFloat()
                }

                embeddings.add(
                    CatalogEmbedding(
                        productName = productName,
                        cls = cls,
                        patches = patches,
                        color = ColorFeature(lab)
                    )
                )
            }

            embeddings
        }
    }

    fun delete() {

        val file = File(
            context.filesDir,
            fileName
        )

        if (file.exists()) {
            file.delete()
        }
    }
}