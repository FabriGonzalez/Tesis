package com.example.tesis_identificador

import android.graphics.Bitmap
import kotlin.math.sqrt

data class CatalogEmbedding(
    val productName: String,
    val cls: FloatArray,
    val patches: Array<FloatArray>,
    val color: ColorFeature
)

data class MatchResult(
    val productName: String,
    val dinoSimilarity: Double,
    val colorSimilarity: Double,
    val finalScore: Double
)

class CatalogMatcher(
    private val context: android.content.Context,
    private val encoder: DinoV2Encoder
) {

    private val catalogEmbeddings =
        mutableListOf<CatalogEmbedding>()

    // --------------------------------------------------
    // PESOS
    // Iguales al pipeline de Colab
    // --------------------------------------------------

    private val weightCls = 0.5
    private val weightPatches = 0.5

    private val weightDino = 0.6
    private val weightColor = 0.4

    // --------------------------------------------------
    // CATÁLOGO
    // --------------------------------------------------

    fun addCatalogEmbedding(
        embedding: CatalogEmbedding
    ) {
        catalogEmbeddings.add(
            embedding
        )
    }

    fun addCatalogEmbeddings(
        embeddings: List<CatalogEmbedding>
    ) {

        catalogEmbeddings.clear()

        catalogEmbeddings.addAll(
            embeddings
        )
    }

    fun getCatalogEmbeddings():
            List<CatalogEmbedding> {

        return catalogEmbeddings.toList()
    }

    fun catalogSize(): Int {
        return catalogEmbeddings.size
    }

    // --------------------------------------------------
    // MATCHING
    // --------------------------------------------------

    fun matchQuery(
        queryBitmap: Bitmap
    ): List<MatchResult> {

        if (catalogEmbeddings.isEmpty()) {

            throw IllegalStateException(
                "El catálogo todavía no tiene embeddings."
            )
        }

        // --------------------------------------------------
        // 1. DINOv2 DE LA QUERY
        // --------------------------------------------------

        val queryDino =
            encoder.encode(
                queryBitmap
            )

        val queryCls =
            queryDino.cls

        val queryPatches =
            queryDino.patches

        // --------------------------------------------------
        // 2. COLOR DE LA QUERY
        // --------------------------------------------------

        val queryColor =
            ColorFeatureExtractor.extract(
                queryBitmap
            )

        val results =
            mutableListOf<MatchResult>()

        // --------------------------------------------------
        // 3. COMPARAR CONTRA CADA PRODUCTO
        // --------------------------------------------------

        for (product in catalogEmbeddings) {

            // --------------------------------------------------
            // 3.1 CLS
            // Equivalente a:
            //
            // torch.dot(q_cls, cat_cls)
            // --------------------------------------------------

            val simCls =
                cosineSimilarity(
                    queryCls,
                    product.cls
                )

            // --------------------------------------------------
            // 3.2 PATCHES
            //
            // Equivalente conceptual a:
            //
            // sim_matrix = q_patches @ cat_patches.T
            //
            // max_sims = sim_matrix.max(dim=1)
            //
            // sim_patches = max_sims.mean()
            // --------------------------------------------------

            val simPatches =
                patchSimilarity(
                    queryPatches,
                    product.patches
                )

            // --------------------------------------------------
            // 3.3 DINO HÍBRIDO
            //
            // 50% CLS
            // 50% PATCHES
            // --------------------------------------------------

            val dinoHybridScore =
                weightCls * simCls +
                        weightPatches * simPatches

            // --------------------------------------------------
            // 3.4 COLOR LAB
            // --------------------------------------------------

            val colorSimilarity =
                ColorFeatureExtractor.similarity(
                    queryColor,
                    product.color
                )

            // --------------------------------------------------
            // 3.5 SCORE FINAL
            //
            // 60% DINO
            // 40% COLOR
            // --------------------------------------------------

            val finalScore =
                weightDino * dinoHybridScore +
                        weightColor * colorSimilarity

            results.add(
                MatchResult(
                    productName =
                        product.productName,

                    dinoSimilarity =
                        dinoHybridScore,

                    colorSimilarity =
                        colorSimilarity,

                    finalScore =
                        finalScore
                )
            )
        }

        // --------------------------------------------------
        // 4. RANKING
        // --------------------------------------------------

        return results.sortedByDescending {
            it.finalScore
        }
    }

    // --------------------------------------------------
    // SIMILITUD COSENO
    // --------------------------------------------------

    private fun cosineSimilarity(
        a: FloatArray,
        b: FloatArray
    ): Double {

        require(
            a.size == b.size
        )

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0

        for (i in a.indices) {

            val ai =
                a[i].toDouble()

            val bi =
                b[i].toDouble()

            dot +=
                ai * bi

            normA +=
                ai * ai

            normB +=
                bi * bi
        }

        val denominator =
            sqrt(normA) *
                    sqrt(normB)

        if (denominator == 0.0) {
            return 0.0
        }

        return dot / denominator
    }

    // --------------------------------------------------
    // SIMILITUD ENTRE PATCHES
    //
    // Equivalente a:
    //
    // sim_matrix = q_patches @ cat_patches.T
    //
    // max_sims = sim_matrix.max(dim=1)
    //
    // sim_patches = max_sims.mean()
    // --------------------------------------------------

    private fun patchSimilarity(
        queryPatches: Array<FloatArray>,
        catalogPatches: Array<FloatArray>
    ): Double {

        require(
            queryPatches.size ==
                    catalogPatches.size
        )

        var sumMaxSimilarities = 0.0

        // Para cada patch de la query
        for (queryPatch in queryPatches) {

            var maxSimilarity =
                Double.NEGATIVE_INFINITY

            // Compararlo contra todos
            // los patches del producto
            for (catalogPatch in catalogPatches) {

                val similarity =
                    cosineSimilarity(
                        queryPatch,
                        catalogPatch
                    )

                if (similarity > maxSimilarity) {

                    maxSimilarity =
                        similarity
                }
            }

            sumMaxSimilarities +=
                maxSimilarity
        }

        return sumMaxSimilarities /
                queryPatches.size
    }
}