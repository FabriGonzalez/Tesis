package com.example.tesis_identificador

import android.graphics.RectF
import android.util.Log
import kotlin.math.max
import kotlin.math.min

object SamFilterPipeline {

    private const val TAG = "SamFilter"

    data class BinarizedMask(
        val idx: Int,
        val bits: BooleanArray, // Matriz aplanada (256x256) binarizada
        val area: Int,
        val bbox: RectF,        // coordenadas inclusivas de píxel
        val density: Float,
        val width: Int = 256,
        val height: Int = 256
    )

    /**
     * Binariza la máscara flotante (> 0.0f) y calcula sus métricas geométricas exactas.
     */
    private fun binarizeMask(idx: Int, mask: Array<FloatArray>): BinarizedMask? {
        val h = mask.size
        val w = mask[0].size
        val bits = BooleanArray(w * h)

        var minX = w
        var maxX = -1
        var minY = h
        var maxY = -1
        var area = 0

        for (y in 0 until h) {
            val row = mask[y]
            val offset = y * w
            for (x in 0 until w) {
                val isFg = row[x] > 0.0f
                bits[offset + x] = isFg
                if (isFg) {
                    area++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        if (area == 0 || maxX < minX || maxY < minY) return null

        val bboxW = (maxX - minX + 1).toFloat()
        val bboxH = (maxY - minY + 1).toFloat()
        val density = area.toFloat() / (bboxW * bboxH)

        return BinarizedMask(
            idx = idx,
            bits = bits,
            area = area,
            bbox = RectF(minX.toFloat(), minY.toFloat(), maxX.toFloat(), maxY.toFloat()),
            density = density,
            width = w,
            height = h
        )
    }

    /** Cantidad de píxeles en común entre dos máscaras. */
    private fun intersectionCount(a: BinarizedMask, b: BinarizedMask): Int {
        var inter = 0
        val size = min(a.bits.size, b.bits.size)
        for (i in 0 until size) {
            if (a.bits[i] && b.bits[i]) inter++
        }
        return inter
    }

    /**
     * Qué fracción del bbox de [cand] queda dentro del bbox de [parent].
     * Sirve para detectar sub-partes cuando la máscara del padre tiene huecos
     * (ej: la abertura del cuello en una remera).
     */
    private fun bboxContainment(cand: BinarizedMask, parent: BinarizedMask): Float {
        val ix1 = max(cand.bbox.left, parent.bbox.left)
        val iy1 = max(cand.bbox.top, parent.bbox.top)
        val ix2 = min(cand.bbox.right, parent.bbox.right)
        val iy2 = min(cand.bbox.bottom, parent.bbox.bottom)

        val iw = ix2 - ix1 + 1f
        val ih = iy2 - iy1 + 1f
        if (iw <= 0f || ih <= 0f) return 0f

        val candArea = (cand.bbox.width() + 1f) * (cand.bbox.height() + 1f)
        return (iw * ih) / candArea
    }

    /**
     * PIPELINE UNIFICADO:
     * 1) binarizar + densidad mínima
     * 2) filtro de forma (franjas finas, área mínima)
     * 3) supresión en UNA sola pasada sobre TODAS las candidatas (mayor a menor área)
     * 4) recorte a topK sobre los ganadores (no antes de deduplicar)
     */
    fun processPipeline(
        rawMasks: List<Array<FloatArray>>,
        topK: Int = 30,
        minDensity: Float = 0.15f,
        minAreaRatio: Float = 0.025f,
        minShapeDensity: Float = 0.40f,
        dedupIouThreshold: Float = 0.60f,
        containmentThreshold: Float = 0.60f,
        subpartBBoxThreshold: Float = 0.95f,
        subpartMaxAreaRatio: Float = 0.35f,
        debug: Boolean = true
    ): List<Int> {

        // 1. Binarizar + densidad mínima
        val binarized = rawMasks.mapIndexedNotNull { idx, mask ->
            val b = binarizeMask(idx, mask)
            if (b != null && b.density >= minDensity) b else null
        }
        if (binarized.isEmpty()) return emptyList()

        // 2. Filtro de forma
        val canvasArea = 256f * 256f
        val shaped = binarized.filter { m ->
            val relWidth = m.bbox.width() / 256f
            val aspectRatio = (m.bbox.height() + 1f) / max(1f, m.bbox.width() + 1f)
            val areaRatio = m.area / canvasArea

            val ok = relWidth >= 0.10f &&
                    aspectRatio <= 4.0f &&
                    areaRatio >= minAreaRatio &&
                    m.density >= minShapeDensity

            if (!ok && debug) {
                Log.d(TAG, "mask ${m.idx} DESCARTADA por forma (relW=%.2f aspect=%.2f areaR=%.3f dens=%.2f)"
                    .format(relWidth, aspectRatio, areaRatio, m.density))
            }
            ok
        }

        // 3. Supresión unificada sobre TODAS las candidatas, de mayor a menor área
        val candidates = shaped.sortedByDescending { it.area }
        val winners = mutableListOf<BinarizedMask>()

        for (cand in candidates) {
            var reason: String? = null
            var maxIou = 0f
            var maxIom = 0f

            for (kept in winners) {
                val inter = intersectionCount(cand, kept)
                val union = cand.area + kept.area - inter
                val iou = if (union == 0) 0f else inter.toFloat() / union
                val iom = inter.toFloat() / min(cand.area, kept.area).toFloat()
                val areaRatio = cand.area.toFloat() / kept.area.toFloat()
                val bboxContain = bboxContainment(cand, kept)

                maxIou = max(maxIou, iou)
                maxIom = max(maxIom, iom)

                if (iou >= dedupIouThreshold) {
                    reason = "IoU=%.2f con mask ${kept.idx}".format(iou)
                } else if (iom >= containmentThreshold) {
                    reason = "contención=%.2f con mask ${kept.idx}".format(iom)
                } else if (areaRatio <= subpartMaxAreaRatio && bboxContain >= subpartBBoxThreshold) {
                    reason = "sub-parte por bbox (contain=%.2f, areaRatio=%.2f) de mask ${kept.idx}"
                        .format(bboxContain, areaRatio)
                }

                if (reason != null) break
            }

            if (reason != null) {
                if (debug) Log.d(TAG, "mask ${cand.idx} (area=${cand.area}) DESCARTADA: $reason")
            } else {
                winners.add(cand)
                if (debug) {
                    Log.d(TAG, "mask ${cand.idx} (area=${cand.area}) CONSERVADA (maxIoU=%.2f maxIoM=%.2f)"
                        .format(maxIou, maxIom))
                }
            }
        }

        // 4. Recorte a topK sobre los ganadores (ya ordenados por área)
        return winners.take(topK).map { it.idx }
    }
}