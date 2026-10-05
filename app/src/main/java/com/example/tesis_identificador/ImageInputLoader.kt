package com.example.tesis_identificador

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import kotlin.math.max
import kotlin.math.roundToInt

object ImageInputLoader {

    private const val TAG = "ImageInput"

    // Tope de seguridad: las fotos de cámara moderna pueden pesar 12 MP o más.
    private const val MAX_DIMENSION = 2048

    fun load(context: Context, uri: Uri): Bitmap {

        val decoded = try {

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                decodeWithImageDecoder(context, uri)
            } else {
                decodeWithBitmapFactory(context, uri)
            }

        } catch (e: Exception) {
            Log.e(TAG, "No se pudo convertir $uri a Bitmap", e)
            throw IllegalStateException(
                "No se pudo leer la imagen seleccionada (${e.javaClass.simpleName}: ${e.message})",
                e
            )
        }

        val bitmap = if (decoded.config == Bitmap.Config.ARGB_8888) {
            decoded
        } else {
            decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
        }

        Log.d(TAG, "Imagen cargada ${bitmap.width}x${bitmap.height} desde $uri")

        return bitmap
    }

    // ------------------------------------------------------------
    // API 28+
    // ImageDecoder ya aplica la rotación EXIF por sí solo.
    // ------------------------------------------------------------

    private fun decodeWithImageDecoder(context: Context, uri: Uri): Bitmap {

        val source = ImageDecoder.createSource(context.contentResolver, uri)

        return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->

            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE

            val width = info.size.width
            val height = info.size.height
            val longestSide = max(width, height)

            if (longestSide > MAX_DIMENSION) {
                val scale = MAX_DIMENSION.toFloat() / longestSide
                decoder.setTargetSize(
                    (width * scale).roundToInt().coerceAtLeast(1),
                    (height * scale).roundToInt().coerceAtLeast(1)
                )
            }
        }
    }

    // ------------------------------------------------------------
    // API 24-27
    // BitmapFactory no respeta EXIF, hay que rotar a mano.
    // ------------------------------------------------------------

    private fun decodeWithBitmapFactory(context: Context, uri: Uri): Bitmap {

        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }

        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }

        val options = BitmapFactory.Options().apply {
            inSampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalStateException("El Uri no entregó ningún dato de imagen")

        val orientation = readExifOrientation(context, uri)

        return applyExifOrientation(bitmap, orientation)
    }

    private fun calculateInSampleSize(width: Int, height: Int): Int {

        if (width <= 0 || height <= 0) return 1

        var sampleSize = 1

        while (max(width / sampleSize, height / sampleSize) > MAX_DIMENSION) {
            sampleSize *= 2
        }

        return sampleSize
    }

    private fun readExifOrientation(context: Context, uri: Uri): Int {

        return try {
            context.contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo leer EXIF de $uri", e)
            ExifInterface.ORIENTATION_NORMAL
        }
    }

    private fun applyExifOrientation(bitmap: Bitmap, orientation: Int): Bitmap {

        val matrix = Matrix()

        when (orientation) {

            ExifInterface.ORIENTATION_NORMAL,
            ExifInterface.ORIENTATION_UNDEFINED -> return bitmap

            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)

            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)

            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)

            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(270f)
                matrix.postScale(-1f, 1f)
            }

            else -> return bitmap
        }

        val rotated = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        if (rotated != bitmap) {
            bitmap.recycle()
        }

        return rotated
    }
}