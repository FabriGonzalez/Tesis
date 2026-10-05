package com.example.tesis_identificador

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import ai.onnxruntime.OrtEnvironment
import com.yalantis.ucrop.UCrop
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "Pipeline"
    }

    private val ui = PipelineUiState()

    private val cameraAvailable: Boolean by lazy {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).resolveActivity(packageManager) != null
    }

    private var pendingCameraUri: Uri? = null
    private var pendingCropUri: Uri? = null
    private var pendingCropLabel: String = ""

    private var cachedMatcher: CatalogMatcher? = null

    private val cropImage = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { activityResult ->

        val source = pendingCropUri
        pendingCropUri = null

        val label = pendingCropLabel
        val data = activityResult.data

        if (activityResult.resultCode != Activity.RESULT_OK || data == null) {
            Log.d(TAG, "Recorte cancelado desde $source")
            ui.statusText = "Recorte cancelado"
            return@registerForActivityResult
        }

        val error = UCrop.getError(data)

        if (error != null) {
            Log.e(TAG, "uCrop devolvió un error", error)
            ui.statusText = "ERROR"
            ui.resultText = "No se pudo recortar: ${error.message}"
            return@registerForActivityResult
        }

        val cropped = UCrop.getOutput(data)

        if (cropped == null) {
            Log.e(TAG, "uCrop no devolvió ninguna Uri de salida")
            ui.statusText = "ERROR"
            ui.resultText = "El recorte no produjo ninguna imagen"
            return@registerForActivityResult
        }

        Log.d(TAG, "Recorte listo desde $source -> $cropped")

        handleImageUri(cropped, source = "$label recortada")
    }

    private val pickVisualMedia = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->

        if (uri == null) {
            Log.d(TAG, "Selección de galería cancelada")
            ui.statusText = "Selección cancelada"
            return@registerForActivityResult
        }

        Log.d(TAG, "Galería devolvió $uri")
        launchCrop(uri, source = "Foto de galería")
    }

    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { saved ->

        val uri = pendingCameraUri
        pendingCameraUri = null

        if (!saved || uri == null) {
            Log.d(TAG, "Captura cancelada")
            ui.statusText = "Captura cancelada"
            return@registerForActivityResult
        }

        Log.d(TAG, "Cámara devolvió $uri")
        launchCrop(uri, source = "Foto de cámara")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "Cámara disponible: $cameraAvailable")

        setContent {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(text = ui.statusText)
                Spacer(modifier = Modifier.height(12.dp))

                if (ui.isProcessing) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(12.dp))
                }

                val image = ui.selectedImage

                if (image != null) {
                    Image(
                        bitmap = image.asImageBitmap(),
                        contentDescription = "Imagen seleccionada",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 320.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(text = "${ui.sourceLabel} · ${image.width}x${image.height} px")
                }

                Spacer(modifier = Modifier.height(20.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Button(
                        onClick = { launchGallery() },
                        enabled = !ui.isProcessing,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(text = "🖼 Elegir foto")
                    }

                    Button(
                        onClick = { launchCamera() },
                        enabled = !ui.isProcessing && cameraAvailable,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(text = "📷 Abrir cámara")
                    }
                }

                if (ui.maskSamples.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = if (ui.isProcessing) {
                            "Elegí una máscara (${ui.maskSamples.size})"
                        } else {
                            "Tocá una máscara para buscar productos (${ui.maskSamples.size})"
                        }
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    Row(modifier = Modifier.horizontalScroll(rememberScrollState())) {
                        ui.maskSamples.forEachIndexed { index, mask ->

                            val selected = ui.selectedMaskIndex == index

                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier
                                    .padding(end = 8.dp)
                                    .clickable(enabled = !ui.isProcessing) {
                                        matchSelectedMask(mask, index)
                                    }
                            ) {
                                Image(
                                    bitmap = mask.asImageBitmap(),
                                    contentDescription = "Máscara ${index + 1}",
                                    modifier = Modifier
                                        .width(160.dp)
                                        .border(
                                            width = if (selected) 3.dp else 1.dp,
                                            color = if (selected) {
                                                MaterialTheme.colorScheme.primary
                                            } else {
                                                Color.Gray
                                            }
                                        )
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(text = "${index + 1}")
                            }
                        }
                    }
                }

                if (ui.resultText.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(text = ui.resultText)
                }
            }
        }
    }

    // ------------------------------------------------------------
    // FASE 2 / FASE 3 / RECORTE
    // Galería y cámara convergen en el mismo recortador, y el
    // recorte termina en el mismo handleImageUri.
    // ------------------------------------------------------------

    private fun launchGallery() {
        pickVisualMedia.launch(
            PickVisualMediaRequest(
                ActivityResultContracts.PickVisualMedia.ImageOnly
            )
        )
    }

    private fun launchCamera() {

        val uri = try {
            createCameraUri()
        } catch (e: Exception) {
            Log.e(TAG, "No se pudo crear el archivo de captura", e)
            ui.statusText = "No se pudo preparar el archivo de captura"
            return
        }

        pendingCameraUri = uri

        try {
            takePicture.launch(uri)
        } catch (e: ActivityNotFoundException) {
            pendingCameraUri = null
            File(uri.path ?: "").delete()
            Log.e(TAG, "No hay ninguna cámara instalada", e)
            ui.statusText = "No hay cámara disponible en el dispositivo"
        }
    }

    private fun createCameraUri(): Uri {

        val directory = File(cacheDir, "camera").apply { mkdirs() }

        directory.listFiles()?.forEach { it.delete() }

        val file = File(directory, "captura_${System.currentTimeMillis()}.jpg")

        return FileProvider.getUriForFile(
            this,
            "$packageName.fileprovider",
            file
        )
    }

    private fun launchCrop(sourceUri: Uri, source: String) {

        pendingCropUri = sourceUri
        pendingCropLabel = source

        val destination = try {
            createCropUri()
        } catch (e: Exception) {
            Log.e(TAG, "No se pudo crear el archivo de recorte", e)
            ui.statusText = "No se pudo preparar el archivo de recorte"
            return
        }

        val options = UCrop.Options().apply {
            // Proporción libre: el recorte no fuerza una forma, así el
            // pipeline recibe exactamente la región elegida.
            setFreeStyleCropEnabled(true)
            setCompressionFormat(Bitmap.CompressFormat.JPEG)
            setCompressionQuality(95)
            setMaxScaleMultiplier(4f)
            setToolbarTitle("Recortar")
        }

        ui.statusText = "Recortando $source..."

        val intent = try {
            UCrop.of(sourceUri, destination).withOptions(options).getIntent(this)
        } catch (e: Exception) {
            pendingCropUri = null
            Log.e(TAG, "No se pudo construir el intent de recorte", e)
            ui.statusText = "No se pudo abrir el recortador"
            return
        }

        try {
            cropImage.launch(intent)
        } catch (e: ActivityNotFoundException) {
            pendingCropUri = null
            Log.e(TAG, "UCropActivity no está disponible", e)
            ui.statusText = "No se pudo abrir el recortador"
        }
    }

    private fun createCropUri(): Uri {

        val directory = File(cacheDir, "crop").apply { mkdirs() }

        directory.listFiles()?.forEach { it.delete() }

        val file = File(directory, "recorte_${System.currentTimeMillis()}.jpg")

        return FileProvider.getUriForFile(
            this,
            "$packageName.fileprovider",
            file
        )
    }

    private fun handleImageUri(uri: Uri, source: String) {

        val bitmap = try {
            ImageInputLoader.load(this, uri)
        } catch (e: Exception) {
            Log.e(TAG, "Fallo al convertir la imagen de $source", e)
            ui.statusText = "ERROR"
            ui.resultText = e.message ?: e.javaClass.simpleName
            return
        }

        ui.selectedImage = bitmap
        ui.sourceLabel = source
        ui.maskSamples = emptyList()
        ui.selectedMaskIndex = null
        ui.resultText = ""

        processImage(bitmap)
    }

    // ------------------------------------------------------------
    // FASE 1
    // Único punto de entrada del pipeline: recibe un Bitmap.
    // ------------------------------------------------------------

    private fun processImage(bitmap: Bitmap) {

        if (ui.isProcessing) {
            Log.w(TAG, "processImage ignorado: ya hay un proceso en curso")
            return
        }

        Log.d(TAG, "processImage recibido: ${bitmap.width}x${bitmap.height}")

        ui.isProcessing = true
        ui.statusText = "Procesando ${bitmap.width}x${bitmap.height}..."

        lifecycleScope.launch {

            try {
                // ----------------------------------------------------
                // 1. MÓDULO MobileSAM (Segmentación Automática)
                // ----------------------------------------------------
                ui.statusText = "Ejecutando MobileSAM..."

                val env = OrtEnvironment.getEnvironment()
                val encoderBytes = assets.open("mobile_sam.encoder.onnx").use { it.readBytes() }
                val decoderBytes = assets.open("mobile_sam.decoder.onnx").use { it.readBytes() }

                val encoderSession = env.createSession(encoderBytes)
                val decoderSession = env.createSession(decoderBytes)

                try {
                    val automaticTest = MobileSamAutomaticTest(
                        encoder = MobileSamEncoder(env = env, session = encoderSession),
                        decoder = MobileSamDecoder(env = env, session = decoderSession),
                        context = this@MainActivity
                    )

                    withContext(Dispatchers.Default) {
                        automaticTest.generate(bitmap = bitmap, pointsPerSide = 8)
                    }
                } finally {
                    encoderSession.close()
                    decoderSession.close()
                }

                val masks = loadMaskSamples()

                Log.d(TAG, "Segmentación lista: ${masks.size} máscaras")

                ui.maskSamples = masks
                ui.selectedMaskIndex = null

                // El pipeline se detiene acá: el usuario tiene que elegir
                // una de las máscaras y recién ese recorte entra a DINOv2.
                ui.statusText = if (masks.isEmpty()) {
                    "No se encontró ningún objeto. Probá con otra foto."
                } else {
                    "Elegí una máscara para buscar productos (${masks.size} candidatas)"
                }

            } catch (e: Exception) {
                Log.e(TAG, "Fallo en processImage", e)
                ui.statusText = "ERROR"
                ui.resultText = "${e.javaClass.simpleName}\n${e.message}"
            } finally {
                ui.isProcessing = false
            }
        }
    }

    private fun loadMaskSamples(): List<Bitmap> {

        return cacheDir
            .listFiles { file ->
                file.name.startsWith(MobileSamAutomaticTest.MASK_PREFIX) &&
                        file.name.endsWith(".png")
            }
            ?.sortedBy { it.name }
            ?.mapNotNull { BitmapFactory.decodeFile(it.absolutePath) }
            ?: emptyList()
    }

    // ------------------------------------------------------------
    // FASE 2
    // El usuario elige una de las máscaras que pasó el filtro y ese
    // recorte es la entrada de DINOv2 + matching.
    // ------------------------------------------------------------

    private fun matchSelectedMask(mask: Bitmap, index: Int) {

        if (ui.isProcessing) {
            Log.w(TAG, "Selección ignorada: ya hay un proceso en curso")
            return
        }

        Log.d(TAG, "matchSelectedMask #$index (${mask.width}x${mask.height})")

        ui.isProcessing = true
        ui.selectedMaskIndex = index
        ui.resultText = ""

        lifecycleScope.launch {

            try {
                ui.statusText = "Cargando DINOv2..."

                val matcher = obtainMatcher { ui.statusText = it }
                val catalogSize = matcher.catalogSize()

                ui.statusText = "Comparando con el catálogo..."

                val results = withContext(Dispatchers.Default) {
                    matcher.matchQuery(mask)
                }

                ui.resultText = buildString {
                    appendLine("MATCHING DINOv2 HÍBRIDO")
                    appendLine("Máscara elegida: #${index + 1} (${mask.width}x${mask.height})")
                    appendLine("Origen: ${ui.sourceLabel}")
                    appendLine("Catálogo: $catalogSize productos")
                    appendLine("Máscaras candidatas: ${ui.maskSamples.size}")
                    appendLine()
                    appendLine("RANKING:")
                    results.forEachIndexed { position, result ->
                        appendLine("${position + 1}. ${result.productName}")
                        appendLine("   DINO híbrido: %.4f".format(result.dinoSimilarity))
                        appendLine("   Color: %.4f".format(result.colorSimilarity))
                        appendLine("   Final: %.4f".format(result.finalScore))
                        appendLine()
                    }
                }

                ui.statusText = "Búsqueda finalizada"

            } catch (e: Exception) {
                Log.e(TAG, "Fallo en matchSelectedMask", e)
                ui.statusText = "ERROR"
                ui.resultText = "${e.javaClass.simpleName}\n${e.message}"
            } finally {
                ui.isProcessing = false
            }
        }
    }

    // El catálogo y el encoder se cargan una sola vez por sesión: tocar
    // varias máscaras no debe volver a correr DINOv2 sobre todo el catálogo.
    private suspend fun obtainMatcher(onStatus: (String) -> Unit): CatalogMatcher {

        cachedMatcher?.let {
            Log.d(TAG, "Reutilizando matcher cacheado (${it.catalogSize()} productos)")
            return it
        }

        val encoder = DinoV2Encoder(this@MainActivity, ExecutionProvider.NNAPI)
        val matcher = CatalogMatcher(context = this@MainActivity, encoder = encoder)
        val store = CatalogEmbeddingStore(this@MainActivity)

        if (store.exists()) {

            onStatus("Cargando catálogo...")
            matcher.addCatalogEmbeddings(withContext(Dispatchers.IO) { store.load() })

        } else {

            onStatus("Generando catálogo...")

            val files = assets.list("catalog")?.filter {
                it.endsWith(".png", true) ||
                        it.endsWith(".jpg", true) ||
                        it.endsWith(".jpeg", true)
            }?.sorted() ?: emptyList()

            if (files.isEmpty()) throw IllegalStateException("No hay imágenes en assets/catalog/")

            val embeddings = mutableListOf<CatalogEmbedding>()

            for ((position, fileName) in files.withIndex()) {

                onStatus("Procesando producto ${position + 1}/${files.size}\n$fileName")

                val catalogBitmap = assets.open("catalog/$fileName").use {
                    BitmapFactory.decodeStream(it)
                } ?: throw IllegalStateException("Error al cargar $fileName")

                val dinoOutput = encoder.encode(catalogBitmap)
                val color = ColorFeatureExtractor.extract(catalogBitmap)

                embeddings.add(
                    CatalogEmbedding(
                        productName = fileName.substringBeforeLast("."),
                        cls = dinoOutput.cls,
                        patches = dinoOutput.patches,
                        color = color
                    )
                )
            }

            matcher.addCatalogEmbeddings(embeddings)

            onStatus("Guardando catálogo...")
            withContext(Dispatchers.IO) { store.save(embeddings) }
        }

        cachedMatcher = matcher
        return matcher
    }
}

private class PipelineUiState {
    var statusText by mutableStateOf("Esperando una imagen...")
    var resultText by mutableStateOf("")
    var isProcessing by mutableStateOf(false)
    var selectedImage by mutableStateOf<Bitmap?>(null)
    var sourceLabel by mutableStateOf("")
    var maskSamples by mutableStateOf<List<Bitmap>>(emptyList())
    var selectedMaskIndex by mutableStateOf<Int?>(null)
}