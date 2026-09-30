package com.example.tesis_identificador

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import ai.onnxruntime.OrtEnvironment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val statusText = remember { mutableStateOf("Preparando prueba...") }
            val resultText = remember { mutableStateOf("") }
            val isProcessing = remember { mutableStateOf(true) }
            val maskSamples = remember { mutableStateOf<List<android.graphics.Bitmap>>(emptyList()) }

            LaunchedEffect(Unit) {
                try {
                    // ----------------------------------------------------
                    // 1. MÓDULO MobileSAM (Segmentación Automática)
                    // ----------------------------------------------------
                    statusText.value = "Ejecutando MobileSAM..."

                    val env = OrtEnvironment.getEnvironment()
                    val encoderBytes = assets.open("mobile_sam.encoder.onnx").use { it.readBytes() }
                    val decoderBytes = assets.open("mobile_sam.decoder.onnx").use { it.readBytes() }

                    val encoderSession = env.createSession(encoderBytes)
                    val decoderSession = env.createSession(decoderBytes)

                    val mobileSamEncoder = MobileSamEncoder(env = env, session = encoderSession)
                    val mobileSamDecoder = MobileSamDecoder(env = env, session = decoderSession)

                    val testBitmap = BitmapFactory.decodeStream(assets.open("queries/mask_test_multi.jpg"))

                    // Ejecutamos únicamente la prueba de grilla automática
                    val automaticTest = MobileSamAutomaticTest(
                        encoder = mobileSamEncoder,
                        decoder = mobileSamDecoder,
                        context = this@MainActivity
                    )

                    withContext(Dispatchers.Default) {
                        automaticTest.generate(bitmap = testBitmap, pointsPerSide = 8)
                    }

                    // Carga de las máscaras generadas para la tira visual de Compose
                    maskSamples.value = cacheDir
                        .listFiles { f ->
                            f.name.startsWith(MobileSamAutomaticTest.MASK_PREFIX) && f.name.endsWith(".png")
                        }
                        ?.sortedBy { it.name }
                        ?.mapNotNull { BitmapFactory.decodeFile(it.absolutePath) }
                        ?: emptyList()

                    // Cerramos sesiones ONNX una vez completado el test
                    encoderSession.close()
                    decoderSession.close()

                    // ----------------------------------------------------
                    // 2. MÓDULO DINOv2 + Matcher (Reconocimiento)
                    // ----------------------------------------------------
                    statusText.value = "Cargando DINOv2..."
                    val encoder = DinoV2Encoder(this@MainActivity, ExecutionProvider.NNAPI)
                    val matcher = CatalogMatcher(context = this@MainActivity, encoder = encoder)
                    val store = CatalogEmbeddingStore(this@MainActivity)

                    val catalogEmbeddings: List<CatalogEmbedding> = if (store.exists()) {
                        statusText.value = "Cargando catálogo..."
                        withContext(Dispatchers.IO) { store.load() }
                    } else {
                        statusText.value = "Generando catálogo..."
                        withContext(Dispatchers.Default) {
                            val embeddings = mutableListOf<CatalogEmbedding>()
                            val files = assets.list("catalog")?.filter {
                                it.endsWith(".png", true) || it.endsWith(".jpg", true) || it.endsWith(".jpeg", true)
                            }?.sorted() ?: emptyList()

                            if (files.isEmpty()) throw IllegalStateException("No hay imágenes en assets/catalog/")

                            for ((index, fileName) in files.withIndex()) {
                                statusText.value = "Procesando producto ${index + 1}/${files.size}\n$fileName"
                                val bitmap = assets.open("catalog/$fileName").use { BitmapFactory.decodeStream(it) }
                                    ?: throw IllegalStateException("Error al cargar $fileName")

                                val dinoOutput = encoder.encode(bitmap)
                                val color = ColorFeatureExtractor.extract(bitmap)

                                embeddings.add(
                                    CatalogEmbedding(
                                        productName = fileName.substringBeforeLast("."),
                                        cls = dinoOutput.cls,
                                        patches = dinoOutput.patches,
                                        color = color
                                    )
                                )
                            }
                            embeddings
                        }
                    }

                    if (!store.exists()) {
                        statusText.value = "Guardando catálogo..."
                        withContext(Dispatchers.IO) { store.save(catalogEmbeddings) }
                    }

                    matcher.addCatalogEmbeddings(catalogEmbeddings)

                    // ----------------------------------------------------
                    // 3. BUSQUEDA HÍBRIDA SOBRE MÁSCARA
                    // ----------------------------------------------------
                    statusText.value = "Comparando con el catálogo..."
                    val queryBitmap = assets.open("queries/mask_01.png").use { BitmapFactory.decodeStream(it) }
                        ?: throw IllegalStateException("No se pudo cargar mask_01.png")

                    val results = withContext(Dispatchers.Default) {
                        matcher.matchQuery(queryBitmap)
                    }

                    // Formatear resultados
                    resultText.value = buildString {
                        appendLine("MATCHING DINOv2 HÍBRIDO")
                        appendLine("Catálogo: ${catalogEmbeddings.size} productos\n")
                        appendLine("RANKING:")
                        results.forEachIndexed { index, result ->
                            appendLine("${index + 1}. ${result.productName}")
                            appendLine("   DINO híbrido: %.4f".format(result.dinoSimilarity))
                            appendLine("   Color: %.4f".format(result.colorSimilarity))
                            appendLine("   Final: %.4f".format(result.finalScore))
                            appendLine()
                        }
                    }

                    statusText.value = "Prueba finalizada"
                    isProcessing.value = false

                } catch (e: Exception) {
                    statusText.value = "ERROR"
                    resultText.value = "${e.javaClass.simpleName}\n${e.message}"
                    isProcessing.value = false
                }
            }

            // Layout UI
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(text = statusText.value)
                Spacer(modifier = Modifier.height(20.dp))

                if (isProcessing.value) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(20.dp))
                }

                if (maskSamples.value.isNotEmpty()) {
                    Text(text = "Primeras máscaras MobileSAM")
                    Spacer(modifier = Modifier.height(16.dp))

                    Row(modifier = Modifier.horizontalScroll(rememberScrollState())) {
                        maskSamples.value.forEachIndexed { index, bitmap ->
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(text = "Mask ${index + 1}")
                                Spacer(modifier = Modifier.height(8.dp))
                                Image(
                                    bitmap = bitmap.asImageBitmap(),
                                    contentDescription = "Mask ${index + 1}",
                                    modifier = Modifier.width(200.dp)
                                )
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }

                Text(text = resultText.value)
            }
        }
    }
}