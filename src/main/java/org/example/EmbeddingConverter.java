package org.example;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.types.TString;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EmbeddingConverter {
    public static EmbeddingConverter INSTANCE;

    private final SavedModelBundle model;

    private EmbeddingConverter() {
        this.model = SavedModelBundle.load("src\\main\\resources");
    }

    public static EmbeddingConverter getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new EmbeddingConverter();
        }
        return INSTANCE;
    }

    public List<List<Float>> convert(String... sentences) {
        // Create input tensor
        Tensor inputTensor = TString.tensorOf(NdArrays.vectorOfObjects(sentences));

        // Run the model
        try (Tensor result = model.session().runner()
                .feed("serving_default_inputs", inputTensor)
                .fetch("StatefulPartitionedCall")
                .run()
                .get(0)) {
            return getEmbeddingsResult(result);
        }
    }

    private static List<List<Float>> getEmbeddingsResult(Tensor result) {
        // Extract the embeddings from the result tensor
        FloatDataBuffer resultData = result.asRawTensor().data().asFloats();

        // Extract the dimensions from the shape
        Shape shape = result.shape();
        int sentenceCount = (int) shape.get(0); // Number of sentences
        int embeddingSize = (int) shape.get(1); // Embedding size, typically 512

        // Iterate through the tensor data and populate the embeddings list
        List<List<Float>> embeddings = new ArrayList<>();
        for (int i = 0; i < sentenceCount; i++) {
            List<Float> embedding = new ArrayList<>();
            for (int j = 0; j < embeddingSize; j++) {
                embedding.add(resultData.getFloat((long) i * embeddingSize + j));
            }
            embeddings.add(embedding);
        }

        return embeddings;
    }
}
