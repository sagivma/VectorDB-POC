package org.example;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Utils {
    public static List<JsonObject> buildVectors(String[] sentences) {
        EmbeddingConverter embeddingConverter = EmbeddingConverter.getInstance();
        List<List<Float>> embeddings = embeddingConverter.convert(sentences);

        List<JsonObject> vectors = new ArrayList<>();
        for (int i = 0; i < sentences.length; i++) {
            System.out.println("Embedding for sentence: " + sentences[i]);
            System.out.println(embeddings.get(i));

            JsonObject jsonObject = buildVector(sentences[i], embeddings.get(i));
            vectors.add(jsonObject);
        }

        return vectors;
    }

    private static JsonObject buildVector(String sentence, List<Float> embedding) {
        Gson gson = new Gson();
        JsonObject vector = new JsonObject();
        vector.add("vector", gson.toJsonTree(embedding));
        vector.addProperty("id", new Random().nextInt());
        vector.addProperty("metadata", sentence);

        return vector;
    }
}
