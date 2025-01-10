package org.example;
import com.google.gson.JsonObject;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Example sentences to embed
        String[] answers = {"My name is Sagiv", "I love to eat pizza"};

        EmbeddingConverter embeddingConverter = new EmbeddingConverter();
        List<List<Float>> answersEmbeddings = embeddingConverter.convert(answers);

        List<JsonObject> answersVectors = new ArrayList<>();
        for (int i = 0; i < answers.length; i++) {
            System.out.println("Embedding for answer: " + answers[i]);
            System.out.println(answersEmbeddings.get(i));

            JsonObject jsonObject = Utils.buildVector(answers[i], answersEmbeddings.get(i));
            answersVectors.add(jsonObject);
        }

        VectorDBClient vectorDBClient = new VectorDBClient();
        String collectionName = "sentence_embeddings";
        vectorDBClient.createCollection(collectionName, 512);
        vectorDBClient.insert(collectionName, answersVectors);

        String question = "What is my name?";
        FloatVec questionEmbedding = new FloatVec(embeddingConverter.convert(question).get(0));
        List<List<SearchResp.SearchResult>> searchResults = vectorDBClient.search(collectionName, questionEmbedding);

        for (List<SearchResp.SearchResult> results : searchResults) {
            for (SearchResp.SearchResult result : results) {
                System.out.printf("ID: %d, Score: %f, %s\n", (long)result.getId(), result.getScore(), result.getEntity().toString());
            }
        }
    }
}

