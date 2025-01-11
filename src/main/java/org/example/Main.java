package org.example;
import com.google.gson.JsonObject;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        VectorDBClient vectorDBClient = new VectorDBClient();
        String collectionName = "coz_pilot_collection";

        if (!vectorDBClient.hasCollection(collectionName)) {
            vectorDBClient.createCollection(collectionName);
            String[] answers = {"My name is Sagiv", "I love to eat pizza", "Israel is the best country in the world", "Only in Haifa there is Maccabi", "My car is blue", "My mother name is Eti"};
            List<JsonObject> answersVectors = Utils.buildVectors(answers);
            vectorDBClient.insert(collectionName, answersVectors);
        }

        vectorDBClient.loadCollection(collectionName);

        String question = "What is my mom name?";
        EmbeddingConverter embeddingConverter = EmbeddingConverter.getInstance();
        FloatVec questionEmbedding = new FloatVec(embeddingConverter.convert(question).get(0));
        List<List<SearchResp.SearchResult>> searchResults = vectorDBClient.search(collectionName, questionEmbedding);

        for (List<SearchResp.SearchResult> results : searchResults) {
            for (SearchResp.SearchResult result : results) {
                System.out.printf("ID: %d, Score: %f, %s\n", (long)result.getId(), result.getScore(), result.getEntity().toString());
            }
        }
    }
}

