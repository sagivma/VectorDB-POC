package org.example;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.util.List;
import java.util.Random;
import java.util.UUID;

public class Utils {
    public static JsonObject buildVector(String sentence, List<Float> embedding) {
        Gson gson = new Gson();
        JsonObject vector = new JsonObject();
        vector.add("vector", gson.toJsonTree(embedding));
        vector.addProperty("id", new Random().nextInt());
        vector.addProperty("metadata", sentence);

        return vector;
    }
}
