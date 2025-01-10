package org.example;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.util.List;

public class Utils {
    public static JsonObject buildVector(String sentence, List<Float> embedding) {
        Gson gson = new Gson();
        JsonObject vector = new JsonObject();
        vector.add("embedding", gson.toJsonTree(embedding));
        vector.addProperty("sentence", sentence);

        return vector;
    }
}
