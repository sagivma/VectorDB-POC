package org.example;

import com.google.gson.JsonObject;
import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.collection.CreateCollectionParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.LoadCollectionReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.BaseVector;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class VectorDBClient {
    private MilvusClientV2 client;

    public VectorDBClient() {
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus") // replace this with your token
                .build();
        this.client = new MilvusClientV2(connectConfig);
        System.out.println("Connected to Milvus!");
    }

    public void createCollection(String collectionName) {
        CreateCollectionReq.CollectionSchema collectionSchema = client.createSchema();
        collectionSchema.addField(AddFieldReq.builder().fieldName("id").dataType(DataType.Int64).isPrimaryKey(Boolean.TRUE).autoID(Boolean.FALSE).description("id").build());
        collectionSchema.addField(AddFieldReq.builder().fieldName("vector").dataType(DataType.FloatVector).dimension(512).build());
        collectionSchema.addField(AddFieldReq.builder().fieldName("metadata").dataType(DataType.VarChar).build());

        IndexParam indexParam = IndexParam.builder()
                .fieldName("vector")
                .metricType(IndexParam.MetricType.COSINE)
                .build();

        CreateCollectionReq createCollectionReq = CreateCollectionReq.builder()
                .collectionName(collectionName)
                .collectionSchema(collectionSchema)
                .indexParams(Collections.singletonList(indexParam))
                .build();
        client.createCollection(createCollectionReq);
    }

    public void loadCollection(String collectionName) {
        LoadCollectionReq loadCollectionReq = LoadCollectionReq.builder()
                .collectionName(collectionName)
                .build();
        client.loadCollection(loadCollectionReq);
    }

    public void insert(String collectionName, List<JsonObject> vectors) {
        InsertReq insertReq = InsertReq.builder()
                .collectionName(collectionName)
                .data(vectors)
                .build();
        client.insert(insertReq);
        System.out.println("Data inserted into Milvus!");
    }

    public List<List<SearchResp.SearchResult>> search(String collectionName, FloatVec queryVector) {
        SearchReq searchReq = SearchReq.builder()
                .collectionName(collectionName)
                .data(Collections.singletonList(queryVector))
                .outputFields(Collections.singletonList("metadata"))
                .topK(10)
                .build();

        return client.search(searchReq).getSearchResults();
    }
}
