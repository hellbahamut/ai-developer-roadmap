# 6.3 Java集成AI模型

## 1. Python微服务 + Java调用

这是最常见、解耦最好的方式。Java负责业务逻辑，Python负责AI推理，通过HTTP/gRPC通信。

### 1.1 Spring Boot调用Python API

```java
@Service
@Slf4j
public class LLMService {

    @Value("${llm.service.base-url}")
    private String baseUrl;

    private final WebClient webClient;

    public LLMService() {
        this.webClient = WebClient.builder()
            .baseUrl(baseUrl)
            .build();
    }

    /**
     * 调用LLM聊天接口
     */
    public String chat(String message, String sessionId) {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("message", message);
            requestBody.put("session_id", sessionId != null ? sessionId : "default");

            String response = webClient.post()
                .uri("/chat")
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.ofSeconds(30));

            // 解析JSON响应...
            return response;

        } catch (Exception e) {
            log.error("调用LLM服务失败", e);
            throw new RuntimeException("LLM服务调用失败");
        }
    }
}
```

## 2. DJL（Deep Java Library）

亚马逊推出的Java深度学习库，支持PyTorch, TensorFlow, MXNet等引擎。

```java
// 依赖
// ai.djl:api, ai.djl.pytorch:pytorch-engine

public void init() throws ModelException, IOException {
    // 加载预训练模型
    Criteria<QAInput, QAOutput> criteria = Criteria.builder()
        .setTypes(QAInput.class, QAOutput.class)
        .optModelUrls("djl://ai.djl.pytorch/bertqa")
        .build();

    model = criteria.loadModel();
    predictor = model.newPredictor();
}
```

## 3. ONNX Runtime（跨平台）

将PyTorch模型导出为ONNX格式，然后在Java中加载运行。

### 3.1 Python导出ONNX

```python
import torch.onnx

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
```

### 3.2 Java加载ONNX

```java
// 依赖: com.microsoft.onnxruntime:onnxruntime

public void init() throws OrtException {
    environment = OrtEnvironment.getEnvironment();
    session = environment.createSession("model.onnx");
}
```
