import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

payload = {
    "inputText": "Hello",
    "textGenerationConfig": {"maxTokenCount": 1}
}

resp = client.invoke_model(
    modelId="amazon.titan-text-express-v1",
    body=json.dumps(payload).encode("utf-8"),
    contentType="application/json",
    accept="application/json",
)

print(resp["body"].read().decode("utf-8"))
