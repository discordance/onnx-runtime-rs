#include <stdio.h>
#include <stdlib.h>
#include "include/onnxruntime_c_api.h"

const OrtApi* g_ort;

void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

int main() {
  printf("Hello, ONNX! \n");

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  OrtEnv* env;
  CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  OrtSessionOptions* session_options;
  CheckStatus(g_ort->CreateSessionOptions(&session_options));
  g_ort->SetIntraOpNumThreads(session_options, 1);

  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  const char* model_path = "/Users/nunja/Rust/onnx-runtime/test_model/one.10.onnx";

  printf("Using Onnxruntime C API\n");
  OrtSession* session;
  CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

  size_t num_input_nodes;
  OrtStatus* status;
  OrtAllocator* allocator;
  CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  return 0;
}