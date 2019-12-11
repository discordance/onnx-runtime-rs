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

  size_t t_s = 16 * 256 * 2;
  float* t_array = (float *)calloc(t_s, sizeof(float));

  for(int i = 0; i < t_s; ++i){
      t_array[i] = 0.0;
  } 

  static const int64_t dim_array[4] = {1, 16, 256, 2};

  // create input tensor object from data values
  OrtMemoryInfo* memory_info;
  CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtValue* input_tensor = NULL;
  CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, t_array, t_s * sizeof(float), dim_array, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  int is_tensor;
  CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
  g_ort->ReleaseMemoryInfo(memory_info);

  // score model & input tensor, get back output tensor
  const char *input_node_names[] = {"import/IteratorGetNext:0"};
  const char *output_node_names[] = {"import/conv2d_19/Sigmoid:0"};

  OrtValue* output_tensor = NULL;
  CheckStatus(g_ort->Run(session, NULL, input_node_names, (const OrtValue* const*)&input_tensor, 1, output_node_names, 1, &output_tensor));
  CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
  
  float* floatarr;
  CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));

  printf("Res1 %.10f \n", floatarr[0]);
  printf("Res2 %.10f \n", floatarr[t_s-1]);

  return 0;
}