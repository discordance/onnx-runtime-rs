#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CString;
use std::os::raw::c_char;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

static EMPTY: [u8; 0] = [0;0];

mod onnx_runtime {
    use super::*;

    // checks and displar ORT status
    pub fn check_ort_status(status: *mut OrtStatus, api: *const OrtApi) {
        if !status.is_null() {
            unsafe {
                let msg = (*api).GetErrorMessage.unwrap()(status) as *mut i8;
                let c_string = CString::from_raw(msg);
                println!(">>>> {:?}", c_string.to_str().unwrap_or("empty c string"));
                (*api).ReleaseStatus.unwrap()(status);
                exit(1);
            }
        }
    }
}

// https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env() {
        unsafe {

            // env is created
            let env = &mut OrtEnv { _unused: EMPTY } as  *mut _;

            let g_ort = OrtGetApiBase();
            let ort_api = (*g_ort).GetApi.unwrap();

            let g_ort_r = ort_api(ORT_API_VERSION);
            let g_ort = *g_ort_r;

            let res = g_ort.CreateEnv.unwrap()(
                OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
                "test".as_bytes().as_ptr() as *const i8,
                env as *mut _,
            );
            onnx_runtime::check_ort_status(res, g_ort_r as *const OrtApi);

            // env is created
              
            // initialize session options if needed
            // &my_struct as *const _
            let session_options = &mut OrtSessionOptions{  _unused: EMPTY } as *mut _;

            let res = g_ort.CreateSessionOptions.unwrap()(session_options as *mut _);
            onnx_runtime::check_ort_status(res, g_ort_r as *const OrtApi);
            
            // Sets graph optimization level
            let res = g_ort.SetSessionGraphOptimizationLevel.unwrap()(
                session_options  as *mut _, 
                GraphOptimizationLevel_ORT_ENABLE_BASIC
            );
            onnx_runtime::check_ort_status(res, g_ort_r as *const OrtApi);

            // thread
            let res = g_ort.SetIntraOpNumThreads.unwrap()(session_options  as *mut _, 1i32);
            onnx_runtime::check_ort_status(res, g_ort_r as *const OrtApi);

            // create session and load model into memory
            let session = &mut OrtSession{  _unused: EMPTY } as *mut _;

            let path = CString::new("Hello, world!").unwrap();
            
            let res = g_ort.CreateSession.unwrap()(
                env as *const _,
                path.as_ptr(), 
                session_options as *const _, 
                session as *mut _);
            // onnx_runtime::check_ort_status(res, g_ort_r as *const OrtApi); 

            println!("lol");
        }
    }
}
