#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{CStr, CString};
// use std::os::raw::c_char;
use std::mem::MaybeUninit;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

static EMPTY: [u8; 0] = [0; 0];

mod onnx_runtime {
    use super::*;

    // simples wrapper possible
    pub struct OnnxRuntimeApi {
        api: Option<*const OrtApi>,

        // keep theses to free after
        env: *mut OrtEnv,
        session_opts: *mut OrtSessionOptions,
        session: *mut OrtSession,
    }

    impl OnnxRuntimeApi {
        pub fn new() -> Self {
            let mut this = Self {
                api: None,
                // init
                env: unsafe { MaybeUninit::zeroed().assume_init() },
                session_opts: unsafe { MaybeUninit::zeroed().assume_init() },
                session: unsafe { MaybeUninit::zeroed().assume_init() },
            };
            unsafe {
                let base = OrtGetApiBase();
                let ort_api = (*base).GetApi.unwrap()(ORT_API_VERSION);
                this.api = Some(ort_api);
            }
            this
        }

        //
        pub fn init_session(&mut self, model_path: &str) {
            unsafe {
                if let Some(api_ptr) = self.api {
                    // env
                    let res = (*api_ptr).CreateEnv.expect("c fn")(
                        OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
                        CString::new("").unwrap().as_ptr(),
                        &mut self.env as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    // session opts
                    let res = (*api_ptr).CreateSessionOptions.expect("c fn")(
                        &mut self.session_opts as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    let path = CString::new(model_path).unwrap();
                    let res = (*api_ptr).CreateSession.unwrap()(
                        self.env as *const _,
                        path.as_ptr(),
                        self.session_opts as *const _,
                        &mut self.session as *mut _ as *mut _,
                    );

                    self.check_ort_status(res as *mut _);
                }
            }
        }

        // get inv as mut mut
        fn get_env_mut(&mut self) -> *mut *mut OrtEnv {
            let mut er = &mut self.env as *mut _;
            er as *mut *mut _
        }

        // get sess opts as mut mut
        fn get_session_opts_mut(&mut self) -> *mut *mut OrtSessionOptions {
            let mut er = &mut self.session_opts as *mut _;
            er as *mut *mut _
        }

        // get sess as mut mut
        fn get_session_mut(&mut self) -> *mut *mut OrtSession {
            let mut er = &mut self.session as *mut _;
            er as *mut *mut _
        }

        // checks and display ORT status
        fn check_ort_status(&self, status: *mut OrtStatus) {
            if !status.is_null() {
                unsafe {
                    if let Some(api_ptr) = self.api {
                        let msg = (*api_ptr).GetErrorMessage.unwrap()(status) as *const i8;
                        let c_str = CStr::from_ptr(msg);
                        println!("ONNX ERROR: {:?}", c_str);
                        (*api_ptr).ReleaseStatus.unwrap()(status);
                        exit(1);
                    }
                }
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
        // init
        let mut onnx_api = onnx_runtime::OnnxRuntimeApi::new();
        onnx_api.init_session("test_model/one.10.onnx");
    }
}
