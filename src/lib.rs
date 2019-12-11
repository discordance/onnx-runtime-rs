#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(rustc_private)]

extern crate libc;

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::time::{Instant};


include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
                env: unsafe { std::ptr::null_mut() },
                session_opts: unsafe { std::ptr::null_mut() },
                session: unsafe { std::ptr::null_mut() },
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

        // print model input layer (node names, types, shape etc.)
        pub fn print_network(&mut self) {

        }

        // test run the network
        pub fn test_run(&mut self) {
            unsafe {
                //
                let input_tensor_size : usize = 16 * 256 * 2;

                // data
                let mut data : Vec<f32> = Vec::with_capacity(input_tensor_size);
                // let mut out_names = vec![];
                for i in 0..input_tensor_size {
                    // data.push(i as f32 / (input_tensor_size + 1) as f32);
                    data.push(0.0);
                }
                data.shrink_to_fit();

                // dims
                let mut dims : Vec<i64> = vec![1, 16, 256, 2];
                dims.shrink_to_fit();

                if let Some(api_ptr) = self.api {

                    let mut mem_info : *mut OrtSessionOptions = std::ptr::null_mut();
                    let mut input_tensor : *mut OrtValue = std::ptr::null_mut();

                    let res = (*api_ptr).CreateCpuMemoryInfo.expect("c fn")(
                        OrtAllocatorType_OrtArenaAllocator, 
                        OrtMemType_OrtMemTypeCPU, 
                        &mut mem_info as *mut _ as *mut _
                    );
                    self.check_ort_status(res as *mut _);

                    // let data_ptr = &mut data as *mut _ as *mut _;
                    let data_ptr = data.as_mut_ptr();
                    let dims_ptr = dims.as_ptr();
                    std::mem::forget(data);
                    std::mem::forget(dims);

                    // create tensor
                    let res = (*api_ptr).CreateTensorWithDataAsOrtValue.expect("c fn")(
                        mem_info as *const _, 
                        data_ptr as *mut _, 
                        input_tensor_size * std::mem::size_of::<f32>(), 
                        dims_ptr as *const _, 
                        4, 
                        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
                        &mut input_tensor as *mut _ as *mut _
                    );
                    self.check_ort_status(res as *mut _);

                    // check
                    let mut is_tensor : *mut i64 = MaybeUninit::zeroed().assume_init();
                    let res = (*api_ptr).IsTensor.expect("c fn")(
                        input_tensor as *const _, 
                        &mut is_tensor as *mut _ as *mut _
                    );
                    self.check_ort_status(res as *mut _);

                    (*api_ptr).ReleaseMemoryInfo.expect("c fn")(mem_info as *mut _);

                    let mut out_tensor : *mut OrtValue = MaybeUninit::zeroed().assume_init();

                    // run
                    let now = Instant::now();

                    let res = (*api_ptr).Run.expect("c fn")(
                        self.session as *mut _,
                        std::ptr::null(),
                        OnnxRuntimeApi::input_names(), 
                        &mut input_tensor as *const _ as *const _,
                        1, 
                        OnnxRuntimeApi::output_names(), 
                        1, 
                        &mut out_tensor as *mut _ as *mut _
                    );
                    self.check_ort_status(res as *mut _);

                    println!("run performed in {} ms", now.elapsed().as_millis());

                    let res = (*api_ptr).IsTensor.expect("c fn")(
                        out_tensor as *const _, 
                        &mut is_tensor as *mut _ as *mut _
                    );
                    self.check_ort_status(res as *mut _);

                    println!("is_tensor {:?}", is_tensor);

                    let mut floatarr: *mut f32 = std::ptr::null_mut();

                    let res = (*api_ptr).GetTensorMutableData.expect("c fn")(
                        out_tensor,
                        &mut floatarr as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    let results = std::slice::from_raw_parts(floatarr, input_tensor_size);

                    println!("OUT 1 {:?}", results[0]);
                    println!("OUT 2 {:?}", results[input_tensor_size-1]);

                    // release
                    (*api_ptr).ReleaseValue.expect("c fn")(out_tensor);
                    (*api_ptr).ReleaseValue.expect("c fn")(input_tensor);
                    (*api_ptr).ReleaseSession.expect("c fn")(self.session);
                    (*api_ptr).ReleaseSessionOptions.expect("c fn")(self.session_opts);
                    (*api_ptr).ReleaseEnv.expect("c fn")(self.env);
                }
            }
        }

        fn input_names() -> *const *const i8 {
            let cstr = CStr::from_bytes_with_nul(b"import/IteratorGetNext:0\0").unwrap();
            let cstr_ptr =  cstr.as_ptr();
            std::mem::forget(cstr);
            let v = vec![
                cstr_ptr
            ];
            let p = v.as_ptr();
            std::mem::forget(v);
            p
        }

        fn output_names() -> *const *const i8 {
            let cstr = CStr::from_bytes_with_nul(b"import/conv2d_19/Sigmoid:0\0").unwrap();
            let cstr_ptr =  cstr.as_ptr();
            std::mem::forget(cstr);
            let v = vec![
                cstr_ptr
            ];
            let p = v.as_ptr();
            std::mem::forget(v);
            p
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
        onnx_api.test_run();
    }
}
