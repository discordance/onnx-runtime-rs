#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(rustc_private)]

extern crate rand;

use std::ffi::{CStr, CString};
use std::time::Instant;
use crate::rand::Rng;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod onnx_runtime {
    use super::*;

    // simple wrapper as possible
    #[derive(Debug)]
    pub struct OnnxRuntimeApi {
        api: Option<*const OrtApi>,

        // re-usable input vector
        input_vec: Vec<f32>,
        // keep theses to free after
        env: *mut OrtEnv,
        session_opts: *mut OrtSessionOptions,
        session: *mut OrtSession,
        input_names_arr: *const *const i8,
        output_names_arr: *const *const i8,
        input_dims_arr: *const i64,
        output_dims_arr: *const i64,
        input_size: usize,
        output_size: usize,
        input_tensor: *mut OrtValue,
    }

    impl OnnxRuntimeApi {
        pub fn new() -> Self {
            let mut this = Self {
                api: None,
                input_vec: vec![],
                // init
                env: unsafe { std::ptr::null_mut() },
                session_opts: unsafe { std::ptr::null_mut() },
                session: unsafe { std::ptr::null_mut() },
                input_names_arr: unsafe { std::ptr::null_mut() },
                output_names_arr: unsafe { std::ptr::null_mut() },
                input_dims_arr: unsafe { std::ptr::null_mut() },
                output_dims_arr: unsafe { std::ptr::null_mut() },
                input_size: 0,
                output_size: 0,
                input_tensor: unsafe { std::ptr::null_mut() },
            };
            unsafe {
                let base = OrtGetApiBase();
                let ort_api = (*base).GetApi.unwrap()(ORT_API_VERSION);
                this.api = Some(ort_api);
            }
            this
        }

        //
        pub fn load_model(&mut self, model_path: &str) {
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

        pub fn setup(
            &mut self,
            input_dims: Vec<i64>,
            output_dims: Vec<i64>,
            input_names: Vec<&str>,
            output_names: Vec<&str>,
        ) {
            if let Some(api_ptr) = self.api {
                // setup names
                self.input_names_arr = OnnxRuntimeApi::to_c_str_vec(&input_names[..]);
                self.output_names_arr = OnnxRuntimeApi::to_c_str_vec(&output_names[..]);

                // setup dims
                let mut in_dims = input_dims.clone();
                in_dims.shrink_to_fit();
                let mut out_dims = output_dims.clone();
                out_dims.shrink_to_fit();

                // calc the flat size
                let size: i64 = in_dims.iter().product();
                self.input_size = size as usize;
                let size: i64 = out_dims.iter().product();
                self.output_size = size as usize;

                // stores pointer
                self.input_dims_arr = in_dims.as_ptr();
                std::mem::forget(in_dims);
                self.output_dims_arr = out_dims.as_ptr();
                std::mem::forget(out_dims);

                // init input vec
                self.input_vec = Vec::with_capacity(self.input_size);
                for i in 0..self.input_size {
                    self.input_vec.push(0.0); // zero init
                }
                self.input_vec.shrink_to_fit();

                unsafe {
                    // take care of the input tensor
                    let mut mem_info: *mut OrtSessionOptions = std::ptr::null_mut();
                    self.input_tensor = std::ptr::null_mut();

                    let res = (*api_ptr).CreateCpuMemoryInfo.expect("c fn")(
                        OrtAllocatorType_OrtArenaAllocator,
                        OrtMemType_OrtMemTypeCPU,
                        &mut mem_info as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    // create tensor
                    let res = (*api_ptr).CreateTensorWithDataAsOrtValue.expect("c fn")(
                        mem_info as *const _,
                        self.input_vec.as_mut_ptr() as *mut _,
                        self.input_size * std::mem::size_of::<f32>(),
                        self.input_dims_arr as *const _,
                        4,
                        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        &mut self.input_tensor as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    // release mem info
                    (*api_ptr).ReleaseMemoryInfo.expect("c fn")(mem_info as *mut _);
                }
            }
        }

        // run inference
        pub fn run(&mut self, data_in: &[f32], data_out: &mut [f32]) {
            if let Some(api_ptr) = self.api {
                // fill input_vector
                self.input_vec.copy_from_slice(data_in);

                // inference
                unsafe {
                    // created by onnx runtime
                    let mut out_tensor: *mut OrtValue = std::ptr::null_mut();

                    // run
                    let res = (*api_ptr).Run.expect("c fn")(
                        self.session as *mut _,
                        std::ptr::null(),
                        OnnxRuntimeApi::to_c_str_vec(&vec!["import/IteratorGetNext:0"][..]),
                        &self.input_tensor as *const _ as *const _,
                        1,
                        OnnxRuntimeApi::to_c_str_vec(&vec!["import/conv2d_19/Sigmoid:0"][..]),
                        1,
                        &mut out_tensor as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    // let mut floatarr: *mut f32 = data_out.as_mut_ptr();
                    let mut floatarr: *mut f32 = std::ptr::null_mut();

                    let res = (*api_ptr).GetTensorMutableData.expect("c fn")(
                        out_tensor,
                        &mut floatarr as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    let results = std::slice::from_raw_parts(floatarr, self.output_size);
                    data_out.copy_from_slice(results);
                }
            }
        }

        // test run the network
        pub fn test_run(&mut self) {
            unsafe {
                //
                let input_tensor_size: usize = 16 * 256 * 2;

                // data
                let mut data: Vec<f32> = Vec::with_capacity(input_tensor_size);
                // let mut out_names = vec![];
                for i in 0..input_tensor_size {
                    // data.push(i as f32 / (input_tensor_size + 1) as f32);
                    data.push(0.0);
                }
                data.shrink_to_fit();

                // dims
                let mut dims: Vec<i64> = vec![1, 16, 256, 2];
                dims.shrink_to_fit();

                if let Some(api_ptr) = self.api {
                    let mut mem_info: *mut OrtSessionOptions = std::ptr::null_mut();
                    let mut input_tensor: *mut OrtValue = std::ptr::null_mut();

                    let res = (*api_ptr).CreateCpuMemoryInfo.expect("c fn")(
                        OrtAllocatorType_OrtArenaAllocator,
                        OrtMemType_OrtMemTypeCPU,
                        &mut mem_info as *mut _ as *mut _,
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
                        &mut input_tensor as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    // check
                    let mut is_tensor: *mut i64 = std::ptr::null_mut();
                    let res = (*api_ptr).IsTensor.expect("c fn")(
                        input_tensor as *const _,
                        &mut is_tensor as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    (*api_ptr).ReleaseMemoryInfo.expect("c fn")(mem_info as *mut _);

                    let mut out_tensor: *mut OrtValue = std::ptr::null_mut();

                    // run
                    let now = Instant::now();

                    let res = (*api_ptr).Run.expect("c fn")(
                        self.session as *mut _,
                        std::ptr::null(),
                        OnnxRuntimeApi::to_c_str_vec(&vec!["import/IteratorGetNext:0"][..]),
                        &mut input_tensor as *const _ as *const _,
                        1,
                        OnnxRuntimeApi::to_c_str_vec(&vec!["import/conv2d_19/Sigmoid:0"][..]),
                        1,
                        &mut out_tensor as *mut _ as *mut _,
                    );
                    self.check_ort_status(res as *mut _);

                    println!("run performed in {} ms", now.elapsed().as_millis());

                    let res = (*api_ptr).IsTensor.expect("c fn")(
                        out_tensor as *const _,
                        &mut is_tensor as *mut _ as *mut _,
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

                    println!("OUT 1 t {:?}", results[0]);
                    println!("OUT 2 t {:?}", results[input_tensor_size - 1]);

                    // release
                    (*api_ptr).ReleaseValue.expect("c fn")(out_tensor);
                    (*api_ptr).ReleaseValue.expect("c fn")(input_tensor);
                    (*api_ptr).ReleaseSession.expect("c fn")(self.session);
                    (*api_ptr).ReleaseSessionOptions.expect("c fn")(self.session_opts);
                    (*api_ptr).ReleaseEnv.expect("c fn")(self.env);
                }
            }
        }

        fn to_c_str_vec(names: &[&str]) -> *const *const i8 {
            let mut v: Vec<*const i8> = vec![]; // init
            for nm in names.iter() {
                let cs = CString::new(*nm).unwrap();
                let cstr_ptr = cs.as_ptr();
                std::mem::forget(cs);
                v.push(cstr_ptr);
            }
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
        onnx_api.load_model("test_model/one.10.onnx");
        // onnx_api.test_run();
        
        onnx_api.setup(vec![1, 16, 256, 2], vec![1, 16, 256, 2], vec!["import/IteratorGetNext:0"], vec!["import/conv2d_19/Sigmoid:0"]);

        // make a fake vec
        let mut rng = rand::thread_rng();
        let numbers: Vec<f32> = (0..8192).map(|_| {
            rng.gen_range(-1.0, 1.0)
            // 0.0
        }).collect();

        let mut results: Vec<f32> = vec![0.0; 8192];

        onnx_api.run(&numbers[..], &mut results[..]);
    }
}
