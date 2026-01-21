use core::ffi::c_void;
pub struct DeviceBuffer<T>
{
    ptr: *mut T, //raw GPU pointer (will come from CUDA)
    len: usize, // number of elements
}
extern "C"
{
    pub fn launch_vector_add(a: *const f32, b: *const f32, c: *mut f32, n: i32);
    pub fn cuda_alloc(ptr: *mut *mut core::ffi::c_void, size: usize) -> i32;
    pub fn cuda_free(ptr: *mut core::ffi::c_void) -> i32;
}

impl<T> DeviceBuffer<T>
{
    pub fn new(len: usize) -> Self
    {
        let mut raw_ptr: *mut c_void = core::ptr::null_mut();
        let size = len * core::mem::size_of::<T>();
        let result = unsafe { cuda_alloc(&mut raw_ptr as *mut *mut c_void, size) };

        if result != 0
          { panic!("cudaMalloc failed with error code {}",result); }

        Self {ptr: raw_ptr as *mut T, len}
    }

    pub fn as_ptr(&self) -> *mut T
    {
        self.ptr
    }

    pub fn len(&self) -> usize
    {
        self.len
    }

}
    
impl<T> Drop for DeviceBuffer<T>
{
    fn drop(&mut self)
    {
        if !self.ptr.is_null()
          { unsafe{cuda_free(self.ptr as *mut c_void);} }
    }
}