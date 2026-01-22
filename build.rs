fn main()
{
  println!("cargo:rustc-link-search=native=./cuda");
  println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  println!("cargo:rustc-link-lib=static=vector_add");
  println!("cargo:rustc-link-lib=dylib=cudart");
}