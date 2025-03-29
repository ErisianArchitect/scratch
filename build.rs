fn main() {
    println!("cargo:rustc-link-lib=dylib=luajit-5.1");
    println!("cargo:rustc-link-search=native=C:/Users/derek/Documents/code/c/LuaJIT/src");
}