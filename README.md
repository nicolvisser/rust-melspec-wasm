

## Build

```sh
rustup target add wasm32-unknown-unknown
```

```sh
cargo build --target wasm32-unknown-unknown --release
```

```sh
cargo install wasm-pack
```

```sh
wasm-pack build --target web
```

## View html example

Build first.

```sh
python3 -m http.server
```

Open the server in your browser.

## Publishing

Build first.

```sh
cd pkg
```

```sh
npm publish
```
