# WebAssembly to compute mel spectrograms

This module provides a WebAssebly compiled library to compute a mel spectrogram from an audio signal. The code is written in Rust.

Mel spectrograms are widely used in audio processing and machine learning applications, particularly in speech and music analysis.
This implementation should (at the time of writing) be considered for display purposes, and not for scientific computing.

## Getting Started

Install the package in your project:

```sh
npm install rust-melspec-wasm
```

Alternatively, clone the project and follow the build instructions below to compile the Rust code into WebAssembly and integrate it into your project.

In JavaScript, import the module:

```js
import { mel_spectrogram_db } from "rust-melspec-wasm";
```

In React applications using Vite, you might need to initialize the wasm module first:

```jsx
import init, { mel_spectrogram_db } from "rust-melspec-wasm";

const App = () => {
  const [wasmReady, setWasmReady] = useState(false);

  useEffect(() => {
    init()
      .then(() => {
        setWasmReady(true);
        console.log("WASM initialized successfully");
      })
      .catch((error) => {
        console.error("Failed to initialize WASM:", error);
      });
  }, []);
};
```

Use the function `mel_spectrogram_db` to compute the mel spectrogram of an audio signal:

```js
const melspec = mel_spectrogram_db(
  sampleRate,
  waveform,
  n_fft,
  win_length,
  hop_length,
  f_min,
  f_max,
  n_mels,
  top_db
);
```

The data types of the arguments in the Rust code are:

```rust
fn mel_spectrogram_db(
    sample_rate: f32,
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
)
```

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

Build first:

```sh
python3 -m http.server
```

Open the server in your browser.

