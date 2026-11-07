use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<()> {
    // 1. Forzar CPU
    let device = Device::Cpu;
    println!("Dispositivo: {:?}", device);

    // 2. Cargar pesos desde SafeTensors (mmap, sin cargar todo en RAM)
    // En candle 0.11 el tercer argumento es el DType y la función es unsafe.
    let model_path = Path::new("weights/model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    // 3. Cargar tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file("tokenizer/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

    // 4. Preparar entrada
    let texto = "Hola, ¿cómo estás?";
    let encoding = tokenizer
        .encode(texto, true)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;

    let ids: Vec<u32> = encoding.get_ids().iter().map(|&x| x as u32).collect();
    let input = Tensor::from_vec(ids, (1, encoding.get_ids().len()), &device)?;

    println!("Tokens de entrada: {:?}", encoding.get_ids());

    // ---------------------------------------------------------
    // 5. Forward pass
    //
    // Aquí se instancia la arquitectura concreta (GPT-2, Llama, etc.).
    //
    // Ejemplo conceptual con candle-transformers 0.11:
    //
    // use candle_transformers::models::gpt2::{GPT2, Config};
    // let config = Config::default();
    // let model = GPT2::load(&config, &vb)?;
    // let logits = model.forward(&input)?;
    //
    // Ajusta las importaciones según la versión instalada.
    // ---------------------------------------------------------

    println!("Motor de inferencia listo para CPU.");
    println!("Entrada procesada: {} tokens", encoding.get_ids().len());

    Ok(())
}
