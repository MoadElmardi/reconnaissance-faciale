use tfhe::{prelude::*, FheUint8, FheUint8Id, Tag};
use tfhe::{ConfigBuilder, generate_keys, set_server_key, CpuFheUint8Array, PublicKey};
use core::panic;
use std::path::{Path};
use std::time::Instant;
use std::{fs, io};

pub fn main() {
    // Initialize the TFHE library
    let config = ConfigBuilder::default().build();
    let (cks, sks) = generate_keys(config);
    
    set_server_key(sks);
    let public_key = PublicKey::new(&cks);

    // let npy_path = PathBuf::from("scripts/orb_out/s4_7.npy");
    // let npy_path2 = PathBuf::from("scripts/orb_out/s2_9.npy");
    // let vecteur = load_orb_npy(&npy_path);
    // let vecteur2 = load_orb_npy(&npy_path2);

    let all = load_all_descriptors(Path::new("scripts/orb_out"));
    println!("Loaded {} images", all.len());

    let mut y_true = Vec::new();
    let mut y_pred = Vec::new();

    let tau = 57; // seuil à calibrer

    let len = all.len() - 200;

    for i in 0..len {
        for j in (i+1)..len {
            println!("Iteration i={}, j={}", i, j);
            let now = Instant::now();
            let (id_i, desc_i) = &all[i];
            let (id_j, desc_j) = &all[j];

            let v_chiffre_i = CpuFheUint8Array::try_encrypt(desc_i.as_slice(), &cks).unwrap();
            let v_chiffre_j = CpuFheUint8Array::try_encrypt(desc_j.as_slice(), &cks).unwrap();

            // vrai label
            let same_subject = id_i.split('_').next() == id_j.split('_').next();

            // distance
            let d = hamming_distance(v_chiffre_i, v_chiffre_j, public_key.clone());
            let d_decrypted: u8 = d.decrypt(&cks);
            let pred_same = d_decrypted <= tau;

            y_true.push(same_subject);
            y_pred.push(pred_same);
            println!("{}", now.elapsed().as_secs_f64());
        }
    }

    // calcul précision = (tp+tn)/total
    let mut tp=0; let mut tn=0; let mut fp=0; let mut fn_=0;
    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        match (yt, yp) {
            (true, true) => tp+=1,
            (false, false) => tn+=1,
            (false, true) => fp+=1,
            (true, false) => fn_+=1,
        }
    }
    let acc = (tp+tn) as f64 / (tp+tn+fp+fn_) as f64;
    println!("TP={tp}, TN={tn}, FP={fp}, FN={fn_}");
    println!("Accuracy={:.3}", acc);
}

fn load_orb_npy(path: &Path) -> Vec<u8> {

    let file = io::BufReader::new(fs::File::open(path).unwrap());
    let npy = npyz::NpyFile::new(file)
        .expect("Failed to read npy file");
    let v: Vec<u8> = npy.into_vec()
        .expect("Failed to convert npy data to vector");
    
    if v.len() != 32 {
        panic!("Expected 32 bytes, got {}", v.len());
    }

    return v;
}

fn load_all_descriptors(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().unwrap_or_default() == "npy" {
            let v = load_orb_npy(&path);
            let id = path.file_stem().unwrap().to_string_lossy().to_string();
            out.push((id, v));
        }
    }
    out
}

fn hamming_distance(a: CpuFheUint8Array, b: CpuFheUint8Array, public_key: PublicKey) -> FheUint8 {
    if a.shape()[0] != b.shape()[0] {
        panic!("Vectors must be of the same length");
    }

    let vector_a = a.as_slice();
    let vector_b = b.as_slice();

    let vector_c = &vector_a ^ &vector_b; // Bitwise XOR 

    // let vec_a: Vec<u8> = a.decrypt(&clientkey);
    // let vec_b: Vec<u8> = b.decrypt(&clientkey);

    // let vec_c = vec_a.iter().zip(vec_b.iter())
    //     .map(|(x, y)| x ^ y) // XOR operation
    //     .collect::<Vec<u8>>();

    // let c: Vec<u8> = vector_c.decrypt(&clientkey);

    // assert_eq!(c, vec_c, "XOR operation did not match expected result");

    // let zero = vec![0u8; 1];
    // let zero_chiffre: Vec<FheUint8> = zero.iter().map(|&x| FheUint8::encrypt(x, &public_key)).collect();
    // let shape = vec![1];
    // let ct_vec: Vec<BaseRadixCiphertext<Ciphertext>> = zero_chiffre
    //     .into_iter()
    //     .map(|x| x.into_raw_parts().0)
    //     .collect();
    // let mut dist = CpuFheUint8Array::new(ct_vec, shape);
    let mut dist = FheUint8::encrypt(0u8, &public_key);

    for i in 0..vector_c.shape()[0] {
        let byte = vector_c.slice(&[i..i+1]); // Get the byte at index i
        let raw_byte = byte.container()[0].clone();
        let scalar_byte = FheUint8::from_raw_parts(raw_byte, FheUint8Id, Tag::default());
        let bits_number = scalar_byte.count_ones().cast_into(); // Count the number of set bits in the byte

        // Count the number of set bits in the byte
        dist = &dist + &bits_number; // Add the count to the total distance
    }
    
    dist
}