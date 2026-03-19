use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Mutex;

static GLOBAL_RNG: Mutex<Option<StdRng>> = Mutex::new(None);

/// Set the global random seed for reproducibility.
pub fn manual_seed(seed: u64) {
    let mut rng = GLOBAL_RNG.lock().unwrap();
    *rng = Some(StdRng::seed_from_u64(seed));
}

/// Get a reference-counted RNG (seeded if manual_seed was called).
pub fn get_rng() -> StdRng {
    let rng = GLOBAL_RNG.lock().unwrap();
    match &*rng {
        Some(r) => r.clone(),
        None => StdRng::from_entropy(),
    }
}
