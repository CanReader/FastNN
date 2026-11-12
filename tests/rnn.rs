use fastnn::prelude::*;

#[test]
#[should_panic(expected = "sequence length is 0")]
fn lstm_zero_length_sequence_panics() {
    let lstm = LSTM::new(2, 3);
    lstm.forward(&Tensor::zeros(&[1, 0, 2]));
}

#[test]
#[should_panic(expected = "sequence length is 0")]
fn gru_zero_length_sequence_panics() {
    let gru = GRU::new(2, 3);
    gru.forward(&Tensor::zeros(&[1, 0, 2]));
}

#[test]
#[should_panic(expected = "stack: needs at least one tensor")]
fn stack_empty_panics() {
    Tensor::stack(&[], 0);
}
