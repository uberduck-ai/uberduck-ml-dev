__all__ = ['run']


import streamlit as st
from collections import OrderedDict
from .generate import _get_inference, MODEL_LIST, MODEL_TYPES


def run():
    st.title("Inference inspector")

    symbol_set = st.selectbox(
        "What symbol set would you like to use?", ("NVIDIA_TACO2_DEFAULTS")
    )
    st.write("You selected:", symbol_set)

    use_arpabet = st.selectbox("Would you like to use arpabet?", ("Yes", "No"))
    st.write("You selected:", use_arpabet)

    # st.text_input("Model file name", "test/fixtures/models/taco2ljdefault")
    # st.text_input("Model format", OrderedDict)
    vocoder_path = st.text_input(
        "Vocoder path", "test/fixtures/models/gen_02640000_studio"
    )
    vocoder_config = st.text_input("Vocoder config", None)
    n_speakers = st.text_input("Number of speakers", 1)
    gate_threshold = st.text_input("Gate threshold", 0.1)

    chosen_model = st.sidebar.selectbox("Select model", MODEL_LIST)
    chosen_type = st.sidebar.selectbox("Select model save type", MODEL_TYPES)
    text = [st.text_input("Text", "Thats silly")]
    speakers = [st.text_input("Speaker_id", 0)]

    hparams = TACOTRON2_DEFAULTS
    hparams.n_speakers = n_speakers
    hparams.gate_threshold = gate_threshold
    if n_speakers > 1:
        hparams.has_speaker_embedding = True
    model = Tacotron2(hparams)
    device = "cuda"
    model = Tacotron2(hparams)
    if chosen_type == "OD":
        model.from_pretrained(model_dict=chosen_model, device=device)
    if chosen_type == "OD":
        model.from_pretrained(warm_start_path=chosen_model, device=device)

    hifigan = HiFiGanGenerator(
        config=vocoder_config,
        checkpoint=vocoder_file,
        cudnn_enabled=True,
    )

    inference = _get_inference(model, vocoder, texts, speakers, symbol_set, arpabet)

    st.audio(inference)


if __name__ == "__main__":
    run()
