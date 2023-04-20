import streamlit as st
import torch
from uberduck_ml_dev.monitoring.generate import _get_inference
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.vocoders.hifigan import HiFiGanGenerator
import scipy
from io import BytesIO
import pandas as pd
import numpy as np

from uberduck_ml_dev.text.symbols import (
    DEFAULT_SYMBOLS,
    IPA_SYMBOLS,
    NVIDIA_TACO2_SYMBOLS,
    GRAD_TTS_SYMBOLS,
)

SYMBOL_LIST = [DEFAULT_SYMBOLS, IPA_SYMBOLS, NVIDIA_TACO2_SYMBOLS, GRAD_TTS_SYMBOLS]
MODEL_FORMAT = ["OD", "D"]
DEVICES = ["cuda", "cpu"]
device = "cuda"

from collections import namedtuple

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get(**kwargs):
        """Gets a SessionState object for the current session.

        Creates a new object if necessary.

        Parameters
        ----------
        **kwargs : any
            Default values you want to add to the session state, if we're creating a
            new one.

        Example
        -------
        >>> session_state = get(user_name='', favorite_color='black')
        >>> session_state.user_name
        ''
        >>> session_state.user_name = 'Mary'
        >>> session_state.favorite_color
        'black'

        Since you set user_name above, next time your script runs this will be the
        result:
        >>> session_state = get(user_name='', favorite_color='black')
        >>> session_state.user_name
        'Mary'

        """
        # Hack to get the session object from Streamlit.

        ctx = ReportThread.get_report_ctx()

        this_session = None

        current_server = Server.get_current()
        if hasattr(current_server, "_session_infos"):
            # Streamlit < 0.56
            session_infos = Server.get_current()._session_infos.values()
        else:
            session_infos = Server.get_current()._session_info_by_id.values()

        for session_info in session_infos:
            s = session_info.session
            if (
                # Streamlit < 0.54.0
                (hasattr(s, "_main_dg") and s._main_dg == ctx.main_dg)
                or
                # Streamlit >= 0.54.0
                (not hasattr(s, "_main_dg") and s.enqueue == ctx.enqueue)
                or
                # Streamlit >= 0.65.2
                (
                    not hasattr(s, "_main_dg")
                    and s._uploaded_file_mgr == ctx.uploaded_file_mgr
                )
            ):
                this_session = s

        if this_session is None:
            raise RuntimeError(
                "Oh noes. Couldn't get your Streamlit Session object. "
                "Are you doing something fancy with threads?"
            )

        # Got the session object! Now let's attach some state into it.

        if not hasattr(this_session, "_custom_session_state"):
            this_session._custom_session_state = SessionState(**kwargs)

        return this_session._custom_session_state


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = list(list_set)
    return unique_list


def run():
    st.title("Inference inspector")

    symbol_set = st.sidebar.selectbox("symbol_set", SYMBOL_LIST)
    arpabet = st.sidebar.number_input("arpabet", 0.0)
    model_format = st.sidebar.selectbox("model_format", MODEL_FORMAT)

    @st.cache(allow_output_mutation=True)
    def get_vocoder():
        return []

    @st.cache(allow_output_mutation=True)
    def get_model():
        return []

    @st.cache(allow_output_mutation=True)
    def get_vc():
        return []

    vocoder_path = st.text_input("Add vocoder path")
    if st.button("Add vocoder"):
        get_vocoder().append(vocoder_path)
    vocoder_path = st.sidebar.selectbox("Vocoder", unique(get_vocoder()))

    model_path = st.text_input("Add model path")
    if st.button("Add model"):
        get_model().append(model_path)
    model_path = st.selectbox("Model", unique(get_model()))

    vc_path = st.text_input("Add vocoder config path")
    if st.button("Add vocoder config"):
        get_vc().append(vc_path)
    vc_path = st.sidebar.selectbox("Vocoder config", unique(get_vc()))

    n_speakers = int(st.sidebar.number_input("n_speakers", 1))
    speaker_id = int(st.sidebar.number_input("speaker_id", 0))
    gate_threshold = float(st.sidebar.number_input("gate_threshold", 0.0))
    device = str(st.sidebar.selectbox("cuda", DEVICES))
    cpu_run = device == "cpu"
    cudnn_enabled = device == "cuda"
    text = st.text_input("Text", "Youssa Jar Jar Binks")

    columns = [
        "model_path",
        "vocoder_path",
        "arpabet",
        "vc_path",
        "n_speakers",
        "gate_threshold",
        "speaker_id",
        "symbol_set",
        "model_format",
    ]
    df = pd.DataFrame(columns=columns)
    session_state = SessionState.get(df=df)

    if st.button("Add"):
        print("add", df.shape)
        session_state.df = session_state.df.append(
            {
                "model_path": model_path,
                "vocoder_path": vocoder_path,
                "vc_path": vc_path,
                "n_speakers": n_speakers,
                "gate_threshold": gate_threshold,
                "arpabet": arpabet,
                "speaker_id": speaker_id,
                "symbol_set": symbol_set,
                "model_format": model_format,
            },
            ignore_index=True,
        )
        st.text("Updated dataframe")
        session_state.df = session_state.df.drop_duplicates(subset=columns)
        st.dataframe(session_state.df)
        print(session_state.df.shape)

    session_state.df = session_state.df.drop_duplicates(subset=columns)
    st.dataframe(session_state.df)
    selected_indices = st.multiselect("Select rows:", session_state.df.index)

    if st.button("Submit"):
        for i in selected_indices:
            n_speakers = session_state.df["n_speakers"].iloc[i]
            arpabet = session_state.df["arpabet"].iloc[i]
            speaker_id = session_state.df["speaker_id"].iloc[i]
            vocoder_path = session_state.df["vocoder_path"].iloc[i]
            vc_path = session_state.df["vc_path"].iloc[i]
            model_path = session_state.df["model_path"].iloc[i]
            symbol_set = session_state.df["symbol_set"].iloc[i]
            model_format = session_state.df["model_format"].iloc[i]

            hparams = TACOTRON2_DEFAULTS
            hparams.n_speakers = n_speakers
            hparams.gate_threshold = gate_threshold
            hparams.cudnn_enabled = cudnn_enabled
            if n_speakers > 1:
                hparams.has_speaker_embedding = True
            model = Tacotron2(hparams)

            if model_format == "OD":
                model.from_pretrained(model_dict=torch.load(model_path), device=device)
            if model_format == "D":
                model.from_pretrained(warm_start_path=model_path, device=device)
            if cudnn_enabled:
                model.cuda()
            hifigan = HiFiGanGenerator(
                config=vc_path,
                checkpoint=vocoder_path,
                cudnn_enabled=True,
            )
            texts = [text]
            speakers = torch.tensor(
                np.repeat(speaker_id, 1), device=device, dtype=torch.long
            )
            inference = _get_inference(
                model, hifigan, texts, speakers, symbol_set, arpabet, cpu_run
            )
            bio = BytesIO()
            scipy.io.wavfile.write(bio, data=inference, rate=22050)
            st.audio(bio)


if __name__ == "__main__":
    run()
