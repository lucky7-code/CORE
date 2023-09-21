from torch import nn

from model.AKT.akt import AKT
from model.AKT.akt_core import AKT_CORE
from model.DKT.dkt import DKT
from model.DKT.dkt_core import DKT_CORE
from model.DKVMN.dkvmn import DKVMN
from model.DKVMN.dkvmn_core import DKVMN_CORE
from model.SAINT.saint import SAINT
from model.SAKT.sakt import SAKT
from model.simple_model import  Concept_Only

def init_model(model_name, CORE, model_config, data_config, emb_type, device):

    global model
    if CORE:
        if model_name == "dkt":
            model = DKT_CORE(data_config["num_c"],data_config["num_q"], model_config["emb_size"], emb_type=emb_type).to(device)
        elif model_name == "dkvmn":
            model = DKVMN_CORE(data_config["num_c"],data_config["num_q"], model_config["dim_s"], model_config["size_m"], emb_type=emb_type).to(device)
        # elif model_name == "sakt":
        #     model = CORE_SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
        # elif model_name == "saint":
        #     model = CORE_SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
        elif model_name == "akt":
            model = AKT_CORE(data_config["num_c"], data_config["num_q"], **model_config, emb_type="qid").to(device)
        else:
            print("The wrong model name was used...")
            return None
    else:
        if model_name == "dkt":
            model = DKT(data_config["num_c"],data_config["num_q"], model_config["emb_size"], emb_type=emb_type).to(device)
        elif model_name == "dkvmn":
            model = DKVMN(data_config["num_c"],data_config["num_q"], model_config["dim_s"], model_config["size_m"], emb_type=emb_type).to(device)
        elif model_name == "sakt":
            model = SAKT(data_config["num_c"], data_config["num_q"], data_config["maxlen"], **model_config, emb_type=emb_type).to(device)
        elif model_name == "saint":
            model = SAINT(data_config["num_q"], data_config["num_c"], data_config["maxlen"], **model_config, emb_type=emb_type).to(device)
        elif model_name == "akt":
            model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type="qid").to(device)
        elif model_name == "simple_net":
            model = Concept_Only(data_config["num_c"],data_config["num_q"]).to(device)
        else:
            print("The wrong model name was used...")
            return None
    return model