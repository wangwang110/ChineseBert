from models.modeling_glycebert import GlyceBertForMaskedLM
path="/data_local/plm_models/ChineseBERT-base/"
chinese_bert = GlyceBertForMaskedLM.from_pretrained(path)
print(chinese_bert)