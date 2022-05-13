config = {
    'train_batch_size': 64,
    'test_batch_size': 16,
    "learn_rate": 0.0002,
    "lr_dc_step": 2,
    "lr_dc": 0.9,
    'bpr_weight': 0.7,
    "train_epoch": 20,
    'temperature': 0.1,
    'max_seq_Q_len': 40,
    'max_seq_A_len': 40,
    "pretrain_epoch": 1,
    "margine": 0.5,
    "num_of_labels": 70,
    "seq_feature_dim": 768,  # 768#384
    "pretrain_model": "sentence-transformers/paraphrase-mpnet-base-v2"
    #bert-base-uncased 768
    #sentence-transformers/all-mpnet-base-v2  768
    #"sentence-transformers/paraphrase-mpnet-base-v2"

    #sentence-transformers/all-MiniLM-L6-v2 384
    #sentence-transformers/paraphrase-MiniLM-L6-v2
}
