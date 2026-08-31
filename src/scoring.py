"""[Rishi] Reconstruct a model from a saved checkpoint and score rows with
it. Moved here from scripts/eval_checkpoint.py so src/models/ensemble.py
can reuse it without importing across the scripts/ boundary (that stayed
script-only, no __init__.py, by design — this is the shared piece both
need).
"""
import torch

from src.models.fm import FM
from src.models.deepfm_mtl import DeepFMMTL


def score(config, state_dict, X, dim, splits=None, split_name=None):
    name = config['model']['name']
    if name == 'fm':
        _, k = state_dict['V'].shape
        m = FM(dim, k=k)
        m.load_state_dict(state_dict)
        return m.predict(X)

    if name == 'deepfm_mtl':
        from src.models.seq_encoder import SeqEncoder
        mcfg = config['model']
        seq_kwargs = mcfg.get('seq_kwargs')
        seq_dim = 0
        seq_encoder = None
        if seq_kwargs is not None:
            seq_encoder = SeqEncoder(**seq_kwargs)
            seq_encoder.load_state_dict(state_dict['seq_encoder'])
            seq_encoder.eval()
            seq_dim = seq_encoder.out_dim

        m = DeepFMMTL(dim, embed_dim=mcfg.get('embed_dim', 16),
                       num_fields=mcfg.get('num_fields', 5), seq_dim=seq_dim)
        m.load_state_dict(state_dict['deepfm'])
        m.eval()

        seq_embedding = None
        if seq_encoder is not None:
            from src.features.sequential import build_sequences, build_video_vocab, sequences_for_rows
            if splits is None or split_name is None:
                raise ValueError('splits/split_name required to rebuild sequences for a seq checkpoint')
            vocab_size = seq_kwargs['num_items']
            # Re-derive the same train-window vocab/sequences build_seq_arrays
            # used at training time — deterministic given the same splits.
            vocab = build_video_vocab(splits)
            if len(vocab) != vocab_size:
                raise ValueError(f"rebuilt vocab size {len(vocab)} != checkpoint's {vocab_size} "
                                  f"— splits/data changed since this checkpoint was trained")
            user_seqs = build_sequences(splits, vocab=vocab)
            seq_ids = sequences_for_rows(splits[split_name], user_seqs, pad_id=vocab_size)
            with torch.no_grad():
                seq_embedding = seq_encoder(torch.from_numpy(seq_ids))

        with torch.no_grad():
            X_t = torch.from_numpy(X).long()
            return m(X_t, seq_embedding=seq_embedding)['primary'].numpy()

    raise NotImplementedError(f"no scoring path registered for model '{name}'")
