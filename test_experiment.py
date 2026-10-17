"""Small checks for data leakage, directional states, and generation."""
import unittest
import torch
from experiment import CONFIG, LanguageModel, prepare, tokenize, generate


class ExperimentTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("Alice’s 12 cats!"), ["alice's", '12', 'cats', '!'])

    def test_sequences_and_train_vocabulary(self):
        datasets, vocab, lookup, stats = prepare()
        self.assertEqual(sum(stats[n]['tokens'] for n in ['train', 'validation', 'test']), stats['total_tokens'])
        for ds, name in zip(datasets, ['train', 'validation', 'test']):
            x, y = ds.tensors
            self.assertEqual(x.shape, (stats[name]['tokens']-12, 12))
            self.assertTrue(torch.equal(x[1:, -1], y[:-1]))
        self.assertEqual(lookup.get('unseenwordxyz', 0), 0)

    def test_model_shapes_gradients_and_bidirectional_state(self):
        x = torch.randint(0, 30, (3, 12))
        for kind in ['RNN', 'BiRNN', 'LSTM', 'GRU']:
            model = LanguageModel(kind, 30)
            logits = model(x)
            self.assertEqual(logits.shape, (3, 30))
            logits.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if kind == 'BiRNN':
                outputs, hidden = model.recurrent(model.embedding(x))
                self.assertTrue(torch.allclose(hidden[-2], outputs[:, -1, :CONFIG['hidden']]))
                self.assertTrue(torch.allclose(hidden[-1], outputs[:, 0, CONFIG['hidden']:]))

    def test_seeded_sampling(self):
        vocab = ['<unk>', 'alice', 'said', '.']
        lookup = {w: i for i, w in enumerate(vocab)}
        model = LanguageModel('GRU', len(vocab))
        args = (model, ' '.join(['alice']*12), 1., vocab, lookup, 42)
        self.assertEqual(generate(*args), generate(*args))
        self.assertNotIn('<unk>', generate(*args)[0])


if __name__ == '__main__':
    unittest.main()
