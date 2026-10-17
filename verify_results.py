"""Check saved results against histories and generated samples after a full run."""
from pathlib import Path
import math
import pandas as pd

root = Path(__file__).resolve().parent
results = pd.read_csv(root/'model_comparison_results.csv')
history = pd.read_csv(root/'training_history.csv')
samples = pd.read_csv(root/'generated_text_samples.csv')
assert set(results.model) == {'RNN', 'BiRNN', 'LSTM', 'GRU'}
assert len(samples) == 108
assert len(history) == 60
assert samples.groupby(['model', 'seed_id', 'temperature']).size().eq(3).all()
assert samples[['unique_token_ratio', 'repeated_bigram_ratio']].ge(0).all().all()
assert samples[['unique_token_ratio', 'repeated_bigram_ratio']].le(1).all().all()
for row in results.itertuples():
    sub = history[history.model == row.model]
    best = sub.loc[sub.validation_loss.idxmin()]
    assert row.best_epoch == best.epoch
    assert math.isclose(row.best_validation_loss, best.validation_loss)
    assert math.isclose(math.exp(row.test_loss), row.test_perplexity)
    assert 0 <= row.test_accuracy <= 1 and row.training_seconds > 0
for name in ['loss_curves.png', 'temperature_diversity.png']:
    assert (root/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
print('Verified four models, 60 training epochs, 108 samples, checkpoint selection, perplexities and plots.')
