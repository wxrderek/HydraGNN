###########################################################################################
# Composite loss: weighted extra terms added to a model's own loss
#
# Enabled from the config with
#
#   "Training": {"loss": {"enabled": true, "provider": "composite",
#                         "terms": [{"name": "<term>", "weight": <float>}, ...],
#                         "include_in_evaluation": true,
#                         "<options group>": {...}}}
#
# With include_in_evaluation false, the terms are still computed and reported in validation and
# test (the model in eval mode), but left out of the returned loss, so the learning rate
# scheduler and the best checkpoint follow the model's own loss only. The default is true.
#
# Each registry row below is (view builder, term function, options group). The view builder
# turns the model output and the batch into the objects the term needs; terms that share a
# builder share one evaluation of it per batch. To add a term, write
#
#   view builder: builder(pred, data, model, options) -> views
#   term function: term(views) -> scalar tensor
#
# and add one row to _TERMS. Registered density matrix terms: electron_count, overlap_metric,
# idempotency, frobenius_total, frobenius_delta (hydragnn/domain_losses/density_matrix.py).
#
# With "density_matrix": {"trace_normalization": true}, a dense head's prediction is scaled to
# the number of electrons before the model's own loss and before the terms (transform_prediction).
#
# The training loop passes the batch to a model's loss only if that loss method has a data
# argument, as Base.loss and SchurDecoderModel.loss do (train_validate_test._loss_data_kwargs).
###########################################################################################

from hydragnn.domain_losses.density_matrix import (
    density_matrix_views,
    trace_normalize_dense_rows,
    electron_count_term,
    frobenius_delta_term,
    frobenius_total_term,
    idempotency_term,
    overlap_metric_term,
)

_TERMS = {
    'electron_count': (density_matrix_views, electron_count_term, 'density_matrix'),
    'overlap_metric': (density_matrix_views, overlap_metric_term, 'density_matrix'),
    'idempotency': (density_matrix_views, idempotency_term, 'density_matrix'),
    'frobenius_total': (density_matrix_views, frobenius_total_term, 'density_matrix'),
    'frobenius_delta': (density_matrix_views, frobenius_delta_term, 'density_matrix'),
}


class CompositeLoss:
    '''weighted sum of registered extra loss terms, with a report per term'''

    # A plain object rather than a torch Module: it holds no parameters or buffers, and as an
    # attribute of the model it changes no state dict key, so existing checkpoints still load

    def __init__(self, config):
        terms = config.get('terms', [])
        names = [term['name'] for term in terms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f'duplicate composite loss terms: {duplicates}')
        unknown = sorted(set(names) - set(_TERMS))
        if unknown:
            raise ValueError(f'unknown composite loss terms: {unknown}; registered: {sorted(_TERMS)}')
        # terms with weight 0 are kept out of the computation entirely
        self.weights = {term['name']: float(term.get('weight', 0.0)) for term in terms}
        self.active = [name for name in names if self.weights[name] != 0.0]
        self.options = {group: dict(config.get(group, {})) for _, _, group in _TERMS.values()}
        self.include_in_evaluation = bool(config.get('include_in_evaluation', True))

    def transform_prediction(self, pred, data, model):
        '''the prediction the model's own loss compares with the target; unchanged unless dense trace normalization is on'''
        options = self.options['density_matrix']
        if not options.get('trace_normalization', False):
            return pred
        if isinstance(pred, dict):
            raise ValueError(
                'density_matrix.trace_normalization is for the dense head; for the Schur decoder set '
                'SchurDecoder.trace_normalization instead'
            )
        num_graphs = int(data.density_matrix_dim.reshape(-1).shape[0])
        rows, _ = trace_normalize_dense_rows(
            pred[0].reshape(num_graphs, -1), data, options['dense_target']
        ) # (num_graphs, padded row length)
        return [rows.reshape(pred[0].shape)] + list(pred[1:])

    def __call__(self, pred, data, model):
        '''the weighted extra loss to add, or None, and a report in the format of hydragnn/loss_reporting.py'''
        total = None
        report = {}
        views = {}
        for name in self.active:
            builder, term, group = _TERMS[name]
            if builder not in views:
                views[builder] = builder(pred, data, model, self.options[group])
            raw = term(views[builder])
            weighted = self.weights[name] * raw
            total = weighted if total is None else total + weighted
            report[f'domain.{name}'] = {
                'raw': raw.detach(),
                'weight': self.weights[name],
                'weighted': weighted.detach(),
            }
        # in eval mode the terms are reported but not added when include_in_evaluation is false
        if not getattr(model, 'training', True) and not self.include_in_evaluation:
            return None, report
        return total, report
