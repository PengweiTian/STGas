mean_values = [104.0136177, 114.0342201, 119.91659325]

expand_param = {
    'expand_prob': 0.5,
    'max_expand_ratio': 4.0,
}

batch_samplers = [{
    'sampler': {},
    'max_trials': 1,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'min_jaccard_overlap': 0.1, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'min_jaccard_overlap': 0.3, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'min_jaccard_overlap': 0.5, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'min_jaccard_overlap': 0.7, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'min_jaccard_overlap': 0.9, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
    'sample_constraint': {'max_jaccard_overlap': 1.0, },
    'max_trials': 50,
    'max_sample': 1,
}, ]
