#!/usr/bin/env python3

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from fcc_diverse_sampler import OUTPUT_COLUMNS, colname, generate_ids, read_source


def models(seed):
    return {
        'Polynomial regression': Pipeline([
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('model', LinearRegression()),
        ]),
        'Gradient boosting': MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=250,
                min_samples_leaf=20,
                learning_rate=0.1,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
            n_jobs=-1,
        ),
        'Neural network': Pipeline([
            ('scale', StandardScaler()),
            (
                'model',
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        alpha=0.01,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=500,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]),
    }


def gradient_boosting(seed):
    return models(seed)['Gradient boosting']


def arrays(rows):
    return (
        np.asarray([row[1] for row in rows], dtype=float),
        np.asarray([row[2] for row in rows], dtype=float),
    )


def split(rows, seed, fraction=0.20):
    rng = np.random.default_rng(seed)
    test_positions = set(
        rng.choice(len(rows), round(len(rows) * fraction), replace=False).tolist()
    )
    training = [row for position, row in enumerate(rows) if position not in test_positions]
    test = [row for position, row in enumerate(rows) if position in test_positions]
    return training, test


def select(rows, size, seed):
    if size >= len(rows):
        return rows
    selected_ids, _, _ = generate_ids(rows, size, seed)
    selected_set = set(selected_ids)
    return [row for row in rows if row[0] in selected_set]


def score(y_true, prediction):
    per_target_r2 = r2_score(y_true, prediction, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, prediction, multioutput='raw_values'))
    standard_deviation = np.std(y_true, axis=0)
    nrmse = np.divide(
        rmse,
        standard_deviation,
        out=np.full_like(rmse, np.nan),
        where=standard_deviation > 0,
    )
    return {
        'mean_r2': float(np.mean(per_target_r2)),
        'minimum_target_r2': float(np.min(per_target_r2)),
        'mean_nrmse': float(np.nanmean(nrmse)),
        'per_target_r2': per_target_r2,
        'per_target_nrmse': nrmse,
    }


def fit_and_score(model, training_rows, test_rows):
    x_train, y_train = arrays(training_rows)
    x_test, y_test = arrays(test_rows)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return score(y_test, prediction), model, prediction


def aggregate(records):
    result = {}
    for model_name in records[0]['models']:
        r2_values = np.asarray([record['models'][model_name]['mean_r2'] for record in records])
        nrmse_values = np.asarray([
            record['models'][model_name]['mean_nrmse'] for record in records
        ])
        result[model_name] = {
            'mean_test_r2': float(np.mean(r2_values)),
            'std_test_r2': float(np.std(r2_values)),
            'minimum_seed_test_r2': float(np.min(r2_values)),
            'maximum_seed_test_r2': float(np.max(r2_values)),
            'mean_test_nrmse': float(np.mean(nrmse_values)),
        }
    return result


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--output', type=Path, default=directory / 'output' / 'reliability.json')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    data, header, _, _ = read_source(args.workbook)
    target_names = [header[column].strip() for column in OUTPUT_COLUMNS]
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    repeated = []
    for seed in range(args.seed, args.seed + 10):
        training_pool, test_rows = split(data, seed)
        training_rows = select(training_pool, 1000, seed)
        seed_result = {'seed': seed, 'models': {}}
        for name, model in models(seed).items():
            values, _, _ = fit_and_score(model, training_rows, test_rows)
            seed_result['models'][name] = {
                'mean_r2': values['mean_r2'],
                'minimum_target_r2': values['minimum_target_r2'],
                'mean_nrmse': values['mean_nrmse'],
            }
        repeated.append(seed_result)
        print(f'repeated split {seed} complete', flush=True)

    base_training_pool, base_test = split(data, args.seed)
    learning_curve = []
    for size in (250, 500, 1000, 2000, 5000, len(base_training_pool)):
        training_rows = select(base_training_pool, size, args.seed)
        values, _, _ = fit_and_score(
            gradient_boosting(args.seed),
            training_rows,
            base_test,
        )
        learning_curve.append({
            'training_rows': len(training_rows),
            'mean_test_r2': values['mean_r2'],
            'minimum_target_r2': values['minimum_target_r2'],
            'mean_test_nrmse': values['mean_nrmse'],
        })
        print(f'learning curve {len(training_rows)} complete', flush=True)

    level_tests = []
    for input_position in range(7):
        levels = sorted({row[1][input_position] for row in data})
        middle_level = levels[len(levels) // 2]
        for test_type, level in (
            ('interpolation', middle_level),
            ('minimum extrapolation', levels[0]),
            ('maximum extrapolation', levels[-1]),
        ):
            training_pool = [
                row for row in data if row[1][input_position] != level
            ]
            test_rows = [
                row for row in data if row[1][input_position] == level
            ]
            training_rows = select(training_pool, 1000, args.seed + input_position)
            for model_name, model in models(args.seed + input_position).items():
                values, _, _ = fit_and_score(model, training_rows, test_rows)
                level_tests.append({
                    'model': model_name,
                    'input': header[input_position],
                    'excel_column': colname(input_position),
                    'test_type': test_type,
                    'held_out_level': level,
                    'test_rows': len(test_rows),
                    'mean_r2': values['mean_r2'],
                    'minimum_target_r2': values['minimum_target_r2'],
                    'mean_nrmse': values['mean_nrmse'],
                })
            print(f'{colname(input_position)} {test_type} complete', flush=True)

    final_training = select(base_training_pool, 1000, args.seed)
    final_values, final_model, test_prediction = fit_and_score(
        gradient_boosting(args.seed),
        final_training,
        base_test,
    )
    _, y_test = arrays(base_test)

    final_ids = {row[0] for row in final_training}
    calibration_pool = [row for row in base_training_pool if row[0] not in final_ids]
    calibration_rng = np.random.default_rng(args.seed)
    calibration_positions = calibration_rng.choice(
        len(calibration_pool),
        5000,
        replace=False,
    )
    calibration_rows = [calibration_pool[position] for position in calibration_positions]
    x_calibration, y_calibration = arrays(calibration_rows)
    calibration_residuals = np.abs(y_calibration - final_model.predict(x_calibration))
    quantile_level = min(
        1.0,
        math.ceil((len(calibration_rows) + 1) * 0.90) / len(calibration_rows),
    )
    interval_half_width = np.quantile(
        calibration_residuals,
        quantile_level,
        axis=0,
        method='higher',
    )
    interval_coverage = np.mean(
        np.abs(y_test - test_prediction) <= interval_half_width,
        axis=0,
    )

    yield_prediction = test_prediction[:, :8]
    yield_actual = y_test[:, :8]
    physical_checks = {
        'negative_prediction_count': int(np.sum(test_prediction < 0)),
        'yield_prediction_outside_0_to_100_count': int(
            np.sum((yield_prediction < 0) | (yield_prediction > 100))
        ),
        'conversion_prediction_outside_0_to_100_count': int(
            np.sum((test_prediction[:, 8] < 0) | (test_prediction[:, 8] > 100))
        ),
        'actual_yield_sum_mean': float(np.mean(np.sum(yield_actual, axis=1))),
        'predicted_yield_sum_mean': float(np.mean(np.sum(yield_prediction, axis=1))),
        'yield_sum_mean_absolute_error': float(np.mean(np.abs(
            np.sum(yield_prediction, axis=1) - np.sum(yield_actual, axis=1)
        ))),
    }

    per_target = []
    for position, name in enumerate(target_names):
        per_target.append({
            'target': name,
            'excel_column': colname(OUTPUT_COLUMNS[position]),
            'test_r2': float(final_values['per_target_r2'][position]),
            'test_nrmse': float(final_values['per_target_nrmse'][position]),
            'interval_90_coverage': float(interval_coverage[position]),
            'interval_90_half_width': float(interval_half_width[position]),
        })

    level_summary = {}
    for model_name in models(args.seed):
        interpolation = [
            row for row in level_tests
            if row['model'] == model_name and row['test_type'] == 'interpolation'
        ]
        extrapolation = [
            row for row in level_tests
            if row['model'] == model_name and row['test_type'] != 'interpolation'
        ]
        level_summary[model_name] = {
            'mean_interpolation_r2': float(np.mean([
                row['mean_r2'] for row in interpolation
            ])),
            'worst_interpolation_r2': float(np.min([
                row['mean_r2'] for row in interpolation
            ])),
            'mean_extrapolation_r2': float(np.mean([
                row['mean_r2'] for row in extrapolation
            ])),
            'worst_extrapolation_r2': float(np.min([
                row['mean_r2'] for row in extrapolation
            ])),
        }
    report = {
        'schema': {
            'inputs': 'A:G',
            'targets': 'T:AA, AC:AI, AO',
            'source_rows': len(data),
            'selected_training_rows': 1000,
        },
        'repeated_split_summary': aggregate(repeated),
        'repeated_split_details': repeated,
        'gradient_boosting_learning_curve': learning_curve,
        'held_out_level_tests': level_tests,
        'held_out_level_summary': level_summary,
        'physical_checks': physical_checks,
        'conformal_intervals': {
            'nominal_coverage': 0.90,
            'calibration_rows': len(calibration_rows),
            'mean_observed_coverage': float(np.mean(interval_coverage)),
            'minimum_target_coverage': float(np.min(interval_coverage)),
        },
        'per_target_final_gradient_boosting': per_target,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
