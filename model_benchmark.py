#!/usr/bin/env python3

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from fcc_diverse_sampler import (
    BLUE_COLUMNS,
    OUTPUT_COLUMNS,
    REQUIRED_COLUMNS,
    colname,
    generate_ids,
    read_selected_full_rows,
    read_source,
    write_csv,
)


def model_searches(seed):
    return {
        'Linear regression': (
            Pipeline([
                ('scale', StandardScaler()),
                ('model', LinearRegression()),
            ]),
            {},
        ),
        'Polynomial regression': (
            Pipeline([
                ('scale', StandardScaler()),
                ('poly', PolynomialFeatures(include_bias=False)),
                ('model', LinearRegression()),
            ]),
            {'poly__degree': [2, 3]},
        ),
        'Ridge regression': (
            Pipeline([
                ('scale', StandardScaler()),
                ('model', Ridge()),
            ]),
            {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        'Decision tree': (
            DecisionTreeRegressor(random_state=seed),
            {
                'max_depth': [6, 10, 14, None],
                'min_samples_leaf': [2, 5, 10, 20],
            },
        ),
        'Random forest': (
            RandomForestRegressor(
                n_estimators=250,
                random_state=seed,
                n_jobs=1,
            ),
            {
                'max_depth': [12, 18, None],
                'min_samples_leaf': [2, 5],
                'max_features': [0.7, 1.0],
            },
        ),
        'Extra trees': (
            ExtraTreesRegressor(
                n_estimators=250,
                random_state=seed,
                n_jobs=1,
            ),
            {
                'max_depth': [12, 18, None],
                'min_samples_leaf': [2, 5],
                'max_features': [0.7, 1.0],
            },
        ),
        'Gradient boosting': (
            MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=250,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
                n_jobs=1,
            ),
            {
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__max_leaf_nodes': [15, 31],
                'estimator__l2_regularization': [1.0, 5.0],
            },
        ),
        'Neural network': (
            Pipeline([
                ('scale', StandardScaler()),
                (
                    'model',
                    TransformedTargetRegressor(
                        regressor=MLPRegressor(
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
            {
                'model__regressor__hidden_layer_sizes': [(32, 32), (64, 64)],
                'model__regressor__alpha': [0.0001, 0.001, 0.01],
            },
        ),
    }


def metrics(y_true, prediction):
    r2 = r2_score(y_true, prediction, multioutput='raw_values')
    mae = mean_absolute_error(y_true, prediction, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, prediction, multioutput='raw_values'))
    scale = np.std(y_true, axis=0)
    nrmse = np.divide(rmse, scale, out=np.full_like(rmse, np.nan), where=scale > 0)
    return r2, mae, rmse, nrmse


def write_final_dataset(workbook, data, selected_ids, header, units, data_dir, seed):
    records = read_selected_full_rows(workbook, selected_ids)
    write_csv(
        data_dir / 'FCC_Final_1000.csv',
        [[header[i] for i in REQUIRED_COLUMNS],
         [units.get(i, '') for i in REQUIRED_COLUMNS]] +
        [[records[row].get(i, '') for i in REQUIRED_COLUMNS] for row in selected_ids],
    )
    (data_dir / 'selected_excel_rows.txt').write_text(
        '\n'.join(map(str, selected_ids)) + '\n',
        encoding='utf-8',
    )
    selected_set = set(selected_ids)
    selected = [row for row in data if row[0] in selected_set]
    source_levels = sum(len({row[1][column] for row in data}) for column in range(7))
    selected_levels = sum(len({row[1][column] for row in selected}) for column in range(7))

    def pairs(rows):
        return {
            (left, row[1][left], right, row[1][right])
            for row in rows
            for left in range(7)
            for right in range(left + 1, 7)
        }

    ranges_preserved = sum(
        min(row[2][column] for row in data) == min(row[2][column] for row in selected)
        and max(row[2][column] for row in data) == max(row[2][column] for row in selected)
        for column in range(len(OUTPUT_COLUMNS))
    )
    report = {
        'mode': 'split-first-regenerate',
        'seed': seed,
        'training_pool_rows': len(data),
        'selected_rows': len(selected),
        'test_reserved_before_selection': True,
        'input_levels_retained': f'{selected_levels}/{source_levels}',
        'pairwise_combinations_retained': f'{len(pairs(selected))}/{len(pairs(data))}',
        'output_ranges_preserved': f'{ranges_preserved}/{len(OUTPUT_COLUMNS)}',
        'excluded_blue_columns': [colname(column) for column in BLUE_COLUMNS],
    }
    return report


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--test-fraction', type=float, default=0.20)
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--data-dir', type=Path, default=directory / 'data')
    parser.add_argument('--output-dir', type=Path, default=directory / 'output')
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error('--test-fraction must be between 0 and 1')

    data, header, units, _ = read_source(args.workbook)
    row_ids = np.asarray([row[0] for row in data])
    x = np.asarray([row[1] for row in data], dtype=float)
    y = np.asarray([row[2] for row in data], dtype=float)

    rng = np.random.default_rng(args.seed)
    test_size = round(len(data) * args.test_fraction)
    test_indices = np.sort(rng.choice(len(data), test_size, replace=False))
    test_mask = np.zeros(len(data), dtype=bool)
    test_mask[test_indices] = True
    candidate_data = [row for position, row in enumerate(data) if not test_mask[position]]

    selected_ids, _, _ = generate_ids(candidate_data, args.sample_size, args.seed)
    selected_id_set = set(selected_ids)
    selected_mask = np.asarray([row_id in selected_id_set for row_id in row_ids])
    if np.any(selected_mask & test_mask):
        raise RuntimeError('Selected training rows overlap the untouched test set')

    x_train = x[selected_mask]
    y_train = y[selected_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]
    test_row_ids = row_ids[test_mask]

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    neighbor_model = NearestNeighbors(n_neighbors=1).fit((x_train - mean) / scale)
    test_distances = neighbor_model.kneighbors(
        (x_test - mean) / scale,
        return_distance=True,
    )[0][:, 0]
    stress_size = max(1, len(x_test) // 10)
    stress_indices = np.argsort(test_distances)[-stress_size:]
    example_position = int(rng.integers(len(x_test)))

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_report = write_final_dataset(
        args.workbook,
        candidate_data,
        selected_ids,
        header,
        units,
        args.data_dir,
        args.seed,
    )
    np.savetxt(args.data_dir / 'untouched_test_excel_rows.txt', test_row_ids, fmt='%d')
    (args.output_dir / 'audit_summary.json').write_text(
        json.dumps(audit_report, indent=2) + '\n',
        encoding='utf-8',
    )

    folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    summary_rows = []
    target_rows = []
    best_parameters = {}
    example_predictions = {}
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    for name, (estimator, parameter_grid) in model_searches(args.seed).items():
        started = time.perf_counter()
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring='r2',
            cv=folds,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        prediction_train = search.predict(x_train)
        prediction_test = search.predict(x_test)
        train_values = metrics(y_train, prediction_train)
        test_values = metrics(y_test, prediction_test)
        stress_values = metrics(
            y_test[stress_indices],
            prediction_test[stress_indices],
        )
        best_index = search.best_index_
        cv_train_r2 = float(search.cv_results_['mean_train_score'][best_index])
        cv_validation_r2 = float(search.best_score_)
        elapsed = time.perf_counter() - started

        summary_rows.append([
            name,
            cv_train_r2,
            cv_validation_r2,
            cv_train_r2 - cv_validation_r2,
            float(np.mean(train_values[0])),
            float(np.mean(test_values[0])),
            float(np.mean(stress_values[0])),
            float(np.nanmean(test_values[3])),
            float(np.nanmean(stress_values[3])),
            elapsed,
        ])
        best_parameters[name] = search.best_params_
        example_predictions[name] = prediction_test[example_position]

        for position, column in enumerate(OUTPUT_COLUMNS):
            target_rows.append([
                header[column],
                colname(column),
                name,
                train_values[0][position],
                test_values[0][position],
                stress_values[0][position],
                test_values[1][position],
                test_values[2][position],
                test_values[3][position],
            ])

        print(
            f'{name}: CV={cv_validation_r2:.5f}, '
            f'test={np.mean(test_values[0]):.5f}, '
            f'stress={np.mean(stress_values[0]):.5f}',
            flush=True,
        )

    with open(args.output_dir / 'model_summary.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Model',
            'CV train R2',
            'CV validation R2',
            'CV overfit gap',
            'Final train R2',
            'Untouched test R2',
            'Distant test R2',
            'Untouched test NRMSE',
            'Distant test NRMSE',
            'Search and evaluation seconds',
        ])
        writer.writerows(summary_rows)

    with open(args.output_dir / 'per_target_scores.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Target',
            'Excel column',
            'Model',
            'Final train R2',
            'Untouched test R2',
            'Distant test R2',
            'Untouched test MAE',
            'Untouched test RMSE',
            'Untouched test NRMSE',
        ])
        writer.writerows(target_rows)

    with open(args.output_dir / 'example_unseen_row.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Excel row',
            'Target',
            'Actual',
            *example_predictions.keys(),
        ])
        for position, column in enumerate(OUTPUT_COLUMNS):
            writer.writerow([
                int(test_row_ids[example_position]),
                header[column],
                y_test[example_position, position],
                *(value[position] for value in example_predictions.values()),
            ])

    (args.output_dir / 'best_parameters.json').write_text(
        json.dumps(best_parameters, indent=2, default=str) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
