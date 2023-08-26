import warnings
import io
import sympy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from math import sin, cos, tan, asin, acos, atan, log, log2, log10, exp
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from scipy.optimize import linprog

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class InputData:
    dataset_file: str
    x_name: str
    u_name: str
    y_name: str
    x_amount: int
    u_amount: int
    y_amount: int


class Models:
    """Create models based on input table .xlsx, where parameters are X1, X2...U1, U2...Y1,Y2..."""

    def __init__(self, params: InputData):
        self.params = params
        self.data = pd.read_excel(self.params.dataset_file)
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.df = self.data.copy()
        self.x_cols = np.array([f'{self.params.x_name}{i}' for i in range(1, self.params.x_amount + 1)])
        self.u_cols = np.array([f'{self.params.u_name}{i}' for i in range(1, self.params.u_amount + 1)])
        self.y_cols = np.array([f'{self.params.y_name}{i}' for i in range(1, self.params.y_amount + 1)])
        self.mins = None
        self.maxes = None
        self.means = None
        self.stds = None
        self.selected_features = None
        self.intercept = None
        self.models = None
        self.bounds = None
        self.x_params = None

    def fill_null(self, method='means'):
        if method == 'means':
            self.df = self.df.fillna(self.df.mean())

    def standardize(self):
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.df = (self.df - self.means) / self.stds

    def normalize(self):
        self.mins = self.df.min()
        self.maxes = self.df.max()
        self.df = ((self.df - self.mins) / (self.maxes - self.mins)) + 1

    def get_real_from_norm(self, param: str, value: float):
        standardized = (value - 1) * (self.maxes[param] - self.mins[param]) + self.mins[param]
        real = standardized * self.stds[param] + self.means[param]
        return real

    def get_norm_from_real(self, param: str, value: float):
        standardized = (value - self.means[param]) / self.stds[param]
        normalized = (standardized - self.mins[param]) / (self.maxes[param] - self.mins[param]) + 1
        return normalized

    def remove_excess(self, lower, upper):
        return self.df.loc[:, ~(self.df < lower).any(axis=0) & ~(self.df > upper).any(axis=0)]

    def extend_data(self, do_reciprocal=True, do_trig=True):
        new_cols = np.array([])
        keys = np.array([])
        cols = 0

        # Multiply by U
        for x_col in self.x_cols:
            for u_col in self.u_cols:
                keys = np.append(keys, f'{u_col}*{x_col}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        # Square
        for x_col in self.x_cols:
            new_x_key = f'{x_col}**2'
            keys = np.append(keys, new_x_key)
            new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: x ** 2))
            cols += 1
            for u_col in self.u_cols:
                keys = np.append(keys, f'{u_col}*{new_x_key}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        # Cube
        for x_col in self.x_cols:
            new_x_key = f'{x_col}**3'
            keys = np.append(keys, new_x_key)
            new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: x ** 3))
            cols += 1
            for u_col in self.u_cols:
                keys = np.append(keys, f'{u_col}*{new_x_key}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        # Square root
        for x_col in self.x_cols:
            if (self.df[x_col] < 0).any():
                continue
            new_x_key = f'{x_col}**(1/2)'
            keys = np.append(keys, new_x_key)
            new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: x ** (1 / 2)))
            cols += 1
            for u_col in self.u_cols:
                keys = np.append(keys, f'{u_col}*{new_x_key}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        # Cube root
        for x_col in self.x_cols:
            new_x_key = f'{x_col}**(1/3)'
            keys = np.append(keys, new_x_key)
            new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: x ** (1 / 3)))
            cols += 1
            for u_col in self.u_cols:
                keys = np.append(keys, f'{u_col}*{new_x_key}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        if do_trig:
            # # Exponential
            # for x_col in self.x_cols:
            #     new_x_key = f'exp({x_col})'
            #     keys = np.append(keys, new_x_key)
            #     new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: exp(x)))
            #     cols += 1
            #     for u_col in self.u_cols:
            #         keys = np.append(keys, f'{u_col}*{new_x_key}')
            #         new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
            #         cols += 1

            # Log (base: e)
            for x_col in self.x_cols:
                if (self.df[x_col] <= 0).any():
                    continue
                new_x_key = f'log({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: log(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # # Log (base: 2)
            # for x_col in self.x_cols:
            #     if (self.df[x_col] <= 0).any():
            #         continue
            #     new_x_key = f'log2({x_col})'
            #     keys = np.append(keys, new_x_key)
            #     new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: log2(x)))
            #     cols += 1
            #     for u_col in self.u_cols:
            #         keys = np.append(keys, f'{u_col}*{new_x_key}')
            #         new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
            #         cols += 1
            #
            # # Log (base: 10)
            # for x_col in self.x_cols:
            #     if (self.df[x_col] <= 0).any():
            #         continue
            #     new_x_key = f'log10({x_col})'
            #     keys = np.append(keys, new_x_key)
            #     new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: log10(x)))
            #     cols += 1
            #     for u_col in self.u_cols:
            #         keys = np.append(keys, f'{u_col}*{new_x_key}')
            #         new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
            #         cols += 1

            # Sine
            for x_col in self.x_cols:
                new_x_key = f'sin({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: sin(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # Cosine
            for x_col in self.x_cols:
                new_x_key = f'cos({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: cos(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # Tangent
            for x_col in self.x_cols:
                new_x_key = f'tan({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: tan(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # ArcSine
            for x_col in self.x_cols:
                if not ((self.df[x_col] >= -1).all() and (self.df[x_col] <= 1).all()):
                    continue
                new_x_key = f'asin({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: asin(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # ArcCosine
            for x_col in self.x_cols:
                if not ((self.df[x_col] >= -1).all() and (self.df[x_col] <= 1).all()):
                    continue
                new_x_key = f'acos({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: acos(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

            # ArcTangent
            for x_col in self.x_cols:
                if not ((self.df[x_col] >= -1).all() and (self.df[x_col] <= 1).all()):
                    continue
                new_x_key = f'atan({x_col})'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: atan(x)))
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                    cols += 1

        # Multiply the input parameters together
        for i, x_col in enumerate(self.x_cols):
            for x_col2 in self.x_cols[i + 1:]:
                new_x_key = f'{x_col}*{x_col2}'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col] * self.df[x_col2])
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] * self.df[x_col2] * self.df[u_col])
                    cols += 1

        # Divide the input parameters together
        for i, x_col in enumerate(self.x_cols):
            for x_col2 in self.x_cols[i + 1:]:
                if (self.df[x_col2] == 0).any():
                    continue
                if x_col == x_col2:
                    continue
                new_x_key = f'{x_col}/{x_col2}'
                keys = np.append(keys, new_x_key)
                new_cols = np.append(new_cols, self.df[x_col] / self.df[x_col2])
                cols += 1
                for u_col in self.u_cols:
                    keys = np.append(keys, f'{u_col}*{new_x_key}')
                    new_cols = np.append(new_cols, self.df[x_col] / self.df[x_col2] * self.df[u_col])
                    cols += 1

        # Add new data to dataframe
        new_cols = new_cols.astype(float)
        new_cols = np.reshape(new_cols, (cols, len(self.df.index)))
        new_cols = np.transpose(new_cols)
        new_data = pd.DataFrame(new_cols, columns=keys)
        self.df = pd.concat([self.df, new_data], axis=1)

        if not do_reciprocal:
            return
        # Do reciprocal to all added columns
        keys = np.array([elem for elem in keys if 'U' not in elem])
        new_cols = np.array([])
        rec_keys = np.array([])
        cols = 0
        for x_col in keys:
            if (self.df[x_col] == 0).any():
                continue
            new_x_key = f'1/{x_col}'
            rec_keys = np.append(rec_keys, new_x_key)
            new_cols = np.append(new_cols, self.df[x_col].apply(lambda x: 1 / x))
            cols += 1
            for u_col in self.u_cols:
                rec_keys = np.append(rec_keys, f'{u_col}*{new_x_key}')
                new_cols = np.append(new_cols, self.df[x_col] * self.df[u_col])
                cols += 1

        new_cols = np.reshape(new_cols, (cols, len(self.df.index)))
        new_cols = np.transpose(new_cols)
        new_data = pd.DataFrame(new_cols, columns=rec_keys)
        self.df = pd.concat([self.df, new_data], axis=1)

    # def build_models(self, amount_of_cols=12, filter_u=False, alpha_in=None, show_results=False, show_alpha=False):
    #     self.selected_features = {}
    #     self.intercept = {}
    #     if alpha_in is None:
    #         alpha_in = [0.01, 0.01]
    #     if len(alpha_in) != 2:
    #         return
    #     x = self.df.loc[:, ~self.df.columns.isin(self.y_cols)]
    #     if filter_u:
    #         x = x.filter(regex='U')
    #     self.selected_features = {}
    #     print('---R coefficient (lasso) ---')
    #     for y_col in self.y_cols:
    #         alpha = alpha_in.copy()
    #         y_out = self.df.loc[:, y_col]
    #         while True:
    #             model = Lasso(alpha=alpha[0])
    #             model.fit(x, y_out)
    #             if sum(model.coef_ != 0) > amount_of_cols:
    #                 if show_alpha:
    #                     print(sum(model.coef_ != 0), alpha[0], end=' | ')
    #                 alpha[0] += alpha[1]
    #                 continue
    #
    #             # Save most valuable parameters with coefficients
    #             coefficients = [i for i in model.coef_.tolist() if i != 0]
    #             columns = x.columns[model.coef_ != 0].tolist()
    #             features = {y_col: [coefficients, columns]}
    #             intercept = {y_col: float(model.intercept_)}
    #             self.selected_features.update(features)
    #             self.intercept.update(intercept)
    #             # Show results of the model
    #             if not show_results:
    #                 break
    #             r = model.score(x, y_out) ** (1 / 2)
    #             print(f'\nR for {y_col}: \033[1m{round(r, 3)}\033[0m')
    #             print('Amount of selected features:', sum(model.coef_ != 0))
    #             print('Alpha:', alpha[0])
    #             # print('Selected features with coefficients:', features)
    #             break

    def build_models(self, amount_of_cols=12, filter_u=False):
        self.selected_features = {}
        self.intercept = {}
        print('---R coefficient (RFE)---')
        x = self.df.loc[:, ~self.df.columns.isin(self.y_cols)]
        if filter_u:
            x = x.filter(regex=self.params.u_name)
        model = LinearRegression()
        for y_col in self.y_cols:
            y = self.df.loc[:, y_col]
            selector = RFE(model, n_features_to_select=amount_of_cols)
            selector.fit(x, y)
            columns = x.columns[selector.support_].tolist()
            coefficients = selector.estimator_.coef_.tolist()
            features = {y_col: [coefficients, columns]}
            intercept = {y_col: float(selector.estimator_.intercept_)}
            self.selected_features.update(features)
            self.intercept.update(intercept)
            r = selector.score(x, y) ** (1 / 2)
            print(f'R for {y_col}: \033[1m{round(r, 3)}\033[0m')

    def show_models(self, output=False):
        self.models = {}
        x_means = self.df.loc[:, self.x_cols].mean()
        symbols = {}
        for col in np.concatenate((self.x_cols, self.u_cols)):
            symbols[col] = sympy.symbols(col)
        if output:
            print('\n---Final Models---')
        for y_col in self.y_cols:
            io_formula = io.StringIO()
            coefficients = self.selected_features.get(y_col)[0]
            features = self.selected_features.get(y_col)[1]
            for cf, cols in zip(coefficients, features):
                io_formula.write(f'{cf}*{cols} + ')
            io_formula.write(f'{self.intercept.get(y_col)}')
            str_formula = io_formula.getvalue()
            model = {y_col: str_formula}
            self.models.update(model)
            if output:
                print(f'Features for {y_col}:', ', '.join(features).replace('**', '^'))
                print(f'Model: {y_col} = {model.get(y_col)}'.replace('**', '^'))
                formula = sympy.sympify(str_formula).subs([(x, x_val) for x, x_val in x_means.items()])
                print(f'Means: {y_col} = {formula}\n')
                print()

    def do_optimization(self, func: str, maximize=False):
        # Define symbols X and U
        symbols_x = {}
        symbols_u = {}
        for x in self.x_cols:
            symbols_x[x] = sympy.symbols(x)
        for u in self.u_cols:
            symbols_u[u] = sympy.symbols(u)

        # Build models from input X parameters
        for key, value in self.bounds.items():
            self.bounds[key][0] = self.get_norm_from_real(key, value[0])
            self.bounds[key][1] = self.get_norm_from_real(key, value[-1])
        for x, x_val in self.x_params.items():
            self.x_params[x] = self.get_norm_from_real(x, x_val)

        models = self.models.copy()
        for y, model in models.items():
            models[y] = sympy.sympify(model).subs([(x, x_val) for x, x_val in self.x_params.items()])

        if maximize:
            models[func] = -models[func]
        # Prepare parameters for linear programming optimization
        # Define c
        c = [models.get(func).coeff(u) for u in symbols_u.values()]

        # Define A
        A = []
        for y, model in models.items():
            if y == func:
                continue
            u_params = [model.coeff(u) for u in symbols_u.values()]
            A.append(u_params)
            A.append(list(map(lambda a: -a, u_params)))

        # Define b
        b = []
        for y, model in models.items():
            if y == func:
                continue
            expr = sympy.sympify(models.get(y))
            intercept, _ = expr.as_coeff_Add(rational=False)
            upper_limit = self.bounds.get(y)[1] - intercept
            lower_limit = intercept - self.bounds.get(y)[0]
            b.append(upper_limit)
            b.append(lower_limit)

        # Define bounds
        u_bounds = [(self.bounds.get(i)[0], self.bounds.get(i)[1]) for i in self.u_cols]

        # Do optimization
        result = linprog(c, A_ub=A, b_ub=b, bounds=u_bounds, method='highs')

        if not result.success:
            return None, None, result.message

        # Output results
        u_found = {u: value for u, value in zip(self.u_cols, result.x)}

        res = {}
        for y, model in models.items():
            temp = sympy.sympify(model).subs([(x, x_val) for x, x_val in zip(symbols_u.values(), u_found.values())])
            if maximize and y == func:
                temp = -temp
            real = self.get_real_from_norm(y, temp)
            res.update({y: real if real > 0 else 0})

        for u, u_value in zip(self.u_cols, u_found.values()):
            real = self.get_real_from_norm(u, u_value)
            res.update({u: real if real > 0 else 0})

        return res, models, result.message

    def do_optimization_real(self, func: str, maximize=False):
        # Define symbols X and U
        symbols_x = {}
        symbols_u = {}
        for x in self.x_cols:
            symbols_x[x] = sympy.symbols(x)
        for u in self.u_cols:
            symbols_u[u] = sympy.symbols(u)

        # Build models from input X parameters
        models = self.models.copy()
        for y, model in models.items():
            models[y] = sympy.sympify(model).subs([(x, x_val) for x, x_val in self.x_params.items()])

        if maximize:
            models[func] = -models[func]
        # Prepare parameters for linear programming optimization
        # Define c
        c = [models.get(func).coeff(u) for u in symbols_u.values()]

        # Define A
        A = []
        for y, model in models.items():
            if y == func:
                continue
            u_params = [model.coeff(u) for u in symbols_u.values()]
            A.append(u_params)
            A.append(list(map(lambda a: -a, u_params)))

        # Define b
        b = []
        for y, model in models.items():
            if y == func:
                continue
            expr = sympy.sympify(models.get(y))
            intercept, _ = expr.as_coeff_Add(rational=False)
            upper_limit = self.bounds.get(y)[1] - intercept
            lower_limit = intercept - self.bounds.get(y)[0]
            b.append(upper_limit)
            b.append(lower_limit)

        # Define bounds
        u_bounds = [(self.bounds.get(i)[0], self.bounds.get(i)[1]) for i in self.u_cols]

        # Do optimization
        result = linprog(c, A_ub=A, b_ub=b, bounds=u_bounds, method='highs')

        if not result.success:
            return None, None, result.message

        # Output results
        u_found = {u: value for u, value in zip(self.u_cols, result.x)}

        res = {}
        for y, model in models.items():
            out = sympy.sympify(model).subs([(x, x_val) for x, x_val in zip(symbols_u.values(), u_found.values())])
            if y == func and maximize:
                out = -out
            res.update({y: out})

        for u, u_value in zip(self.u_cols, u_found.values()):
            res.update({u: u_value})

        return res, models, result.message


def main():
    # # lab_1_1
    # parameters = InputData(dataset_file='data/lab1_1.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=32, u_amount=0, y_amount=5)

    # # AC_1
    # parameters = InputData(dataset_file='data/AC_1.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=6, u_amount=3, y_amount=5)

    # # CH_2
    # parameters = InputData(dataset_file='data/CH_2.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=6, u_amount=4, y_amount=5)

    # # pvl
    # parameters = InputData(dataset_file='data/pvl.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=4, u_amount=2, y_amount=4)

    # # vlr
    # parameters = InputData(dataset_file='data/vlr.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=5, u_amount=2, y_amount=4)

    # Ania
    parameters = InputData(dataset_file='data/ania.xlsx', x_name='X', u_name='U', y_name='Y',
                           x_amount=11, u_amount=2, y_amount=5)

    # # Michael
    # parameters = InputData(dataset_file='data/michael.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=10, u_amount=2, y_amount=7)

    # # Liza
    # parameters = InputData(dataset_file='data/liza.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=7, u_amount=2, y_amount=3)

    # # vika
    # parameters = InputData(dataset_file='data/vika.xlsx', x_name='x', u_name='u', y_name='y',
    #                        x_amount=13, u_amount=3, y_amount=5)

    # # R
    # parameters = InputData(dataset_file='data/r.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=5, u_amount=3, y_amount=5)

    # # Anderx
    # parameters = InputData(dataset_file='data/anderx.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=5, u_amount=3, y_amount=5)

    # # Vlad
    # parameters = InputData(dataset_file='data/vlad.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=12, u_amount=2, y_amount=10)

    # # Eugene
    # parameters = InputData(dataset_file='data/eugene.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=5, u_amount=3, y_amount=5)

    # # basa
    # parameters = InputData(dataset_file='data/base.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=11, u_amount=2, y_amount=5)

    # # smth
    # parameters = InputData(dataset_file='data/smth.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=11, u_amount=2, y_amount=4)

    # # alex
    # parameters = InputData(dataset_file='data/alex.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=5, u_amount=3, y_amount=5)

    # # sasha
    # parameters = InputData(dataset_file='data/sasha.xlsx', x_name='X', u_name='U', y_name='Y',
    #                        x_amount=6, u_amount=2, y_amount=4)

    model = Models(parameters)
    # model.fill_null()
    # model.standardize()
    # model.normalize()
    # model.extend_data(do_reciprocal=False, do_trig=False)
    # model.df.to_excel('normal.xlsx', index=False)
    # model.df.to_excel('temp.xlsx', index=False)
    model.build_models(amount_of_cols=13, filter_u=False)
    # model.build_models(amount_of_cols=10, filter_u=False, alpha_in=[0.001, 0.001], show_results=True, show_alpha=False)
    model.show_models(output=True)
    print(model.models)
    return
    model.x_params = model.data.loc[0, model.x_cols]
    model.bounds = {'Y1': [0, 1], 'Y2': [0, 1], 'Y4': [0, 120], 'Y5': [0, 0.04],
                    'U1': [504, 800], 'U2': [84, 150], 'U3': [840, 1300]}
    for key, value in model.bounds.items():
        model.bounds[key][1] *= 1
    a, b, c = model.do_optimization_real(func='Y3', maximize=False)
    print(a)
    print(c)


if __name__ == "__main__":
    main()
