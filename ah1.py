import pandas as pd
import numpy as np
import sympy
from scipy.optimize import linprog
from PySide6.QtWidgets import QMainWindow, QApplication, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from dataclasses import dataclass


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
    def __init__(self, params: InputData, models: dict):
        # Data for model evaluation
        self.x_params = None
        self.params = params
        self.data = pd.read_excel(self.params.dataset_file)
        self.df = self.data.copy()
        self.x_cols = np.array([f'{self.params.x_name}{i}' for i in range(1, self.params.x_amount + 1)])
        self.u_cols = np.array([f'{self.params.u_name}{i}' for i in range(1, self.params.u_amount + 1)])
        self.y_cols = np.array([f'{self.params.y_name}{i}' for i in range(1, self.params.y_amount + 1)])
        self.mins = None
        self.maxes = None
        self.means = None
        self.stds = None


        self.fill_null()
        self.standardize()
        self.normalize()
        self.x_params = self.data.loc[:, self.x_cols].mean()

        # Data for optimization
        self.models = models
        self.bounds = {'Y1': [0.28, 0.68], 'Y2': [0, 1.7], 'Y4': [0, 100], 'Y5': [0, 0.04],
                       'U1': [0, 480], 'U2': [0, 75], 'U3': [0, 1800]}

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

    def do_optimization(self, func: str):
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
            self.bounds[key][-1] = self.get_norm_from_real(key, value[-1])
        for x, x_val in self.x_params.items():
            self.x_params[x] = self.get_norm_from_real(x, x_val)

        models = self.models.copy()
        for y, model in models.items():
            models[y] = sympy.sympify(model).subs([(x, x_val) for x, x_val in self.x_params.items()])

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
            upper_limit = self.bounds.get(y)[-1] - intercept
            lower_limit = intercept - self.bounds.get(y)[0]
            b.append(upper_limit)
            b.append(lower_limit)

        # Define bounds
        u_bounds = [(self.bounds.get(i)[0], self.bounds.get(i)[-1]) for i in self.u_cols]

        # Do optimization
        result = linprog(c, A_ub=A, b_ub=b, bounds=u_bounds, method='highs')

        if not result.success:
            return None, None, result.message

        # Output results
        u_found = {u: value for u, value in zip(self.u_cols, result.x)}

        res = {}
        for y, model in models.items():
            temp = sympy.sympify(model).subs([(x, x_val) for x, x_val in zip(symbols_u.values(), u_found.values())])
            real = self.get_real_from_norm(y, temp)
            res.update({y: real if real > 0 else 0})

        for u, u_value in zip(self.u_cols, u_found.values()):
            real = self.get_real_from_norm(u, u_value)
            res.update({u: real})

        return res, models, result.message


class MainWindow(QMainWindow):
    def __init__(self, file_name: str, models: Models):
        super().__init__()

        # Load the UI file
        loader = QUiLoader()
        ui_file = QFile(file_name)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        # Set window parameters
        self.resize(1040, 640)
        self.setWindowTitle("Optimize")
        self.models = models
        self.ui.button_optimize.clicked.connect(self.do_calculation)
        self.ui.button_exit.clicked.connect(self.close)

        # Set the UI as the central widget of the window
        self.setCentralWidget(self.ui)
        self.initialize()

    def initialize(self):
        for i, x_val in enumerate(self.models.x_params):
            getattr(self.ui, f'input{i}').setText(f'{x_val:.2f}')
        for i, val in enumerate(self.models.bounds.values()):
            getattr(self.ui, f'min_limit{i}').setText(f'{val[0]}')
            getattr(self.ui, f'max_limit{i}').setText(f'{val[-1]}')

    def do_calculation(self):
        for i, x in enumerate(self.models.x_params.keys()):
            self.models.x_params[x] = float(getattr(self.ui, f'input{i}').text())
        for i, val in enumerate(self.models.bounds.keys()):
            self.models.bounds[val][0] = float(getattr(self.ui, f'min_limit{i}').text())
            self.models.bounds[val][-1] = float(getattr(self.ui, f'max_limit{i}').text())

        result, models, info = self.models.do_optimization('Y3')
        msg_box = QMessageBox()
        msg_box.setText(info)
        msg_box.exec()
        if result is None:
            for i in range(8):
                getattr(self.ui, f'output{i}').setText('0')
            for i in range(5):
                getattr(self.ui, f'model{i}').setText('0')
            return
        for i, val in enumerate(models.items()):
            getattr(self.ui, f'model{i}').setText(f'{val[-1]}')
        for i, val in enumerate(result.items()):
            getattr(self.ui, f'output{i}').setText(f'{val[-1]:.2f}')


def main():
    # AC_1
    parameters = InputData(dataset_file='../data/AC_1.xlsx', x_name='X', u_name='U', y_name='Y',
                           x_amount=6, u_amount=3, y_amount=5)
    models = {
        'Y1': '-0.09705609966746001*U3*X1*X6 + 0.09546330453607535*U1*X2*X6 + 0.0746046206732374*U1*X1/X2 + 0.07460462067323785*U2*X1/X2 + -0.913007824572722*1/X2**3 + -0.09705609966746026*U1*1/X1*X6 + -0.0970560996674608*U3*1/X1*X6 + -0.10608469038965304*U3*1/X2*X3 + 0.09546330453607593*U1*1/X2*X6 + 0.0954633045360749*U2*1/X2*X6 + 0.07460462067323774*U1*1/X1/X2 + 0.0746046206732378*U2*1/X1/X2 + 1.7747432358378985',
        'Y2': '0.12498564978704829*X1**3 + 0.13818117181749678*tan(X5) + -0.059056919964059824*U1*X1*X6 + -0.059056919964060546*U3*X1*X6 + -0.28032766720493685*X2*X5 + 0.503653994637326*X5*X6 + -0.02615049279346964*U2*X1/X2 + -0.05905691996405955*U1*1/X1*X6 + -0.05905691996406048*U2*1/X1*X6 + 0.10956163921312578*U1*1/X5*X6 + -0.026150492793470197*U1*1/X1/X2 + -0.026150492793469968*U3*1/X1/X2 + 0.9533467625774998',
        'Y3': '0.07109008012624708*U2*X4*X6 + 0.07109008012624744*U3*X4*X6 + -0.0924240608512252*U2*X5*X6 + -0.09242406085122554*U3*X5*X6 + 0.14367361260662925*U1*1/exp(X5) + 0.0710900801262468*U1*1/X4*X6 + 0.07109008012624696*U2*1/X4*X6 + 0.07109008012624728*U3*1/X4*X6 + -0.09242406085122544*U1*1/X5*X6 + -0.09242406085122534*U2*1/X5*X6 + -0.09242406085122569*U3*1/X5*X6 + -0.17618341631101517*U1*1/X2/X4 + 1.0601835895680627',
        'Y4': '-0.01586948742684438*X2**3 + -0.10607229414888275*X5**3 + 0.37726292145383566*exp(X5) + 0.06922186042876136*X1*X6 + -0.010862193385225608*U1*X3*X6 + -0.010862193385225836*U2*X3*X6 + -0.010862193385223012*U3*X3*X6 + 0.0050371308437736*U1*1/X5**3 + 0.030619939649099398*1/cos(X5) + -0.010862193385222707*U1*1/X3*X6 + -0.010862193385225392*U2*1/X3*X6 + -0.010862193385223234*U3*1/X3*X6 + 0.12328847467710524',
        'Y5': '-0.4525283668197151*X2**3 + -0.05349984319604768*U1*X1*X4 + 0.3111811535618773*U3*1/X1**3 + 0.33541597374374554*U1*1/X2**3 + -0.14547122761306308*U2*1/exp(X1) + -0.14547122761306305*U3*1/exp(X1) + 0.042880243257313114*U2*1/tan(X2) + -0.01607388504192636*U3*1/tan(X5) + -0.05349984319605008*U1*1/X1*X4 + -0.05349984319605137*U2*1/X1*X4 + -0.053499843196051826*U3*1/X1*X4 + -0.022784837518347957*U1*1/X2*X6 + 2.43169979490201'}
    model = Models(params=parameters, models=models)
    app = QApplication([])
    window = MainWindow("../interface/ah1.ui", model)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
